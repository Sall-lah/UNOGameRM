"""Master Agent Harness orchestrating Zone 2 decision pipeline and Zone 3 audit logging.

Executes the per-turn sequence: PlugMem retrieve -> LLM Call #1 (prior/pruning) ->
MARS MCTS lookahead -> Most-Visited commit -> LLM Call #2 (risk) -> Stage 1 freeze ->
server action dispatch -> HorizonMonitor -> Stage 2 evaluation -> PlugMem update.
"""

from __future__ import annotations

import random
import time
from typing import Any, Optional, Sequence, Union

from src.config import AppConfig, load_config
from src.core.cards import (
    COLOURS,
    UNOAction,
    parse_card_string,
)
from src.env.barrier import ClientServerBarrier, LegalObservation
from src.env.uno_engine import ModifiedUNOServerPlatform
from src.llm.call1_prior import Call1Result, run_call1
from src.llm.call2_risk import Call2Result, run_call2
from src.llm.client import FakeLLMClient, LLMClient
from src.memory.graph import PlugMemConstraint
from src.memory.plugmem import UNOPlugMem
from src.search.mars_mcts import MARSBudgetAwareMCTS, SearchResult, action_from_id
from src.audit.dataset import (
    DatasetWriter,
    DecisionRecord,
    GameResourcesGroup,
    IdentityGroup,
    MemoryPlugMemGroup,
    ObservationOtGroup,
    SearchMCTSGroup,
    Stage1FreezeLogGroup,
    Stage2PostHorizonGroup,
    SystemReliabilityGroup,
)
from src.audit.horizon import HorizonEvaluation, HorizonMonitor
from src.audit.stage1 import Stage1FreezeResult, Stage1Writer
from src.audit.stage2 import Stage2Evaluator, TimestampInvariantError


class AgentHarness:
    """Orchestrates per-turn agent decisions, budget accounting, and audit logging."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        server_platform: Optional[ModifiedUNOServerPlatform] = None,
        plugmem: Optional[UNOPlugMem] = None,
        mcts: Optional[MARSBudgetAwareMCTS] = None,
        llm_client: Optional[LLMClient] = None,
        stage1_writer: Optional[Stage1Writer] = None,
        horizon_monitor: Optional[HorizonMonitor] = None,
        stage2_evaluator: Optional[Stage2Evaluator] = None,
        dataset_writer: Optional[DatasetWriter] = None,
        starting_player: str = "AGENT",
        rng: Optional[random.Random] = None,
    ) -> None:
        """Initializes harness subsystems and dependency injection references."""
        self.config = config or load_config()
        self.server_platform = server_platform or ModifiedUNOServerPlatform()
        self.plugmem = plugmem or UNOPlugMem(config=self.config.memory)
        self.rng = rng or random.Random(42)
        self.mcts = mcts or MARSBudgetAwareMCTS(config=self.config.mcts, rng=self.rng)
        self.llm_client = llm_client or FakeLLMClient()

        self.stage1_writer = stage1_writer or Stage1Writer(
            freeze_dir=f"{self.config.audit.output_root}/stage1_freeze"
        )
        self.horizon_monitor = horizon_monitor or HorizonMonitor()
        self.stage2_evaluator = stage2_evaluator or Stage2Evaluator(
            strict_invariant=self.config.audit.strict_invariant
        )
        self.dataset_writer = dataset_writer or DatasetWriter(
            output_root=self.config.audit.output_root,
            dataset_split=self.config.audit.dataset_split,
        )
        self.starting_player = starting_player

        self.cumulative_cards_drawn_round: int = 0
        self.last_dispatch_result: Optional[dict[str, Any]] = None
        self.last_next_obs: Optional[LegalObservation] = None
        self.last_record: Optional[DecisionRecord] = None

    def on_agent_turn(self, obs: LegalObservation) -> DecisionRecord:
        """Executes the full Zone 2 pipeline and Zone 3 freeze/dispatch/evaluation flow.

        Why: Coordinates retrieval, prior generation, bounded search, action commitment,
        risk prediction, tamper-evident freeze, server dispatch, and dataset persistence
        within the strict turn time budget.
        """
        turn_start_s = time.perf_counter()

        t_turn_s = self.config.budget.t_turn_s
        delta_buffer_s = self.config.budget.delta_buffer_s
        llm2_reserve_s = self.config.budget.llm2_reserve_s
        effective_budget_s = max(0.01, t_turn_s - delta_buffer_s)
        effective_deadline_s = turn_start_s + effective_budget_s

        # Detect non-play decisions
        # 1. First-turn Wild colour choice (exactly the four colour declarations)
        is_first_turn_wild_choice = (
            len(obs.legal_actions) == 4
            and all(a.startswith("WILD card, set to") for a in obs.legal_actions)
        )

        # 2. CPU Wild +4 response decision (accept vs challenge)
        is_d4_response_decision = set(obs.legal_actions) == {"draw", "challenge"}

        # --- Phase 1: PlugMem Retrieve ---
        t_retrieve_start = time.perf_counter()
        retrieval = self.plugmem.retrieve(obs)
        distilled_context = retrieval.distilled_context
        retrieval_ms = (time.perf_counter() - t_retrieve_start) * 1000.0

        memory_constraints = self.plugmem.get_determinization_constraints()

        # --- Phase 2: LLM Call #1 Prior & Pruning ---
        t_call1_start = time.perf_counter()
        if is_d4_response_decision:
            # Deterministic policy override for D4 challenge response
            chosen_d4 = "draw" if self.config.agent.d4_response == "accept" else "challenge"
            call1_res = Call1Result(
                pruned_actions=[chosen_d4],
                action_priors={chosen_d4: 1.0},
                pruning_note=f"Policy-fixed D4 response: {chosen_d4}",
                invalid_json_call1=False,
                latency_ms=0.0,
            )
        else:
            time_left_for_call1 = max(
                0.01, effective_deadline_s - time.perf_counter() - llm2_reserve_s
            )
            call1_timeout = min(self.config.llm.call1_timeout_s, time_left_for_call1)
            call1_res = run_call1(
                obs=obs,
                distilled_context=distilled_context,
                client=self.llm_client,
                max_candidates=self.config.llm.max_candidates,
                timeout_s=call1_timeout,
            )
        call1_ms = (time.perf_counter() - t_call1_start) * 1000.0

        # --- Phase 3: MARS MCTS Lookahead Search ---
        t_mcts_start = time.perf_counter()
        if is_d4_response_decision:
            # Policy-fixed D4 response directly commits chosen action
            locked_action = action_from_id(chosen_d4)
            mcts_result = SearchResult(
                locked_action=locked_action,
                visit_counts_N={chosen_d4: 1},
                action_values_Q={chosen_d4: 0.0},
                simulations_N_total=1,
                unseen_pool_size_Ut=0,
                search_wall_time_ms=0.0,
                fallback_used=False,
                visit_shares={chosen_d4: 1.0},
                mean_raw_utilities={chosen_d4: 0.0},
                rollout_apen_rates={chosen_d4: 0.0},
                rollout_opp_win_rates={chosen_d4: 0.0},
            )
        else:
            elapsed_before_mcts = time.perf_counter() - turn_start_s
            remaining_mcts_budget_s = effective_budget_s - elapsed_before_mcts - llm2_reserve_s
            mcts_deadline = time.perf_counter() + max(0.0, remaining_mcts_budget_s)

            mcts_result = self.mcts.search(
                obs=obs,
                priors=call1_res.action_priors,
                memory_constraints=memory_constraints,
                deadline=mcts_deadline,
            )
        mcts_ms = (time.perf_counter() - t_mcts_start) * 1000.0

        # --- Phase 4: Most-Visited Action Commitment ---
        t_commit_start = time.perf_counter()
        locked_action: UNOAction = mcts_result.locked_action
        locked_action_id: str = locked_action.action_id
        commit_ms = (time.perf_counter() - t_commit_start) * 1000.0

        # --- Phase 5: LLM Call #2 Risk Elicitation & Rationale ---
        t_call2_start = time.perf_counter()
        elapsed_before_call2 = time.perf_counter() - turn_start_s
        remaining_call2_budget_s = max(0.001, effective_budget_s - elapsed_before_call2)
        call2_timeout = min(self.config.llm.call2_timeout_s, remaining_call2_budget_s)

        call2_res: Call2Result = run_call2(
            locked_action=locked_action,
            obs=obs,
            distilled_context=distilled_context,
            mcts_stats=mcts_result,
            client=self.llm_client,
            timeout_s=call2_timeout,
        )
        call2_ms = (time.perf_counter() - t_call2_start) * 1000.0

        # Budget accounting
        turn_total_time_ms = (time.perf_counter() - turn_start_s) * 1000.0
        timeout_exceeded = (turn_total_time_ms / 1000.0) >= effective_budget_s

        # Prepare audit dictionaries
        action_wild_held = sum(
            1 for c_str in obs.agent_hand if parse_card_string(c_str).is_action_or_wild
        )
        obs_ot_dict = {
            "agent_hand": list(obs.agent_hand),
            "agent_hand_count": len(obs.agent_hand),
            "action_wild_count": action_wild_held,
            "top_discard": obs.top_discard,
            "active_colour": obs.active_colour,
            "opp_card_count": obs.opp_card_count,
            "legal_actions": list(obs.legal_actions),
        }

        memory_dict = {
            "retrieved_node_ids": list(retrieval.retrieved_node_ids),
            "opponent_draw_constraints": [
                {"colour": col, "confidence": conf}
                for col, conf in retrieval.active_opp_lack_colours.items()
            ],
            "opponent_wild_declarations": [
                {"colour": col, "origin_turn": turn}
                for turn, col in retrieval.active_opp_wild_colours
            ],
            "relevance_scores": list(retrieval.relevance_scores),
            "distilled_context_tokens": retrieval.token_estimate,
        }

        # --- Zone 3: Stage 1 Pre-Execution Freeze ---
        freeze_res: Stage1FreezeResult = self.stage1_writer.freeze(
            round_id=obs.round_id,
            turn_idx=obs.turn_idx,
            observation_ot=obs_ot_dict,
            memory_plugmem=memory_dict,
            p_prior=call1_res.action_priors,
            visit_counts_N=mcts_result.visit_counts_N,
            action_values_Q=mcts_result.action_values_Q,
            locked_action_at=locked_action_id,
            p_t_A=call2_res.p_t_A,
            p_t_B=call2_res.p_t_B,
            short_rationale=call2_res.short_rationale,
        )

        t_freeze_ns = freeze_res.freeze_timestamp_ns

        # --- Zone 3: Server Action Dispatch ---
        t_dispatch_ns = time.monotonic_ns()
        dispatch_error = False

        try:
            dispatch_result = self.server_platform.dispatch_agent_action_and_observe_horizon(
                locked_action_id
            )
        except Exception:
            dispatch_error = True
            raise

        self.last_dispatch_result = dispatch_result
        horizon_events = dispatch_result.get("horizon_events", ())
        next_raw_state = dispatch_result.get("next_state")
        round_over = dispatch_result.get("round_over", False)

        # Monotonic timestamp ordering assertion
        t_eval_ns = time.monotonic_ns()
        if self.config.audit.strict_invariant:
            if not (t_freeze_ns < t_dispatch_ns <= t_eval_ns):
                raise TimestampInvariantError(
                    f"Timestamp invariant violation: freeze={t_freeze_ns} < "
                    f"dispatch={t_dispatch_ns} <= eval={t_eval_ns}"
                )

        # --- Zone 3: Horizon Evaluation & Post-Horizon Stage 2 ---
        agent_played_retention = (
            locked_action.card.is_turn_retention_in_2p if locked_action.card else False
        )
        horizon_eval: HorizonEvaluation = self.horizon_monitor.process_events(
            events=horizon_events,
            agent_played_retention_card=agent_played_retention,
        )

        # Track cumulative cards drawn by agent
        self.cumulative_cards_drawn_round += (
            horizon_eval.A_pen_cards_drawn + horizon_eval.A_nop_voluntary_drawn
        )

        # Extract next LegalObservation if round continues
        next_obs: Optional[LegalObservation] = None
        if next_raw_state and not round_over:
            next_obs = ClientServerBarrier.extract_observation(next_raw_state)
        self.last_next_obs = next_obs

        # Compile data groups for DecisionRecord
        identity_group = IdentityGroup(
            round_id=obs.round_id,
            turn_idx=obs.turn_idx,
            starting_player=self.starting_player,
            config_version="v1.0-frozen",
            first_turn_wild_colour=is_first_turn_wild_choice,
        )

        search_mcts_group = SearchMCTSGroup(
            pruned_candidates=call1_res.pruned_actions,
            p_prior=call1_res.action_priors,
            simulations_N_total=mcts_result.simulations_N_total,
            visit_counts_N=mcts_result.visit_counts_N,
            action_values_Q=mcts_result.action_values_Q,
            unseen_pool_size_Ut=mcts_result.unseen_pool_size_Ut,
            search_wall_time_ms=mcts_result.search_wall_time_ms,
        )

        freeze_log_group = Stage1FreezeLogGroup(
            locked_action_at=locked_action_id,
            p_t_A=call2_res.p_t_A,
            p_t_B=call2_res.p_t_B,
            short_rationale=call2_res.short_rationale,
            freeze_timestamp_ns=t_freeze_ns,
            stage1_sha256=freeze_res.stage1_sha256,
        )

        reliability_group = SystemReliabilityGroup(
            turn_total_time_ms=turn_total_time_ms,
            budget_limit_ms=effective_budget_s * 1000.0,
            timeout_exceeded=timeout_exceeded,
            mcts_fallback_used=mcts_result.fallback_used,
            llm_call1_invalid_json=call1_res.invalid_json_call1,
            llm_call2_invalid_json=call2_res.fallback_triggered_call2,
            dispatch_error=dispatch_error,
        )

        auto_uno = len(obs.agent_hand) == 2 and locked_action_id != "draw"

        decision_record = self.stage2_evaluator.evaluate(
            stage1_file_path=freeze_res.file_path,
            stage1_sha256=freeze_res.stage1_sha256,
            freeze_timestamp_ns=t_freeze_ns,
            dispatch_timestamp_ns=t_dispatch_ns,
            horizon_eval=horizon_eval,
            dispatched_command=locked_action_id,
            auto_uno_declared=auto_uno,
            identity_data=identity_group,
            observation_ot_data=obs_ot_dict,
            memory_plugmem_data=memory_dict,
            search_mcts_data=search_mcts_group,
            stage1_freeze_log_data=freeze_log_group,
            system_reliability_data=reliability_group,
            cumulative_cards_drawn_round=self.cumulative_cards_drawn_round,
            eval_timestamp_ns=t_eval_ns,
            next_obs=next_obs,
        )

        # --- Zone 3: PlugMem Update ---
        self.plugmem.update(
            obs=obs,
            horizon_events=horizon_events,
            locked_action=locked_action_id,
            t=obs.turn_idx,
        )

        # Emit d_{r,t} record to dataset file
        self.dataset_writer.append(decision_record)
        self.last_record = decision_record

        return decision_record
