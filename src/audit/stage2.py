"""Stage 2 Post-Horizon Evaluator and DecisionRecord assembler.

Enforces the hard monotonic timestamp ordering invariant (freeze < dispatch <= eval),
performs cryptographic tamper-detection re-hashing of Stage 1 freeze records, computes
game resource trajectories at t+1, and compiles the final verified DecisionRecord.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from src.core.cards import parse_card_string
from src.env.barrier import LegalObservation
from src.audit.dataset import (
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
from src.audit.horizon import HorizonEvaluation
from src.audit.stage1 import Stage1FreezeResult


class TimestampInvariantError(AssertionError):
    """Raised when the hard invariant freeze_ns < dispatch_ns <= eval_ns is violated."""


class TamperError(ValueError):
    """Raised when persisted Stage 1 bytes fail cryptographic SHA-256 integrity verification."""


class Stage2Evaluator:
    """Evaluates post-horizon state, verifies Stage 1 integrity, and builds DecisionRecords."""

    def __init__(self, strict_invariant: bool = True) -> None:
        """Initializes evaluator with strict invariant enforcement settings."""
        self.strict_invariant = strict_invariant

    def evaluate(
        self,
        stage1_file_path: Union[str, Path],
        stage1_sha256: str,
        freeze_timestamp_ns: int,
        dispatch_timestamp_ns: int,
        horizon_eval: HorizonEvaluation,
        dispatched_command: str,
        auto_uno_declared: bool,
        identity_data: Union[IdentityGroup, Mapping[str, Any]],
        observation_ot_data: Union[ObservationOtGroup, Mapping[str, Any]],
        memory_plugmem_data: Union[MemoryPlugMemGroup, Mapping[str, Any]],
        search_mcts_data: Union[SearchMCTSGroup, Mapping[str, Any]],
        stage1_freeze_log_data: Union[Stage1FreezeLogGroup, Mapping[str, Any]],
        system_reliability_data: Union[SystemReliabilityGroup, Mapping[str, Any]],
        cumulative_cards_drawn_round: int,
        eval_timestamp_ns: Optional[int] = None,
        next_obs: Optional[LegalObservation] = None,
        terminal_agent_hand_count: Optional[int] = None,
    ) -> DecisionRecord:
        """Verifies integrity, computes t+1 resource deltas, and constructs DecisionRecord.

        Why: Ensures no uncommitted predictions or post-hoc altered records enter the
        evaluation dataset, preserving strict methodological validity.
        """
        eval_timestamp = eval_timestamp_ns if eval_timestamp_ns is not None else time.monotonic_ns()

        # 1. Enforce Hard Monotonic Timestamp Ordering Invariant: freeze < dispatch <= eval
        if self.strict_invariant:
            if not (freeze_timestamp_ns < dispatch_timestamp_ns <= eval_timestamp):
                raise TimestampInvariantError(
                    f"Timestamp invariant violation: expected freeze ({freeze_timestamp_ns}) < "
                    f"dispatch ({dispatch_timestamp_ns}) <= eval ({eval_timestamp})"
                )

        # 2. Tamper-Detection: Re-read persisted Stage 1 bytes and verify SHA-256
        path = Path(stage1_file_path)
        if not path.exists():
            raise TamperError(f"Stage 1 freeze record not found on disk at {path}")

        persisted_bytes = path.read_bytes()
        recomputed_sha256 = hashlib.sha256(persisted_bytes).hexdigest()

        if recomputed_sha256 != stage1_sha256:
            raise TamperError(
                f"Stage 1 integrity violation: expected hash {stage1_sha256}, got {recomputed_sha256}"
            )

        # 3. Compute Game Resource Deltas at t+1
        obs_ot = (
            observation_ot_data
            if isinstance(observation_ot_data, ObservationOtGroup)
            else ObservationOtGroup.model_validate(observation_ot_data)
        )
        agent_hand_count_t = obs_ot.agent_hand_count

        if next_obs is not None:
            agent_hand_count_next = len(next_obs.agent_hand)
            action_wild_count_next = sum(
                1 for c_str in next_obs.agent_hand if parse_card_string(c_str).is_action_or_wild
            )
            delta_hand_size_next = agent_hand_count_next - agent_hand_count_t
        elif terminal_agent_hand_count is not None:
            agent_hand_count_next = terminal_agent_hand_count
            action_wild_count_next = 0
            delta_hand_size_next = agent_hand_count_next - agent_hand_count_t
        else:
            # Round terminated without next observation
            if horizon_eval.y_t_B_opp_win == 0 and horizon_eval.round_status == "TERMINATED":
                agent_hand_count_next = 0
            else:
                agent_hand_count_next = max(
                    0, agent_hand_count_t - 1 + horizon_eval.A_pen_cards_drawn + horizon_eval.A_nop_voluntary_drawn
                )
            action_wild_count_next = 0
            delta_hand_size_next = agent_hand_count_next - agent_hand_count_t

        used_action_or_wild = 0
        if dispatched_command not in ("draw", "challenge"):
            try:
                card = parse_card_string(dispatched_command)
                if card.is_action_or_wild:
                    used_action_or_wild = 1
            except ValueError:
                pass

        identity_group = (
            identity_data
            if isinstance(identity_data, IdentityGroup)
            else IdentityGroup.model_validate(identity_data)
        )

        game_resources = GameResourcesGroup(
            delta_hand_size_next=delta_hand_size_next,
            agent_hand_count_next=agent_hand_count_next,
            used_action_or_wild_at_t=used_action_or_wild,
            action_wild_count_next=action_wild_count_next,
            cumulative_cards_drawn_round=cumulative_cards_drawn_round,
            act_opportunities_so_far=identity_group.turn_idx,
        )

        stage2_post_horizon = Stage2PostHorizonGroup(
            dispatched_command=dispatched_command,
            auto_uno_declared=auto_uno_declared,
            dispatch_timestamp_ns=dispatch_timestamp_ns,
            eval_timestamp_ns=eval_timestamp,
            y_t_A_penalty_draw=horizon_eval.y_t_A_penalty_draw,
            A_pen_cards_drawn=horizon_eval.A_pen_cards_drawn,
            A_nop_voluntary_drawn=horizon_eval.A_nop_voluntary_drawn,
            y_t_B_opp_win=horizon_eval.y_t_B_opp_win,
            opp_turns_in_horizon=horizon_eval.opp_turns_in_horizon,
            stratum_horizon=horizon_eval.stratum_horizon,
            round_status=horizon_eval.round_status,
            challenge_penalty=horizon_eval.challenge_penalty,
        )

        return DecisionRecord(
            identity=identity_group,
            observation_ot=obs_ot,
            memory_plugmem=(
                memory_plugmem_data
                if isinstance(memory_plugmem_data, MemoryPlugMemGroup)
                else MemoryPlugMemGroup.model_validate(memory_plugmem_data)
            ),
            search_mcts=(
                search_mcts_data
                if isinstance(search_mcts_data, SearchMCTSGroup)
                else SearchMCTSGroup.model_validate(search_mcts_data)
            ),
            stage1_freeze_log=(
                stage1_freeze_log_data
                if isinstance(stage1_freeze_log_data, Stage1FreezeLogGroup)
                else Stage1FreezeLogGroup.model_validate(stage1_freeze_log_data)
            ),
            stage2_post_horizon=stage2_post_horizon,
            game_resources=game_resources,
            system_reliability=(
                system_reliability_data
                if isinstance(system_reliability_data, SystemReliabilityGroup)
                else SystemReliabilityGroup.model_validate(system_reliability_data)
            ),
        )
