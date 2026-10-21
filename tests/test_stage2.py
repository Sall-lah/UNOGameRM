"""Tests for Stage 2 Evaluator (Task 7.4).

Verifies monotonic timestamp ordering invariant enforcement (freeze < dispatch <= eval),
cryptographic tamper detection re-hashing, and game resource delta calculations.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.env.barrier import LegalObservation
from src.audit.dataset import (
    IdentityGroup,
    MemoryPlugMemGroup,
    ObservationOtGroup,
    SearchMCTSGroup,
    Stage1FreezeLogGroup,
    SystemReliabilityGroup,
)
from src.audit.horizon import HorizonEvaluation
from src.audit.stage1 import Stage1Writer
from src.audit.stage2 import (
    Stage2Evaluator,
    TamperError,
    TimestampInvariantError,
)


@pytest.fixture
def base_evaluator_data(tmp_path: Path):
    """Provides valid Stage 1 freeze file and metadata for Stage 2 evaluator tests."""
    writer = Stage1Writer(freeze_dir=tmp_path)
    res = writer.freeze(
        round_id=1,
        turn_idx=1,
        observation_ot={"agent_hand": ["red 5", "blue 3"], "agent_hand_count": 2, "action_wild_count": 0, "top_discard": "red 1", "active_colour": "red", "opp_card_count": 5, "legal_actions": ["red 5", "draw"]},
        memory_plugmem={"retrieved_node_ids": [], "opponent_draw_constraints": [], "opponent_wild_declarations": [], "relevance_scores": [], "distilled_context_tokens": 0},
        p_prior={"red 5": 1.0},
        visit_counts_N={"red 5": 10},
        action_values_Q={"red 5": 0.5},
        locked_action_at="red 5",
        p_t_A=0.1,
        p_t_B=0.05,
        short_rationale="Play red 5.",
    )

    identity = IdentityGroup(round_id=1, turn_idx=1, starting_player="AGENT")
    obs_ot = ObservationOtGroup(
        agent_hand=["red 5", "blue 3"],
        agent_hand_count=2,
        action_wild_count=0,
        top_discard="red 1",
        active_colour="red",
        opp_card_count=5,
        legal_actions=["red 5", "draw"],
    )
    memory = MemoryPlugMemGroup(
        retrieved_node_ids=[],
        opponent_draw_constraints=[],
        opponent_wild_declarations=[],
        relevance_scores=[],
        distilled_context_tokens=0,
    )
    search = SearchMCTSGroup(
        pruned_candidates=["red 5"],
        p_prior={"red 5": 1.0},
        simulations_N_total=10,
        visit_counts_N={"red 5": 10},
        action_values_Q={"red 5": 0.5},
        unseen_pool_size_Ut=100,
        search_wall_time_ms=50.0,
    )
    freeze_log = Stage1FreezeLogGroup(
        locked_action_at="red 5",
        p_t_A=0.1,
        p_t_B=0.05,
        short_rationale="Play red 5.",
        freeze_timestamp_ns=res.freeze_timestamp_ns,
        stage1_sha256=res.stage1_sha256,
    )
    reliability = SystemReliabilityGroup(
        turn_total_time_ms=100.0,
        budget_limit_ms=2000.0,
        timeout_exceeded=False,
        mcts_fallback_used=False,
        llm_call1_invalid_json=False,
        llm_call2_invalid_json=False,
        dispatch_error=False,
    )

    horizon_eval = HorizonEvaluation(
        y_t_A_penalty_draw=0,
        A_pen_cards_drawn=0,
        A_nop_voluntary_drawn=0,
        y_t_B_opp_win=0,
        opp_turns_in_horizon=1,
        stratum_horizon=2,
        round_status="ONGOING",
    )

    return {
        "freeze_res": res,
        "identity": identity,
        "obs_ot": obs_ot,
        "memory": memory,
        "search": search,
        "freeze_log": freeze_log,
        "reliability": reliability,
        "horizon_eval": horizon_eval,
    }


def test_timestamp_invariant_passes_in_order_and_rejects_out_of_order(base_evaluator_data):
    """Verify freeze < dispatch <= eval invariant enforcement."""
    evaluator = Stage2Evaluator(strict_invariant=True)
    d = base_evaluator_data
    freeze_ns = d["freeze_res"].freeze_timestamp_ns

    # Valid: freeze < dispatch <= eval
    dispatch_ns = freeze_ns + 1000
    eval_ns = dispatch_ns + 500
    rec = evaluator.evaluate(
        stage1_file_path=d["freeze_res"].file_path,
        stage1_sha256=d["freeze_res"].stage1_sha256,
        freeze_timestamp_ns=freeze_ns,
        dispatch_timestamp_ns=dispatch_ns,
        horizon_eval=d["horizon_eval"],
        dispatched_command="red 5",
        auto_uno_declared=False,
        identity_data=d["identity"],
        observation_ot_data=d["obs_ot"],
        memory_plugmem_data=d["memory"],
        search_mcts_data=d["search"],
        stage1_freeze_log_data=d["freeze_log"],
        system_reliability_data=d["reliability"],
        cumulative_cards_drawn_round=0,
        eval_timestamp_ns=eval_ns,
    )
    assert rec.stage2_post_horizon.dispatch_timestamp_ns == dispatch_ns

    # Invalid: dispatch <= freeze
    with pytest.raises(TimestampInvariantError):
        evaluator.evaluate(
            stage1_file_path=d["freeze_res"].file_path,
            stage1_sha256=d["freeze_res"].stage1_sha256,
            freeze_timestamp_ns=freeze_ns,
            dispatch_timestamp_ns=freeze_ns,  # Not strictly greater
            horizon_eval=d["horizon_eval"],
            dispatched_command="red 5",
            auto_uno_declared=False,
            identity_data=d["identity"],
            observation_ot_data=d["obs_ot"],
            memory_plugmem_data=d["memory"],
            search_mcts_data=d["search"],
            stage1_freeze_log_data=d["freeze_log"],
            system_reliability_data=d["reliability"],
            cumulative_cards_drawn_round=0,
            eval_timestamp_ns=eval_ns,
        )


def test_tamper_detection_rehash_detects_modified_on_disk_bytes(base_evaluator_data):
    """Verify modifying Stage 1 file bytes on disk raises TamperError."""
    evaluator = Stage2Evaluator()
    d = base_evaluator_data
    file_path = d["freeze_res"].file_path

    # Modify file bytes to simulate post-hoc tampering
    tampered_bytes = d["freeze_res"].canonical_bytes.replace(b"red 5", b"blue 9")
    file_path.write_bytes(tampered_bytes)

    with pytest.raises(TamperError, match="integrity violation"):
        evaluator.evaluate(
            stage1_file_path=file_path,
            stage1_sha256=d["freeze_res"].stage1_sha256,  # Original hash
            freeze_timestamp_ns=d["freeze_res"].freeze_timestamp_ns,
            dispatch_timestamp_ns=d["freeze_res"].freeze_timestamp_ns + 1000,
            horizon_eval=d["horizon_eval"],
            dispatched_command="red 5",
            auto_uno_declared=False,
            identity_data=d["identity"],
            observation_ot_data=d["obs_ot"],
            memory_plugmem_data=d["memory"],
            search_mcts_data=d["search"],
            stage1_freeze_log_data=d["freeze_log"],
            system_reliability_data=d["reliability"],
            cumulative_cards_drawn_round=0,
            eval_timestamp_ns=d["freeze_res"].freeze_timestamp_ns + 2000,
        )


def test_game_resource_deltas_with_next_obs_and_terminal_boundary(base_evaluator_data):
    """Verify delta_hand_size_next and action_wild_count_next calculations."""
    evaluator = Stage2Evaluator()
    d = base_evaluator_data
    freeze_ns = d["freeze_res"].freeze_timestamp_ns

    # Case 1: Next observation present
    next_obs = LegalObservation(
        round_id=1,
        turn_idx=2,
        agent_hand=("blue 3",),  # Dropped from 2 to 1
        top_discard="red 5",
        active_colour="red",
        opp_card_count=4,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    rec1 = evaluator.evaluate(
        stage1_file_path=d["freeze_res"].file_path,
        stage1_sha256=d["freeze_res"].stage1_sha256,
        freeze_timestamp_ns=freeze_ns,
        dispatch_timestamp_ns=freeze_ns + 100,
        horizon_eval=d["horizon_eval"],
        dispatched_command="red 5",
        auto_uno_declared=False,
        identity_data=d["identity"],
        observation_ot_data=d["obs_ot"],
        memory_plugmem_data=d["memory"],
        search_mcts_data=d["search"],
        stage1_freeze_log_data=d["freeze_log"],
        system_reliability_data=d["reliability"],
        cumulative_cards_drawn_round=0,
        eval_timestamp_ns=freeze_ns + 200,
        next_obs=next_obs,
    )
    # Agent hand went from 2 to 1 -> delta = -1
    assert rec1.game_resources.delta_hand_size_next == -1
    assert rec1.game_resources.agent_hand_count_next == 1

    # Case 2: Round ended terminal (agent emptied hand to win)
    terminal_eval = HorizonEvaluation(
        y_t_A_penalty_draw=0,
        A_pen_cards_drawn=0,
        A_nop_voluntary_drawn=0,
        y_t_B_opp_win=0,
        opp_turns_in_horizon=0,
        stratum_horizon=1,
        round_status="TERMINATED",
    )
    rec_term = evaluator.evaluate(
        stage1_file_path=d["freeze_res"].file_path,
        stage1_sha256=d["freeze_res"].stage1_sha256,
        freeze_timestamp_ns=freeze_ns,
        dispatch_timestamp_ns=freeze_ns + 100,
        horizon_eval=terminal_eval,
        dispatched_command="red 5",
        auto_uno_declared=False,
        identity_data=d["identity"],
        observation_ot_data=d["obs_ot"],
        memory_plugmem_data=d["memory"],
        search_mcts_data=d["search"],
        stage1_freeze_log_data=d["freeze_log"],
        system_reliability_data=d["reliability"],
        cumulative_cards_drawn_round=0,
        eval_timestamp_ns=freeze_ns + 200,
        terminal_agent_hand_count=0,
    )
    assert rec_term.game_resources.agent_hand_count_next == 0
    assert rec_term.game_resources.delta_hand_size_next == -2
