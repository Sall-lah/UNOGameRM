"""Tests for DecisionRecord schema and DatasetWriter JSONL persistence (Task 7.5).

Verifies validation rejection of missing groups, British 'colour' key normalization,
JSONL serialization and round-trip parsing, and pilot vs main dataset routing.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

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


@pytest.fixture
def sample_valid_record() -> DecisionRecord:
    """Constructs a fully populated, valid DecisionRecord for testing."""
    return DecisionRecord(
        identity=IdentityGroup(round_id=1, turn_idx=1, starting_player="AGENT"),
        observation_ot=ObservationOtGroup(
            agent_hand=["red 5"],
            agent_hand_count=1,
            action_wild_count=0,
            top_discard="red 2",
            active_colour="red",
            opp_card_count=3,
            legal_actions=["red 5", "draw"],
        ),
        memory_plugmem=MemoryPlugMemGroup(
            retrieved_node_ids=["prop_1"],
            opponent_draw_constraints=[{"colour": "green", "origin_turn": 1, "confidence": 0.85}],
            opponent_wild_declarations=[{"colour": "blue", "origin_turn": 1}],
            relevance_scores=[0.95],
            distilled_context_tokens=16,
        ),
        search_mcts=SearchMCTSGroup(
            pruned_candidates=["red 5"],
            p_prior={"red 5": 1.0},
            simulations_N_total=20,
            visit_counts_N={"red 5": 20},
            action_values_Q={"red 5": 0.8},
            unseen_pool_size_Ut=104,
            search_wall_time_ms=12.5,
        ),
        stage1_freeze_log=Stage1FreezeLogGroup(
            locked_action_at="red 5",
            p_t_A=0.05,
            p_t_B=0.02,
            short_rationale="Play red 5.",
            freeze_timestamp_ns=1000,
            stage1_sha256="abc123sha",
        ),
        stage2_post_horizon=Stage2PostHorizonGroup(
            dispatched_command="red 5",
            auto_uno_declared=True,
            dispatch_timestamp_ns=2000,
            eval_timestamp_ns=3000,
            y_t_A_penalty_draw=0,
            A_pen_cards_drawn=0,
            A_nop_voluntary_drawn=0,
            y_t_B_opp_win=0,
            opp_turns_in_horizon=1,
            stratum_horizon=2,
            round_status="ONGOING",
            challenge_penalty=0,
        ),
        game_resources=GameResourcesGroup(
            delta_hand_size_next=-1,
            agent_hand_count_next=0,
            used_action_or_wild_at_t=0,
            action_wild_count_next=0,
            cumulative_cards_drawn_round=0,
            act_opportunities_so_far=1,
        ),
        system_reliability=SystemReliabilityGroup(
            turn_total_time_ms=45.0,
            budget_limit_ms=2000.0,
            timeout_exceeded=False,
            mcts_fallback_used=False,
            llm_call1_invalid_json=False,
            llm_call2_invalid_json=False,
            dispatch_error=False,
        ),
    )


def test_decision_record_rejects_missing_groups(sample_valid_record: DecisionRecord):
    """Verify DecisionRecord validation fails when any of the eight groups is missing."""
    data = sample_valid_record.model_dump()
    data.pop("system_reliability")

    with pytest.raises(ValidationError):
        DecisionRecord.model_validate(data)


def test_colour_key_normalization():
    """Verify MemoryPlugMemGroup normalises American 'color' to British 'colour'."""
    mem_data = {
        "retrieved_node_ids": [],
        "opponent_draw_constraints": [{"color": "green", "origin_turn": 1, "confidence": 0.85}],
        "opponent_wild_declarations": [{"color": "red", "origin_turn": 2}],
        "relevance_scores": [],
        "distilled_context_tokens": 0,
    }
    group = MemoryPlugMemGroup.model_validate(mem_data)

    assert group.opponent_draw_constraints[0]["colour"] == "green"
    assert "color" not in group.opponent_draw_constraints[0]
    assert group.opponent_wild_declarations[0]["colour"] == "red"
    assert "color" not in group.opponent_wild_declarations[0]


def test_dataset_writer_jsonl_round_trip(tmp_path: Path, sample_valid_record: DecisionRecord):
    """Verify DatasetWriter writes valid JSON lines that parse back into identical records."""
    writer = DatasetWriter(output_root=str(tmp_path), dataset_split="pilot")
    writer.append(sample_valid_record)

    records = writer.read_all()
    assert len(records) == 1
    loaded = records[0]

    assert loaded.identity.round_id == sample_valid_record.identity.round_id
    assert loaded.stage1_freeze_log.locked_action_at == "red 5"
    assert loaded.stage2_post_horizon.stratum_horizon == 2


def test_dataset_routing_pilot_vs_main(tmp_path: Path, sample_valid_record: DecisionRecord):
    """Verify pilot and main experiment files are written to separate subdirectories."""
    pilot_writer = DatasetWriter(output_root=str(tmp_path), dataset_split="pilot")
    pilot_writer.append(sample_valid_record)

    main_writer = DatasetWriter(output_root=str(tmp_path), dataset_split="main")
    main_writer.append(sample_valid_record)

    pilot_file = tmp_path / "pilot" / "dataset_drt.jsonl"
    main_file = tmp_path / "main_experiment" / "dataset_drt.jsonl"

    assert pilot_file.exists()
    assert main_file.exists()
    assert pilot_file != main_file
