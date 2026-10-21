"""Pytest fixtures and synthetic record generators for analysis tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import pytest

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def make_synthetic_record_dict(
    round_id: int = 1,
    turn_idx: int = 1,
    p_t_A: float = 0.2,
    p_t_B: float = 0.1,
    y_t_A: int = 0,
    y_t_B: int = 0,
    stratum: int = 2,
    timeout: bool = False,
    mcts_fallback: bool = False,
    locked_action: str = "red 5",
    p_prior: dict[str, float] | None = None,
    visit_counts: dict[str, int] | None = None,
    rationale: str = "Playing red 5 to keep control.",
    agent_hand: list[str] | None = None,
    opp_draw_constraints: list[dict[str, Any]] | None = None,
    opp_wild_declarations: list[dict[str, Any]] | None = None,
    delta_hand_size_next: int = -1,
    used_action_or_wild: int = 0,
    cumulative_cards_drawn: int = 0,
    opp_card_count: int = 5,
    top_discard: str = "red 1",
) -> dict[str, Any]:
    """Generates a schema-valid doc-06 §3 decision record dictionary."""
    if p_prior is None:
        p_prior = {locked_action: 0.7, "draw": 0.3}
    if visit_counts is None:
        visit_counts = {locked_action: 50, "draw": 10}
    if agent_hand is None:
        agent_hand = [locked_action, "blue 3"]
    if opp_draw_constraints is None:
        opp_draw_constraints = []
    if opp_wild_declarations is None:
        opp_wild_declarations = []

    return {
        "identity": {
            "round_id": round_id,
            "turn_idx": turn_idx,
            "starting_player": "AGENT",
            "config_version": "v1.0-frozen",
            "first_turn_wild_colour": False,
        },
        "observation_ot": {
            "agent_hand": agent_hand,
            "agent_hand_count": len(agent_hand),
            "action_wild_count": 0,
            "top_discard": top_discard,
            "active_colour": "red",
            "opp_card_count": opp_card_count,
            "legal_actions": list(p_prior.keys()),
        },
        "memory_plugmem": {
            "retrieved_node_ids": ["node_1"],
            "opponent_draw_constraints": opp_draw_constraints,
            "opponent_wild_declarations": opp_wild_declarations,
            "relevance_scores": [0.85],
            "distilled_context_tokens": 120,
        },
        "search_mcts": {
            "pruned_candidates": list(p_prior.keys()),
            "p_prior": p_prior,
            "simulations_N_total": sum(visit_counts.values()),
            "visit_counts_N": visit_counts,
            "action_values_Q": {k: 0.6 for k in visit_counts.keys()},
            "unseen_pool_size_Ut": 40,
            "search_wall_time_ms": 45.2,
        },
        "stage1_freeze_log": {
            "locked_action_at": locked_action,
            "p_t_A": p_t_A,
            "p_t_B": p_t_B,
            "short_rationale": rationale,
            "freeze_timestamp_ns": 1000000000,
            "stage1_sha256": "fake_sha256_for_test",
        },
        "stage2_post_horizon": {
            "dispatched_command": locked_action,
            "auto_uno_declared": False,
            "dispatch_timestamp_ns": 1000005000,
            "eval_timestamp_ns": 1000010000,
            "y_t_A_penalty_draw": y_t_A,
            "A_pen_cards_drawn": 2 if y_t_A else 0,
            "A_nop_voluntary_drawn": 0,
            "y_t_B_opp_win": y_t_B,
            "opp_turns_in_horizon": 1,
            "stratum_horizon": stratum,
            "round_status": "TERMINATED" if y_t_B else "IN_PROGRESS",
            "challenge_penalty": 0,
        },
        "game_resources": {
            "delta_hand_size_next": delta_hand_size_next,
            "agent_hand_count_next": max(0, len(agent_hand) + delta_hand_size_next),
            "used_action_or_wild_at_t": used_action_or_wild,
            "action_wild_count_next": 0,
            "cumulative_cards_drawn_round": cumulative_cards_drawn,
            "act_opportunities_so_far": turn_idx,
        },
        "system_reliability": {
            "turn_total_time_ms": 120.5,
            "budget_limit_ms": 5000.0,
            "timeout_exceeded": timeout,
            "mcts_fallback_used": mcts_fallback,
            "llm_call1_invalid_json": False,
            "llm_call2_invalid_json": False,
            "dispatch_error": False,
        },
    }


@pytest.fixture
def synthetic_jsonl_dataset(tmp_path: Path) -> Path:
    """Creates a temporary multi-round JSONL dataset file with varying strata and risks."""
    records = []
    # Round 1: 3 turns (stratum 2, then stratum 1, then stratum 2)
    records.append(make_synthetic_record_dict(round_id=1, turn_idx=1, stratum=2, p_t_A=0.1, y_t_A=0, p_t_B=0.05, y_t_B=0))
    records.append(make_synthetic_record_dict(round_id=1, turn_idx=2, stratum=1, locked_action="red skip", p_t_A=0.3, y_t_A=0, p_t_B=0.05, y_t_B=0))
    records.append(make_synthetic_record_dict(round_id=1, turn_idx=3, stratum=2, p_t_A=0.2, y_t_A=1, p_t_B=0.1, y_t_B=0))

    # Round 2: 2 turns (stratum 3, then stratum 2 with fallback)
    records.append(make_synthetic_record_dict(round_id=2, turn_idx=1, stratum=3, p_t_A=0.4, y_t_A=1, p_t_B=0.2, y_t_B=0))
    records.append(make_synthetic_record_dict(round_id=2, turn_idx=2, stratum=2, timeout=True, p_t_A=0.5, y_t_A=0, p_t_B=0.8, y_t_B=1))

    # Round 3: 2 turns
    records.append(make_synthetic_record_dict(round_id=3, turn_idx=1, stratum=2, p_t_A=0.15, y_t_A=0, p_t_B=0.1, y_t_B=0))
    records.append(make_synthetic_record_dict(round_id=3, turn_idx=2, stratum=2, p_t_A=0.25, y_t_A=0, p_t_B=0.9, y_t_B=1))

    file_path = tmp_path / "dataset_drt.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return file_path
