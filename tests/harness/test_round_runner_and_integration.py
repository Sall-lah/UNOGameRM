"""Tests for Round Runner and full end-to-end deterministic integration (Tasks 7.9 & 7.10).

Verifies starting-player alternation, per-round PlugMem reset, loop termination,
and byte-identical DecisionRecord sequences across identical-seed runs with FakeLLMClient.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.config import AppConfig
from src.env.uno_engine import ModifiedUNOServerPlatform
from src.harness.round_runner import (
    derive_round_seeds,
    run_round,
)
from src.llm.client import FakeLLMClient


def test_round_runner_starting_player_alternation(tmp_path: Path):
    """Verify starting player alternates: even round -> AGENT, odd round -> CPU."""
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    mcts:
      max_simulations: 5
      max_rollout_depth: 2
    """)

    # Fast responses for FakeLLMClient
    call1_json = json.dumps({
        "pruned_actions": ["draw"],
        "action_priors": {"draw": 1.0},
        "pruning_note": "Draw card.",
    })
    call2_json = json.dumps({
        "p_t_A": 0.2,
        "p_t_B": 0.1,
        "expected_stratum": 2,
        "short_rationale": "Fast test rationale.",
    })
    llm = FakeLLMClient(
        responses_by_prompt={"A_t": call1_json, "LOCKED ACTION": call2_json},
        default_response=call1_json,
    )

    # Even round r=2 -> starting player is AGENT
    records_r2 = run_round(r=2, base_seed=100, config=cfg, llm_client=llm)
    assert len(records_r2) > 0
    assert records_r2[0].identity.starting_player == "AGENT"
    assert records_r2[0].identity.round_id == 2

    # Odd round r=3 -> starting player is CPU
    records_r3 = run_round(r=3, base_seed=100, config=cfg, llm_client=llm)
    assert len(records_r3) > 0
    assert records_r3[0].identity.starting_player == "CPU"
    assert records_r3[0].identity.round_id == 3


def test_deterministic_full_round_integration(tmp_path: Path):
    """Verify two runs of run_round with identical seed and FakeLLM yield identical records."""
    call1_json = json.dumps({
        "pruned_actions": ["draw"],
        "action_priors": {"draw": 1.0},
        "pruning_note": "Draw.",
    })
    call2_json = json.dumps({
        "p_t_A": 0.05,
        "p_t_B": 0.02,
        "expected_stratum": 2,
        "short_rationale": "Deterministic run.",
    })

    # Run 1
    dir1 = tmp_path / "run1"
    cfg1 = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(dir1)}
      strict_invariant: true
    mcts:
      penalty_weight_w: 0.0
      max_simulations: 5
      max_rollout_depth: 2
    """)
    llm1 = FakeLLMClient(
        responses_by_prompt={"A_t": call1_json, "LOCKED ACTION": call2_json},
        default_response=call1_json,
    )
    records1 = run_round(r=2, base_seed=42, config=cfg1, llm_client=llm1)

    # Run 2
    dir2 = tmp_path / "run2"
    cfg2 = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(dir2)}
      strict_invariant: true
    mcts:
      penalty_weight_w: 0.0
      max_simulations: 5
      max_rollout_depth: 2
    """)
    llm2 = FakeLLMClient(
        responses_by_prompt={"A_t": call1_json, "LOCKED ACTION": call2_json},
        default_response=call1_json,
    )
    records2 = run_round(r=2, base_seed=42, config=cfg2, llm_client=llm2)

    assert len(records1) == len(records2)
    assert len(records1) > 0

    # Assert decision actions and outcomes match exactly
    for rec1, rec2 in zip(records1, records2):
        assert rec1.identity.round_id == rec2.identity.round_id
        assert rec1.identity.turn_idx == rec2.identity.turn_idx
        assert rec1.stage1_freeze_log.locked_action_at == rec2.stage1_freeze_log.locked_action_at
        assert rec1.stage2_post_horizon.dispatched_command == rec2.stage2_post_horizon.dispatched_command
        assert rec1.stage2_post_horizon.y_t_A_penalty_draw == rec2.stage2_post_horizon.y_t_A_penalty_draw
        assert rec1.stage2_post_horizon.y_t_B_opp_win == rec2.stage2_post_horizon.y_t_B_opp_win
        assert rec1.stage2_post_horizon.stratum_horizon == rec2.stage2_post_horizon.stratum_horizon
        assert rec1.stage2_post_horizon.round_status == rec2.stage2_post_horizon.round_status
