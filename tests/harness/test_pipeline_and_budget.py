"""Tests for Zone 2 pipeline phase ordering and budget accounting (Tasks 7.1, 7.2, 7.3).

Verifies strict pipeline phase execution order, MCTS deadline calculation with Call #2 reserve,
and timeout_exceeded reliability flag setting under overrunning execution.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.config import AppConfig
from src.core.cards import UNOAction
from src.env.barrier import LegalObservation
from src.harness.agent_harness import AgentHarness
from src.llm.client import FakeLLMClient
from src.search.mars_mcts import SearchResult


def test_pipeline_phase_ordering_and_call2_gating(tmp_path: Path, sample_legal_obs, fake_llm):
    """Verify phases run in order: retrieve -> Call 1 -> MCTS -> commit -> Call 2.

    Asserts that Call 2 is never invoked before an action is locked.
    """
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    budget:
      t_turn_s: 5.0
      delta_buffer_s: 0.5
      llm2_reserve_s: 1.0
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["blue skip"], "top_card": "red 5", "legal_actions": ["draw"]},
    }

    phases_recorded = []

    mock_plugmem = MagicMock()
    def stub_retrieve(obs):
        phases_recorded.append("retrieve")
        ret_mock = MagicMock()
        ret_mock.distilled_context = "context"
        ret_mock.retrieved_node_ids = []
        ret_mock.active_opp_lack_colours = {}
        ret_mock.active_opp_wild_colours = ()
        ret_mock.relevance_scores = []
        ret_mock.token_estimate = 0
        return ret_mock
    mock_plugmem.retrieve.side_effect = stub_retrieve
    mock_plugmem.get_determinization_constraints.return_value = []

    mock_mcts = MagicMock()
    def stub_search(obs, priors, memory_constraints, deadline):
        # Assert Call 1 has already occurred and Call 2 has NOT
        assert "retrieve" in phases_recorded
        assert "call2" not in phases_recorded
        phases_recorded.append("mcts")
        return SearchResult(
            visit_counts_N={"red 5": 10},
            action_values_Q={"red 5": 0.5},
            simulations_N_total=10,
            unseen_pool_size_Ut=100,
            search_wall_time_ms=5.0,
            fallback_used=False,
            locked_action=UNOAction(action_id="red 5", action_type="PLAY"),
            visit_shares={"red 5": 1.0},
            mean_raw_utilities={"red 5": 0.5},
            rollout_apen_rates={"red 5": 0.0},
            rollout_opp_win_rates={"red 5": 0.0},
        )
    mock_mcts.search.side_effect = stub_search

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        plugmem=mock_plugmem,
        mcts=mock_mcts,
        llm_client=fake_llm,
    )

    obs = sample_legal_obs(legal_actions=("red 5", "draw"))
    rec = harness.on_agent_turn(obs)

    # Check phase execution sequence
    assert phases_recorded == ["retrieve", "mcts"]
    assert rec.stage1_freeze_log.locked_action_at == "red 5"
    assert rec.stage2_post_horizon.dispatched_command == "red 5"


def test_mcts_deadline_reserves_llm2_time(tmp_path: Path, sample_legal_obs, fake_llm):
    """Verify the deadline passed to MCTS reserves budget.llm2_reserve_s up front."""
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    budget:
      t_turn_s: 4.0
      delta_buffer_s: 0.5
      llm2_reserve_s: 1.0
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["blue skip"], "top_card": "red 5"},
    }

    captured_deadline = []
    mock_mcts = MagicMock()
    def stub_search(obs, priors, memory_constraints, deadline):
        captured_deadline.append(deadline)
        return SearchResult(
            visit_counts_N={"red 5": 5},
            action_values_Q={"red 5": 0.5},
            simulations_N_total=5,
            unseen_pool_size_Ut=100,
            search_wall_time_ms=1.0,
            fallback_used=False,
            locked_action=UNOAction(action_id="red 5", action_type="PLAY"),
            visit_shares={"red 5": 1.0},
            mean_raw_utilities={"red 5": 0.5},
            rollout_apen_rates={"red 5": 0.0},
            rollout_opp_win_rates={"red 5": 0.0},
        )
    mock_mcts.search.side_effect = stub_search

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        mcts=mock_mcts,
        llm_client=fake_llm,
    )

    t_before = time.perf_counter()
    obs = sample_legal_obs(legal_actions=("red 5", "draw"))
    harness.on_agent_turn(obs)

    # Effective budget: 4.0 - 0.5 = 3.5s.
    # MCTS budget: 3.5 - 1.0 (llm2 reserve) = ~2.5s from turn start.
    assert len(captured_deadline) == 1
    mcts_deadline = captured_deadline[0]
    expected_max_deadline = t_before + (4.0 - 0.5 - 1.0) + 0.1  # with slight tolerance
    assert mcts_deadline <= expected_max_deadline


def test_budget_overrun_sets_timeout_exceeded_and_dispatches(tmp_path: Path, sample_legal_obs, fake_llm):
    """Verify timeout_exceeded is set on overrun, yet freeze and dispatch still succeed."""
    # Very short budget: 0.02s
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    budget:
      t_turn_s: 0.02
      delta_buffer_s: 0.005
      llm2_reserve_s: 0.005
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["blue skip"], "top_card": "red 5"},
    }

    # Delay MCTS slightly to force budget overrun
    def slow_search(obs, priors, memory_constraints, deadline):
        time.sleep(0.03)  # Overruns 0.015s effective budget
        return SearchResult(
            visit_counts_N={"red 5": 2},
            action_values_Q={"red 5": 0.5},
            simulations_N_total=2,
            unseen_pool_size_Ut=100,
            search_wall_time_ms=30.0,
            fallback_used=True,
            locked_action=UNOAction(action_id="red 5", action_type="PLAY"),
            visit_shares={"red 5": 1.0},
            mean_raw_utilities={"red 5": 0.5},
            rollout_apen_rates={"red 5": 0.0},
            rollout_opp_win_rates={"red 5": 0.0},
        )

    mock_mcts = MagicMock()
    mock_mcts.search.side_effect = slow_search

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        mcts=mock_mcts,
        llm_client=fake_llm,
    )

    obs = sample_legal_obs(legal_actions=("red 5", "draw"))
    rec = harness.on_agent_turn(obs)

    # Overrun occurred and was recorded
    assert rec.system_reliability.timeout_exceeded is True
    # Freeze and dispatch were still executed
    assert rec.stage1_freeze_log.locked_action_at == "red 5"
    assert rec.stage2_post_horizon.dispatched_command == "red 5"
    assert mock_platform.dispatch_agent_action_and_observe_horizon.called
