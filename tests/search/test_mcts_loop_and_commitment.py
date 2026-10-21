"""Tests for MCTS simulation loop bounds, action commitment, and exports (Tasks 5.5, 5.6, 5.9).

Verifies K_max termination, deadline bounds, determinism under fixed seeds,
most-visited action commitment with strict tie-breaking, schema export alignment,
visit share normalization, and single-legal-action fast-path.
"""

from __future__ import annotations

import random
import time
import pytest

from src.config import MCTSConfig
from src.core.cards import parse_card_string
from src.env.barrier import LegalObservation
from src.search.mars_mcts import MARSBudgetAwareMCTS, SearchResult


def test_search_stops_at_simulation_cap(make_legal_obs):
    """Verify search loop terminates exactly at K_max when no deadline is reached."""
    cfg = MCTSConfig(max_simulations=10, max_rollout_depth=2)
    mcts = MARSBudgetAwareMCTS(config=cfg, rng=random.Random(42))

    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "red 2"),
        top_discard="red 3",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("red 1", "red 2", "draw"),
    )

    res = mcts.search(obs)
    assert res.simulations_N_total == 10
    assert sum(res.visit_counts_N.values()) == 10
    assert not res.fallback_used


def test_search_stops_at_deadline(make_legal_obs):
    """Verify search does not start a new simulation once the deadline has passed."""
    cfg = MCTSConfig(max_simulations=100000)  # Huge cap
    mcts = MARSBudgetAwareMCTS(config=cfg, rng=random.Random(42))

    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "red 2"),
        top_discard="red 3",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("red 1", "red 2", "draw"),
    )

    # 15 ms deadline
    budget_s = 0.015
    deadline = time.perf_counter() + budget_s

    res = mcts.search(obs, deadline=deadline)
    # Search should stop near the deadline, well before 100,000 simulations
    assert res.simulations_N_total < 100000
    assert res.simulations_N_total > 0


def test_deterministic_search_under_fixed_seed(make_legal_obs):
    """Verify identical inputs, priors, and seeds produce byte-identical search results."""
    cfg = MCTSConfig(max_simulations=20, max_rollout_depth=3)
    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "blue 2"),
        top_discard="red 5",
        active_colour="red",
        opp_card_count=5,
        legal_actions=("red 1", "draw"),
    )
    priors = {"red 1": 0.8, "draw": 0.2}

    def make_timer():
        cur = [0.0]
        def _clock():
            cur[0] += 0.0005
            return cur[0]
        return _clock

    mcts1 = MARSBudgetAwareMCTS(config=cfg, rng=random.Random(9999), timer=make_timer())
    res1 = mcts1.search(obs, priors=priors)

    mcts2 = MARSBudgetAwareMCTS(config=cfg, rng=random.Random(9999), timer=make_timer())
    res2 = mcts2.search(obs, priors=priors)

    assert res1.simulations_N_total == res2.simulations_N_total
    assert res1.visit_counts_N == res2.visit_counts_N
    assert res1.action_values_Q == res2.action_values_Q
    assert res1.locked_action == res2.locked_action
    assert res1.fallback_used == res2.fallback_used



def test_commit_most_visited_action():
    """Verify strictly most-visited action is committed."""
    mcts = MARSBudgetAwareMCTS()
    candidates = ["red 1", "red 2", "draw"]
    visit_counts = {"red 1": 10, "red 2": 25, "draw": 5}
    action_values = {"red 1": 0.9, "red 2": 0.4, "draw": 0.1}
    priors = {"red 1": 0.33, "red 2": 0.33, "draw": 0.33}

    committed = mcts._commit_most_visited_action(
        candidates=candidates,
        visit_counts=visit_counts,
        action_values_Q=action_values,
        priors=priors,
    )
    assert committed == "red 2"


def test_commit_tie_break_order():
    """Verify tie-break order: N -> Q -> Prior -> Lexicographic action_id."""
    mcts = MARSBudgetAwareMCTS()

    # Tie on N, broken by higher Q
    candidates = ["red 1", "red 2"]
    v_counts = {"red 1": 10, "red 2": 10}
    q_vals = {"red 1": 0.5, "red 2": 0.7}
    priors = {"red 1": 0.5, "red 2": 0.5}
    assert mcts._commit_most_visited_action(candidates, v_counts, q_vals, priors) == "red 2"

    # Tie on N and Q, broken by higher Prior
    q_vals_eq = {"red 1": 0.6, "red 2": 0.6}
    priors_diff = {"red 1": 0.3, "red 2": 0.7}
    assert mcts._commit_most_visited_action(candidates, v_counts, q_vals_eq, priors_diff) == "red 2"

    # Tie on N, Q, and Prior, broken by lexicographic order ("blue 1" < "red 1")
    candidates_lex = ["red 1", "blue 1"]
    v_counts_lex = {"red 1": 10, "blue 1": 10}
    q_vals_lex = {"red 1": 0.5, "blue 1": 0.5}
    priors_lex = {"red 1": 0.5, "blue 1": 0.5}
    assert mcts._commit_most_visited_action(candidates_lex, v_counts_lex, q_vals_lex, priors_lex) == "blue 1"


def test_result_export_schema_and_visit_shares(make_legal_obs):
    """Verify SearchResult fields match search_mcts schema and visit shares sum to 1.0."""
    cfg = MCTSConfig(max_simulations=15, max_rollout_depth=2)
    mcts = MARSBudgetAwareMCTS(config=cfg, rng=random.Random(42))

    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "red 2"),
        top_discard="red 3",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("red 1", "red 2", "draw"),
    )

    res: SearchResult = mcts.search(obs)

    # Check schema keys
    for act_id in obs.legal_actions:
        assert act_id in res.visit_counts_N
        assert act_id in res.action_values_Q
        assert act_id in res.visit_shares
        assert act_id in res.mean_raw_utilities
        assert act_id in res.rollout_apen_rates
        assert act_id in res.rollout_opp_win_rates

    assert res.simulations_N_total == 15
    assert res.unseen_pool_size_Ut > 0
    assert res.search_wall_time_ms >= 0.0
    assert isinstance(res.fallback_used, bool)
    assert res.locked_action.action_id in obs.legal_actions

    # Visit shares must sum to 1.0
    assert abs(sum(res.visit_shares.values()) - 1.0) < 1e-6


def test_single_legal_action_commits_without_wasted_search(make_legal_obs):
    """Verify single legal action commits immediately with N_total = 0 and no search loops."""
    mcts = MARSBudgetAwareMCTS()
    obs: LegalObservation = make_legal_obs(
        legal_actions=("draw",),  # Only draw is legal
    )

    res = mcts.search(obs)

    assert res.locked_action.action_id == "draw"
    assert res.simulations_N_total == 0
    assert res.fallback_used is False
    assert res.visit_counts_N == {"draw": 0}
    assert res.visit_shares == {"draw": 1.0}
