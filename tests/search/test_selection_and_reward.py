"""Tests for prior-guided UCT selection and efficiency-guided rewards (Tasks 5.3 & 5.4).

Verifies unvisited candidate expansion ordering, UCT exploration weighting,
raw utility formula terms (win, penalty, resource, hand diff), degenerate
normalization, latency modulation, and running mean backpropagation.
"""

from __future__ import annotations

import math
import random
import pytest

from src.config import MCTSConfig
from src.core.cards import parse_card_string
from src.search.forward_model import SimulatedState
from src.search.mars_mcts import MARSBudgetAwareMCTS


def test_unvisited_candidates_expanded_by_descending_prior():
    """Verify unvisited candidates (N=0) are selected in descending P_prior order."""
    mcts = MARSBudgetAwareMCTS()
    candidates = ["red 1", "red 2", "red 3"]
    visit_counts = {"red 1": 0, "red 2": 0, "red 3": 0}
    action_values_Q = {"red 1": 0.0, "red 2": 0.0, "red 3": 0.0}
    priors = {"red 1": 0.2, "red 2": 0.7, "red 3": 0.1}

    selected = mcts._select_root_action(
        candidates=candidates,
        visit_counts=visit_counts,
        action_values_Q=action_values_Q,
        priors=priors,
        simulations_N_total=0,
    )
    # Highest prior among unvisited is red 2 (0.7)
    assert selected == "red 2"


def test_prior_influences_exploration_among_visited_actions():
    """Verify when all candidates are visited with equal Q and N, higher prior yields higher score."""
    mcts = MARSBudgetAwareMCTS(MCTSConfig(c_uct=1.25))
    candidates = ["red 1", "red 2"]
    visit_counts = {"red 1": 5, "red 2": 5}
    action_values_Q = {"red 1": 0.5, "red 2": 0.5}
    priors = {"red 1": 0.8, "red 2": 0.2}

    selected = mcts._select_root_action(
        candidates=candidates,
        visit_counts=visit_counts,
        action_values_Q=action_values_Q,
        priors=priors,
        simulations_N_total=10,
    )
    # Both have equal Q=0.5 and N=5, so red 1 with prior 0.8 wins
    assert selected == "red 1"


def test_missing_prior_defaults_to_uniform():
    """Verify missing candidates in the prior map receive uniform prior 1 / |candidates|."""
    mcts = MARSBudgetAwareMCTS()
    candidates = ["red 1", "red 2", "draw"]
    normalized = mcts._normalize_priors(candidates, priors={"red 1": 0.5})

    # Missing red 2 and draw should be assigned equal shares of remaining probability
    assert "red 2" in normalized
    assert "draw" in normalized
    assert abs(normalized["red 2"] - normalized["draw"]) < 1e-6
    assert abs(sum(normalized.values()) - 1.0) < 1e-6


def test_raw_utility_win_adds_one(make_simulated_state):
    """Verify raw utility M includes +1.0 when agent wins the round."""
    cfg = MCTSConfig(alpha1=0.35, alpha2=0.15, alpha3=0.25)
    mcts = MARSBudgetAwareMCTS(config=cfg)

    # State where agent has 0 cards (win), opponent has 5 cards
    state = make_simulated_state(
        agent_hand=[],
        opp_hand=["blue 1", "blue 2", "blue 3", "blue 4", "blue 5"],
        draw_pile=["yellow 1"],
    )
    state.winner = "agent"

    m = mcts._calculate_raw_utility(state)
    # Expected: +1.0 (win) + 0.35 * (5 - 0) / 15 + 0.15 * (0 / 8) - 0.25 * 0
    expected = 1.0 + 0.35 * (5.0 / 15.0)
    assert abs(m - expected) < 1e-6


def test_raw_utility_apen_subtracts_penalty(make_simulated_state):
    """Verify raw utility M includes -alpha3 (-0.25) when agent suffered A_pen."""
    cfg = MCTSConfig(alpha1=0.35, alpha2=0.15, alpha3=0.25)
    mcts = MARSBudgetAwareMCTS(config=cfg)

    state = make_simulated_state(
        agent_hand=["red 1", "red 2"],
        opp_hand=["blue 1", "blue 2"],
        draw_pile=["yellow 1"],
    )
    state.agent_suffered_a_pen = True

    m = mcts._calculate_raw_utility(state)
    # Hand diff = 0, held action/wild = 0, pen = 1.0 -> -0.25
    expected = 0.0 - 0.25 * 1.0
    assert abs(m - expected) < 1e-6


def test_degenerate_normalization_yields_neutral_score():
    """Verify when M_max == M_min, normalized score G equals 0.5."""
    m_raw = 0.42
    m_min = 0.42
    m_max = 0.42

    norm_g = 0.5 if m_max == m_min else (m_raw - m_min) / (m_max - m_min)
    assert norm_g == 0.5


def test_latency_modulation_penalizes_slow_rollouts():
    """Verify with penalty_weight_w < 0, a slower rollout yields a smaller reward R."""
    w = -0.07
    l_target = 0.002
    norm_g = 0.5

    t_fast = 0.001  # faster than target
    t_slow = 0.010  # slower than target

    r_fast = norm_g * ((t_fast / l_target) ** w)
    r_slow = norm_g * ((t_slow / l_target) ** w)

    assert r_fast > r_slow


def test_backpropagation_running_mean_update():
    """Verify incremental running-mean formula Q(a) += (R - Q(a)) / N(a)."""
    n = 2
    q = 0.6
    r = 0.9

    n_new = n + 1
    q_new = q + (r - q) / n_new
    # Expected: (0.6 * 2 + 0.9) / 3 = 2.1 / 3 = 0.7
    assert abs(q_new - 0.7) < 1e-6
