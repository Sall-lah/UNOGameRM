"""Tests for deterministic fallback policy and opponent rollout isolation (Tasks 5.7 & 5.8).

Verifies fallback triggering under expired deadlines, sub-visit thresholds,
priority ordering (colour-matching high-point non-wild -> wild -> draw),
and strict barrier isolation from UNO.py AIPlayer.
"""

from __future__ import annotations

import inspect
import random
import time
import pytest

from src.config import MCTSConfig
from src.env.barrier import LegalObservation
from src.search.mars_mcts import MARSBudgetAwareMCTS, action_from_id


def test_fallback_on_already_exceeded_deadline(make_legal_obs):
    """Verify when deadline is already past on entry, fallback triggers immediately."""
    mcts = MARSBudgetAwareMCTS()
    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 5", "blue 3"),
        top_discard="red 1",
        active_colour="red",
        legal_actions=("red 5", "draw"),
    )

    # Deadline in the past
    past_deadline = time.perf_counter() - 1.0
    res = mcts.search(obs, deadline=past_deadline)

    assert res.fallback_used is True
    assert res.simulations_N_total == 0
    assert res.locked_action.action_id in obs.legal_actions
    assert res.locked_action.action_id == "red 5"  # Colour match preferred over draw


def test_fallback_prefers_colour_matching_non_wild_highest_point_value(make_legal_obs):
    """Verify fallback selects non-Wild matching active colour with highest point value."""
    mcts = MARSBudgetAwareMCTS()
    # Holds red 2 (2 pts), red 9 (9 pts), red skip (20 pts), WILD card (50 pts, but wild)
    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 2", "red 9", "red skip", "WILD card, set to red"),
        top_discard="red 0",
        active_colour="red",
        legal_actions=("red 2", "red 9", "red skip", "WILD card, set to red", "draw"),
    )

    # Force fallback by passing expired deadline
    res = mcts.search(obs, deadline=time.perf_counter() - 0.1)

    assert res.fallback_used is True
    # red skip has 20 points, beating red 9 (9 points) and red 2 (2 points)
    assert res.locked_action.action_id == "red skip"


def test_fallback_prefers_wild_on_most_frequent_colour_when_no_colour_match(make_legal_obs):
    """Verify fallback plays Wild on most frequent hand colour when no non-wild matches."""
    mcts = MARSBudgetAwareMCTS()
    # Active colour is green, but agent has only blue cards and a Wild
    obs: LegalObservation = make_legal_obs(
        agent_hand=("blue 1", "blue 2", "blue 3", "WILD card, set to blue"),
        top_discard="green 7",
        active_colour="green",
        legal_actions=("WILD card, set to blue", "WILD card, set to red", "draw"),
    )

    res = mcts.search(obs, deadline=time.perf_counter() - 0.1)

    assert res.fallback_used is True
    # Most frequent colour in hand is blue (3 blue cards); should pick WILD set to blue
    assert res.locked_action.action_id == "WILD card, set to blue"


def test_fallback_draws_when_no_card_is_playable(make_legal_obs):
    """Verify fallback commits draw action when no playable non-wild or wild exists."""
    mcts = MARSBudgetAwareMCTS()
    # Active colour is yellow, but agent holds blue cards with no matching values or wilds
    obs: LegalObservation = make_legal_obs(
        agent_hand=("blue 1", "blue 2"),
        top_discard="yellow 7",
        active_colour="yellow",
        legal_actions=("draw",),
    )

    res = mcts.search(obs, deadline=time.perf_counter() - 0.1)

    assert res.fallback_used is True
    assert res.locked_action.action_id == "draw"


def test_fallback_triggers_when_sub_one_visit_per_candidate(make_legal_obs):
    """Verify fallback triggers if search finishes without at least one visit per candidate."""
    # 3 candidates ("red 1", "red 2", "draw"), but max_simulations = 2
    cfg = MCTSConfig(max_simulations=2, max_rollout_depth=2)
    mcts = MARSBudgetAwareMCTS(config=cfg, rng=random.Random(42))

    obs: LegalObservation = make_legal_obs(
        agent_hand=("red 1", "red 2"),
        top_discard="red 5",
        active_colour="red",
        legal_actions=("red 1", "red 2", "draw"),
    )

    res = mcts.search(obs)
    # Total sims was 2, which is less than 3 candidates -> fallback_used must be True
    assert res.simulations_N_total == 2
    assert res.fallback_used is True
    assert res.locked_action.action_id in obs.legal_actions


def test_rollout_opponent_approximation_isolation():
    """Verify rollout opponent policy never references UNO.py AIPlayer or hidden server state."""
    import src.search.forward_model as fm
    import src.search.mars_mcts as mm
    import src.search.determinization as dt

    # Check imported symbols and source in all search modules
    for mod in (fm, mm, dt):
        assert not hasattr(mod, "AIPlayer"), f"Forbidden AIPlayer imported in {mod.__name__}"
        source = inspect.getsource(mod)
        assert "AIPlayer" not in source, f"Forbidden AIPlayer reference found in {mod.__name__}"
        assert "cpu_hand" not in source, f"Forbidden cpu_hand reference found in {mod.__name__}"
        assert "opponent_cards" not in source, f"Forbidden opponent_cards reference found in {mod.__name__}"

