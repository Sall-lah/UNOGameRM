"""Pytest fixtures for LLM decision call unit tests.

Provides sample LegalObservation instances, canned JSON responses, and mock MCTS stats.
"""

from __future__ import annotations

import json
from typing import Sequence
import pytest

from src.env.barrier import LegalObservation


@pytest.fixture
def sample_obs() -> LegalObservation:
    """Provides a realistic LegalObservation for LLM prompt rendering and call tests."""
    return LegalObservation(
        round_id=1,
        turn_idx=2,
        agent_hand=("red 5", "blue skip", "WILD card, set to green"),
        top_discard="red 3",
        active_colour="red",
        opp_card_count=6,
        legal_actions=(
            "red 5",
            "WILD card, set to green",
            "WILD card, set to red",
            "draw",
        ),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=("yellow 2", "green 4"),
    )


@pytest.fixture
def valid_call1_json() -> str:
    """Provides a valid, schema-compliant JSON response for LLM Call #1."""
    return json.dumps({
        "pruned_actions": ["red 5", "WILD card, set to green"],
        "action_priors": {
            "red 5": 0.65,
            "WILD card, set to green": 0.35,
        },
        "pruning_note": "Keep red 5 to match active colour, and Wild as colour switch.",
    })


@pytest.fixture
def valid_call2_json() -> str:
    """Provides a valid, schema-compliant JSON response for LLM Call #2."""
    return json.dumps({
        "p_t_A": 0.15,
        "p_t_B": 0.05,
        "expected_stratum": 2,
        "short_rationale": "Opponent holds 6 cards, so win probability is low. Red play exposes to red +2.",
    })


@pytest.fixture
def make_legal_obs():
    """Factory fixture generating valid LegalObservation instances with customizable state."""

    def _factory(
        round_id: int = 1,
        turn_idx: int = 1,
        agent_hand: Sequence[str] = ("red 5", "blue skip", "WILD card, set to red"),
        top_discard: str = "red 3",
        active_colour: str = "red",
        opp_card_count: int = 7,
        legal_actions: Sequence[str] = ("red 5", "draw"),
        public_events: Sequence[dict] = (),
        known_discards: Sequence[str] = (),
    ) -> LegalObservation:
        return LegalObservation(
            round_id=round_id,
            turn_idx=turn_idx,
            agent_hand=tuple(agent_hand),
            top_discard=top_discard,
            active_colour=active_colour,
            opp_card_count=opp_card_count,
            legal_actions=tuple(legal_actions),
            public_events_since_last_turn=tuple(public_events),
            known_discards_since_reshuffle=tuple(known_discards),
        )

    return _factory

