"""Pytest fixtures for MCTS unit tests and forward model validation.

Provides seeded generators, observation factories, and mock game states built
exclusively from src.core.cards and src.env.barrier primitives.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence
import pytest

from src.core.cards import (
    COLOURS,
    UNOCard,
    parse_card_string,
)
from src.env.barrier import LegalObservation
from src.search.forward_model import SimulatedState


@pytest.fixture
def seeded_rng() -> random.Random:
    """Provides a fresh seeded random.Random instance for deterministic test assertions."""
    return random.Random(42)


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
        legal_actions: Optional[Sequence[str]] = None,
        public_events: Sequence[dict] = (),
        known_discards: Sequence[str] = (),
    ) -> LegalObservation:
        if legal_actions is None:
            # Generate default legal actions matching top discard / hand
            actions = ["red 5", "draw"]
            legal_actions = tuple(actions)
        else:
            legal_actions = tuple(legal_actions)

        return LegalObservation(
            round_id=round_id,
            turn_idx=turn_idx,
            agent_hand=tuple(agent_hand),
            top_discard=top_discard,
            active_colour=active_colour,
            opp_card_count=opp_card_count,
            legal_actions=legal_actions,
            public_events_since_last_turn=tuple(public_events),
            known_discards_since_reshuffle=tuple(known_discards),
        )

    return _factory


@pytest.fixture
def make_simulated_state():
    """Factory fixture generating SimulatedState instances for forward model testing."""

    def _factory(
        agent_hand: Sequence[str] = ("red 5", "blue skip"),
        opp_hand: Sequence[str] = ("green 2", "yellow 7"),
        draw_pile: Sequence[str] = ("yellow 1", "blue 3", "red 9"),
        top_discard: str = "red 3",
        active_colour: str = "red",
        current_player: str = "agent",
    ) -> SimulatedState:
        return SimulatedState(
            agent_hand=[parse_card_string(c) for c in agent_hand],
            opp_hand=[parse_card_string(c) for c in opp_hand],
            draw_pile=[parse_card_string(c) for c in draw_pile],
            top_discard=parse_card_string(top_discard),
            active_colour=active_colour,
            current_player=current_player,
            agent_suffered_a_pen=False,
            agent_uno_flag=len(agent_hand) == 1,
            opp_uno_flag=len(opp_hand) == 1,
            winner=None,
        )

    return _factory
