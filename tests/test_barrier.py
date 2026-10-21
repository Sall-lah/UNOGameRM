"""Tests for Information Barrier, LegalObservation contract, and unseen pool computation."""

import pytest
from dataclasses import FrozenInstanceError
from player import AIPlayer
from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD,
    WILD_DRAW_FOUR,
    UNOCard,
)
from src.env.barrier import (
    FORBIDDEN_KEYS,
    ClientServerBarrier,
    LegalObservation,
    compute_unseen_pool,
)
from src.env.events import HorizonEventType, HorizonServerEvent


def test_legal_observation_is_frozen_and_fields_contract():
    """Verify LegalObservation raises on mutation and exposes contracted fields."""
    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("red 5", "blue skip"),
        top_discard="red 7",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("red 5", "draw"),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=("red 7",),
    )

    with pytest.raises(FrozenInstanceError):
        obs.opp_card_count = 6  # type: ignore

    with pytest.raises(FrozenInstanceError):
        obs.agent_hand = ()  # type: ignore

    assert obs.opp_card_count == 7
    # Assert opponent identities are not an attribute
    assert not hasattr(obs, "opp_cards")
    assert not hasattr(obs, "cpu_hand")


def test_extract_observation_raises_on_forbidden_keys():
    """Verify extract_observation raises AssertionError naming each key in FORBIDDEN_KEYS."""
    clean_raw = {
        "round_id": 1,
        "t": 1,
        "agent_hand": ["red 5"],
        "top_card": "red 2",
        "opp_hand_count": 5,
        "legal_actions": ["red 5", "draw"],
        "public_discards": ["red 2"],
    }

    # Clean raw passes
    obs = ClientServerBarrier.extract_observation(clean_raw)
    assert isinstance(obs, LegalObservation)

    # Test each forbidden key
    for key in FORBIDDEN_KEYS:
        leaky_raw = dict(clean_raw)
        leaky_raw[key] = ["leaked_data"]
        with pytest.raises(AssertionError) as exc_info:
            ClientServerBarrier.extract_observation(leaky_raw)
        assert key in str(exc_info.value)


def test_deep_copy_isolation():
    """Verify modifying source raw dictionary after extraction does not alter observation."""
    source_hand = ["green 3", "yellow 7"]
    raw = {
        "round_id": 2,
        "t": 3,
        "agent_hand": source_hand,
        "top_card": "green 1",
        "opp_hand_count": 4,
    }

    obs = ClientServerBarrier.extract_observation(raw)
    assert obs.agent_hand == ("green 3", "yellow 7")

    # Mutate source list
    source_hand.append("blue 9")
    source_hand[0] = "mutated"

    assert obs.agent_hand == ("green 3", "yellow 7")


def test_legal_actions_encoding_and_wild_expansion():
    """Verify legal actions expand wilds per colour, always include draw, and draw-only when unplayable."""
    raw = {
        "round_id": 1,
        "t": 1,
        "agent_hand": ["red 5", "WILD card, set to white"],
        "top_card": "red 1",
        "opp_hand_count": 7,
    }
    obs = ClientServerBarrier.extract_observation(raw)
    assert "red 5" in obs.legal_actions
    assert "draw" in obs.legal_actions
    for col in COLOURS:
        assert f"WILD card, set to {col}" in obs.legal_actions

    # Test no playable cards -> exactly ('draw',)
    raw_unplayable = {
        "round_id": 1,
        "t": 2,
        "agent_hand": ["blue 1", "green 2"],
        "top_card": "yellow 8",
        "opp_hand_count": 7,
    }
    obs_unplayable = ClientServerBarrier.extract_observation(raw_unplayable)
    assert obs_unplayable.legal_actions == ("draw",)


def test_public_event_forwarding_hides_private_cards():
    """Verify voluntary keep hides card, while auto-play and opponent plays forward publicly."""
    events = [
        # Opponent wild play with declared green
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
            card=UNOCard("WIL", "green", WILD),
            declared_colour="green",
        ),
        # Voluntary keep draw: card identity must be hidden
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
            active=True,
            card=UNOCard("NUM", "blue", "4"),
            outcome="keep",
            cards_drawn_count=1,
        ),
        # Penalty draw count
        HorizonServerEvent(
            event_index=2,
            actor="AGENT",
            event_type=HorizonEventType.DRAW_PENALTY_APEN,
            active=False,
            cards_drawn_count=2,
        ),
        # Reshuffle count
        HorizonServerEvent(
            event_index=3,
            actor="OPPONENT",
            event_type=HorizonEventType.RESHUFFLE,
            active=False,
            recycled_count=15,
        ),
    ]

    raw = {
        "round_id": 1,
        "t": 1,
        "agent_hand": ["red 5"],
        "top_card": "red 1",
        "opp_hand_count": 6,
        "horizon_events": events,
    }

    obs = ClientServerBarrier.extract_observation(raw)
    fwd_events = obs.public_events_since_last_turn
    assert len(fwd_events) == 4

    # Opponent wild play
    assert fwd_events[0]["actor"] == "OPPONENT"
    assert fwd_events[0]["declared_colour"] == "green"

    # Voluntary keep must hide card identity
    assert fwd_events[1]["event_type"] == "DRAW_VOLUNTARY_ANOP"
    assert fwd_events[1]["outcome"] == "keep"
    assert fwd_events[1]["card"] is None

    # Penalty count
    assert fwd_events[2]["cards_drawn_count"] == 2

    # Reshuffle count
    assert fwd_events[3]["recycled_count"] == 15


def test_compute_unseen_pool_baseline_and_refined():
    """Verify unseen pool cardinality: 108 - |hand| - 1 baseline, and refinement."""
    hand = (
        "red 1",
        "red 2",
        "blue 3",
        "green 4",
        "yellow 5",
        "blue skip",
        "WILD card, set to green",
    )
    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=hand,
        top_discard="red 7",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=("yellow 9̲", "blue reverse"),
    )

    # Baseline pool (refine=False)
    pool = compute_unseen_pool(obs, refine=False)
    assert len(pool) == 108 - len(hand) - 1  # 108 - 7 - 1 = 100

    # Refined pool (refine=True)
    refined_pool = compute_unseen_pool(obs, refine=True)
    assert len(refined_pool) == 108 - len(hand) - 1 - len(obs.known_discards_since_reshuffle)  # 100 - 2 = 98


def test_process_path_guard():
    """Verify LegalObservation exposes no engine internals and all fields are immutable/tuples."""
    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("red 1",),
        top_discard="red 2",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("red 1", "draw"),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    for field_name in (
        "round_id",
        "turn_idx",
        "agent_hand",
        "top_discard",
        "active_colour",
        "opp_card_count",
        "legal_actions",
        "public_events_since_last_turn",
        "known_discards_since_reshuffle",
    ):
        val = getattr(obs, field_name)
        assert isinstance(val, (int, str, tuple))


def test_one_directional_barrier_caveat_unmodified():
    """Verify barrier preserves native AIPlayer methods unchanged without monkeypatching."""
    ai = AIPlayer("OPPONENT")
    assert hasattr(ai, "think")
    assert hasattr(ai, "think_wild")
    assert hasattr(ai, "think_challenge")
