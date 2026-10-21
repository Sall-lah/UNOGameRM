"""Tests for HorizonServerEvent schema and HorizonEventType enum membership."""

from src.core.cards import UNOCard
from src.env.events import HorizonEventType, HorizonServerEvent


def test_event_type_exact_membership():
    """Verify enum members match the exact required set of 10 events."""
    expected_members = {
        "PLAY_CARD",
        "DRAW_VOLUNTARY_ANOP",
        "DRAW_PENALTY_APEN",
        "D4_CHALLENGE_WON",
        "D4_CHALLENGE_LOST",
        "TURN_FORFEIT",
        "WILD_COLOUR_DECLARED",
        "UNO_DECLARED",
        "RESHUFFLE",
        "ROUND_WIN",
    }
    actual_members = {e.name for e in HorizonEventType}
    assert actual_members == expected_members


def test_horizon_server_event_construction():
    """Verify HorizonServerEvent fields, actor, active flag, and payload assignment."""
    card = UNOCard("NUM", "red", "5")
    evt = HorizonServerEvent(
        event_index=0,
        actor="AGENT",
        event_type=HorizonEventType.PLAY_CARD,
        active=True,
        card=card,
        agent_hand_size=6,
        opp_hand_size=7,
    )
    assert evt.event_index == 0
    assert evt.actor == "AGENT"
    assert evt.event_type == HorizonEventType.PLAY_CARD
    assert evt.active is True
    assert evt.card == card
    assert evt.agent_hand_size == 6


def test_draw_events_payloads():
    """Verify distinct voluntary and penalty draw payloads."""
    voluntary_evt = HorizonServerEvent(
        event_index=1,
        actor="AGENT",
        event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
        active=True,
        cards_drawn_count=1,
        outcome="keep",
    )
    assert voluntary_evt.event_type == HorizonEventType.DRAW_VOLUNTARY_ANOP
    assert voluntary_evt.outcome == "keep"

    penalty_evt = HorizonServerEvent(
        event_index=2,
        actor="AGENT",
        event_type=HorizonEventType.DRAW_PENALTY_APEN,
        active=False,
        cards_drawn_count=2,
    )
    assert penalty_evt.event_type == HorizonEventType.DRAW_PENALTY_APEN
    assert penalty_evt.cards_drawn_count == 2
