"""Tests for strict separation between voluntary draws (A_nop) and penalty draws (A_pen)."""

from card import Card
from src.core.cards import DRAW_TWO
from src.env.events import HorizonEventType
from src.env.uno_engine import ModifiedUNOServerPlatform


def test_voluntary_draw_keep():
    """Verify non-playable voluntary draw emits DRAW_VOLUNTARY_ANOP with outcome keep."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=303)

    # Top card red 5
    platform.game.deck.discard_pile.append(Card("NUM", "red", "5"))
    # Next card in deck is unplayable blue 9
    platform.game.deck.cards.append(Card("NUM", "blue", "9̲"))
    # Opponent plays normal card red 3
    platform.game.opp.hand = [Card("NUM", "red", "3")]

    res = platform.dispatch_agent_action_and_observe_horizon("draw")
    draw_events = [
        e
        for e in res["horizon_events"]
        if e.event_type == HorizonEventType.DRAW_VOLUNTARY_ANOP
        and e.actor == "AGENT"
    ]
    assert len(draw_events) == 1
    assert draw_events[0].outcome == "keep"
    assert draw_events[0].cards_drawn_count == 1
    # Ensure no penalty event was emitted for agent
    assert not any(
        e.event_type == HorizonEventType.DRAW_PENALTY_APEN and e.actor == "AGENT"
        for e in res["horizon_events"]
    )


def test_voluntary_draw_auto_play():
    """Verify playable voluntary draw emits DRAW_VOLUNTARY_ANOP with outcome auto_play and PLAY_CARD."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=404)

    # Top card red 5
    platform.game.deck.discard_pile.append(Card("NUM", "red", "5"))
    # Prepare deck draw pile top so next drawn card is red 7 (playable on red 5)
    platform.game.deck.cards.append(Card("NUM", "red", "7"))
    platform.game.opp.hand = [Card("NUM", "red", "2")]

    res = platform.dispatch_agent_action_and_observe_horizon("draw")
    voluntary_events = [
        e
        for e in res["horizon_events"]
        if e.event_type == HorizonEventType.DRAW_VOLUNTARY_ANOP
        and e.actor == "AGENT"
    ]
    assert len(voluntary_events) == 1
    assert voluntary_events[0].outcome == "auto_play"

    # Must be accompanied by a PLAY_CARD event for the drawn card
    play_events = [
        e
        for e in res["horizon_events"]
        if e.event_type == HorizonEventType.PLAY_CARD
        and e.actor == "AGENT"
        and str(e.card) == "red 7"
    ]
    assert len(play_events) == 1


def test_penalty_draw_apen():
    """Verify opponent +2 forces agent to draw 2 with DRAW_PENALTY_APEN."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=505)

    # Agent plays yellow 3
    platform.game.agent.hand.append(Card("NUM", "yellow", "3"))
    platform.game.deck.discard_pile.append(Card("NUM", "yellow", "1"))

    # Force opponent to hold yellow +2 as playable card, and unplayable blue 9
    platform.game.opp.hand = [Card("ACT", "yellow", DRAW_TWO), Card("NUM", "blue", "9̲")]

    res = platform.dispatch_agent_action_and_observe_horizon("yellow 3")
    penalty_events = [
        e
        for e in res["horizon_events"]
        if e.event_type == HorizonEventType.DRAW_PENALTY_APEN
        and e.actor == "AGENT"
    ]
    assert len(penalty_events) == 1
    assert penalty_events[0].cards_drawn_count == 2
