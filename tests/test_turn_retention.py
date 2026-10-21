"""Tests for 2-player turn retention rules and stratum classification."""

from card import Card
from src.core.cards import DRAW_TWO, REVERSE, SKIP
from src.env.events import HorizonEventType
from src.env.uno_engine import ModifiedUNOServerPlatform


def test_agent_action_cards_cause_zero_active_opponent_turns():
    """Verify agent action cards yield Stratum 1 (0 active opponent turns)."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=101)

    # Test with Skip
    platform.game.agent.hand.append(Card("ACT", "red", SKIP))
    platform.game.deck.discard_pile.append(Card("NUM", "red", "5"))

    res = platform.dispatch_agent_action_and_observe_horizon("red skip")
    active_opp_turns = [
        e for e in res["horizon_events"] if e.actor == "OPPONENT" and e.active
    ]
    assert len(active_opp_turns) == 0
    assert any(
        e.event_type == HorizonEventType.TURN_FORFEIT and e.actor == "OPPONENT"
        for e in res["horizon_events"]
    )

    # Test with Draw Two
    platform.game.agent.hand.append(Card("ACT", "red", DRAW_TWO))
    platform.game.deck.discard_pile.append(Card("NUM", "red", "5"))

    res_d2 = platform.dispatch_agent_action_and_observe_horizon("red + 2")
    active_opp_d2 = [
        e for e in res_d2["horizon_events"] if e.actor == "OPPONENT" and e.active
    ]
    assert len(active_opp_d2) == 0
    assert any(
        e.event_type == HorizonEventType.DRAW_PENALTY_APEN
        and e.actor == "OPPONENT"
        and e.cards_drawn_count == 2
        for e in res_d2["horizon_events"]
    )


def test_opponent_action_card_chains_another_turn():
    """Verify opponent action card causes agent penalty/forfeit and chains another opponent turn."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=202)

    # Agent plays a plain green 5
    platform.game.agent.hand.append(Card("NUM", "green", "5"))
    platform.game.deck.discard_pile.append(Card("NUM", "green", "2"))

    # Force opponent to hold green SKIP as only playable card, and red 1 as non-playable
    platform.game.opp.hand = [Card("ACT", "green", SKIP), Card("NUM", "red", "1")]

    res = platform.dispatch_agent_action_and_observe_horizon("green 5")
    # Agent turn forfeit event should be emitted
    assert any(
        e.event_type == HorizonEventType.TURN_FORFEIT and e.actor == "AGENT"
        for e in res["horizon_events"]
    )
    # Opponent took at least two plays / active turns
    opp_plays = [
        e
        for e in res["horizon_events"]
        if e.actor == "OPPONENT" and e.event_type in (HorizonEventType.PLAY_CARD, HorizonEventType.DRAW_VOLUNTARY_ANOP)
    ]
    assert len(opp_plays) >= 2
