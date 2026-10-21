"""Tests for automatic UNO declaration on hand size 2 to 1 transitions."""

from card import Card
from src.env.events import HorizonEventType
from src.env.uno_engine import ModifiedUNOServerPlatform


def test_auto_uno_on_card_play():
    """Verify playing a card while holding 2 cards sets uno=True and emits UNO_DECLARED."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=606)

    # Set agent hand to exactly 2 cards
    card_to_play = Card("NUM", "blue", "3")
    other_card = Card("NUM", "red", "8")
    platform.game.agent.hand = [card_to_play, other_card]
    platform.game.agent.uno = False
    platform.game.deck.discard_pile.append(Card("NUM", "blue", "1"))
    # Opponent plays normal card
    platform.game.opp.hand = [Card("NUM", "blue", "5")]

    res = platform.dispatch_agent_action_and_observe_horizon("blue 3")

    assert platform.game.agent.uno is True
    assert any(
        e.event_type == HorizonEventType.UNO_DECLARED and e.actor == "AGENT"
        for e in res["horizon_events"]
    )


def test_auto_uno_on_draw_with_one_card():
    """Verify drawing when holding 1 card declares UNO, and if auto-played preserves uno=True."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=707)

    # Set agent hand to 1 card
    platform.game.agent.hand = [Card("NUM", "green", "2")]
    platform.game.agent.uno = False
    platform.game.deck.discard_pile.append(Card("NUM", "green", "4"))
    # Next card in deck is playable green 7 (auto-play)
    platform.game.deck.cards.append(Card("NUM", "green", "7"))
    platform.game.opp.hand = [Card("NUM", "green", "9̲")]

    res = platform.dispatch_agent_action_and_observe_horizon("draw")

    assert platform.game.agent.uno is True
    assert any(
        e.event_type == HorizonEventType.UNO_DECLARED and e.actor == "AGENT"
        for e in res["horizon_events"]
    )
