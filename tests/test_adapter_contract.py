"""Tests for ModifiedUNOServerPlatform contract, shapes, and information hiding."""

from card import Card
from src.core.cards import WILD
from src.env.events import HorizonEventType
from src.env.uno_engine import ModifiedUNOServerPlatform


def test_contract_shapes_and_hidden_state_protection():
    """Verify raw state dictionary keys and ensure hidden cards are not exposed."""
    platform = ModifiedUNOServerPlatform()
    raw_state = platform.start_new_round(round_id=10, starting_player=0, seed=808)

    # State shape validation
    assert "agent_hand" in raw_state
    assert "top_card" in raw_state
    assert "opp_hand_count" in raw_state
    assert "legal_actions" in raw_state
    assert "public_discards" in raw_state

    # Hidden state must NOT be in raw_state dict
    assert "opp_hand" not in raw_state
    assert "deck_cards" not in raw_state
    assert "discard_pile" not in raw_state

    # Dispatch result dictionary shape
    res = platform.dispatch_agent_action_and_observe_horizon("draw")
    expected_dispatch_keys = {
        "server_timeout_exceeded",
        "dispatch_latency_sec",
        "round_over",
        "winner",
        "horizon_events",
        "agent_hand_after_horizon",
        "opp_hand_count_after_horizon",
    }
    assert set(res.keys()) == expected_dispatch_keys


def test_legal_actions_expands_wilds_and_includes_draw():
    """Verify legal actions expansion includes all 4 colours for wild and includes 'draw'."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=909)

    platform.game.agent.hand = [
        Card("WIL", "white", WILD),
        Card("NUM", "red", "5"),
    ]
    platform.game.deck.discard_pile.append(Card("NUM", "red", "2"))

    legal = platform.get_raw_agent_state()["legal_actions"]

    assert "draw" in legal
    assert "red 5" in legal
    assert "WILD card, set to red" in legal
    assert "WILD card, set to green" in legal
    assert "WILD card, set to yellow" in legal
    assert "WILD card, set to blue" in legal


def test_reshuffle_event_emission():
    """Verify RESHUFFLE event is emitted with recycled count when deck is drawn dry."""
    platform = ModifiedUNOServerPlatform()
    platform.start_new_round(round_id=1, starting_player=0, seed=111)

    # Clear deck cards so next draw forces a reshuffle of discard pile
    platform.game.deck.cards.clear()
    platform.game.deck.discard_pile = [
        Card("NUM", "blue", "1"),
        Card("NUM", "blue", "2"),
        Card("NUM", "blue", "3"),
        Card("NUM", "blue", "4"),  # top card
    ]

    # Agent draws
    res = platform.dispatch_agent_action_and_observe_horizon("draw")

    reshuffle_events = [
        e for e in res["horizon_events"] if e.event_type == HorizonEventType.RESHUFFLE
    ]
    assert len(reshuffle_events) == 1
    assert reshuffle_events[0].recycled_count == 3  # 4 cards - 1 top card = 3 recycled
