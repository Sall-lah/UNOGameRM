"""Tests ensuring the human interactive Game class remains untouched and inert."""

from deck import Deck
from player import AIPlayer, Player
from uno import Game


def test_default_game_construction():
    """Verify default Game instance has original properties and human entry point intact."""
    game = Game()
    assert hasattr(game, "start")
    assert hasattr(game, "play_turn")
    assert hasattr(game, "setup")
    assert game.players == []
    assert isinstance(game.deck, Deck)
    assert game.first_turn is True


def test_default_deck_and_player():
    """Verify default Deck and AIPlayer construction."""
    deck = Deck()
    assert len(deck.cards) == 108
    assert len(deck.discard_pile) == 0

    ai = AIPlayer("TEST")
    assert ai.name == "TEST"
    assert ai.intellect in (1, 2, 3)
    assert isinstance(ai, Player)
