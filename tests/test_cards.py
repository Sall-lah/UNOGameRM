"""Tests for UNOCard parity against upstream UNO.py Card, immutability, and predicates."""

import pytest
from card import Card
from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    UNDERLINE_COMBINING,
    WILD,
    WILD_DRAW_FOUR,
    UNOCard,
    build_full_uno_deck,
)


def test_str_parity_for_all_deck_cards():
    """Verify UNOCard.__str__ matches upstream UNO.py Card.__str__ for all 108 deck cards."""
    deck = build_full_uno_deck()
    assert len(deck) == 108
    for uno_card in deck:
        py_card = uno_card.to_uno_py()
        assert str(uno_card) == str(py_card)


def test_str_parity_for_declared_wilds():
    """Verify UNOCard.__str__ matches declared wild cards of all colours."""
    for col in COLOURS:
        w = UNOCard("WIL", col, WILD)
        w4 = UNOCard("WIL", col, WILD_DRAW_FOUR)
        assert str(w) == str(Card("WIL", col, WILD))
        assert str(w4) == str(Card("WIL", col, WILD_DRAW_FOUR))


def test_predicates():
    """Verify is_wild, is_action_or_wild, and is_turn_retention_in_2p."""
    num = UNOCard("NUM", "red", "5")
    skip = UNOCard("ACT", "blue", SKIP)
    rev = UNOCard("ACT", "green", REVERSE)
    d2 = UNOCard("ACT", "yellow", DRAW_TWO)
    w = UNOCard("WIL", "white", WILD)
    w4 = UNOCard("WIL", "white", WILD_DRAW_FOUR)

    assert not num.is_wild
    assert not num.is_action_or_wild
    assert not num.is_turn_retention_in_2p

    assert not skip.is_wild
    assert skip.is_action_or_wild
    assert skip.is_turn_retention_in_2p

    assert not rev.is_wild
    assert rev.is_action_or_wild
    assert rev.is_turn_retention_in_2p

    assert not d2.is_wild
    assert d2.is_action_or_wild
    assert d2.is_turn_retention_in_2p

    assert w.is_wild
    assert w.is_action_or_wild
    assert not w.is_turn_retention_in_2p

    assert w4.is_wild
    assert w4.is_action_or_wild
    assert w4.is_turn_retention_in_2p


def test_is_playable_on():
    """Verify UNO.py playability matching rules."""
    top = UNOCard("NUM", "red", "5")
    assert UNOCard("NUM", "red", "7").is_playable_on(top)
    assert UNOCard("NUM", "blue", "5").is_playable_on(top)
    assert not UNOCard("NUM", "blue", "7").is_playable_on(top)
    assert UNOCard("WIL", "white", WILD).is_playable_on(top)
    assert UNOCard("WIL", "white", WILD_DRAW_FOUR).is_playable_on(top)


def test_immutability_and_hashability():
    """Verify UNOCard is frozen and hashable as set/dict keys."""
    card = UNOCard("NUM", "red", "5")
    with pytest.raises(Exception):
        card.colour = "blue"  # type: ignore

    s = {card, UNOCard("NUM", "red", "5")}
    assert len(s) == 1
    d = {card: "val"}
    assert d[UNOCard("NUM", "red", "5")] == "val"


def test_round_trip_conversion():
    """Verify lossless conversion between UNOCard and UNO.py Card."""
    py_card = Card("ACT", "green", REVERSE)
    uno_card = UNOCard.from_uno_py(py_card)
    assert uno_card.type == "ACT"
    assert uno_card.colour == "green"
    assert uno_card.value == REVERSE

    reconverted = uno_card.to_uno_py()
    assert reconverted.type == py_card.type
    assert reconverted.colour == py_card.colour
    assert reconverted.value == py_card.value
    assert str(reconverted) == str(py_card)
