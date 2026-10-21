"""Tests for full deck construction and inverse parsing of card strings."""

import pytest
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
    parse_card_string,
)


def test_build_full_uno_deck_multiplicities():
    """Verify 108 cards total: 76 NUM, 24 ACT, 8 WIL (all white)."""
    deck = build_full_uno_deck()
    assert len(deck) == 108

    num_cards = [c for c in deck if c.type == "NUM"]
    act_cards = [c for c in deck if c.type == "ACT"]
    wil_cards = [c for c in deck if c.type == "WIL"]

    assert len(num_cards) == 76
    assert len(act_cards) == 24
    assert len(wil_cards) == 8

    # All wilds built as white
    assert all(c.colour == "white" for c in wil_cards)

    # Per colour breakdown
    for col in COLOURS:
        col_nums = [c for c in num_cards if c.colour == col]
        assert len(col_nums) == 19
        zeros = [c for c in col_nums if c.value == "0"]
        assert len(zeros) == 1

        col_acts = [c for c in act_cards if c.colour == col]
        assert len(col_acts) == 6
        assert len([c for c in col_acts if c.value == SKIP]) == 2
        assert len([c for c in col_acts if c.value == REVERSE]) == 2
        assert len([c for c in col_acts if c.value == DRAW_TWO]) == 2


def test_parse_card_string_inverts_all_deck_renderings():
    """Verify parse_card_string correctly reconstructs every card in the deck."""
    deck = build_full_uno_deck()
    for card in deck:
        rendered = str(card)
        parsed = parse_card_string(rendered)
        assert parsed == card


def test_parse_card_string_declared_wilds():
    """Verify parse_card_string parses declared wild cards into colored UNOCards."""
    for col in COLOURS:
        w_str = f"WILD card, set to {col}"
        parsed_w = parse_card_string(w_str)
        assert parsed_w == UNOCard("WIL", col, WILD)

        w4_str = f"WILD + 4 card, set to {col}"
        parsed_w4 = parse_card_string(w4_str)
        assert parsed_w4 == UNOCard("WIL", col, WILD_DRAW_FOUR)


def test_parse_card_string_invalid():
    """Verify parse_card_string raises ValueError on invalid strings and 'draw'."""
    with pytest.raises(ValueError):
        parse_card_string("draw")

    with pytest.raises(ValueError):
        parse_card_string("invalid string")

    with pytest.raises(ValueError):
        parse_card_string("purple 5")

    with pytest.raises(ValueError):
        parse_card_string(123)  # type: ignore
