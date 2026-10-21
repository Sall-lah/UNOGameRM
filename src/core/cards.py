"""Canonical UNO card and action representations matching the upstream UNO.py engine.

This module provides immutable, frozen value types for cards and actions, ensuring
that downstream components (barrier, MCTS, PlugMem, audit log) operate on verified,
leak-free domain objects that faithfully reproduce the rules of UNO.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

# Canonical colours used throughout the upstream UNO.py implementation.
COLOURS: tuple[str, ...] = ("red", "green", "yellow", "blue")

# Explicit unicode escapes for underlined numbers and action glyphs matching UNO.py.
UNDERLINE_COMBINING: str = "\u0332"
SKIP: str = "\u2298"
REVERSE: str = "\u21c4"
DRAW_TWO: str = "+ 2"
WILD: str = "\u2a01"
WILD_DRAW_FOUR: str = "+ 4"

# Value representations used when expressing cards as human-readable strings.
_VAL_EXPRESSIONS: dict[str, str] = {
    f"6{UNDERLINE_COMBINING}": "6",
    f"9{UNDERLINE_COMBINING}": "9",
    SKIP: "skip",
    REVERSE: "reverse",
}

# Reverse mapping for string parsing back into canonical card tokens.
_INVERSE_VAL_EXPRESSIONS: dict[str, tuple[str, str]] = {
    "0": ("NUM", "0"),
    "1": ("NUM", "1"),
    "2": ("NUM", "2"),
    "3": ("NUM", "3"),
    "4": ("NUM", "4"),
    "5": ("NUM", "5"),
    "6": ("NUM", f"6{UNDERLINE_COMBINING}"),
    f"6{UNDERLINE_COMBINING}": ("NUM", f"6{UNDERLINE_COMBINING}"),
    "7": ("NUM", "7"),
    "8": ("NUM", "8"),
    "9": ("NUM", f"9{UNDERLINE_COMBINING}"),
    f"9{UNDERLINE_COMBINING}": ("NUM", f"9{UNDERLINE_COMBINING}"),
    "skip": ("ACT", SKIP),
    SKIP: ("ACT", SKIP),
    "reverse": ("ACT", REVERSE),
    REVERSE: ("ACT", REVERSE),
    "+ 2": ("ACT", DRAW_TWO),
    DRAW_TWO: ("ACT", DRAW_TWO),
}

_WILD_STRING_PATTERN = re.compile(
    r"^WILD(?: (\+ 4))? card, set to (red|green|yellow|blue|white)$"
)


@dataclass(frozen=True, slots=True)
class UNOCard:
    """Immutable representation of an individual UNO card.

    Frozen dataclass ensures cards are safe to share across concurrent evaluation
    threads, hashable for state caching, and cannot have their identities mutated
    by downstream simulation or logging logic.
    """

    type: str  # "NUM", "ACT", "WIL"
    colour: str  # "red", "green", "yellow", "blue", or "white"
    value: str  # Literal card value token matching UNO.py

    @property
    def is_wild(self) -> bool:
        """Determines if the card is a wild or wild draw four."""
        return self.type == "WIL"

    @property
    def is_action_or_wild(self) -> bool:
        """Identifies action or wild cards that alter turn order or carry special effects."""
        return self.type != "NUM"

    @property
    def is_turn_retention_in_2p(self) -> bool:
        """Identifies cards that skip the opponent in 2-player rules.

        In 2-player UNO, Skip, Reverse, Draw Two, and Wild Draw Four all cause the
        acting player to immediately retain the turn rather than switching active players.
        """
        return self.value in (SKIP, REVERSE, DRAW_TWO, WILD_DRAW_FOUR)

    def is_playable_on(self, top_card: Any) -> bool:
        """Evaluates whether this card can be legally placed on top of top_card.

        Wilds are always legally playable. Non-wild cards require either color or value
        alignment with the current active top card, strictly adhering to UNO.py rules.
        """
        if self.type == "WIL":
            return True
        return self.colour == top_card.colour or self.value == top_card.value

    def __str__(self) -> str:
        """Produces a string representation byte-identical to UNO.py Card.__str__."""
        if self.type == "WIL":
            suffix = " + 4" if self.value == WILD_DRAW_FOUR else ""
            return f"WILD{suffix} card, set to {self.colour}"
        expr = _VAL_EXPRESSIONS.get(self.value, self.value)
        return f"{self.colour} {expr}"

    @classmethod
    def from_uno_py(cls, card: Any) -> UNOCard:
        """Converts an upstream UNO.py Card object into an immutable UNOCard.

        Enables boundary decoupling so internal research components never hold mutable
        references to objects modified in-place by the engine loop.
        """
        return cls(type=card.type, colour=card.colour, value=card.value)

    def to_uno_py(self) -> Any:
        """Converts this UNOCard into an upstream UNO.py Card instance.

        Allows passing adapter-generated card states into the upstream engine
        functions without altering upstream class signatures.
        """
        from card import Card

        return Card(self.type, self.colour, self.value)


@dataclass(frozen=True, slots=True)
class UNOAction:
    """Represents a discrete action taken by an agent or opponent.

    Guarantees consistent serialization of player decisions across MCTS exploration,
    barrier filtering, and audit log generation.
    """

    action_id: str
    action_type: str  # "PLAY", "DRAW", "D4_RESPONSE"
    card: Optional[UNOCard] = None
    declared_colour: Optional[str] = None


def build_full_uno_deck() -> list[UNOCard]:
    """Constructs the canonical 108-card UNO deck according to official rules.

    Provides a clean, deterministic baseline deck for seeding games, verifying
    resampling distributions, and running offline simulation rollouts.
    """
    num_card_values = (
        ("1", "2", "3", "4", "5", f"6{UNDERLINE_COMBINING}", "7", "8", f"9{UNDERLINE_COMBINING}") * 2
        + ("0",)
    )
    act_card_values = (SKIP, REVERSE, DRAW_TWO) * 2
    wil_card_values = (WILD, WILD_DRAW_FOUR) * 4

    deck: list[UNOCard] = []
    for c in COLOURS:
        for v in num_card_values:
            deck.append(UNOCard(type="NUM", colour=c, value=v))
        for v in act_card_values:
            deck.append(UNOCard(type="ACT", colour=c, value=v))
    for v in wil_card_values:
        deck.append(UNOCard(type="WIL", colour="white", value=v))

    return deck


def parse_card_string(card_str: str) -> UNOCard:
    """Parses a rendered card string back into its canonical UNOCard representation.

    Reconstructs card entities from string action identifiers produced by agents
    or logging interfaces while strictly rejecting invalid or action-only strings.
    """
    if not isinstance(card_str, str):
        raise ValueError(f"Card string must be a string, got {type(card_str)}")

    wild_match = _WILD_STRING_PATTERN.match(card_str)
    if wild_match:
        is_d4, colour = wild_match.groups()
        value = WILD_DRAW_FOUR if is_d4 else WILD
        return UNOCard(type="WIL", colour=colour, value=value)

    parts = card_str.split(" ", 1)
    if len(parts) != 2 or parts[0] not in COLOURS:
        raise ValueError(f"Invalid card string: '{card_str}'")

    colour, val_part = parts[0], parts[1]
    if val_part not in _INVERSE_VAL_EXPRESSIONS:
        raise ValueError(f"Unrecognized card value expression: '{val_part}' in '{card_str}'")

    card_type, canonical_val = _INVERSE_VAL_EXPRESSIONS[val_part]
    return UNOCard(type=card_type, colour=colour, value=canonical_val)
