"""Information barrier (Zone 1 to Zone 2) enforcing client-server isolation.

This module guarantees that the agent harness only ever receives a sanitized,
frozen LegalObservation. It acts as an enforceable boundary tripwire asserting
the total absence of hidden engine structures (cpu hand, deck order, future states).

NOTE ON ONE-DIRECTIONAL BARRIER ASYMMETRY:
The information barrier strips hidden opponent state and deck ordering from the agent,
ensuring Zone 2 cannot leak ground truth cards. However, the upstream UNO.py AIPlayer
opponent logic (think_wild, think_challenge) peeks at the agent's hand by design.
This is a documented one-directional barrier limitation and validity caveat, not
a defect to be patched here, as modifying the opponent would alter the fixed reference policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional

from src.config import AppConfig, load_config
from src.core.cards import (
    COLOURS,
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD,
    WILD_DRAW_FOUR,
    UNOCard,
    build_full_uno_deck,
    parse_card_string,
)
from src.env.events import HorizonEventType, HorizonServerEvent

# Explicit denylist of internal keys that must never appear in raw engine states.
FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {"cpu_hand", "opponent_cards", "draw_pile", "deck_order", "future_state"}
)


@dataclass(frozen=True, slots=True)
class LegalObservation:
    """Immutable, leak-free observation exported to the agent decision harness.

    All collection attributes are frozen tuples containing only strings or primitives,
    preventing both accidental mutation and hidden object reference smuggling.
    """

    round_id: int
    turn_idx: int
    agent_hand: tuple[str, ...]
    top_discard: str
    active_colour: str
    opp_card_count: int
    legal_actions: tuple[str, ...]
    public_events_since_last_turn: tuple[dict[str, Any], ...]
    known_discards_since_reshuffle: tuple[str, ...]


class ClientServerBarrier:
    """Enforces information filtering between the engine server and the agent client."""

    @classmethod
    def extract_observation(cls, raw_engine_state: dict[str, Any]) -> LegalObservation:
        """Inspects raw engine state, verifies assertions, and builds a LegalObservation.

        Raises AssertionError if any forbidden internal key is present, and deep-copies
        all fields into immutable tuples.
        """
        # Tripwire check against sensitive internal engine leaks
        for forbidden_key in FORBIDDEN_KEYS:
            if forbidden_key in raw_engine_state:
                raise AssertionError(
                    f"Information barrier violation: forbidden key '{forbidden_key}' detected in raw state."
                )

        isolated_state = copy.deepcopy(raw_engine_state)

        round_id = int(isolated_state.get("round_id", 0))
        turn_idx = int(isolated_state.get("t", isolated_state.get("turn_idx", 1)))
        agent_hand = tuple(str(c) for c in isolated_state.get("agent_hand", ()))
        top_discard = str(isolated_state.get("top_card", isolated_state.get("discard_pile_top", "")))

        # Determine active colour from top discard
        active_colour = cls._derive_active_colour(top_discard)

        opp_card_count = int(isolated_state.get("opp_hand_count", 0))

        # Build or normalize legal actions
        raw_legal = isolated_state.get("legal_actions")
        if raw_legal is not None:
            legal_actions = tuple(cls._normalize_legal_actions(raw_legal))
        else:
            legal_actions = tuple(cls._compute_legal_actions(agent_hand, top_discard))

        # Forward public events without leaking private cards
        events_input = isolated_state.get("horizon_events", isolated_state.get("public_events_since_last_turn", ()))
        public_events = tuple(cls._sanitize_public_events(events_input))

        # Public discards tracked since last reshuffle
        raw_discards = isolated_state.get("public_discards", isolated_state.get("known_discards_since_reshuffle", ()))
        known_discards = tuple(str(c) for c in raw_discards)

        return LegalObservation(
            round_id=round_id,
            turn_idx=turn_idx,
            agent_hand=agent_hand,
            top_discard=top_discard,
            active_colour=active_colour,
            opp_card_count=opp_card_count,
            legal_actions=legal_actions,
            public_events_since_last_turn=public_events,
            known_discards_since_reshuffle=known_discards,
        )

    @staticmethod
    def _derive_active_colour(top_discard: str) -> str:
        """Derives the active playable colour from a top card representation."""
        if top_discard.startswith("WILD"):
            parts = top_discard.split("set to ")
            if len(parts) == 2 and parts[1] in COLOURS:
                return parts[1]
            return "red"
        col = top_discard.split(" ")[0]
        if col in COLOURS:
            return col
        return "red"

    @classmethod
    def _compute_legal_actions(cls, agent_hand: tuple[str, ...], top_discard: str) -> list[str]:
        """Calculates legal actions from agent hand cards and top discard string."""
        actions: list[str] = []
        top_card_obj = parse_card_string(top_discard) if top_discard else None

        for card_str in agent_hand:
            card = parse_card_string(card_str)
            if top_card_obj is None or card.is_playable_on(top_card_obj):
                if card.is_wild:
                    suffix = " + 4" if card.value == WILD_DRAW_FOUR else ""
                    for col in COLOURS:
                        actions.append(f"WILD{suffix} card, set to {col}")
                else:
                    actions.append(card_str)

        actions.append("draw")
        return cls._normalize_legal_actions(actions)

    @staticmethod
    def _normalize_legal_actions(actions: list[str]) -> list[str]:
        """Collapses duplicate actions while ensuring draw is always present at the end."""
        seen = set()
        deduped = []
        for a in actions:
            if a != "draw" and a not in seen:
                seen.add(a)
                deduped.append(a)
        deduped.append("draw")
        return deduped

    @staticmethod
    def _sanitize_public_events(events: Any) -> list[dict[str, Any]]:
        """Strips hidden card identities from public event streams."""
        sanitized: list[dict[str, Any]] = []
        for e in events:
            if isinstance(e, HorizonServerEvent):
                # Hide drawn card for voluntary keep draws
                card_repr = str(e.card) if e.card and e.outcome != "keep" else None
                evt_dict = {
                    "event_index": e.event_index,
                    "actor": e.actor,
                    "event_type": e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type),
                    "active": e.active,
                    "card": card_repr,
                    "declared_colour": e.declared_colour,
                    "cards_drawn_count": e.cards_drawn_count,
                    "recycled_count": e.recycled_count,
                    "outcome": e.outcome,
                    "agent_hand_size": e.agent_hand_size,
                    "opp_hand_size": e.opp_hand_size,
                    "winner": e.winner,
                }
                sanitized.append(evt_dict)
            elif isinstance(e, dict):
                clean_e = copy.deepcopy(e)
                if clean_e.get("event_type") == "DRAW_VOLUNTARY_ANOP" and clean_e.get("outcome") == "keep":
                    clean_e.pop("card", None)
                sanitized.append(clean_e)
        return sanitized


def compute_unseen_pool(
    obs: LegalObservation,
    refine: Optional[bool] = None,
    config: Optional[AppConfig] = None,
) -> list[UNOCard]:
    """Computes the unseen card multiset U_t from public observation fields only.

    Derives U_t = D_108 \\ (agent_hand ∪ {top_discard}), with optional refinement
    subtracting known discards since reshuffle when gated on config flag.
    """
    if refine is None:
        cfg = config or load_config()
        refine = cfg.barrier.refine_unseen_pool

    deck = build_full_uno_deck()
    pool = list(deck)

    def _remove_card_from_pool(card_to_remove: UNOCard) -> None:
        for idx, c in enumerate(pool):
            # For wilds in the deck, match on type and value, ignoring colour which is 'white'
            if card_to_remove.is_wild and c.is_wild and c.value == card_to_remove.value:
                pool.pop(idx)
                return
            elif c == card_to_remove:
                pool.pop(idx)
                return

    # Subtract agent hand cards
    for card_str in obs.agent_hand:
        card = parse_card_string(card_str)
        _remove_card_from_pool(card)

    # Subtract top discard instance
    if obs.top_discard:
        top_card = parse_card_string(obs.top_discard)
        _remove_card_from_pool(top_card)

    # If refined variant requested, subtract public discards since reshuffle
    if refine:
        for discard_str in obs.known_discards_since_reshuffle:
            d_card = parse_card_string(discard_str)
            _remove_card_from_pool(d_card)

    return pool
