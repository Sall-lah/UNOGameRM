"""Typed telemetry and server event definitions for the UNO environment horizon.

This module formalizes the event contract across the horizon H_t = [t, t+1).
Structuring events with explicit actors, activity flags, and disambiguated draw types
enables downstream audit logs (change 6) to partition penalty draws from voluntary draws
and accurately measure player latency and stratum transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from src.core.cards import UNOCard


class HorizonEventType(str, Enum):
    """The canonical set of server events that can occur during a decision horizon.

    Restricted to exactly ten variants to prevent ambiguous state transitions
    from polluting downstream statistical analysis and audit logging.
    """

    PLAY_CARD = "PLAY_CARD"
    DRAW_VOLUNTARY_ANOP = "DRAW_VOLUNTARY_ANOP"
    DRAW_PENALTY_APEN = "DRAW_PENALTY_APEN"
    D4_CHALLENGE_WON = "D4_CHALLENGE_WON"
    D4_CHALLENGE_LOST = "D4_CHALLENGE_LOST"
    TURN_FORFEIT = "TURN_FORFEIT"
    WILD_COLOUR_DECLARED = "WILD_COLOUR_DECLARED"
    UNO_DECLARED = "UNO_DECLARED"
    RESHUFFLE = "RESHUFFLE"
    ROUND_WIN = "ROUND_WIN"


ActorType = Literal["AGENT", "OPPONENT"]
DrawOutcomeType = Literal["keep", "auto_play"]


@dataclass(frozen=True, slots=True)
class HorizonServerEvent:
    """Represents a single verifiable atomic event observed in the game engine.

    Captured over the decision horizon H_t = [t, t+1) to track engine modifications,
    evaluate audit invariants, and derive target labels y_{t,A} and y_{t,B}.
    """

    event_index: int
    actor: str  # "AGENT" or "OPPONENT"
    event_type: HorizonEventType
    active: bool  # True if the event reflects an active decision turn by the actor
    card: Optional[UNOCard] = None
    declared_colour: Optional[str] = None
    cards_drawn_count: Optional[int] = None
    recycled_count: Optional[int] = None
    outcome: Optional[str] = None  # "keep" or "auto_play" for voluntary draws
    agent_hand_size: Optional[int] = None
    opp_hand_size: Optional[int] = None
    winner: Optional[str] = None
    reason: Optional[str] = None
