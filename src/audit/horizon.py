"""HorizonMonitor and post-horizon event stream label evaluation.

Processes the typed HorizonServerEvent stream across H_t = [t, t+1) to derive
active opponent turn counts, horizon strata (1, 2, 3), Risk A adverse penalty labels,
Risk B opponent round wins, and disambiguated voluntary vs. penalty card quantities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Union

from src.env.events import HorizonEventType, HorizonServerEvent


@dataclass(frozen=True, slots=True)
class HorizonEvaluation:
    """Evaluated post-horizon outcome metrics and stratum classification.

    Why: Serves as the ground truth label container for Stage 2 evaluation and
    calibration dataset generation.
    """

    y_t_A_penalty_draw: int
    A_pen_cards_drawn: int
    A_nop_voluntary_drawn: int
    y_t_B_opp_win: int
    opp_turns_in_horizon: int
    stratum_horizon: int
    round_status: str
    challenge_penalty: int = 0


def classify_stratum(
    agent_played_retention_card: bool,
    opp_turns_in_horizon: int,
    round_terminated: bool,
) -> int:
    """Classifies the effective horizon duration into Stratum 1, 2, or 3.

    Why: Partitioning horizons by opponent active agency (0 turns, 1 turn, >=2 turns/terminal)
    enables unbiased statistical conditioning of adverse risk predictions.
    """
    if round_terminated:
        return 3

    if opp_turns_in_horizon >= 2:
        return 3

    if agent_played_retention_card and opp_turns_in_horizon == 0:
        return 1

    if opp_turns_in_horizon == 1:
        return 2

    # Default fallback for 0 opponent turns without explicit retention
    return 1 if opp_turns_in_horizon == 0 else 2


class HorizonMonitor:
    """Monitors and evaluates server events observed during horizon H_t = [t, t+1)."""

    def process_events(
        self,
        events: Sequence[Union[HorizonServerEvent, dict[str, Any]]],
        agent_played_retention_card: bool,
    ) -> HorizonEvaluation:
        """Processes event sequence and computes outcome labels and stratum classification.

        Args:
            events: Sequence of HorizonServerEvents or sanitized dictionaries from server.
            agent_played_retention_card: True if agent's locked action was Skip, Reverse, +2, or Wild+4.
        """
        opp_active_turns = 0
        y_t_A = 0
        a_pen_count = 0
        a_nop_count = 0
        y_t_B = 0
        round_terminated = False
        challenge_pen = 0

        for evt in events:
            # Extract attributes regardless of object or dict representation
            if isinstance(evt, HorizonServerEvent):
                actor = evt.actor
                evt_type = (
                    evt.event_type.value
                    if hasattr(evt.event_type, "value")
                    else str(evt.event_type)
                )
                drawn = evt.cards_drawn_count or 0
                winner = evt.winner
            elif isinstance(evt, dict):
                actor = evt.get("actor", "")
                raw_type = evt.get("event_type", "")
                evt_type = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
                drawn = int(evt.get("cards_drawn_count") or 0)
                winner = evt.get("winner")
            else:
                continue

            # 1. Active Opponent Turn Counting: Only count actions where opponent actively played
            if actor == "OPPONENT":
                if evt_type in (
                    HorizonEventType.PLAY_CARD.value,
                    HorizonEventType.DRAW_VOLUNTARY_ANOP.value,
                ):
                    opp_active_turns += 1

            # 2. Risk A (Forced Penalty Draw): Only set by DRAW_PENALTY_APEN targeting agent
            if actor == "AGENT":
                if evt_type == HorizonEventType.DRAW_PENALTY_APEN.value:
                    y_t_A = 1
                    a_pen_count += drawn
                elif evt_type == HorizonEventType.DRAW_VOLUNTARY_ANOP.value:
                    a_nop_count += (drawn if drawn > 0 else 1)
                elif evt_type == HorizonEventType.D4_CHALLENGE_LOST.value:
                    # Agent's own +4 was challenged successfully: penalty logged separately
                    challenge_pen += (drawn if drawn > 0 else 4)

            # 3. Risk B (Opponent Round Win)
            if evt_type == HorizonEventType.ROUND_WIN.value:
                round_terminated = True
                if winner == "OPPONENT" or actor == "OPPONENT":
                    y_t_B = 1

        stratum = classify_stratum(
            agent_played_retention_card=agent_played_retention_card,
            opp_turns_in_horizon=opp_active_turns,
            round_terminated=round_terminated,
        )

        return HorizonEvaluation(
            y_t_A_penalty_draw=y_t_A,
            A_pen_cards_drawn=a_pen_count,
            A_nop_voluntary_drawn=a_nop_count,
            y_t_B_opp_win=y_t_B,
            opp_turns_in_horizon=opp_active_turns,
            stratum_horizon=stratum,
            round_status="TERMINATED" if round_terminated else "ONGOING",
            challenge_penalty=challenge_pen,
        )
