"""Tests for HorizonMonitor turn counting and stratum classification (Task 7.2).

Verifies active opponent turn accumulation excluding forfeits/penalties,
and Stratum 1/2/3 classification under all horizon conditions.
"""

from __future__ import annotations

import pytest

from src.env.events import HorizonEventType, HorizonServerEvent
from src.audit.horizon import (
    HorizonMonitor,
    classify_stratum,
)


def test_active_turn_counting_excludes_forfeits_and_penalties():
    """Verify TURN_FORFEIT and DRAW_PENALTY_APEN do not increment opp_turns_in_horizon."""
    monitor = HorizonMonitor()

    events = [
        # Opponent was skipped: TURN_FORFEIT
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.TURN_FORFEIT,
            active=False,
        ),
        # Opponent forced to draw: penalty-consumed turn
        HorizonServerEvent(
            event_index=2,
            actor="OPPONENT",
            event_type=HorizonEventType.DRAW_PENALTY_APEN,
            active=False,
            cards_drawn_count=2,
        ),
    ]

    res = monitor.process_events(events, agent_played_retention_card=True)
    # Both turns were non-active; count must be 0
    assert res.opp_turns_in_horizon == 0


def test_voluntary_draw_and_play_are_counted_as_active_turns():
    """Verify PLAY_CARD and DRAW_VOLUNTARY_ANOP by opponent count as active turns."""
    monitor = HorizonMonitor()

    events = [
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
            active=True,
            cards_drawn_count=1,
            outcome="keep",
        ),
    ]

    res = monitor.process_events(events, agent_played_retention_card=False)
    assert res.opp_turns_in_horizon == 1


def test_stratum_1_agent_retains_turn():
    """Verify agent turn-retention with 0 active opponent turns yields Stratum 1."""
    monitor = HorizonMonitor()
    events = [
        # Opponent skipped
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.TURN_FORFEIT,
            active=False,
        )
    ]
    res = monitor.process_events(events, agent_played_retention_card=True)
    assert res.opp_turns_in_horizon == 0
    assert res.stratum_horizon == 1


def test_stratum_2_single_opponent_turn():
    """Verify single active opponent turn without termination yields Stratum 2."""
    monitor = HorizonMonitor()
    events = [
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
        )
    ]
    res = monitor.process_events(events, agent_played_retention_card=False)
    assert res.opp_turns_in_horizon == 1
    assert res.stratum_horizon == 2


def test_stratum_3_chained_opponent_turns():
    """Verify 2 or more active opponent turns yield Stratum 3."""
    monitor = HorizonMonitor()
    events = [
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
        ),
        HorizonServerEvent(
            event_index=2,
            actor="OPPONENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
        ),
    ]
    res = monitor.process_events(events, agent_played_retention_card=False)
    assert res.opp_turns_in_horizon == 2
    assert res.stratum_horizon == 3


def test_stratum_3_on_round_termination():
    """Verify round termination yields Stratum 3 regardless of opponent turn count."""
    monitor = HorizonMonitor()
    events = [
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.ROUND_WIN,
            active=True,
            winner="AGENT",
        )
    ]
    res = monitor.process_events(events, agent_played_retention_card=False)
    assert res.opp_turns_in_horizon == 0
    assert res.stratum_horizon == 3
    assert res.round_status == "TERMINATED"
