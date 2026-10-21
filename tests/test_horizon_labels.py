"""Tests for HorizonMonitor target labels y_{t,A} and y_{t,B} (Task 7.3).

Verifies Risk A penalty draw isolation from voluntary draws, Risk B opponent win labels,
and the challenge-event label policy (accepted CPU +4 vs lost agent challenge).
"""

from __future__ import annotations

import pytest

from src.env.events import HorizonEventType, HorizonServerEvent
from src.audit.horizon import HorizonMonitor


def test_risk_a_set_only_by_penalty_draw():
    """Verify DRAW_PENALTY_APEN sets y_t_A_penalty_draw = 1 and accumulates drawn count."""
    monitor = HorizonMonitor()
    events = [
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.DRAW_PENALTY_APEN,
            active=False,
            cards_drawn_count=2,
        )
    ]
    res = monitor.process_events(events, agent_played_retention_card=False)
    assert res.y_t_A_penalty_draw == 1
    assert res.A_pen_cards_drawn == 2
    assert res.A_nop_voluntary_drawn == 0


def test_voluntary_draw_never_sets_risk_a():
    """Verify voluntary draw DRAW_VOLUNTARY_ANOP accumulates A_nop and leaves y_t_A = 0."""
    monitor = HorizonMonitor()
    events = [
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
            active=True,
            cards_drawn_count=1,
            outcome="keep",
        )
    ]
    res = monitor.process_events(events, agent_played_retention_card=False)
    assert res.y_t_A_penalty_draw == 0
    assert res.A_pen_cards_drawn == 0
    assert res.A_nop_voluntary_drawn == 1


def test_risk_b_set_only_by_opponent_round_win():
    """Verify ROUND_WIN with actor OPPONENT sets y_t_B_opp_win = 1; AGENT win does not."""
    monitor = HorizonMonitor()

    # Case 1: Opponent wins
    events_opp = [
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.ROUND_WIN,
            active=True,
            winner="OPPONENT",
        )
    ]
    res_opp = monitor.process_events(events_opp, agent_played_retention_card=False)
    assert res_opp.y_t_B_opp_win == 1
    assert res_opp.round_status == "TERMINATED"

    # Case 2: Agent wins
    events_agent = [
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.ROUND_WIN,
            active=True,
            winner="AGENT",
        )
    ]
    res_agent = monitor.process_events(events_agent, agent_played_retention_card=False)
    assert res_agent.y_t_B_opp_win == 0
    assert res_agent.round_status == "TERMINATED"


def test_challenge_label_policy_accepted_plus4_vs_lost_challenge():
    """Verify accepted CPU +4 sets y_t_A=1, but lost agent challenge records separately."""
    monitor = HorizonMonitor()

    # Accepted CPU +4: Agent draws 4 via DRAW_PENALTY_APEN
    events_accepted = [
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.DRAW_PENALTY_APEN,
            active=False,
            cards_drawn_count=4,
        )
    ]
    res_accepted = monitor.process_events(events_accepted, agent_played_retention_card=False)
    assert res_accepted.y_t_A_penalty_draw == 1
    assert res_accepted.A_pen_cards_drawn == 4
    assert res_accepted.challenge_penalty == 0

    # Lost agent challenge: D4_CHALLENGE_LOST
    events_challenge_lost = [
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.D4_CHALLENGE_LOST,
            active=False,
            cards_drawn_count=4,
        )
    ]
    res_challenge = monitor.process_events(events_challenge_lost, agent_played_retention_card=False)
    assert res_challenge.y_t_A_penalty_draw == 0  # Does NOT count as Risk A
    assert res_challenge.challenge_penalty == 4
