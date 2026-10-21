"""Tests for Rule 5 procedural skill induction scoring formula."""

from src.core.cards import UNOCard
from src.env.barrier import LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.memory.plugmem import UNOPlugMem


def test_stratum_1_scores_higher_than_adverse_stratum_2():
    """Verify Stratum 1 success scores higher than Stratum 2 with y_A=1, both in [1, 10]."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("red 1",),
        top_discard="red 5",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    # 1. Stratum 1 success (0 active opp turns, y_A=0, y_B=0)
    events_strat1 = [
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.TURN_FORFEIT,
            active=False,
        )
    ]
    memory.update(obs, events_strat1, "red skip", t=1)
    presc_strat1 = list(memory.graph.prescription_nodes.values())[-1]

    # 2. Stratum 2 adverse (1 active opp turn, y_A=1)
    events_strat2 = [
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
            card=UNOCard("ACT", "red", "+ 2"),
        ),
        HorizonServerEvent(
            event_index=1,
            actor="AGENT",
            event_type=HorizonEventType.DRAW_PENALTY_APEN,
            active=False,
            cards_drawn_count=2,
        ),
    ]
    memory.update(obs, events_strat2, "red 5", t=2)
    presc_strat2 = list(memory.graph.prescription_nodes.values())[-1]

    assert presc_strat1.return_score > presc_strat2.return_score
    assert 1.0 <= presc_strat1.return_score <= 10.0
    assert 1.0 <= presc_strat2.return_score <= 10.0
