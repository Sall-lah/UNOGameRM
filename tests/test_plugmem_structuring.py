"""Tests for Rule-grounded structuring (Rules 1-3)."""

from src.core.cards import UNOCard, WILD
from src.env.barrier import LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.memory.plugmem import UNOPlugMem


def test_rules_1_to_3_induction():
    """Verify non-penalty draw (Rule 1), wild declaration (Rule 2), and depletion alert (Rule 3)."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("red 3",),
        top_discard="red 9",
        active_colour="red",
        opp_card_count=2,  # Depleted
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    events = [
        # Rule 1: opponent non-penalty voluntary draw
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
            active=True,
            outcome="keep",
            cards_drawn_count=1,
            opp_hand_size=2,
        ),
        # Rule 2: opponent declared wild colour
        HorizonServerEvent(
            event_index=1,
            actor="OPPONENT",
            event_type=HorizonEventType.WILD_COLOUR_DECLARED,
            active=True,
            declared_colour="yellow",
            opp_hand_size=2,
        ),
    ]

    epi = memory.update(obs, events, "draw", t=1)
    assert epi.node_id == "epi_r1_t1"

    # Verify Rule 1 proposition
    props = list(memory.graph.proposition_nodes.values())
    draw_props = [p for p in props if p.category == "OPPONENT_DRAW_CONSTRAINT"]
    assert len(draw_props) == 1
    assert "constraint:opp_lacks_colour" in draw_props[0].concept_tags
    assert "colour:red" in draw_props[0].concept_tags
    assert draw_props[0].confidence == 0.85
    assert draw_props[0].metadata["lacked_colour"] == "red"

    # Verify Rule 2 proposition
    wild_props = [p for p in props if p.category == "WILD_COLOUR_DECLARATION"]
    assert len(wild_props) == 1
    assert "constraint:opp_declared_wild" in wild_props[0].concept_tags
    assert "colour:yellow" in wild_props[0].concept_tags
    assert wild_props[0].confidence == 0.90

    # Verify Rule 3 depletion alert
    depletion_props = [p for p in props if p.category == "DEPLETION_ALERT"]
    assert len(depletion_props) == 1
    assert "intent:prevent_risk_B" in depletion_props[0].concept_tags
