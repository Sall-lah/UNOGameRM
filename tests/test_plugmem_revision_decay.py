"""Tests for belief revision (contradiction) and gamma^k draw decay/pruning."""

import pytest
from src.core.cards import UNOCard
from src.env.barrier import LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.memory.graph import PropositionNode
from src.memory.plugmem import UNOPlugMem


def test_contradiction_soft_deactivates_constraint():
    """Verify playing a previously lacked colour sets active=False and confidence=0.0."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    # Manually seed an active lack-colour constraint for 'blue'
    prop_id = memory.graph.next_prop_id()
    prop = PropositionNode(
        node_id=prop_id,
        round_id=1,
        origin_turn=1,
        statement="Opponent lacks blue.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("constraint:opp_lacks_colour", "colour:blue"),
        confidence=0.85,
        active=True,
        metadata={"lacked_colour": "blue"},
    )
    memory.graph.add_proposition(prop)

    obs = LegalObservation(
        round_id=1,
        turn_idx=2,
        agent_hand=("red 1",),
        top_discard="blue 5",
        active_colour="blue",
        opp_card_count=6,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    # Opponent plays blue 7
    events = [
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.PLAY_CARD,
            active=True,
            card=UNOCard("NUM", "blue", "7"),
        )
    ]
    memory.update(obs, events, "draw", t=2)

    assert prop.active is False
    assert prop.confidence == 0.0
    assert prop.deactivation_turn == 2


def test_draw_confidence_decay_and_pruning():
    """Verify confidence decays by gamma^k and prunes below prune_threshold."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    # Proposition with 0.40 confidence. gamma = 0.75, prune_threshold = 0.35
    # Drawing 2 cards: 0.40 * (0.75^2) = 0.225 -> below threshold -> soft-deactivated
    prop_low = PropositionNode(
        node_id=memory.graph.next_prop_id(),
        round_id=1,
        origin_turn=1,
        statement="Opponent lacks yellow.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("constraint:opp_lacks_colour", "colour:yellow"),
        confidence=0.40,
        active=True,
        metadata={"lacked_colour": "yellow"},
    )
    memory.graph.add_proposition(prop_low)

    # Proposition with 0.85 confidence. Drawing 1 card: 0.85 * 0.75 = 0.6375 -> above threshold -> retained
    prop_high = PropositionNode(
        node_id=memory.graph.next_prop_id(),
        round_id=1,
        origin_turn=1,
        statement="Opponent lacks green.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("constraint:opp_lacks_colour", "colour:green"),
        confidence=0.85,
        active=True,
        metadata={"lacked_colour": "green"},
    )
    memory.graph.add_proposition(prop_high)

    obs = LegalObservation(
        round_id=1,
        turn_idx=2,
        agent_hand=("red 1",),
        top_discard="red 5",
        active_colour="red",
        opp_card_count=8,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    # Opponent draws 2 cards
    events = [
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.DRAW_PENALTY_APEN,
            active=False,
            cards_drawn_count=2,
        )
    ]
    memory.update(obs, events, "draw", t=2)

    # Low proposition pruned
    assert prop_low.active is False
    assert prop_low.confidence == pytest.approx(0.225)

    # High proposition retained (active=True, decayed confidence 0.85 * 0.75^2 = 0.478125 >= 0.35)
    assert prop_high.active is True
    assert prop_high.confidence == pytest.approx(0.85 * (0.75**2))
