"""Tests for PlugMem round lifecycle isolation and counter reset."""

from src.env.barrier import LegalObservation
from src.env.events import HorizonEventType, HorizonServerEvent
from src.memory.plugmem import UNOPlugMem


def test_reset_clears_prior_round_state():
    """Verify reset_for_new_round clears nodes and restarts ID counters."""
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
    events = [
        HorizonServerEvent(
            event_index=0,
            actor="OPPONENT",
            event_type=HorizonEventType.DRAW_VOLUNTARY_ANOP,
            active=True,
            outcome="keep",
            cards_drawn_count=1,
        )
    ]
    memory.update(obs, events, "draw", t=1)
    assert len(memory.graph.proposition_nodes) > 0

    # Reset for round 2
    memory.reset_for_new_round(round_id=2)

    # In round 2, only seeded priors exist (no round 1 propositions)
    assert len(memory.graph.proposition_nodes) == 0
    assert len(memory.graph.episodic_nodes) == 0
    assert memory.graph._prop_counter == 0


def test_no_cross_round_leakage_in_retrieval():
    """Verify retrieve() returns no node IDs encoding prior rounds."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    obs_r1 = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("blue 2",),
        top_discard="blue 5",
        active_colour="blue",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    memory.update(obs_r1, [], "draw", t=1)

    memory.reset_for_new_round(round_id=2)
    obs_r2 = LegalObservation(
        round_id=2,
        turn_idx=1,
        agent_hand=("blue 2",),
        top_discard="blue 5",
        active_colour="blue",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    res = memory.retrieve(obs_r2)

    for nid in res.retrieved_node_ids:
        assert "r1" not in nid
