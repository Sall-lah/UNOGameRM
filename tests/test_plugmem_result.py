"""Tests for RetrievalResult structure, provenance, and distilled context."""

from src.env.barrier import LegalObservation
from src.memory.graph import PropositionNode
from src.memory.plugmem import UNOPlugMem


def test_retrieval_result_structured_summaries_and_distillation():
    """Verify active_opp_lack_colours and active_opp_wild_colours are populated with provenance."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    prop1 = PropositionNode(
        node_id="prop_r1_0",
        round_id=1,
        origin_turn=1,
        statement="Opponent lacks green.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("constraint:opp_lacks_colour", "colour:green"),
        confidence=0.85,
        active=True,
        provenance_epi_ids=("epi_r1_t1",),
        metadata={"lacked_colour": "green"},
    )
    prop2 = PropositionNode(
        node_id="prop_r1_1",
        round_id=1,
        origin_turn=2,
        statement="Opponent declared yellow.",
        category="WILD_COLOUR_DECLARATION",
        concept_tags=("constraint:opp_declared_wild", "colour:yellow"),
        confidence=0.90,
        active=True,
        provenance_epi_ids=("epi_r1_t2",),
        metadata={"declared_colour": "yellow"},
    )
    memory.graph.add_proposition(prop1)
    memory.graph.add_proposition(prop2)

    obs = LegalObservation(
        round_id=1,
        turn_idx=3,
        agent_hand=("green 5",),
        top_discard="green 2",
        active_colour="green",
        opp_card_count=6,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    res = memory.retrieve(obs)

    assert "green" in res.active_opp_lack_colours
    assert res.active_opp_lack_colours["green"] == 0.85
    assert (2, "yellow") in res.active_opp_wild_colours

    # Verify provenance IDs
    for item in res.items:
        if item.node_id == "prop_r1_0":
            assert "epi_r1_t1" in item.provenance_ids
        elif item.node_id == "prop_r1_1":
            assert "epi_r1_t2" in item.provenance_ids

    # Verify distilled context is structured and bounded
    assert "=== VERIFIED OPPONENT CONSTRAINTS ===" in res.distilled_context
    assert "=== TACTICAL PROCEDURAL PRESCRIPTIONS ===" in res.distilled_context
    assert res.token_estimate > 0


def test_empty_memory_distilled_context():
    """Verify empty memory returns well-formed default distilled context."""
    from src.config import MemoryConfig

    cfg = MemoryConfig(seed_procedural_priors=False)
    memory = UNOPlugMem(config=cfg)
    memory.reset_for_new_round(round_id=1)

    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("red 1",),
        top_discard="red 2",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    res = memory.retrieve(obs)

    assert "[PlugMem Intra-Round Context: No intra-round opponent constraints" in res.distilled_context
    assert res.token_estimate > 0
