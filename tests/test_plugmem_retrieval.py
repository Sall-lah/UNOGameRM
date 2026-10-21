"""Tests for interleaved retrieval, value function reranking, and Top-K pruning."""

from src.config import MemoryConfig
from src.env.barrier import LegalObservation
from src.memory.embedding import LocalFallbackEmbedder
from src.memory.graph import PropositionNode
from src.memory.plugmem import UNOPlugMem


def test_retrieval_excludes_inactive_and_future_turn_nodes():
    """Verify retrieve() returns only active prior-turn nodes (origin_turn < t)."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)

    # Active prior turn node (t=1)
    p_valid = PropositionNode(
        node_id="prop_r1_valid",
        round_id=1,
        origin_turn=1,
        statement="Valid prior belief.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("colour:red",),
        confidence=0.85,
        active=True,
    )
    # Inactive node
    p_inactive = PropositionNode(
        node_id="prop_r1_inactive",
        round_id=1,
        origin_turn=1,
        statement="Inactive belief.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("colour:red",),
        confidence=0.0,
        active=False,
    )
    # Same-turn node (t=2)
    p_same_turn = PropositionNode(
        node_id="prop_r1_same",
        round_id=1,
        origin_turn=2,
        statement="Same turn belief.",
        category="OPPONENT_DRAW_CONSTRAINT",
        concept_tags=("colour:red",),
        confidence=0.85,
        active=True,
    )
    memory.graph.add_proposition(p_valid)
    memory.graph.add_proposition(p_inactive)
    memory.graph.add_proposition(p_same_turn)

    obs = LegalObservation(
        round_id=1,
        turn_idx=2,  # turn is 2
        agent_hand=("red 1",),
        top_discard="red 5",
        active_colour="red",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    res = memory.retrieve(obs)

    assert "prop_r1_valid" in res.retrieved_node_ids
    assert "prop_r1_inactive" not in res.retrieved_node_ids
    assert "prop_r1_same" not in res.retrieved_node_ids


def test_top_k_pruning_and_ordering():
    """Verify Top-K pruning limits results and orders them by descending score."""
    cfg = MemoryConfig(top_k_semantic=2)
    memory = UNOPlugMem(config=cfg)
    memory.reset_for_new_round(round_id=1)

    for i in range(5):
        prop = PropositionNode(
            node_id=f"prop_r1_{i}",
            round_id=1,
            origin_turn=1,
            statement=f"Belief {i}",
            category="OPPONENT_DRAW_CONSTRAINT",
            concept_tags=("colour:blue",),
            confidence=0.5 + (i * 0.1),
            active=True,
        )
        memory.graph.add_proposition(prop)

    obs = LegalObservation(
        round_id=1,
        turn_idx=3,
        agent_hand=("blue 1",),
        top_discard="blue 5",
        active_colour="blue",
        opp_card_count=7,
        legal_actions=("draw",),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    res = memory.retrieve(obs)

    semantic_items = [item for item in res.items if item.node_type == "semantic"]
    assert len(semantic_items) <= 2
    # Verify descending relevance order
    scores = [item.relevance_score for item in semantic_items]
    assert scores == sorted(scores, reverse=True)


def test_local_embedder_deterministic_and_offline():
    """Verify LocalFallbackEmbedder returns identical vectors and works without network."""
    embedder = LocalFallbackEmbedder()
    v1 = embedder.encode("The opponent drew a card on red 5.")
    v2 = embedder.encode("The opponent drew a card on red 5.")
    assert (v1 == v2).all()
