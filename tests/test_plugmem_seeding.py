"""Tests for PlugMem procedural seeding on and off."""

from src.config import MemoryConfig
from src.env.barrier import LegalObservation
from src.memory.plugmem import UNOPlugMem


def test_turn_1_procedural_guidance_present_when_seeding_on():
    """Verify turn-1 retrieval contains procedural guidance with origin_turn=0 when seeding is on."""
    cfg = MemoryConfig(seed_procedural_priors=True)
    memory = UNOPlugMem(config=cfg)
    memory.reset_for_new_round(round_id=1)

    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("green 4",),
        top_discard="green 7",
        active_colour="green",
        opp_card_count=7,
        legal_actions=("green 4", "draw"),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )
    res = memory.retrieve(obs)

    procedural_items = [item for item in res.items if item.node_type == "procedural"]
    assert len(procedural_items) >= 1
    assert any(item.origin_turn == 0 for item in procedural_items)


def test_empty_procedural_graph_when_seeding_off():
    """Verify no prescriptions exist immediately after reset when seeding is disabled."""
    cfg = MemoryConfig(seed_procedural_priors=False)
    memory = UNOPlugMem(config=cfg)
    memory.reset_for_new_round(round_id=1)

    assert len(memory.graph.prescription_nodes) == 0
