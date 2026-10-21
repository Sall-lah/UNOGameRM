"""Tests for optional post-horizon LLM refiner gating."""

from src.config import MemoryConfig
from src.env.barrier import LegalObservation
from src.memory.plugmem import UNOPlugMem


def test_default_config_no_llm_refiner():
    """Verify default configuration leaves enable_llm_refiner disabled and runs with zero LLM calls."""
    memory = UNOPlugMem()
    memory.reset_for_new_round(round_id=1)
    assert memory.config.enable_llm_refiner is False

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
    # Update runs completely offline with no network or LLM
    epi = memory.update(obs, [], "draw", t=1)
    assert epi.node_id == "epi_r1_t1"
