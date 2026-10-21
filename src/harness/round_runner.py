"""Deterministic Round Runner for driving whole rounds under the Two-Stage Audit Protocol.

Executes run_round(r) with starting-player alternation (even r -> AGENT, odd r -> CPU),
deterministic seed derivation, PlugMem intra-round reset, and barrier-enforced turn loops.
"""

from __future__ import annotations

import random
from typing import Optional

from src.config import AppConfig, load_config
from src.env.barrier import ClientServerBarrier, LegalObservation
from src.env.uno_engine import ModifiedUNOServerPlatform
from src.harness.agent_harness import AgentHarness
from src.llm.client import FakeLLMClient, LLMClient
from src.memory.plugmem import UNOPlugMem
from src.search.mars_mcts import MARSBudgetAwareMCTS
from src.audit.dataset import DatasetWriter, DecisionRecord


def derive_round_seeds(base_seed: int, round_id: int) -> tuple[int, int]:
    """Derives deterministic, independent engine and agent seeds from base_seed and round_id.

    Why: Ensures engine card shuffling and agent search determinization draw from
    orthogonal pseudo-random sequences while maintaining 100% reproducible execution.
    """
    engine_seed = (base_seed * 10007 + round_id * 37) & 0x7FFFFFFF
    agent_seed = (base_seed * 10007 + round_id * 37 + 101) & 0x7FFFFFFF
    return engine_seed, agent_seed


def run_round(
    r: int,
    base_seed: int = 42,
    config: Optional[AppConfig] = None,
    llm_client: Optional[LLMClient] = None,
    dataset_writer: Optional[DatasetWriter] = None,
    server_platform: Optional[ModifiedUNOServerPlatform] = None,
) -> list[DecisionRecord]:
    """Runs a complete round r until termination, collecting all emitted DecisionRecords.

    Why: Serves as the top-level execution entry point for pilot calibrations and
    main experimental trial rounds, enforcing starting player alternation and barrier isolation.
    """
    cfg = config or load_config()
    engine_seed, agent_seed = derive_round_seeds(base_seed, r)

    # Starting player alternates: even r -> AGENT, odd r -> CPU
    starting_player = "AGENT" if r % 2 == 0 else "CPU"
    starting_player_idx = 0 if starting_player == "AGENT" else 1

    # Instantiate server platform with isolated engine seed
    platform = server_platform or ModifiedUNOServerPlatform(
        config=cfg,
    )
    platform.start_new_round(round_id=r, starting_player=starting_player_idx, seed=engine_seed)

    # Initialize PlugMem and reset intra-round memory state
    plugmem = UNOPlugMem(config=cfg.memory)
    plugmem.reset_for_new_round(r)

    # Seeded agent RNG and MCTS
    agent_rng = random.Random(agent_seed)
    mcts = MARSBudgetAwareMCTS(config=cfg.mcts, rng=agent_rng)

    client = llm_client or FakeLLMClient()

    harness = AgentHarness(
        config=cfg,
        server_platform=platform,
        plugmem=plugmem,
        mcts=mcts,
        llm_client=client,
        dataset_writer=dataset_writer,
        starting_player=starting_player,
        rng=agent_rng,
    )

    # Retrieve initial sanitized observation across information barrier
    raw_state = platform.get_raw_agent_state()
    obs = ClientServerBarrier.extract_observation(raw_state)

    records: list[DecisionRecord] = []

    # Main turn loop driving agent until round termination
    while True:
        record = harness.on_agent_turn(obs)
        records.append(record)

        if record.stage2_post_horizon.round_status == "TERMINATED":
            break

        # Advance to next agent observation
        if harness.last_next_obs is not None:
            obs = harness.last_next_obs
        else:
            raw_state = platform.get_raw_agent_state()
            if raw_state.get("game_over", False):
                break
            obs = ClientServerBarrier.extract_observation(raw_state)

    return records
