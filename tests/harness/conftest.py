"""Pytest fixtures for Agent Harness and Round Runner unit tests.

Provides mock server platforms, test doubles, and observation factories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence
import pytest

from src.config import AppConfig
from src.env.barrier import LegalObservation
from src.llm.client import FakeLLMClient


@pytest.fixture
def test_app_config(tmp_path: Path) -> AppConfig:
    """Provides AppConfig directed to a temporary test directory."""
    return AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      dataset_split: pilot
      strict_invariant: true
    budget:
      t_turn_s: 5.0
      delta_buffer_s: 0.5
      llm2_reserve_s: 1.0
    mcts:
      max_simulations: 20
      max_rollout_depth: 3
    """)


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    """Provides a FakeLLMClient pre-configured with valid Call 1 and Call 2 JSON responses."""
    call1_json = json.dumps({
        "pruned_actions": ["red 5", "draw"],
        "action_priors": {"red 5": 0.8, "draw": 0.2},
        "pruning_note": "Play red 5 or draw.",
    })
    call2_json = json.dumps({
        "p_t_A": 0.1,
        "p_t_B": 0.05,
        "expected_stratum": 2,
        "short_rationale": "Moderate risk; red 5 safe.",
    })

    return FakeLLMClient(
        responses_by_prompt={
            "A_t": call1_json,
            "LOCKED ACTION": call2_json,
        },
        default_response=call1_json,
    )


@pytest.fixture
def sample_legal_obs():
    """Factory fixture generating valid LegalObservation instances for harness tests."""

    def _factory(
        round_id: int = 1,
        turn_idx: int = 1,
        agent_hand: Sequence[str] = ("red 5", "blue skip"),
        top_discard: str = "red 3",
        active_colour: str = "red",
        opp_card_count: int = 5,
        legal_actions: Sequence[str] = ("red 5", "draw"),
    ) -> LegalObservation:
        return LegalObservation(
            round_id=round_id,
            turn_idx=turn_idx,
            agent_hand=tuple(agent_hand),
            top_discard=top_discard,
            active_colour=active_colour,
            opp_card_count=opp_card_count,
            legal_actions=tuple(legal_actions),
            public_events_since_last_turn=(),
            known_discards_since_reshuffle=(),
        )

    return _factory
