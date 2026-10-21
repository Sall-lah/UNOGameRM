"""Tests for non-play decision handling in AgentHarness (Tasks 7.7 & 7.8).

Verifies first-turn Wild colour decision detection and record flagging,
and agent.d4_response policy adherence (default "accept" -> "draw", vs "challenge").
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.config import AppConfig
from src.env.barrier import LegalObservation
from src.harness.agent_harness import AgentHarness
from src.llm.client import FakeLLMClient


def test_first_turn_wild_colour_decision_flagged_and_dispatched(tmp_path: Path, fake_llm):
    """Verify first-turn Wild colour choice is flagged first_turn_wild_colour=True and dispatched."""
    cfg = AppConfig.from_yaml(f"""
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["red 1"], "top_card": "WILD card, set to green"},
    }

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        llm_client=fake_llm,
    )

    # First turn Wild colour decision: legal actions are the 4 wild color declarations
    obs = LegalObservation(
        round_id=1,
        turn_idx=1,
        agent_hand=("red 1", "blue 2"),
        top_discard="WILD card, set to white",
        active_colour="white",
        opp_card_count=7,
        legal_actions=(
            "WILD card, set to red",
            "WILD card, set to green",
            "WILD card, set to blue",
            "WILD card, set to yellow",
        ),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    rec = harness.on_agent_turn(obs)

    assert rec.identity.first_turn_wild_colour is True
    assert rec.stage2_post_horizon.dispatched_command.startswith("WILD card, set to")
    assert mock_platform.dispatch_agent_action_and_observe_horizon.called


def test_d4_response_default_accept_draws_four(tmp_path: Path, fake_llm):
    """Verify when d4_response is 'accept' (default), harness dispatches 'draw'."""
    cfg = AppConfig.from_yaml(f"""
    agent:
      d4_response: accept
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["red 1"], "top_card": "WILD + 4 card, set to green"},
    }

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        llm_client=fake_llm,
    )

    obs = LegalObservation(
        round_id=1,
        turn_idx=2,
        agent_hand=("red 1",),
        top_discard="WILD + 4 card, set to green",
        active_colour="green",
        opp_card_count=3,
        legal_actions=("draw", "challenge"),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    rec = harness.on_agent_turn(obs)

    # Policy 'accept' must commit and dispatch 'draw'
    assert rec.stage1_freeze_log.locked_action_at == "draw"
    assert rec.stage2_post_horizon.dispatched_command == "draw"
    mock_platform.dispatch_agent_action_and_observe_horizon.assert_called_with("draw")


def test_d4_response_configured_challenge_honoured(tmp_path: Path, fake_llm):
    """Verify when d4_response is 'challenge', harness dispatches 'challenge'."""
    cfg = AppConfig.from_yaml(f"""
    agent:
      d4_response: challenge
    audit:
      output_root: {str(tmp_path)}
      strict_invariant: true
    """)

    mock_platform = MagicMock()
    mock_platform.dispatch_agent_action_and_observe_horizon.return_value = {
        "horizon_events": (),
        "round_over": False,
        "next_state": {"agent_hand": ["red 1"], "top_card": "WILD + 4 card, set to green"},
    }

    harness = AgentHarness(
        config=cfg,
        server_platform=mock_platform,
        llm_client=fake_llm,
    )

    obs = LegalObservation(
        round_id=1,
        turn_idx=2,
        agent_hand=("red 1",),
        top_discard="WILD + 4 card, set to green",
        active_colour="green",
        opp_card_count=3,
        legal_actions=("draw", "challenge"),
        public_events_since_last_turn=(),
        known_discards_since_reshuffle=(),
    )

    rec = harness.on_agent_turn(obs)

    # Policy 'challenge' must commit and dispatch 'challenge'
    assert rec.stage1_freeze_log.locked_action_at == "challenge"
    assert rec.stage2_post_horizon.dispatched_command == "challenge"
    mock_platform.dispatch_agent_action_and_observe_horizon.assert_called_with("challenge")
