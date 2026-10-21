"""Tests for LLM Call #2 execution, locked action gating, and risk fallback (Tasks 7.5, 7.6, 7.7).

Verifies locked-action prerequisite assertion, schema-valid risk response handling,
timeout/parse fallback with action-derived stratum, and latency measurement.
"""

from __future__ import annotations

import json
import pytest

from src.env.barrier import LegalObservation
from src.llm.call2_risk import run_call2
from src.llm.client import FakeLLMClient


def test_call2_rejects_missing_locked_action(sample_obs: LegalObservation):
    """Verify Call #2 raises ValueError immediately when locked_action is None without model calls."""
    client = FakeLLMClient()

    with pytest.raises(ValueError, match="locked_action must not be None"):
        run_call2(
            locked_action=None,  # type: ignore
            obs=sample_obs,
            distilled_context="",
            mcts_stats={},
            client=client,
        )

    # Client was never called
    assert len(client.call_history) == 0


def test_call2_valid_response_clears_fallback_flag(
    sample_obs: LegalObservation,
    valid_call2_json: str,
):
    """Verify valid response returns parsed probabilities and sets fallback_triggered_call2 = False."""
    client = FakeLLMClient(default_response=valid_call2_json)
    res = run_call2(
        locked_action="red 5",
        obs=sample_obs,
        distilled_context="No prior history.",
        mcts_stats={"red 5": (10, 0.65)},
        client=client,
    )

    assert res.fallback_triggered_call2 is False
    assert res.p_t_A == 0.15
    assert res.p_t_B == 0.05
    assert res.expected_stratum == 2
    assert res.latency_ms >= 0.0


def test_call2_timeout_triggers_deterministic_fallback(sample_obs: LegalObservation):
    """Verify simulated timeout returns p_t_A=p_t_B=0.5 and derives stratum from action."""
    client = FakeLLMClient(simulate_timeout=True)

    # Test with turn-retention action (Stratum 1)
    res_retention = run_call2(
        locked_action="blue skip",
        obs=sample_obs,
        distilled_context="",
        mcts_stats={},
        client=client,
    )

    assert res_retention.fallback_triggered_call2 is True
    assert res_retention.p_t_A == 0.5
    assert res_retention.p_t_B == 0.5
    assert res_retention.expected_stratum == 1  # Skip is turn-retention in 2P
    assert res_retention.latency_ms >= 0.0

    # Test with normal number card (Stratum 2)
    res_normal = run_call2(
        locked_action="red 5",
        obs=sample_obs,
        distilled_context="",
        mcts_stats={},
        client=client,
    )
    assert res_normal.fallback_triggered_call2 is True
    assert res_normal.expected_stratum == 2


def test_call2_parse_failure_triggers_deterministic_fallback(sample_obs: LegalObservation):
    """Verify non-JSON response returns deterministic fallback with action-derived stratum."""
    client = FakeLLMClient(default_response="This is an unparseable response.")

    res = run_call2(
        locked_action="WILD + 4 card, set to red",
        obs=sample_obs,
        distilled_context="",
        mcts_stats={},
        client=client,
    )

    assert res.fallback_triggered_call2 is True
    assert res.p_t_A == 0.5
    assert res.p_t_B == 0.5
    assert res.expected_stratum == 1  # Wild +4 is turn-retention in 2P
    assert res.latency_ms >= 0.0
