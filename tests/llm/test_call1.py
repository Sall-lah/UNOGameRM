"""Tests for LLM Call #1 execution, fallbacks, and latency measurement (Tasks 7.4 & 7.7).

Verifies valid response handling, unparseable response uniform fallback,
strictly-positive weight renormalization, |A_t| <= 2 preservation, and latency reporting.
"""

from __future__ import annotations

import json
import pytest

from src.env.barrier import LegalObservation
from src.llm.call1_prior import run_call1
from src.llm.client import FakeLLMClient


def test_call1_valid_response_clears_flag(sample_obs: LegalObservation, valid_call1_json: str):
    """Verify valid response returns pruned prior and sets invalid_json_call1 = False."""
    client = FakeLLMClient(default_response=valid_call1_json)
    res = run_call1(
        obs=sample_obs,
        distilled_context="No prior history.",
        client=client,
        max_candidates=4,
    )

    assert res.invalid_json_call1 is False
    assert res.pruned_actions == ["red 5", "WILD card, set to green"]
    assert res.action_priors["red 5"] == 0.65
    assert res.latency_ms >= 0.0


def test_call1_unparseable_response_falls_back_to_uniform(sample_obs: LegalObservation):
    """Verify non-JSON output triggers uniform fallback across all legal actions."""
    client = FakeLLMClient(default_response="I think you should play red 5.")
    res = run_call1(
        obs=sample_obs,
        distilled_context="",
        client=client,
        max_candidates=4,
    )

    assert res.invalid_json_call1 is True
    assert set(res.pruned_actions) == set(sample_obs.legal_actions)
    expected_uniform = 1.0 / len(sample_obs.legal_actions)
    for p in res.action_priors.values():
        assert abs(p - expected_uniform) < 1e-6
    assert res.latency_ms >= 0.0


def test_call1_misnormalized_positive_weights_renormalized(sample_obs: LegalObservation):
    """Verify strictly positive weights that sum to != 1.0 are renormalized and flagged."""
    bad_sum_response = json.dumps({
        "pruned_actions": ["red 5", "draw"],
        "action_priors": {"red 5": 1.0, "draw": 0.5},  # Sums to 1.5
        "pruning_note": "Positive but unnormalized.",
    })
    client = FakeLLMClient(default_response=bad_sum_response)

    res = run_call1(
        obs=sample_obs,
        distilled_context="",
        client=client,
        max_candidates=4,
    )

    assert res.invalid_json_call1 is True
    assert res.pruned_actions == ["red 5", "draw"]
    # 1.0 / 1.5 = 0.66667, 0.5 / 1.5 = 0.33333
    assert abs(res.action_priors["red 5"] - (1.0 / 1.5)) < 1e-4
    assert abs(res.action_priors["draw"] - (0.5 / 1.5)) < 1e-4
    assert abs(sum(res.action_priors.values()) - 1.0) < 1e-6
    assert res.latency_ms >= 0.0


def test_call1_small_action_set_keeps_all_actions(make_legal_obs):
    """Verify when |A_t| <= 2, all legal actions are preserved regardless of model suggestion."""
    small_obs = make_legal_obs(
        agent_hand=("red 1", "blue 2"),
        top_discard="red 5",
        active_colour="red",
        legal_actions=("red 1", "draw"),  # |A_t| == 2
    )
    # Model attempts to prune "draw"
    prune_attempt_json = json.dumps({
        "pruned_actions": ["red 1"],
        "action_priors": {"red 1": 1.0},
        "pruning_note": "Attempted to drop draw.",
    })
    client = FakeLLMClient(default_response=prune_attempt_json)

    res = run_call1(
        obs=small_obs,
        distilled_context="",
        client=client,
        max_candidates=4,
    )

    # Must preserve both actions
    assert set(res.pruned_actions) == {"red 1", "draw"}
    assert res.latency_ms >= 0.0
