"""Tests for Pydantic v2 schemas for Call #1 and Call #2 (Task 7.3).

Verifies validation constraints, prior key matching, sum tolerances,
probability bounds in [0.0, 1.0], and stratum enum validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm.schemas import (
    LLMCall1Output,
    LLMCall2Output,
)


def test_valid_call1_output_passes_validation():
    """Verify schema-compliant Call #1 response passes validation."""
    payload = {
        "pruned_actions": ["red 5", "draw"],
        "action_priors": {"red 5": 0.8, "draw": 0.2},
        "pruning_note": "Play red 5 or draw.",
    }
    parsed = LLMCall1Output.model_validate(
        payload,
        context={"legal_actions": ["red 5", "blue 1", "draw"], "max_candidates": 4},
    )
    assert parsed.pruned_actions == ["red 5", "draw"]
    assert parsed.action_priors["red 5"] == 0.8


def test_call1_prior_keys_mismatch_fails():
    """Verify validation failure when action_priors keys do not match pruned_actions."""
    payload = {
        "pruned_actions": ["red 5", "draw"],
        "action_priors": {"red 5": 1.0},  # Missing "draw"
        "pruning_note": "Mismatched keys.",
    }
    with pytest.raises(ValidationError):
        LLMCall1Output.model_validate(
            payload,
            context={"legal_actions": ["red 5", "draw"], "max_candidates": 4},
        )


def test_call1_priors_bad_sum_fails():
    """Verify validation failure when priors do not sum to 1.0 +- 1e-3."""
    payload = {
        "pruned_actions": ["red 5", "draw"],
        "action_priors": {"red 5": 0.5, "draw": 0.2},  # Sums to 0.7
        "pruning_note": "Bad sum.",
    }
    with pytest.raises(ValidationError):
        LLMCall1Output.model_validate(
            payload,
            context={"legal_actions": ["red 5", "draw"], "max_candidates": 4},
        )


def test_valid_call2_output_passes_validation():
    """Verify schema-compliant Call #2 response passes validation."""
    payload = {
        "p_t_A": 0.12,
        "p_t_B": 0.08,
        "expected_stratum": 2,
        "short_rationale": "Opponent hand is moderate. No immediate penalty risk.",
    }
    parsed = LLMCall2Output.model_validate(payload)
    assert parsed.p_t_A == 0.12
    assert parsed.p_t_B == 0.08
    assert parsed.expected_stratum == 2


def test_call2_probability_out_of_range_fails():
    """Verify validation failure when p_t_A or p_t_B is outside [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        LLMCall2Output.model_validate({
            "p_t_A": 1.25,  # > 1.0
            "p_t_B": 0.10,
            "expected_stratum": 2,
            "short_rationale": "Risk high.",
        })

    with pytest.raises(ValidationError):
        LLMCall2Output.model_validate({
            "p_t_A": -0.05,  # < 0.0
            "p_t_B": 0.10,
            "expected_stratum": 2,
            "short_rationale": "Risk negative.",
        })


def test_call2_invalid_stratum_fails():
    """Verify validation failure when expected_stratum is not in {1, 2, 3}."""
    with pytest.raises(ValidationError):
        LLMCall2Output.model_validate({
            "p_t_A": 0.10,
            "p_t_B": 0.10,
            "expected_stratum": 4,  # Invalid
            "short_rationale": "Stratum 4.",
        })
