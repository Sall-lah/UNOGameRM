"""Tests for Call #1 and Call #2 prompt template rendering (Task 7.2).

Verifies observation formatting, PlugMem context injection, candidate caps,
MCTS stats integration, and explicit locked action statements.
"""

from __future__ import annotations

import pytest

from src.env.barrier import LegalObservation
from src.llm.prompts import (
    render_call1_prompts,
    render_call2_prompts,
)


def test_call1_prompt_fills_observation_and_memory_fields(sample_obs: LegalObservation):
    """Verify Call #1 prompt injects public observation, PlugMem context, and candidate bounds."""
    context = "Opponent showed constraint on green in Turn 1."
    system_prompt, user_prompt = render_call1_prompts(
        obs=sample_obs,
        distilled_context=context,
        max_candidates=3,
    )

    assert "expert 2-player UNO strategist" in system_prompt
    assert "Round ID: 1 | Decision Index (t): 2" in user_prompt
    assert "Your Hand (3 cards):" in user_prompt
    assert "red 5" in user_prompt
    assert "Top Discard Card: red 3 | Active Color: red" in user_prompt
    assert "Opponent Hand Size: 6 cards" in user_prompt
    assert "Legal Actions Available (A_t):" in user_prompt
    assert context in user_prompt
    assert "at most 3 actions" in user_prompt


def test_call2_prompt_includes_mcts_stats_and_locked_action(sample_obs: LegalObservation):
    """Verify Call #2 prompt renders MCTS stats, locked action, and states action is locked."""
    context = "Opponent lacks red."
    mcts_stats = {"red 5": (12, 0.72), "draw": (3, 0.31)}
    locked_action = "red 5"

    system_prompt, user_prompt = render_call2_prompts(
        obs=sample_obs,
        distilled_context=context,
        locked_action=locked_action,
        mcts_stats=mcts_stats,
    )

    assert "ALREADY LOCKED its action" in system_prompt
    assert "LOCKED ACTION (a_t): red 5" in user_prompt
    assert "red 5: N=12, Q=0.720" in user_prompt
    assert "draw: N=3, Q=0.310" in user_prompt
    assert "Opponent Hand Size BEFORE a_t: 6 cards" in user_prompt
    assert context in user_prompt
