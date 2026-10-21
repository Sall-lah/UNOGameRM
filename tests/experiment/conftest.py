"""Pytest fixtures for experiment runbook tests."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import AppConfig
from src.llm.client import FakeLLMClient


@pytest.fixture
def fake_llm():
    """Provides a deterministic FakeLLMClient with canned responses."""
    import json
    call1_json = json.dumps({
        "pruned_actions": ["draw"],
        "action_priors": {"draw": 1.0},
        "pruning_note": "Test pruning",
    })
    call2_json = json.dumps({
        "p_t_A": 0.05,
        "p_t_B": 0.02,
        "expected_stratum": 2,
        "short_rationale": "Test risk",
    })
    return FakeLLMClient(
        responses_by_prompt={"A_t": call1_json, "LOCKED ACTION": call2_json},
        default_response=call1_json,
    )
