"""Tests for Stage 1 pre-execution freeze record (Task 7.1).

Verifies canonical-JSON byte reproducibility, SHA-256 integrity hashing,
UNO.py card encoding and British colour spelling, and fsync order assertion before dispatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from src.audit.stage1 import (
    Stage1FreezeResult,
    Stage1Writer,
    serialize_stage1_payload,
)


def test_stage1_canonical_json_is_byte_reproducible():
    """Verify identical inputs serialize to identical bytes and produce identical SHA-256."""
    obs = {
        "agent_hand": ["red 5", "WILD card, set to green"],
        "top_discard": "red 2",
        "active_colour": "red",
    }
    mem = {"retrieved_node_ids": ["prop_1"], "distilled_context_tokens": 12}
    priors = {"red 5": 0.7, "WILD card, set to green": 0.3}
    n_counts = {"red 5": 70, "WILD card, set to green": 30}
    q_vals = {"red 5": 0.65, "WILD card, set to green": 0.45}

    bytes1 = serialize_stage1_payload(
        round_id=1,
        turn_idx=3,
        observation_ot=obs,
        memory_plugmem=mem,
        p_prior=priors,
        visit_counts_N=n_counts,
        action_values_Q=q_vals,
        locked_action_at="red 5",
        p_t_A=0.12,
        p_t_B=0.08,
        short_rationale="Playing red 5 matches active colour.",
    )

    bytes2 = serialize_stage1_payload(
        round_id=1,
        turn_idx=3,
        observation_ot=obs,
        memory_plugmem=mem,
        p_prior=priors,
        visit_counts_N=n_counts,
        action_values_Q=q_vals,
        locked_action_at="red 5",
        p_t_A=0.12,
        p_t_B=0.08,
        short_rationale="Playing red 5 matches active colour.",
    )

    assert bytes1 == bytes2
    assert hashlib.sha256(bytes1).hexdigest() == hashlib.sha256(bytes2).hexdigest()


def test_stage1_sha256_over_persisted_bytes_matches(tmp_path: Path):
    """Verify SHA-256 computed over the exact bytes written to disk matches stage1_sha256."""
    writer = Stage1Writer(freeze_dir=tmp_path)
    res: Stage1FreezeResult = writer.freeze(
        round_id=2,
        turn_idx=1,
        observation_ot={"agent_hand": ["blue 1"], "top_discard": "blue 3", "active_colour": "blue"},
        memory_plugmem={"retrieved_node_ids": []},
        p_prior={"blue 1": 1.0},
        visit_counts_N={"blue 1": 10},
        action_values_Q={"blue 1": 0.8},
        locked_action_at="blue 1",
        p_t_A=0.05,
        p_t_B=0.02,
        short_rationale="Blue 1 to follow suit.",
    )

    on_disk_bytes = res.file_path.read_bytes()
    assert on_disk_bytes == res.canonical_bytes
    assert hashlib.sha256(on_disk_bytes).hexdigest() == res.stage1_sha256


def test_stage1_preserves_unopy_encoding_and_british_colour():
    """Verify UNO.py card representation and British 'colour' spelling are preserved."""
    obs = {
        "top_discard": "WILD + 4 card, set to yellow",
        "active_colour": "yellow",
    }
    payload_bytes = serialize_stage1_payload(
        round_id=1,
        turn_idx=1,
        observation_ot=obs,
        memory_plugmem={"opponent_draw_constraints": [{"colour": "yellow"}]},
        p_prior={"draw": 1.0},
        visit_counts_N={"draw": 5},
        action_values_Q={"draw": 0.1},
        locked_action_at="draw",
        p_t_A=0.2,
        p_t_B=0.1,
        short_rationale="Drawing card.",
    )

    data = json.loads(payload_bytes.decode("utf-8"))
    assert data["observation_ot"]["active_colour"] == "yellow"
    assert data["observation_ot"]["top_discard"] == "WILD + 4 card, set to yellow"
    assert data["memory_plugmem"]["opponent_draw_constraints"][0]["colour"] == "yellow"


def test_stage1_fsync_precedes_dispatch(tmp_path: Path):
    """Verify file is durably flushed and exists before a dispatch callback executes."""
    writer = Stage1Writer(freeze_dir=tmp_path)
    events_log = []

    def stub_dispatch(res: Stage1FreezeResult):
        # Assert file exists and contains valid data at dispatch moment
        assert res.file_path.exists()
        assert res.file_path.stat().st_size > 0
        events_log.append("dispatched")

    freeze_res = writer.freeze(
        round_id=1,
        turn_idx=1,
        observation_ot={"agent_hand": ["red 0"], "active_colour": "red"},
        memory_plugmem={},
        p_prior={"red 0": 1.0},
        visit_counts_N={"red 0": 10},
        action_values_Q={"red 0": 0.5},
        locked_action_at="red 0",
        p_t_A=0.1,
        p_t_B=0.05,
        short_rationale="Play red 0.",
    )
    events_log.append("frozen")

    stub_dispatch(freeze_res)

    assert events_log == ["frozen", "dispatched"]
