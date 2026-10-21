"""Tests for analysis dataset loader and validation (Task 10.1)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.analysis.dataset import DatasetValidationError, load_dataset
from tests.analysis.conftest import make_synthetic_record_dict


def test_load_valid_jsonl_dataset(synthetic_jsonl_dataset: Path):
    """Verifies that loading a valid JSONL yields correct row count and dtypes."""
    df = load_dataset(synthetic_jsonl_dataset)

    assert len(df) == 7
    expected_cols = [
        "round_id",
        "turn_idx",
        "p_tA",
        "p_tB",
        "y_tA",
        "y_tB",
        "stratum_horizon",
        "agent_hand_count",
        "opp_card_count",
        "used_action_or_wild_at_t",
        "delta_hand_size_next",
        "agent_hand_count_next",
        "action_wild_count_next",
        "cumulative_cards_drawn_round",
        "timeout_exceeded",
        "mcts_fallback_used",
        "fallback_triggered",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    assert df["round_id"].dtype == int
    assert df["turn_idx"].dtype == int
    assert df["p_tA"].dtype == float
    assert df["p_tB"].dtype == float
    assert df["y_tA"].dtype == int
    assert df["y_tB"].dtype == int
    assert df["timeout_exceeded"].dtype == bool
    assert df["fallback_triggered"].dtype == bool


def test_derive_binary_labels_from_stage2_fields(tmp_path: Path):
    """Verifies that y_tA and y_tB are correctly derived from stage2 post-horizon fields."""
    rec = make_synthetic_record_dict(round_id=1, turn_idx=1, y_t_A=0, y_t_B=1)
    # y_t_A is 0, y_t_B is 1
    file_path = tmp_path / "test_labels.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    df = load_dataset(file_path)
    assert len(df) == 1
    assert df.iloc[0]["y_tA"] == 0
    assert df.iloc[0]["y_tB"] == 1


def test_malformed_record_raises_with_identifiers(tmp_path: Path):
    """Verifies that a record missing a required field raises DatasetValidationError naming round_id, turn_idx, and field."""
    rec = make_synthetic_record_dict(round_id=42, turn_idx=7)
    # Remove a required field
    del rec["stage1_freeze_log"]["p_t_A"]

    file_path = tmp_path / "malformed.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    with pytest.raises(DatasetValidationError) as exc_info:
        load_dataset(file_path)

    err = exc_info.value
    assert err.round_id == 42
    assert err.turn_idx == 7
    assert "stage1_freeze_log.p_t_A" in err.field_name or "p_t_A" in err.field_name
    assert "42" in str(err)
    assert "7" in str(err)
