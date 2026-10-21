"""Tests for primary vs sensitivity fallback split (Task 10.5)."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.analysis.dataset import load_dataset
from src.analysis.split import split_fallback


def test_split_fallback_partitions_dataset(synthetic_jsonl_dataset: Path):
    """Verify primary receives non-fallback turns, sensitivity receives fallback turns, and counts sum to total."""
    df = load_dataset(synthetic_jsonl_dataset)

    primary_df, sensitivity_df = split_fallback(df)

    assert len(primary_df) + len(sensitivity_df) == len(df)
    assert len(primary_df) > 0
    assert len(sensitivity_df) > 0

    # Primary turns must have fallback_triggered == False
    assert (primary_df["fallback_triggered"] == False).all()
    assert (primary_df["timeout_exceeded"] == False).all()
    assert (primary_df["mcts_fallback_used"] == False).all()

    # Sensitivity turns must have fallback_triggered == True
    assert (sensitivity_df["fallback_triggered"] == True).all()
