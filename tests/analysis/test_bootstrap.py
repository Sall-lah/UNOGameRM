"""Tests for round-level cluster bootstrap (Task 10.3)."""

from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import pytest

from src.analysis.bootstrap import round_level_cluster_bootstrap
from src.analysis.dataset import load_dataset


def _dummy_metric(df: pd.DataFrame) -> dict[str, float]:
    """Simple test metric computing mean p_tA."""
    if df.empty:
        return {"mean_p": float("nan")}
    return {"mean_p": float(df["p_tA"].mean())}


def test_bootstrap_reproducibility(synthetic_jsonl_dataset: Path):
    """Verify that running cluster bootstrap twice with the same seed yields identical results."""
    df = load_dataset(synthetic_jsonl_dataset)

    res1 = round_level_cluster_bootstrap(df, _dummy_metric, n_boot=200, seed=42)
    res2 = round_level_cluster_bootstrap(df, _dummy_metric, n_boot=200, seed=42)

    assert res1["mean_p"]["estimate"] == res2["mean_p"]["estimate"]
    assert res1["mean_p"]["ci_95_low"] == res2["mean_p"]["ci_95_low"]
    assert res1["mean_p"]["ci_95_high"] == res2["mean_p"]["ci_95_high"]


def test_bootstrap_whole_round_cluster_resampling():
    """Verify that whole rounds are resampled as clusters (all turns of selected rounds included)."""
    # Create two rounds: round 1 has 3 turns, round 2 has 5 turns
    records = []
    for t in range(1, 4):
        records.append({"round_id": 1, "turn_idx": t, "val": 10.0})
    for t in range(1, 6):
        records.append({"round_id": 2, "turn_idx": t, "val": 20.0})
    df = pd.DataFrame(records)

    observed_sizes = []

    def check_size(resampled_df: pd.DataFrame) -> dict[str, float]:
        observed_sizes.append(len(resampled_df))
        return {"dummy": 1.0}

    # If round 1 (3 turns) is chosen 2 times -> 6 turns
    # If round 2 (5 turns) is chosen 2 times -> 10 turns
    # If round 1 and round 2 are chosen 1 time each -> 8 turns
    round_level_cluster_bootstrap(df, check_size, n_boot=20, seed=123)

    # Every resampled size must be in {6, 8, 10}
    for sz in observed_sizes:
        assert sz in (6, 8, 10), f"Unexpected resampled size: {sz}"


def test_ci_brackets_point_estimate(synthetic_jsonl_dataset: Path):
    """Verify that ci_low <= estimate <= ci_high holds."""
    df = load_dataset(synthetic_jsonl_dataset)

    res = round_level_cluster_bootstrap(df, _dummy_metric, n_boot=500, seed=42)
    est = res["mean_p"]["estimate"]
    ci_low = res["mean_p"]["ci_95_low"]
    ci_high = res["mean_p"]["ci_95_high"]

    assert not math.isnan(est)
    assert not math.isnan(ci_low)
    assert not math.isnan(ci_high)
    assert ci_low <= est <= ci_high
