"""Tests for RQ1 stratified calibration evaluation (Task 10.4)."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.dataset import load_dataset
from src.analysis.rq1 import evaluate_rq1


def test_rq1_evaluates_present_strata(synthetic_jsonl_dataset: Path):
    """Verify that RQ1 evaluates Aggregate and all present strata with Risk A and Risk B."""
    df = load_dataset(synthetic_jsonl_dataset)
    cfg = AnalysisConfig(bootstrap_B=100, bootstrap_seed=42)

    results = evaluate_rq1(df, cfg)

    assert "Aggregate" in results
    assert "Stratum_1" in results
    assert "Stratum_2" in results
    assert "Stratum_3" in results

    # Check that both Risk_A and Risk_B blocks exist in Aggregate
    for risk in ("Risk_A", "Risk_B"):
        assert risk in results["Aggregate"]
        risk_res = results["Aggregate"][risk]
        assert "BS" in risk_res
        assert "BSS" in risk_res
        assert "ECE" in risk_res
        assert "MPB" in risk_res
        assert "prevalence" in risk_res
        assert "N_decisions" in risk_res


def test_rq1_empty_stratum_is_skipped(synthetic_jsonl_dataset: Path):
    """Verify that when a stratum has no records, its scope is omitted rather than erroring."""
    df = load_dataset(synthetic_jsonl_dataset)
    # Filter out stratum 1 records
    no_s1_df = df[df["stratum_horizon"] != 1].copy()

    cfg = AnalysisConfig(bootstrap_B=50, bootstrap_seed=42)
    results = evaluate_rq1(no_s1_df, cfg)

    assert "Stratum_1" not in results
    assert "Aggregate" in results
    assert "Stratum_2" in results
