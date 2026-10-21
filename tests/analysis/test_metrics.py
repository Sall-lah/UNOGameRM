"""Tests for calibration and scoring rules metrics (Task 10.2)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.analysis.metrics import compute_calibration_metrics


def test_known_values_bs_and_mpb():
    """Verify Brier Score and MPB on known values: p=[0.2, 0.8], y=[0, 1]."""
    p = [0.2, 0.8]
    y = [0, 1]

    res = compute_calibration_metrics(p, y)

    # (0.2 - 0)^2 = 0.04; (0.8 - 1)^2 = 0.04; mean = 0.04
    assert pytest.approx(res["BS"], abs=1e-6) == 0.04
    # (0.2 - 0) + (0.8 - 1) = 0.2 - 0.2 = 0.0; mean = 0.0
    assert pytest.approx(res["MPB"], abs=1e-6) == 0.0
    assert res["N_decisions"] == 2


def test_ece_right_edge_binning_includes_one():
    """Verify that p=1.0 is assigned to the final bin and not omitted."""
    # 10 bins on [0, 1]: last bin [0.9, 1.0]
    p = [0.0, 0.5, 1.0]
    y = [0, 1, 1]

    res = compute_calibration_metrics(p, y, n_bins=10)
    assert res["N_decisions"] == 3
    assert not math.isnan(res["ECE"])
    assert res["ECE"] >= 0.0


def test_perfect_calibration_yields_zero_ece():
    """Verify that when each bin's mean predicted prob equals observed rate, ECE is zero."""
    # 4 bins: [0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1.0]
    # Bin 0: p=[0.0, 0.0], y=[0, 0] -> mean p = 0, mean y = 0
    # Bin 3: p=[1.0, 1.0], y=[1, 1] -> mean p = 1, mean y = 1
    p = [0.0, 0.0, 1.0, 1.0]
    y = [0, 0, 1, 1]

    res = compute_calibration_metrics(p, y, n_bins=4)
    assert pytest.approx(res["ECE"], abs=1e-9) == 0.0


def test_degenerate_prevalence_yields_nan_bss():
    """Verify that when all outcomes are identical (BS_ref = 0), BSS is NaN without raising."""
    p = [0.1, 0.2, 0.3]
    y = [0, 0, 0]  # ybar = 0 -> BS_ref = 0

    res = compute_calibration_metrics(p, y)
    assert math.isnan(res["BSS"])
    assert res["BS_ref"] == 0.0


def test_empty_input_returns_nan():
    """Verify that empty inputs yield N_decisions=0 and NaN metrics without raising."""
    res = compute_calibration_metrics([], [])
    assert res["N_decisions"] == 0
    assert math.isnan(res["BS"])
    assert math.isnan(res["BSS"])
    assert math.isnan(res["ECE"])
    assert math.isnan(res["MPB"])
    assert math.isnan(res["prevalence"])
