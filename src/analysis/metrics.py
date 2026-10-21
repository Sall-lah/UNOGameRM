"""Calibration and scoring rules metrics for probabilistic risk predictions.

Why: Implements standard statistical risk evaluation metrics (Brier Score, Brier Skill
Score vs empirical baseline, Expected Calibration Error with boundary-inclusive binning,
and Mean Prediction Bias) matching doc-07 §2 specifications.
"""

from __future__ import annotations

from typing import Any, Sequence, Union
import numpy as np


def compute_calibration_metrics(
    p: Union[np.ndarray, Sequence[float]],
    y: Union[np.ndarray, Sequence[Union[int, float]]],
    n_bins: int = 10,
) -> dict[str, float]:
    """Computes Brier Score, BSS, ECE, MPB, and empirical prevalence.

    Why: Evaluates the reliability and resolution of risk forecasts p relative to
    binary outcomes y, handling degenerate edge cases (empty data, zero variance) gracefully.

    Args:
        p: Array of predicted probabilities in [0, 1].
        y: Array of binary ground-truth outcomes in {0, 1}.
        n_bins: Number of equal-width bins for ECE calculation (default: 10).

    Returns:
        Dictionary containing N_decisions, prevalence, BS, BS_ref, BSS, ECE, and MPB.
    """
    p_arr = np.asarray(p, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if len(p_arr) != len(y_arr):
        raise ValueError(f"Lengths of p ({len(p_arr)}) and y ({len(y_arr)}) must match")

    n = len(p_arr)
    if n == 0:
        return {
            "N_decisions": 0,
            "prevalence": float("nan"),
            "BS": float("nan"),
            "BS_ref": float("nan"),
            "BSS": float("nan"),
            "ECE": float("nan"),
            "MPB": float("nan"),
        }

    # Brier Score and Mean Prediction Bias
    diff = p_arr - y_arr
    bs = float(np.mean(diff ** 2))
    mpb = float(np.mean(diff))

    # Empirical Prevalence and Baseline Brier
    ybar = float(np.mean(y_arr))
    bs_ref = float(ybar * (1.0 - ybar))

    # Brier Skill Score
    if bs_ref <= 1e-12:
        bss = float("nan")
    else:
        bss = float(1.0 - (bs / bs_ref))

    # Expected Calibration Error (ECE) with M equal-width bins
    # Bins [low, high) for m < M-1; [low, high] for final bin m == M-1
    ece = 0.0
    bin_width = 1.0 / n_bins

    for m in range(n_bins):
        low = m * bin_width
        high = (m + 1) * bin_width

        if m == n_bins - 1:
            mask = (p_arr >= low) & (p_arr <= high)
        else:
            mask = (p_arr >= low) & (p_arr < high)

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_p_mean = float(np.mean(p_arr[mask]))
            bin_y_mean = float(np.mean(y_arr[mask]))
            ece += (bin_count / n) * abs(bin_y_mean - bin_p_mean)

    return {
        "N_decisions": n,
        "prevalence": ybar,
        "BS": bs,
        "BS_ref": bs_ref,
        "BSS": bss,
        "ECE": float(ece),
        "MPB": mpb,
    }
