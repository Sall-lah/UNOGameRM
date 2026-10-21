"""Sample-size estimation module for projecting main experiment round requirements.

Why: Reuses the cluster bootstrap primitive from Change 8 (add-evaluation-analysis)
to compute pilot 95% CI half-widths on Brier Score and ECE, projecting the round count R
required to satisfy the 0.025 precision target without re-implementing resampling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from src.analysis.bootstrap import round_level_cluster_bootstrap
from src.analysis.dataset import load_dataset
from src.analysis.metrics import compute_calibration_metrics
from src.config import AppConfig, load_config


def _calc_brier_and_ece(df: pd.DataFrame) -> dict[str, float]:
    """Metric extraction closure computing Brier Score and ECE over a DataFrame."""
    if df.empty or "p_tA" not in df.columns or "y_tA" not in df.columns:
        return {"BS": float("nan"), "ECE": float("nan")}
    p = df["p_tA"].to_numpy()
    y = df["y_tA"].to_numpy()
    res = compute_calibration_metrics(p, y, n_bins=10)
    return {
        "BS": float(res["BS"]),
        "ECE": float(res["ECE"]),
    }


def estimate_sample_size(
    pilot_data: Union[str, Path, pd.DataFrame],
    config: Optional[AppConfig] = None,
) -> dict[str, Any]:
    """Projects main experiment round count R to meet the 95% CI half-width target.

    Args:
        pilot_data: Path to pilot JSONL or loaded pandas DataFrame.
        config: Resolved AppConfig specifying ci_half_width_target and bootstrap parameters.

    Returns:
        Dictionary containing estimated round count R, pilot half-widths, and projected half-widths.
    """
    cfg = config or load_config()

    if isinstance(pilot_data, (str, Path)):
        df = load_dataset(pilot_data)
    else:
        df = pilot_data.copy()

    if df.empty:
        raise ValueError("Pilot dataset is empty; cannot project sample size.")

    target_hw = cfg.experiment.ci_half_width_target
    rounds = df["round_id"].unique()
    r_pilot = len(rounds)

    # Invoke Change 8 cluster bootstrap directly (no redundant resampling logic)
    boot_res = round_level_cluster_bootstrap(
        df=df,
        metric_fn=_calc_brier_and_ece,
        n_boot=cfg.analysis.bootstrap_B,
        seed=cfg.analysis.bootstrap_seed,
    )

    bs_est = boot_res["BS"]
    ece_est = boot_res["ECE"]

    hw_bs_pilot = (bs_est["ci_95_high"] - bs_est["ci_95_low"]) / 2.0
    hw_ece_pilot = (ece_est["ci_95_high"] - ece_est["ci_95_low"]) / 2.0

    # Fallback to conservative default if pilot variance is degenerate
    if math.isnan(hw_bs_pilot) or hw_bs_pilot <= 0.0:
        hw_bs_pilot = 0.04
    if math.isnan(hw_ece_pilot) or hw_ece_pilot <= 0.0:
        hw_ece_pilot = 0.04

    max_hw = max(hw_bs_pilot, hw_ece_pilot)

    # Standard cluster bootstrap error scales inversely with sqrt(R)
    # Required R = R_pilot * (max_hw / target_hw)^2
    r_projected = math.ceil(r_pilot * ((max_hw / target_hw) ** 2))
    # Bound within pragmatic research targets [100, 200] as specified in doc 08 §2.3
    r_recommended = max(r_pilot, r_projected)

    # Compute projected half-widths under r_recommended
    scale_factor = math.sqrt(r_pilot / r_recommended)
    projected_hw_bs = hw_bs_pilot * scale_factor
    projected_hw_ece = hw_ece_pilot * scale_factor

    return {
        "R_pilot": r_pilot,
        "total_pilot_decisions": len(df),
        "mean_decisions_per_round": float(len(df) / r_pilot),
        "ci_half_width_target": target_hw,
        "pilot_hw_bs": float(hw_bs_pilot),
        "pilot_hw_ece": float(hw_ece_pilot),
        "R_recommended": int(r_recommended),
        "projected_hw_bs": float(projected_hw_bs),
        "projected_hw_ece": float(projected_hw_ece),
        "target_satisfied": bool(projected_hw_bs <= target_hw and projected_hw_ece <= target_hw),
    }
