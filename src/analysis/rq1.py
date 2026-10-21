"""RQ1: Stratified calibration evaluation across action horizons and risk types.

Why: Evaluates the statistical calibration of Risk A (penalty draw) and Risk B (opponent win)
forecasts in aggregate and across action horizons (Stratum 1: turn-retention, Stratum 2: standard
single-turn exchange, Stratum 3: multi-turn / challenge horizons) using cluster bootstrap CIs.
"""

from __future__ import annotations

from typing import Any, Optional
import pandas as pd

from src.analysis.bootstrap import round_level_cluster_bootstrap
from src.analysis.config import AnalysisConfig
from src.analysis.metrics import compute_calibration_metrics


def _make_metric_fn(p_col: str, y_col: str, n_bins: int):
    """Factory creating a metric evaluation closure for a specific risk column pair."""

    def _metric_fn(data: pd.DataFrame) -> dict[str, float]:
        if data.empty or p_col not in data.columns or y_col not in data.columns:
            return {
                "N_decisions": 0,
                "prevalence": float("nan"),
                "BS": float("nan"),
                "BSS": float("nan"),
                "ECE": float("nan"),
                "MPB": float("nan"),
            }
        p = data[p_col].to_numpy()
        y = data[y_col].to_numpy()
        res = compute_calibration_metrics(p, y, n_bins=n_bins)
        return {
            "N_decisions": res["N_decisions"],
            "prevalence": res["prevalence"],
            "BS": res["BS"],
            "BSS": res["BSS"],
            "ECE": res["ECE"],
            "MPB": res["MPB"],
        }

    return _metric_fn


def evaluate_rq1(
    df: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
) -> dict[str, Any]:
    """Evaluates RQ1 calibration metrics with cluster bootstrap CIs across strata.

    Why: Provides doc-07 statistical claims with 95% confidence intervals, skipping
    empty strata to avoid degenerate bootstrap estimation.

    Args:
        df: Loaded decision dataset DataFrame.
        cfg: AnalysisConfig specifying bootstrap parameters and ECE bins.

    Returns:
        Dictionary keyed by scope ("Aggregate", "Stratum_1", etc.) and risk type
        ("Risk_A", "Risk_B"), with summary estimates and bootstrap CI bounds.
    """
    if cfg is None:
        cfg = AnalysisConfig()

    results: dict[str, Any] = {}
    rows_for_df: list[dict[str, Any]] = []

    # Define scopes to evaluate
    scopes: list[tuple[str, pd.DataFrame]] = [("Aggregate", df)]

    for stratum in (1, 2, 3):
        scope_name = f"Stratum_{stratum}"
        sub_df = df[df["stratum_horizon"] == stratum]
        if not sub_df.empty:
            scopes.append((scope_name, sub_df))

    risk_pairs = [
        ("Risk_A", "p_tA", "y_tA"),
        ("Risk_B", "p_tB", "y_tB"),
    ]

    for scope_name, scope_df in scopes:
        results[scope_name] = {}

        for risk_name, p_col, y_col in risk_pairs:
            metric_fn = _make_metric_fn(p_col, y_col, n_bins=cfg.ece_bins)

            boot_res = round_level_cluster_bootstrap(
                df=scope_df,
                metric_fn=metric_fn,
                n_boot=cfg.bootstrap_B,
                seed=cfg.bootstrap_seed,
            )

            results[scope_name][risk_name] = boot_res

            # Flatten into a summary row for tabular display
            row_dict = {
                "scope": scope_name,
                "risk_type": risk_name,
                "N_decisions": int(boot_res["N_decisions"]["estimate"])
                if not pd.isna(boot_res["N_decisions"]["estimate"])
                else 0,
                "prevalence": boot_res["prevalence"]["estimate"],
                "prevalence_ci_low": boot_res["prevalence"]["ci_95_low"],
                "prevalence_ci_high": boot_res["prevalence"]["ci_95_high"],
                "BS": boot_res["BS"]["estimate"],
                "BS_ci_low": boot_res["BS"]["ci_95_low"],
                "BS_ci_high": boot_res["BS"]["ci_95_high"],
                "BSS": boot_res["BSS"]["estimate"],
                "BSS_ci_low": boot_res["BSS"]["ci_95_low"],
                "BSS_ci_high": boot_res["BSS"]["ci_95_high"],
                "ECE": boot_res["ECE"]["estimate"],
                "ECE_ci_low": boot_res["ECE"]["ci_95_low"],
                "ECE_ci_high": boot_res["ECE"]["ci_95_high"],
                "MPB": boot_res["MPB"]["estimate"],
                "MPB_ci_low": boot_res["MPB"]["ci_95_low"],
                "MPB_ci_high": boot_res["MPB"]["ci_95_high"],
            }
            rows_for_df.append(row_dict)

    summary_df = pd.DataFrame(rows_for_df)
    results["summary_table"] = summary_df
    return results
