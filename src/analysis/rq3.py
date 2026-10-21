"""RQ3: Prediction error residuals, adverse-event anticipation, and resource trajectories.

Why: Evaluates the operational consequences of miscalibration by computing Brier residual
quartiles and signed residuals, contrasting anticipated (TP) vs unanticipated (FN) adverse events,
and tracking cross-turn card economy dynamics shifted within round boundaries.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from src.analysis.config import AnalysisConfig


def evaluate_rq3(
    df: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
) -> dict[str, Any]:
    """Computes residual loss quartiles, TP/FN adverse event classification, and next-turn shifts.

    Why: Addresses Research Question 3 by assessing whether unanticipated adverse events (FN)
    correlate with severe resource degradation in subsequent decision steps compared to
    anticipated events (TP) where the agent prepared defensive actions.

    Args:
        df: Loaded decision dataset DataFrame.
        cfg: AnalysisConfig specifying tau_risk.

    Returns:
        Dictionary containing residual summaries, TP/FN partitions, and resource shifts.
    """
    if cfg is None:
        cfg = AnalysisConfig()

    tau = cfg.tau_risk

    if df.empty:
        return {
            "residuals": {},
            "adverse_contrast": {},
            "enriched_df": df.copy(),
        }

    # Ensure sorted order within rounds for accurate temporal shifts
    sorted_df = df.sort_values(by=["round_id", "turn_idx"]).reset_index(drop=True)

    # 1. Residual Calculations
    # Risk A: penalty draw
    p_A = sorted_df["p_tA"].to_numpy()
    y_A = sorted_df["y_tA"].to_numpy()
    signed_res_A = p_A - y_A
    loss_A = (p_A - y_A) ** 2

    # Risk B: opponent win
    p_B = sorted_df["p_tB"].to_numpy()
    y_B = sorted_df["y_tB"].to_numpy()
    signed_res_B = p_B - y_B
    loss_B = (p_B - y_B) ** 2

    sorted_df["signed_res_A"] = signed_res_A
    sorted_df["brier_loss_A"] = loss_A
    sorted_df["signed_res_B"] = signed_res_B
    sorted_df["brier_loss_B"] = loss_B

    # Adverse-event classification
    # For y == 1: TP if p >= tau else FN
    def _classify_adverse(p: float, y: int) -> Optional[str]:
        if y != 1:
            return None
        return "TP" if p >= tau else "FN"

    sorted_df["adverse_class_A"] = [
        _classify_adverse(p, y) for p, y in zip(p_A, y_A)
    ]
    sorted_df["adverse_class_B"] = [
        _classify_adverse(p, y) for p, y in zip(p_B, y_B)
    ]

    # 2. Next-turn resource outcomes shifted strictly within round_id
    if "legal_actions" in sorted_df.columns:
        sorted_df["legal_action_count"] = sorted_df["legal_actions"].apply(
            lambda acts: len(acts) if isinstance(acts, (list, tuple)) else np.nan
        )
    else:
        sorted_df["legal_action_count"] = np.nan

    grouped_round = sorted_df.groupby("round_id")

    sorted_df["next_turn_defensive_action"] = grouped_round["used_action_or_wild_at_t"].shift(-1)
    sorted_df["next_turn_cumulative_drawn"] = grouped_round["cumulative_cards_drawn_round"].shift(-1)
    sorted_df["next_turn_legal_action_count"] = grouped_round["legal_action_count"].shift(-1)
    sorted_df["next_turn_delta_hand"] = grouped_round["delta_hand_size_next"].shift(-1)

    # 3. Compute Residual Quartiles
    def _quartiles(arr: np.ndarray) -> dict[str, float]:
        if len(arr) == 0:
            return {"Q1": float("nan"), "Q2": float("nan"), "Q3": float("nan"), "Q4": float("nan")}
        return {
            "Q1": float(np.percentile(arr, 25)),
            "Q2": float(np.percentile(arr, 50)),
            "Q3": float(np.percentile(arr, 75)),
            "Q4": float(np.percentile(arr, 100)),
        }

    residuals_summary = {
        "Risk_A": {
            "mean_signed_residual": float(np.mean(signed_res_A)),
            "mean_brier_loss": float(np.mean(loss_A)),
            "loss_quartiles": _quartiles(loss_A),
        },
        "Risk_B": {
            "mean_signed_residual": float(np.mean(signed_res_B)),
            "mean_brier_loss": float(np.mean(loss_B)),
            "loss_quartiles": _quartiles(loss_B),
        },
    }

    # 4. Compute Adverse Contrast (TP vs FN) on resource metrics
    contrast_summary: dict[str, Any] = {}

    for risk_label, class_col in [("Risk_A", "adverse_class_A"), ("Risk_B", "adverse_class_B")]:
        risk_contrast: dict[str, Any] = {}
        for group_label in ("TP", "FN"):
            subset = sorted_df[sorted_df[class_col] == group_label]
            n_events = len(subset)
            if n_events == 0:
                risk_contrast[group_label] = {
                    "count": 0,
                    "mean_delta_hand_next": float("nan"),
                    "mean_next_turn_defensive": float("nan"),
                    "mean_cumulative_drawn": float("nan"),
                    "mean_next_legal_actions": float("nan"),
                }
            else:
                risk_contrast[group_label] = {
                    "count": n_events,
                    "mean_delta_hand_next": float(subset["delta_hand_size_next"].mean()),
                    "mean_next_turn_defensive": float(subset["next_turn_defensive_action"].dropna().mean())
                    if not subset["next_turn_defensive_action"].dropna().empty
                    else float("nan"),
                    "mean_cumulative_drawn": float(subset["cumulative_cards_drawn_round"].mean()),
                    "mean_next_legal_actions": float(subset["next_turn_legal_action_count"].dropna().mean())
                    if not subset["next_turn_legal_action_count"].dropna().empty
                    else float("nan"),
                }
        contrast_summary[risk_label] = risk_contrast

    return {
        "residuals": residuals_summary,
        "adverse_contrast": contrast_summary,
        "enriched_df": sorted_df,
    }
