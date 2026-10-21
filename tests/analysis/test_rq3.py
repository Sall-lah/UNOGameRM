"""Tests for RQ3 residual loss and resource trajectories (Task 10.7)."""

from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.dataset import load_dataset
from src.analysis.rq3 import evaluate_rq3


def test_signed_residual_and_adverse_classification():
    """Verify signed residual direction (p - y) and adverse TP/FN classification."""
    # Turn with p = 0.2 and y = 1 -> e = 0.2 - 1 = -0.8 (underprediction, FN under tau=0.5)
    df = pd.DataFrame([
        {
            "round_id": 1,
            "turn_idx": 1,
            "p_tA": 0.2,
            "y_tA": 1,
            "p_tB": 0.8,
            "y_tB": 1,
            "used_action_or_wild_at_t": 0,
            "delta_hand_size_next": 1,
            "cumulative_cards_drawn_round": 2,
            "legal_actions": ["red 1", "draw"],
        }
    ])

    cfg = AnalysisConfig(tau_risk=0.5)
    res = evaluate_rq3(df, cfg)
    enriched = res["enriched_df"]

    assert pytest.approx(enriched.iloc[0]["signed_res_A"], abs=1e-6) == -0.8
    assert enriched.iloc[0]["adverse_class_A"] == "FN"

    assert pytest.approx(enriched.iloc[0]["signed_res_B"], abs=1e-6) == -0.2
    assert enriched.iloc[0]["adverse_class_B"] == "TP"


def test_tp_fn_partition_of_adverse_events():
    """Verify all y==1 events are partitioned into exactly TP or FN."""
    df = pd.DataFrame([
        {
            "round_id": 1,
            "turn_idx": 1,
            "p_tA": 0.7,
            "y_tA": 1,  # TP
            "p_tB": 0.1,
            "y_tB": 0,  # Not adverse
            "used_action_or_wild_at_t": 0,
            "delta_hand_size_next": -1,
            "cumulative_cards_drawn_round": 0,
            "legal_actions": ["red 1"],
        },
        {
            "round_id": 1,
            "turn_idx": 2,
            "p_tA": 0.3,
            "y_tA": 1,  # FN
            "p_tB": 0.1,
            "y_tB": 0,
            "used_action_or_wild_at_t": 0,
            "delta_hand_size_next": -1,
            "cumulative_cards_drawn_round": 0,
            "legal_actions": ["red 1"],
        },
    ])

    cfg = AnalysisConfig(tau_risk=0.5)
    res = evaluate_rq3(df, cfg)
    adverse = res["adverse_contrast"]["Risk_A"]

    assert adverse["TP"]["count"] == 1
    assert adverse["FN"]["count"] == 1


def test_next_turn_shift_respects_round_boundaries(synthetic_jsonl_dataset: Path):
    """Verify that next-turn shifted variables do not leak across round boundaries (last turn is NaN)."""
    df = load_dataset(synthetic_jsonl_dataset)
    cfg = AnalysisConfig(tau_risk=0.5)

    res = evaluate_rq3(df, cfg)
    enriched = res["enriched_df"]

    # For each round, find the last turn and assert shifted variables are NaN
    for round_id, grp in enriched.groupby("round_id"):
        last_turn_row = grp.sort_values("turn_idx").iloc[-1]
        assert math.isnan(last_turn_row["next_turn_defensive_action"])
        assert math.isnan(last_turn_row["next_turn_cumulative_drawn"])
        assert math.isnan(last_turn_row["next_turn_delta_hand"])
