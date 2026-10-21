"""Tests for RQ2 alignment, search shift, and rationale rubric (Task 10.6)."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.dataset import load_dataset
from src.analysis.rq2 import deterministic_argmax, evaluate_rq2
from src.analysis.rubric import export_audit_sample, score_rationale
from tests.analysis.conftest import make_synthetic_record_dict


def test_deterministic_argmax_and_tie_breaking():
    """Verify deterministic argmax breaks score ties alphabetically."""
    # "red 1" vs "blue 1" with equal score -> "blue 1" wins alphabetically
    scores = {"red 1": 0.5, "blue 1": 0.5}
    winner = deterministic_argmax(scores)
    assert winner == "blue 1"


def test_top1_agreement_and_search_shift():
    """Verify top-1 agreement and search shift calculations."""
    rec = make_synthetic_record_dict(
        locked_action="red 5",
        p_prior={"red 5": 0.65, "draw": 0.35},
        visit_counts={"red 5": 80, "draw": 20},
    )
    df = pd.DataFrame([
        {
            "round_id": rec["identity"]["round_id"],
            "turn_idx": rec["identity"]["turn_idx"],
            "locked_action_at": rec["stage1_freeze_log"]["locked_action_at"],
            "short_rationale": rec["stage1_freeze_log"]["short_rationale"],
            "p_prior": rec["search_mcts"]["p_prior"],
            "visit_counts_N": rec["search_mcts"]["visit_counts_N"],
            "opponent_draw_constraints": rec["memory_plugmem"]["opponent_draw_constraints"],
            "opponent_wild_declarations": rec["memory_plugmem"]["opponent_wild_declarations"],
            "agent_hand": rec["observation_ot"]["agent_hand"],
            "top_discard": rec["observation_ot"]["top_discard"],
            "opp_card_count": rec["observation_ot"]["opp_card_count"],
            "action_values_Q": rec["search_mcts"]["action_values_Q"],
        }
    ])

    res = evaluate_rq2(df)
    overall = res["groups"]["Overall"]

    assert overall["N_decisions"] == 1
    assert overall["top1_agreement_rate"] == 1.0
    assert pytest.approx(overall["mean_search_shift"], abs=1e-6) == 0.65


def test_expt_stratification():
    """Verify stratification by Exp_t empty vs non-empty."""
    # Row 1: Exp_t empty
    rec1 = make_synthetic_record_dict(round_id=1, turn_idx=1)
    # Row 2: Exp_t non-empty
    rec2 = make_synthetic_record_dict(
        round_id=1,
        turn_idx=2,
        opp_draw_constraints=[{"colour": "red", "confidence": 1.0}],
    )

    df = pd.DataFrame([
        {
            "round_id": r["identity"]["round_id"],
            "turn_idx": r["identity"]["turn_idx"],
            "locked_action_at": r["stage1_freeze_log"]["locked_action_at"],
            "short_rationale": r["stage1_freeze_log"]["short_rationale"],
            "p_prior": r["search_mcts"]["p_prior"],
            "visit_counts_N": r["search_mcts"]["visit_counts_N"],
            "opponent_draw_constraints": r["memory_plugmem"]["opponent_draw_constraints"],
            "opponent_wild_declarations": r["memory_plugmem"]["opponent_wild_declarations"],
            "agent_hand": r["observation_ot"]["agent_hand"],
            "top_discard": r["observation_ot"]["top_discard"],
            "opp_card_count": r["observation_ot"]["opp_card_count"],
            "action_values_Q": r["search_mcts"]["action_values_Q"],
        }
        for r in (rec1, rec2)
    ])

    res = evaluate_rq2(df)
    assert res["groups"]["Exp_t_empty"]["N_decisions"] == 1
    assert res["groups"]["Exp_t_non_empty"]["N_decisions"] == 1
    assert res["groups"]["Overall"]["N_decisions"] == 2


def test_rubric_hallucinated_card_fails_v_obs():
    """Verify that referencing a card not held fails Observation Fidelity (V_obs == 0)."""
    rec = make_synthetic_record_dict(
        agent_hand=["red 1", "blue 2"],
        top_discard="red 3",
        locked_action="red 1",
        rationale="Holding green 9 which protects us against counter-attack.",
    )
    score = score_rationale(rec["stage1_freeze_log"] | rec["observation_ot"] | rec["search_mcts"] | rec["memory_plugmem"])

    assert score["V_obs"] == 0
    assert score["RV"] in (0, 1, 2)


def test_rubric_turn_retention_inconsistency_fails_v_obs():
    """Verify claiming opponent acts after playing a turn-retention card fails V_obs."""
    rec = make_synthetic_record_dict(
        agent_hand=["red skip"],
        locked_action="red skip",
        rationale="Playing red skip, opponent will act next with high danger.",
    )
    score = score_rationale(rec["stage1_freeze_log"] | rec["observation_ot"] | rec["search_mcts"] | rec["memory_plugmem"])

    assert score["V_obs"] == 0


def test_rubric_fabricated_constraint_fails_v_mem():
    """Verify asserting opponent lacks a colour when Exp_t is empty fails Memory Fidelity (V_mem == 0)."""
    rec = make_synthetic_record_dict(
        opp_draw_constraints=[],  # Exp_t empty
        rationale="Opponent has no green so we safely switch colours.",
    )
    score = score_rationale(rec["stage1_freeze_log"] | rec["observation_ot"] | rec["search_mcts"] | rec["memory_plugmem"])

    assert score["V_mem"] == 0


def test_audit_sample_deterministic_and_bounded(synthetic_jsonl_dataset: Path):
    """Verify that export_audit_sample is deterministic and safely bounded by available turns."""
    df = load_dataset(synthetic_jsonl_dataset)

    cfg = AnalysisConfig(audit_sample_size=3, audit_sample_seed=99)
    sample1 = export_audit_sample(df, cfg)
    sample2 = export_audit_sample(df, cfg)

    assert len(sample1) == 3
    assert (sample1["round_id"].values == sample2["round_id"].values).all()
    assert (sample1["turn_idx"].values == sample2["turn_idx"].values).all()

    # Bounded by available turns when sample_size > len(df)
    large_cfg = AnalysisConfig(audit_sample_size=1000, audit_sample_seed=99)
    large_sample = export_audit_sample(df, large_cfg)
    assert len(large_sample) == len(df)
