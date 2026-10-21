"""RQ2: Memory and search utilisation, alignment, and rationale evaluation.

Why: Evaluates the influence of PlugMem constraints and MARS MCTS search on agent decisions,
measuring prior-to-commitment top-1 agreement and search shift stratified by active memory,
and summarizing the automated rationale validity rubric.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np
import pandas as pd

from src.analysis.config import AnalysisConfig
from src.analysis.rubric import score_rationale


def deterministic_argmax(scores: dict[str, Union[int, float]]) -> Optional[str]:
    """Deterministically identifies the highest-scoring action key with stable tie-breaking.

    Why: Statistical reproducibility requires consistent tie resolution across runs;
    breaks score ties alphabetically by action identifier string.
    """
    if not scores:
        return None
    # Sort descending by score, ascending by key string
    sorted_items = sorted(scores.items(), key=lambda item: (-float(item[1]), str(item[0])))
    return sorted_items[0][0]


def evaluate_rq2(
    df: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
) -> dict[str, Any]:
    """Computes top-1 agreement, search shift, and rationale scores stratified by Exp_t.

    Why: Addresses Research Question 2 by quantifying how search shifts probability mass
    away from LLM priors, and testing whether memory presence correlates with higher alignment.

    Args:
        df: Loaded decision dataset DataFrame.
        cfg: AnalysisConfig specifying parameters.

    Returns:
        Dictionary containing metric summaries overall and stratified by Exp_t empty/non-empty.
    """
    if cfg is None:
        cfg = AnalysisConfig()

    if df.empty:
        empty_table = pd.DataFrame(
            columns=[
                "group",
                "N_decisions",
                "top1_agreement_rate",
                "mean_search_shift",
                "mean_V_obs",
                "mean_V_mem",
                "mean_V_mcts",
                "mean_RV",
            ]
        )
        return {"summary_table": empty_table, "groups": {}}

    # Compute turn-level RQ2 indicators
    top1_agreements: list[int] = []
    search_shifts: list[float] = []
    exp_t_non_empty: list[bool] = []
    v_obs_list: list[int] = []
    v_mem_list: list[int] = []
    v_mcts_list: list[int] = []
    rv_list: list[int] = []

    for _, row in df.iterrows():
        p_prior = dict(row.get("p_prior", {}))
        visit_counts = dict(row.get("visit_counts_N", {}))
        locked_action = str(row.get("locked_action_at", ""))

        # Top-1 agreement
        prior_argmax = deterministic_argmax(p_prior)
        mcts_argmax = deterministic_argmax(visit_counts)
        agreement = 1 if (prior_argmax is not None and prior_argmax == mcts_argmax) else 0
        top1_agreements.append(agreement)

        # Search shift magnitude: prior mass on locked action
        shift_val = float(p_prior.get(locked_action, 0.0))
        search_shifts.append(shift_val)

        # Memory constraints present (Exp_t non-empty)
        opp_draw = list(row.get("opponent_draw_constraints", []))
        opp_wild = list(row.get("opponent_wild_declarations", []))
        has_mem = (len(opp_draw) > 0) or (len(opp_wild) > 0)
        exp_t_non_empty.append(has_mem)

        # Rationale Rubric
        rubric_res = score_rationale(row)
        v_obs_list.append(rubric_res["V_obs"])
        v_mem_list.append(rubric_res["V_mem"])
        v_mcts_list.append(rubric_res["V_mcts"])
        rv_list.append(rubric_res["RV"])

    enriched_df = df.copy()
    enriched_df["top1_agreement"] = top1_agreements
    enriched_df["search_shift"] = search_shifts
    enriched_df["exp_t_non_empty"] = exp_t_non_empty
    enriched_df["V_obs"] = v_obs_list
    enriched_df["V_mem"] = v_mem_list
    enriched_df["V_mcts"] = v_mcts_list
    enriched_df["RV"] = rv_list

    groups_data = [
        ("Overall", enriched_df),
        ("Exp_t_empty", enriched_df[~enriched_df["exp_t_non_empty"]]),
        ("Exp_t_non_empty", enriched_df[enriched_df["exp_t_non_empty"]]),
    ]

    summary_rows: list[dict[str, Any]] = []
    groups_dict: dict[str, dict[str, Any]] = {}

    for name, grp_df in groups_data:
        n = len(grp_df)
        if n == 0:
            summary_dict = {
                "group": name,
                "N_decisions": 0,
                "top1_agreement_rate": float("nan"),
                "mean_search_shift": float("nan"),
                "mean_V_obs": float("nan"),
                "mean_V_mem": float("nan"),
                "mean_V_mcts": float("nan"),
                "mean_RV": float("nan"),
            }
        else:
            summary_dict = {
                "group": name,
                "N_decisions": n,
                "top1_agreement_rate": float(grp_df["top1_agreement"].mean()),
                "mean_search_shift": float(grp_df["search_shift"].mean()),
                "mean_V_obs": float(grp_df["V_obs"].mean()),
                "mean_V_mem": float(grp_df["V_mem"].mean()),
                "mean_V_mcts": float(grp_df["V_mcts"].mean()),
                "mean_RV": float(grp_df["RV"].mean()),
            }

        summary_rows.append(summary_dict)
        groups_dict[name] = summary_dict

    summary_table = pd.DataFrame(summary_rows)
    return {
        "summary_table": summary_table,
        "groups": groups_dict,
        "enriched_df": enriched_df,
    }
