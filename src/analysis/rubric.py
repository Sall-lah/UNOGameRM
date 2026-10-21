"""Automated rationale validity rubric (V_obs, V_mem, V_mcts, RV) and audit sample export.

Why: Evaluates the factual fidelity and mechanical coherence of LLM Call #2 rationales
against frozen Stage-1 observations, PlugMem constraints, and MCTS statistics without
accessing hidden state, producing an automated proxy score and human audit export.
"""

from __future__ import annotations

import re
from typing import Any, Optional
import numpy as np
import pandas as pd

from src.analysis.config import AnalysisConfig
from src.core.cards import COLOURS

_COLOUR_PATTERN = re.compile(r"\b(red|green|blue|yellow)\b", re.IGNORECASE)
_CARD_PATTERN = re.compile(
    r"\b(red|green|blue|yellow)\s+(0|1|2|3|4|5|6|7|8|9|skip|reverse|\+ 2|\+2)\b",
    re.IGNORECASE,
)
_OPP_LACK_PATTERN = re.compile(
    r"opponent (?:lacks|has no|does not have)\s+(red|green|blue|yellow)",
    re.IGNORECASE,
)
_OPP_ACT_PATTERN = re.compile(
    r"opponent (?:will act|acts|plays|takes a turn|gets a turn)|gives opponent a turn|opponent's turn",
    re.IGNORECASE,
)
_STACK_PATTERN = re.compile(r"\bstack(?:ing|ed|s)?\b", re.IGNORECASE)


def score_rationale(row: Any) -> dict[str, int]:
    """Scores a single turn's rationale on observation, memory, and search fidelity.

    Why: Computes three verifiable binary scores in {0, 1} strictly from frozen
    Stage-1 facts without peeking at hidden opponent state.

    Args:
        row: Row-like object (pandas Series or dict) containing Stage 1 facts and rationale.

    Returns:
        Dictionary with V_obs, V_mem, V_mcts, and composite RV in {0..3}.
    """
    rationale = str(row.get("short_rationale", "") if hasattr(row, "get") else row["short_rationale"])
    agent_hand = list(row.get("agent_hand", []) if hasattr(row, "get") else row["agent_hand"])
    top_discard = str(row.get("top_discard", "") if hasattr(row, "get") else row["top_discard"])
    opp_card_count = int(row.get("opp_card_count", 0) if hasattr(row, "get") else row["opp_card_count"])
    locked_action = str(row.get("locked_action_at", "") if hasattr(row, "get") else row["locked_action_at"])
    visit_counts = dict(row.get("visit_counts_N", {}) if hasattr(row, "get") else row["visit_counts_N"])
    action_values = dict(row.get("action_values_Q", {}) if hasattr(row, "get") else row["action_values_Q"])
    opp_draw_constraints = list(
        row.get("opponent_draw_constraints", []) if hasattr(row, "get") else row["opponent_draw_constraints"]
    )

    # 1. Observation Fidelity (V_obs)
    v_obs = 1

    # Check for hallucinated hand card mentions
    # If a specific color-rank card is mentioned, it must be in agent hand, locked action, or top discard
    for match in _CARD_PATTERN.finditer(rationale):
        mentioned_card = match.group(0).lower()
        # Normalise "+2" to "+ 2"
        normalised_card = mentioned_card.replace("+2", "+ 2")
        hand_lower = [c.lower() for c in agent_hand]
        if (
            normalised_card not in hand_lower
            and normalised_card not in top_discard.lower()
            and normalised_card not in locked_action.lower()
        ):
            v_obs = 0
            break

    # Turn-retention consistency:
    # If locked_action is turn-retaining, the opponent does NOT act
    is_turn_retaining = any(
        kw in locked_action.lower() for kw in ("skip", "reverse", "+ 2", "+ 4")
    )
    if is_turn_retaining and _OPP_ACT_PATTERN.search(rationale):
        v_obs = 0

    # Opponent card count consistency (e.g. "opponent has X cards")
    opp_count_match = re.search(r"opponent has (\d+) card", rationale, re.IGNORECASE)
    if opp_count_match:
        claimed_count = int(opp_count_match.group(1))
        if claimed_count != opp_card_count:
            v_obs = 0

    # 2. Memory Fidelity (V_mem)
    v_mem = 1
    # Check for claimed opponent lack of colours
    active_lack_colours = {
        item.get("colour", "").lower()
        for item in opp_draw_constraints
        if isinstance(item, dict)
    }

    lack_matches = _OPP_LACK_PATTERN.findall(rationale)
    if lack_matches:
        for claimed_col in lack_matches:
            if claimed_col.lower() not in active_lack_colours:
                v_mem = 0
                break

    # If rationale claims opponent lacks something but Exp_t is completely empty
    if len(opp_draw_constraints) == 0:
        if re.search(r"opponent (?:lacks|has no|does not have)\s+(?:any\s+)?(red|green|blue|yellow)", rationale, re.IGNORECASE):
            v_mem = 0

    # 3. Search / Mechanics Fidelity (V_mcts)
    v_mcts = 1

    # Check stacking illegality
    if _STACK_PATTERN.search(rationale):
        v_mcts = 0

    # Check visit count consistency if mentioned
    if "highest visit" in rationale.lower() or "most visited" in rationale.lower():
        if visit_counts:
            max_v = max(visit_counts.values())
            if visit_counts.get(locked_action, 0) < max_v:
                v_mcts = 0

    rv = v_obs + v_mem + v_mcts
    return {
        "V_obs": v_obs,
        "V_mem": v_mem,
        "V_mcts": v_mcts,
        "RV": rv,
    }


def export_audit_sample(
    df: pd.DataFrame,
    cfg: Optional[AnalysisConfig] = None,
) -> pd.DataFrame:
    """Exports a reproducible random sample of scored rationales for human audit.

    Why: Supports doc-07 qualitative validation of the automated rubric against
    human evaluator judgment using seeded, bounded sampling.

    Args:
        df: DataFrame containing decision turns.
        cfg: AnalysisConfig specifying sample size and random seed.

    Returns:
        DataFrame containing audit fields and automated rubric scores.
    """
    if cfg is None:
        cfg = AnalysisConfig()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "round_id",
                "turn_idx",
                "short_rationale",
                "agent_hand",
                "opp_card_count",
                "locked_action_at",
                "opponent_draw_constraints",
                "opponent_wild_declarations",
                "V_obs",
                "V_mem",
                "V_mcts",
                "RV",
            ]
        )

    # Compute scores for all rows
    scores = [score_rationale(row) for _, row in df.iterrows()]
    scores_df = pd.DataFrame(scores)

    combined_df = df.copy()
    for col in ("V_obs", "V_mem", "V_mcts", "RV"):
        combined_df[col] = scores_df[col].values

    target_n = min(len(combined_df), cfg.audit_sample_size)
    rng = np.random.default_rng(cfg.audit_sample_seed)

    sampled_indices = rng.choice(len(combined_df), size=target_n, replace=False)
    # Sort sampled indices for deterministic row ordering
    sampled_indices.sort()

    sample_df = combined_df.iloc[sampled_indices].copy()

    export_cols = [
        "round_id",
        "turn_idx",
        "short_rationale",
        "agent_hand",
        "opp_card_count",
        "locked_action_at",
        "opponent_draw_constraints",
        "opponent_wild_declarations",
        "V_obs",
        "V_mem",
        "V_mcts",
        "RV",
    ]
    existing_cols = [c for c in export_cols if c in sample_df.columns]
    return sample_df[existing_cols].reset_index(drop=True)
