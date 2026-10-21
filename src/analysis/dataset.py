"""Dataset loading, validation, and canonical column flattening for analysis.

Why: Enforces doc-06 §3 schema validation over clustered decision records (d_{r,t}),
derives binary labels and fallback flags, and flattens nested JSONL structures
into a typed pandas DataFrame with uniform column naming for downstream RQ modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.audit.dataset import DecisionRecord


class DatasetValidationError(ValueError):
    """Raised when a JSONL record fails schema validation or data integrity constraints.

    Why: Pinpoints exact (round_id, turn_idx) and offending field names to prevent
    silent propagation of corrupted or incomplete audit logs into statistical claims.
    """

    def __init__(self, round_id: Any, turn_idx: Any, field_name: str, message: str) -> None:
        super().__init__(
            f"Record (round_id={round_id}, turn_idx={turn_idx}) validation failed "
            f"on field '{field_name}': {message}"
        )
        self.round_id = round_id
        self.turn_idx = turn_idx
        self.field_name = field_name


def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """Loads and validates a clustered decision JSONL file into a typed pandas DataFrame.

    Why: Provides a single, authoritative point of ingestion that flattens doc-06 §3
    groups into canonical columns, enforces strict typing, and computes derived indicators.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    rows: list[dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                raw_dict = json.loads(line_str)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(
                    round_id="unknown",
                    turn_idx=line_num,
                    field_name="json_syntax",
                    message=f"Line {line_num} contains invalid JSON: {exc}",
                ) from exc

            # Extract identity hints for error reporting if possible
            ident = raw_dict.get("identity", {}) if isinstance(raw_dict, dict) else {}
            r_id = ident.get("round_id", f"line_{line_num}")
            t_idx = ident.get("turn_idx", f"line_{line_num}")

            try:
                record = DecisionRecord.model_validate(raw_dict)
            except ValidationError as val_err:
                # Find the first failing field
                first_err = val_err.errors()[0]
                loc_str = ".".join(str(elem) for elem in first_err.get("loc", ()))
                msg = first_err.get("msg", "Validation error")
                raise DatasetValidationError(
                    round_id=r_id,
                    turn_idx=t_idx,
                    field_name=loc_str,
                    message=msg,
                ) from val_err

            # Derive binary labels strictly in {0, 1}
            y_tA = int(1 if record.stage2_post_horizon.y_t_A_penalty_draw != 0 else 0)
            y_tB = int(1 if record.stage2_post_horizon.y_t_B_opp_win != 0 else 0)

            # Derive fallback disjunction
            timeout_exceeded = bool(record.system_reliability.timeout_exceeded)
            mcts_fallback_used = bool(record.system_reliability.mcts_fallback_used)
            fallback_triggered = timeout_exceeded or mcts_fallback_used

            flattened_row = {
                # Identity
                "round_id": int(record.identity.round_id),
                "turn_idx": int(record.identity.turn_idx),
                "starting_player": str(record.identity.starting_player),
                "first_turn_wild_colour": bool(record.identity.first_turn_wild_colour),
                # Probabilities & Labels
                "p_tA": float(record.stage1_freeze_log.p_t_A),
                "p_tB": float(record.stage1_freeze_log.p_t_B),
                "y_tA": y_tA,
                "y_tB": y_tB,
                "stratum_horizon": int(record.stage2_post_horizon.stratum_horizon),
                # Observation
                "agent_hand_count": int(record.observation_ot.agent_hand_count),
                "opp_card_count": int(record.observation_ot.opp_card_count),
                "agent_hand": list(record.observation_ot.agent_hand),
                "legal_actions": list(record.observation_ot.legal_actions),
                "top_discard": str(record.observation_ot.top_discard),
                "active_colour": str(record.observation_ot.active_colour),
                # Resources & Economy
                "used_action_or_wild_at_t": int(record.game_resources.used_action_or_wild_at_t),
                "delta_hand_size_next": int(record.game_resources.delta_hand_size_next),
                "agent_hand_count_next": int(record.game_resources.agent_hand_count_next),
                "action_wild_count_next": int(record.game_resources.action_wild_count_next),
                "cumulative_cards_drawn_round": int(
                    record.game_resources.cumulative_cards_drawn_round
                ),
                # System Reliability & Fallbacks
                "timeout_exceeded": timeout_exceeded,
                "mcts_fallback_used": mcts_fallback_used,
                "fallback_triggered": fallback_triggered,
                # Frozen Stage 1 & MCTS Statistics
                "locked_action_at": str(record.stage1_freeze_log.locked_action_at),
                "short_rationale": str(record.stage1_freeze_log.short_rationale),
                "p_prior": dict(record.search_mcts.p_prior),
                "visit_counts_N": dict(record.search_mcts.visit_counts_N),
                "action_values_Q": dict(record.search_mcts.action_values_Q),
                "simulations_N_total": int(record.search_mcts.simulations_N_total),
                # PlugMem Memory Constraints
                "retrieved_node_ids": list(record.memory_plugmem.retrieved_node_ids),
                "opponent_draw_constraints": list(
                    record.memory_plugmem.opponent_draw_constraints
                ),
                "opponent_wild_declarations": list(
                    record.memory_plugmem.opponent_wild_declarations
                ),
            }
            rows.append(flattened_row)

    df = pd.DataFrame(rows)

    # If empty DataFrame, initialize columns with expected types
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "round_id",
                "turn_idx",
                "starting_player",
                "first_turn_wild_colour",
                "p_tA",
                "p_tB",
                "y_tA",
                "y_tB",
                "stratum_horizon",
                "agent_hand_count",
                "opp_card_count",
                "agent_hand",
                "legal_actions",
                "top_discard",
                "active_colour",
                "used_action_or_wild_at_t",
                "delta_hand_size_next",
                "agent_hand_count_next",
                "action_wild_count_next",
                "cumulative_cards_drawn_round",
                "timeout_exceeded",
                "mcts_fallback_used",
                "fallback_triggered",
                "locked_action_at",
                "short_rationale",
                "p_prior",
                "visit_counts_N",
                "action_values_Q",
                "simulations_N_total",
                "retrieved_node_ids",
                "opponent_draw_constraints",
                "opponent_wild_declarations",
            ]
        )

    # Cast numeric dtypes explicitly
    numeric_int_cols = [
        "round_id",
        "turn_idx",
        "y_tA",
        "y_tB",
        "stratum_horizon",
        "agent_hand_count",
        "opp_card_count",
        "used_action_or_wild_at_t",
        "delta_hand_size_next",
        "agent_hand_count_next",
        "action_wild_count_next",
        "cumulative_cards_drawn_round",
        "simulations_N_total",
    ]
    for col in numeric_int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    numeric_float_cols = ["p_tA", "p_tB"]
    for col in numeric_float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)

    bool_cols = ["timeout_exceeded", "mcts_fallback_used", "fallback_triggered", "first_turn_wild_colour"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    return df
