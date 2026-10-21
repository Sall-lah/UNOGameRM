"""Primary vs sensitivity analysis dataset partitioning.

Why: Evaluates agent calibration under normal operating conditions (primary) separately
from budget overrun or component degradation episodes (sensitivity), preserving the complete
empirical audit trace without discarding edge cases.
"""

from __future__ import annotations

import pandas as pd


def split_fallback(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions dataset into non-fallback (primary) and fallback/timeout (sensitivity) subsets.

    Why: Statistical claims regarding LLM risk calibration require an uncontaminated primary
    sample, while safety auditing demands explicit characterisation of fallback behavior.

    Args:
        df: Loaded DataFrame containing 'timeout_exceeded' and 'mcts_fallback_used' flags.

    Returns:
        Tuple of (primary_df, sensitivity_df).
    """
    if "fallback_triggered" in df.columns:
        fallback_mask = df["fallback_triggered"].astype(bool)
    else:
        fallback_mask = df["timeout_exceeded"].astype(bool) | df["mcts_fallback_used"].astype(bool)

    primary_df = df[~fallback_mask].copy().reset_index(drop=True)
    sensitivity_df = df[fallback_mask].copy().reset_index(drop=True)

    if len(primary_df) + len(sensitivity_df) != len(df):
        raise AssertionError(
            f"Row count mismatch in fallback split: primary ({len(primary_df)}) + "
            f"sensitivity ({len(sensitivity_df)}) != total ({len(df)})"
        )

    return primary_df, sensitivity_df
