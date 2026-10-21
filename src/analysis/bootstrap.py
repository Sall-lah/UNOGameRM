"""Round-level cluster bootstrap for clustered decision trajectory data.

Why: Turn-level observations within a game round violate independence due to game state
and inventory dependencies. Resampling whole round clusters with replacement preserves
intra-round temporal correlations and yields valid, reproducible confidence intervals.
"""

from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd


def round_level_cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Estimates point values and percentile confidence intervals via cluster bootstrap.

    Why: Computes standard 95% CIs while respecting round clustering, deterministically
    seeding the generator and gracefully handling NaN replications.

    Args:
        df: Clustered DataFrame containing at least a 'round_id' column.
        metric_fn: Function mapping a DataFrame to a dictionary of scalar metric values.
        n_boot: Number of bootstrap replications (default: 2000).
        ci_level: Desired two-sided confidence interval width (default: 0.95).
        seed: Deterministic integer seed for np.random.default_rng.

    Returns:
        Dict mapping each metric name to {"estimate": ..., "ci_95_low": ..., "ci_95_high": ...,
        "ci_low": ..., "ci_high": ...}.
    """
    point_est = metric_fn(df)

    if df.empty or "round_id" not in df.columns:
        return {
            k: {
                "estimate": v,
                "ci_95_low": float("nan"),
                "ci_95_high": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            for k, v in point_est.items()
        }

    rounds = df["round_id"].unique()
    num_rounds = len(rounds)

    if num_rounds == 0:
        return {
            k: {
                "estimate": v,
                "ci_95_low": float("nan"),
                "ci_95_high": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
            }
            for k, v in point_est.items()
        }

    # Group DataFrame by round_id once for fast cluster reconstruction
    round_groups = {r: grp for r, grp in df.groupby("round_id")}

    rng = np.random.default_rng(seed)

    # Accumulators for each metric key
    replications: dict[str, list[float]] = {k: [] for k in point_est.keys()}

    for _ in range(n_boot):
        # Sample rounds with replacement
        sampled_rounds = rng.choice(rounds, size=num_rounds, replace=True)
        # Concatenate selected round clusters
        resampled_df = pd.concat([round_groups[r] for r in sampled_rounds], ignore_index=True)
        rep_metrics = metric_fn(resampled_df)

        for k in replications.keys():
            replications[k].append(float(rep_metrics.get(k, float("nan"))))

    alpha = (1.0 - ci_level) / 2.0
    percentile_low = 100.0 * alpha
    percentile_high = 100.0 * (1.0 - alpha)

    results: dict[str, dict[str, float]] = {}

    for k, est in point_est.items():
        vals = replications.get(k, [])
        valid_vals = [v for v in vals if not np.isnan(v)]

        if len(valid_vals) == 0:
            ci_low = float("nan")
            ci_high = float("nan")
        elif len(valid_vals) == 1:
            ci_low = valid_vals[0]
            ci_high = valid_vals[0]
        else:
            ci_low = float(np.percentile(valid_vals, percentile_low))
            ci_high = float(np.percentile(valid_vals, percentile_high))

        results[k] = {
            "estimate": float(est) if not np.isnan(est) else float("nan"),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return results
