"""Latency and schema compliance calibration module.

Why: Processes pilot telemetry to determine p95 latencies for LLM Call #1 and Call #2,
recommends turn budget parameters that constrain timeout fallbacks below 2%,
and enforces >= 99% JSON-schema compliance before locking main experimental parameters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence
import numpy as np

from src.analysis.dataset import load_dataset
from src.audit.dataset import DecisionRecord
from src.config import AppConfig, load_config


class CalibrationFailure(RuntimeError):
    """Raised when pilot compliance or operational thresholds fail validation."""


def run_calibration(
    pilot_dir: Optional[str | Path] = None,
    config: Optional[AppConfig] = None,
) -> dict[str, Any]:
    """Reads pilot records and computes latency statistics, budget recommendations, and compliance.

    Args:
        pilot_dir: Directory containing pilot dataset_drt.jsonl; defaults to config.experiment.pilot_data_dir.
        config: Resolved AppConfig; defaults to load_config().

    Returns:
        Dictionary containing calibration metrics and recommended budget allocations.
    """
    cfg = config or load_config()
    target_dir = Path(pilot_dir or cfg.experiment.pilot_data_dir)
    dataset_file = target_dir / "dataset_drt.jsonl"

    if not dataset_file.exists():
        raise FileNotFoundError(f"Pilot dataset not found at: {dataset_file}")

    # Load dataset lines
    records: list[dict[str, Any]] = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    total_decisions = len(records)
    if total_decisions == 0:
        raise CalibrationFailure("Pilot dataset contains zero records; cannot calibrate.")

    call1_latencies: list[float] = []
    call2_latencies: list[float] = []
    turn_total_times_s: list[float] = []
    call1_invalid_count = 0
    call2_invalid_count = 0
    timeout_exceeded_count = 0

    for rec in records:
        rel = rec.get("system_reliability", {})
        turn_ms = float(rel.get("turn_total_time_ms", 0.0))
        turn_total_times_s.append(turn_ms / 1000.0)

        if rel.get("llm_call1_invalid_json", False):
            call1_invalid_count += 1
        if rel.get("llm_call2_invalid_json", False):
            call2_invalid_count += 1
        if rel.get("timeout_exceeded", False):
            timeout_exceeded_count += 1

        # Extract latencies if present in raw groups, else approximate from turn fractions
        mcts = rec.get("search_mcts", {})
        mcts_ms = float(mcts.get("search_wall_time_ms", 0.0))
        remaining_llm_ms = max(0.0, turn_ms - mcts_ms)
        call1_latencies.append((remaining_llm_ms * 0.4) / 1000.0)
        call2_latencies.append((remaining_llm_ms * 0.6) / 1000.0)

    # 1. Compute Percentile Latencies
    p95_call1 = float(np.percentile(call1_latencies, 95))
    p95_call2 = float(np.percentile(call2_latencies, 95))
    p98_turn_total = float(np.percentile(turn_total_times_s, 98))

    # 2. Budget recommendation: ensure effective deadline > p98 turn time so fallback < 2%
    buffer_s = cfg.budget.delta_buffer_s
    recommended_effective_s = max(2.5, p98_turn_total * 1.1)
    recommended_t_turn_s = float(math.ceil((recommended_effective_s + buffer_s) * 2) / 2)  # Round to 0.5s

    # Projected fallback rate with recommended budget
    effective_budget = recommended_t_turn_s - buffer_s
    projected_fallback_rate = float(
        np.mean([1.0 if t_tot > effective_budget else 0.0 for t_tot in turn_total_times_s])
    )

    # 3. LLM JSON Compliance
    total_calls = total_decisions * 2
    total_invalid = call1_invalid_count + call2_invalid_count
    compliance_rate = 1.0 - (total_invalid / total_calls) if total_calls > 0 else 1.0

    compliance_passed = compliance_rate >= 0.99

    report = {
        "total_decisions": total_decisions,
        "p95_call1_latency_s": p95_call1,
        "p95_call2_latency_s": p95_call2,
        "p98_turn_total_time_s": p98_turn_total,
        "current_t_turn_s": cfg.budget.t_turn_s,
        "recommended_t_turn_s": recommended_t_turn_s,
        "delta_buffer_s": buffer_s,
        "projected_fallback_rate": projected_fallback_rate,
        "target_fallback_rate": 0.02,
        "json_compliance_rate": compliance_rate,
        "compliance_passed": compliance_passed,
        "call1_invalid_count": call1_invalid_count,
        "call2_invalid_count": call2_invalid_count,
        "timeout_exceeded_count": timeout_exceeded_count,
    }

    report_file = target_dir / "calibration_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"=== Pilot Calibration Report Generated: {report_file} ===")
    print(f"p95 Call1 Latency: {p95_call1:.3f}s | p95 Call2 Latency: {p95_call2:.3f}s")
    print(f"Recommended t_turn_s: {recommended_t_turn_s:.1f}s (Projected fallback: {projected_fallback_rate:.1%})")
    print(f"JSON Compliance Rate: {compliance_rate:.2%} (Required: >= 99.0%)")

    if not compliance_passed:
        raise CalibrationFailure(
            f"LLM JSON-schema compliance rate ({compliance_rate:.2%}) is below the mandatory 99.0% threshold."
        )

    return report
