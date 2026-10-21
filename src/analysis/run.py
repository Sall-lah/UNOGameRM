"""CLI runner executing the complete evaluation analysis suite.

Why: Generates reproducible, byte-identical CSV and JSON statistical report tables
for RQ1, RQ2, and RQ3, partitioned across primary and sensitivity subsets as specified in doc 07.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd

from src.analysis.config import AnalysisConfig
from src.analysis.dataset import load_dataset
from src.analysis.rq1 import evaluate_rq1
from src.analysis.rq2 import evaluate_rq2
from src.analysis.rq3 import evaluate_rq3
from src.analysis.rubric import export_audit_sample
from src.analysis.split import split_fallback


def _to_json_safe(obj: Any) -> Any:
    """Converts numpy and pandas scalars/objects to JSON-serializable primitives."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "item"):
        val = obj.item()
        if pd.isna(val):
            return None
        return val
    if pd.isna(obj):
        return None
    return obj


def run_analysis(
    dataset_path: str | Path,
    cfg: AnalysisConfig,
) -> dict[str, Any]:
    """Runs primary and sensitivity evaluations and persists sorted CSV/JSON artifacts.

    Why: Orchestrates RQ1, RQ2, RQ3, and audit sample extraction, guaranteeing deterministic
    output ordering so reruns yield byte-identical reports.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    full_df = load_dataset(dataset_path)

    # 2. Partition primary vs sensitivity
    primary_df, sensitivity_df = split_fallback(full_df)

    # 3. Evaluate Primary Subset
    rq1_primary = evaluate_rq1(primary_df, cfg)
    rq2_primary = evaluate_rq2(primary_df, cfg)
    rq3_primary = evaluate_rq3(primary_df, cfg)
    audit_sample = export_audit_sample(primary_df, cfg)

    # 4. Evaluate Sensitivity Subset
    rq1_sensitivity = evaluate_rq1(sensitivity_df, cfg)
    rq2_sensitivity = evaluate_rq2(sensitivity_df, cfg)
    rq3_sensitivity = evaluate_rq3(sensitivity_df, cfg)

    # 5. Persist CSV Tables
    # Primary tables
    if "summary_table" in rq1_primary:
        rq1_table = rq1_primary["summary_table"].sort_values(by=["scope", "risk_type"]).reset_index(drop=True)
        rq1_table.to_csv(out_dir / "rq1_primary_calibration.csv", index=False)

    if "summary_table" in rq2_primary:
        rq2_table = rq2_primary["summary_table"].sort_values(by=["group"]).reset_index(drop=True)
        rq2_table.to_csv(out_dir / "rq2_primary_alignment.csv", index=False)

    # Sensitivity tables
    if "summary_table" in rq1_sensitivity:
        rq1_sens_table = rq1_sensitivity["summary_table"].sort_values(by=["scope", "risk_type"]).reset_index(drop=True)
        rq1_sens_table.to_csv(out_dir / "rq1_sensitivity_calibration.csv", index=False)

    if "summary_table" in rq2_sensitivity:
        rq2_sens_table = rq2_sensitivity["summary_table"].sort_values(by=["group"]).reset_index(drop=True)
        rq2_sens_table.to_csv(out_dir / "rq2_sensitivity_alignment.csv", index=False)

    # Audit sample CSV
    if not audit_sample.empty:
        sorted_audit = audit_sample.sort_values(by=["round_id", "turn_idx"]).reset_index(drop=True)
        sorted_audit.to_csv(out_dir / "rationale_audit_sample.csv", index=False)

    # 6. Persist Complete JSON Summary Report
    json_report = {
        "metadata": {
            "dataset_path": str(dataset_path),
            "total_records": len(full_df),
            "primary_records": len(primary_df),
            "sensitivity_records": len(sensitivity_df),
            "tau_risk": cfg.tau_risk,
            "bootstrap_B": cfg.bootstrap_B,
            "bootstrap_seed": cfg.bootstrap_seed,
            "ece_bins": cfg.ece_bins,
            "audit_sample_size": cfg.audit_sample_size,
            "audit_sample_seed": cfg.audit_sample_seed,
        },
        "primary": {
            "rq1": {k: v for k, v in rq1_primary.items() if k != "summary_table"},
            "rq2": rq2_primary.get("groups", {}),
            "rq3_residuals": rq3_primary.get("residuals", {}),
            "rq3_adverse_contrast": rq3_primary.get("adverse_contrast", {}),
        },
        "sensitivity": {
            "rq1": {k: v for k, v in rq1_sensitivity.items() if k != "summary_table"},
            "rq2": rq2_sensitivity.get("groups", {}),
            "rq3_residuals": rq3_sensitivity.get("residuals", {}),
            "rq3_adverse_contrast": rq3_sensitivity.get("adverse_contrast", {}),
        },
    }

    clean_json = _to_json_safe(json_report)
    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(clean_json, f, indent=2, sort_keys=True)

    return clean_json


def main() -> None:
    """Entry point for python -m src.analysis.run command line execution."""
    parser = argparse.ArgumentParser(
        description="Run doc-07 evaluation analysis over clustered UNO decision records."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/main_experiment/dataset_drt.jsonl",
        help="Path to clustered decision JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory where CSV and JSON reports will be saved.",
    )
    parser.add_argument(
        "--bootstrap-b",
        type=int,
        default=2000,
        help="Number of round-level cluster bootstrap iterations.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Seed for the cluster bootstrap RNG.",
    )
    parser.add_argument(
        "--tau-risk",
        type=float,
        default=0.5,
        help="Adverse event classification threshold.",
    )

    args = parser.parse_args()

    cfg = AnalysisConfig(
        bootstrap_B=args.bootstrap_b,
        bootstrap_seed=args.bootstrap_seed,
        tau_risk=args.tau_risk,
        output_dir=args.output_dir,
    )

    print(f"Loading dataset from: {args.dataset}")
    report = run_analysis(dataset_path=args.dataset, cfg=cfg)
    print(f"Analysis complete. Results written to: {args.output_dir}")
    print(f"Primary records: {report['metadata']['primary_records']}")
    print(f"Sensitivity records: {report['metadata']['sensitivity_records']}")


if __name__ == "__main__":
    main()
