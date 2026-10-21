"""Tests for evaluation analysis CLI runner (Task 10.8)."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.analysis.config import AnalysisConfig
from src.analysis.run import run_analysis


def test_cli_writes_tables_and_json(synthetic_jsonl_dataset: Path, tmp_path: Path):
    """Verify run_analysis writes CSV and JSON reports to output_dir."""
    out_dir = tmp_path / "reports_test"
    cfg = AnalysisConfig(
        bootstrap_B=50,
        bootstrap_seed=42,
        audit_sample_size=3,
        audit_sample_seed=42,
        output_dir=str(out_dir),
    )

    report = run_analysis(synthetic_jsonl_dataset, cfg)

    assert (out_dir / "rq1_primary_calibration.csv").exists()
    assert (out_dir / "rq2_primary_alignment.csv").exists()
    assert (out_dir / "rq1_sensitivity_calibration.csv").exists()
    assert (out_dir / "rq2_sensitivity_alignment.csv").exists()
    assert (out_dir / "rationale_audit_sample.csv").exists()
    assert (out_dir / "evaluation_report.json").exists()

    assert report["metadata"]["total_records"] == 7
    assert report["metadata"]["primary_records"] == 6
    assert report["metadata"]["sensitivity_records"] == 1


def test_cli_runs_are_byte_identical(synthetic_jsonl_dataset: Path, tmp_path: Path):
    """Verify two runs with identical input and seeds produce byte-identical files."""
    out_dir1 = tmp_path / "run_1"
    out_dir2 = tmp_path / "run_2"

    cfg1 = AnalysisConfig(
        bootstrap_B=50,
        bootstrap_seed=123,
        audit_sample_size=3,
        audit_sample_seed=456,
        output_dir=str(out_dir1),
    )
    cfg2 = AnalysisConfig(
        bootstrap_B=50,
        bootstrap_seed=123,
        audit_sample_size=3,
        audit_sample_seed=456,
        output_dir=str(out_dir2),
    )

    run_analysis(synthetic_jsonl_dataset, cfg1)
    run_analysis(synthetic_jsonl_dataset, cfg2)

    files_to_compare = [
        "rq1_primary_calibration.csv",
        "rq2_primary_alignment.csv",
        "rq1_sensitivity_calibration.csv",
        "rq2_sensitivity_alignment.csv",
        "rationale_audit_sample.csv",
        "evaluation_report.json",
    ]

    for fname in files_to_compare:
        content1 = (out_dir1 / fname).read_bytes()
        content2 = (out_dir2 / fname).read_bytes()
        assert content1 == content2, f"File {fname} is not byte-identical between runs"
