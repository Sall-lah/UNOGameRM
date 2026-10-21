"""Evaluation analysis package for statistical risk scoring, calibration, and audit.

Why: Provides a clean public API for loading clustered UNO decision datasets,
computing calibration metrics (BS, BSS, ECE, MPB), executing round-level cluster
bootstrapping, evaluating RQ1/RQ2/RQ3, scoring rationales, and generating reports.
"""

from __future__ import annotations

from src.analysis.bootstrap import round_level_cluster_bootstrap
from src.analysis.config import AnalysisConfig
from src.analysis.dataset import DatasetValidationError, load_dataset
from src.analysis.metrics import compute_calibration_metrics
from src.analysis.rq1 import evaluate_rq1
from src.analysis.rq2 import evaluate_rq2
from src.analysis.rq3 import evaluate_rq3
from src.analysis.rubric import export_audit_sample, score_rationale
from src.analysis.run import run_analysis
from src.analysis.split import split_fallback

__all__ = [
    "AnalysisConfig",
    "DatasetValidationError",
    "compute_calibration_metrics",
    "evaluate_rq1",
    "evaluate_rq2",
    "evaluate_rq3",
    "export_audit_sample",
    "load_dataset",
    "round_level_cluster_bootstrap",
    "run_analysis",
    "score_rationale",
    "split_fallback",
]
