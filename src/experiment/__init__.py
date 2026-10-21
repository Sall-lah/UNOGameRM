"""Experiment runbook package orchestrating validate, pilot, calibrate, and main runs.

Why: Exposes the complete protocol execution interfaces conforming to docs/08.
"""

from __future__ import annotations

from src.experiment.calibrate import run_calibration
from src.experiment.main_run import run_main_experiment
from src.experiment.manifest import write_manifest
from src.experiment.pilot import run_pilot
from src.experiment.sample_size import estimate_sample_size
from src.experiment.validate import run_all_validation_checks

__all__ = [
    "estimate_sample_size",
    "run_all_validation_checks",
    "run_calibration",
    "run_main_experiment",
    "run_pilot",
    "write_manifest",
]
