"""Tests for pilot runner, latency calibration, and sample size estimation (Tasks 8.5, 8.6, 8.7)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest

from src.analysis.bootstrap import round_level_cluster_bootstrap
from src.audit.dataset import DatasetWriter, DecisionRecord
from src.config import AppConfig
from src.experiment.calibrate import CalibrationFailure, run_calibration
from src.experiment.pilot import run_pilot
from src.experiment.sample_size import estimate_sample_size
from tests.analysis.conftest import make_synthetic_record_dict


def test_pilot_runs_specified_rounds_in_pilot_dir(tmp_path: Path, fake_llm):
    """Verify pilot executes r_pilot rounds strictly under pilot_data_dir and touches nothing in main_data_dir."""
    pilot_dir = tmp_path / "pilot"
    main_dir = tmp_path / "main_experiment"

    cfg = AppConfig.from_yaml(f"""
    audit:
      strict_invariant: true
    mcts:
      max_simulations: 5
      max_rollout_depth: 2
    experiment:
      r_pilot: 2
      seed_base: 5000
      pilot_data_dir: {str(pilot_dir)}
      main_data_dir: {str(main_dir)}
    """)

    records = run_pilot(config=cfg, llm_client=fake_llm)

    assert len(records) > 0
    assert (pilot_dir / "dataset_drt.jsonl").exists()
    assert (pilot_dir / "run_manifest.json").exists()

    # Invariant: Nothing written under main_dir
    assert not main_dir.exists() or len(list(main_dir.rglob("*"))) == 0


def test_pilot_seed_reproducibility(tmp_path: Path, fake_llm):
    """Verify running pilot twice with identical seed_base yields byte-identical datasets."""
    p1 = tmp_path / "p1"
    p2 = tmp_path / "p2"

    cfg1 = AppConfig.from_yaml(f"""
    audit:
      strict_invariant: true
    mcts:
      max_simulations: 5
      max_rollout_depth: 2
      penalty_weight_w: 0.0
    experiment:
      r_pilot: 2
      seed_base: 8888
      pilot_data_dir: {str(p1)}
    """)
    cfg2 = AppConfig.from_yaml(f"""
    audit:
      strict_invariant: true
    mcts:
      max_simulations: 5
      max_rollout_depth: 2
      penalty_weight_w: 0.0
    experiment:
      r_pilot: 2
      seed_base: 8888
      pilot_data_dir: {str(p2)}
    """)

    records1 = run_pilot(config=cfg1, llm_client=fake_llm)
    records2 = run_pilot(config=cfg2, llm_client=fake_llm)

    assert len(records1) == len(records2)
    assert len(records1) > 0

    for rec1, rec2 in zip(records1, records2):
        assert rec1.identity.round_id == rec2.identity.round_id
        assert rec1.identity.turn_idx == rec2.identity.turn_idx
        assert rec1.stage1_freeze_log.locked_action_at == rec2.stage1_freeze_log.locked_action_at
        assert rec1.stage2_post_horizon.dispatched_command == rec2.stage2_post_horizon.dispatched_command
        assert rec1.stage2_post_horizon.y_t_A_penalty_draw == rec2.stage2_post_horizon.y_t_A_penalty_draw
        assert rec1.stage2_post_horizon.y_t_B_opp_win == rec2.stage2_post_horizon.y_t_B_opp_win
        assert rec1.stage2_post_horizon.stratum_horizon == rec2.stage2_post_horizon.stratum_horizon


def test_calibration_budget_and_compliance(tmp_path: Path):
    """Verify calibration calculates latencies, recommends budget meeting <2% target, and checks compliance."""
    pilot_dir = tmp_path / "pilot_cal"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 20 compliant records with 1.0s turn time
    for i in range(1, 21):
        records.append(make_synthetic_record_dict(round_id=1, turn_idx=i))

    file_path = pilot_dir / "dataset_drt.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = AppConfig.from_yaml(f"""
    experiment:
      pilot_data_dir: {str(pilot_dir)}
    budget:
      t_turn_s: 5.0
      delta_buffer_s: 0.5
    """)

    report = run_calibration(pilot_dir=pilot_dir, config=cfg)
    assert report["total_decisions"] == 20
    assert report["compliance_passed"] is True
    assert report["projected_fallback_rate"] < 0.02
    assert (pilot_dir / "calibration_report.json").exists()


def test_calibration_fails_on_sub_99_compliance(tmp_path: Path):
    """Verify calibration raises CalibrationFailure when JSON compliance is below 99%."""
    pilot_dir = tmp_path / "pilot_bad_json"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 10 records where multiple have invalid JSON flags
    for i in range(1, 11):
        rec = make_synthetic_record_dict(round_id=1, turn_idx=i)
        if i in (1, 2):
            rec["system_reliability"]["llm_call1_invalid_json"] = True
        records.append(rec)

    with open(pilot_dir / "dataset_drt.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cfg = AppConfig.from_yaml(f"""
    experiment:
      pilot_data_dir: {str(pilot_dir)}
    """)

    with pytest.raises(CalibrationFailure) as exc_info:
        run_calibration(pilot_dir=pilot_dir, config=cfg)
    assert "below the mandatory 99.0% threshold" in str(exc_info.value)


def test_sample_size_estimation_reuses_analysis_bootstrap(tmp_path: Path):
    """Verify estimate_sample_size returns R meeting target and invokes Change 8 bootstrap."""
    records = []
    for r in range(1, 6):
        for t in range(1, 4):
            records.append(make_synthetic_record_dict(round_id=r, turn_idx=t))

    file_path = tmp_path / "sample_size_pilot.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    cfg = AppConfig.from_yaml("""
    analysis:
      bootstrap_B: 50
      bootstrap_seed: 42
    experiment:
      ci_half_width_target: 0.025
    """)

    # Spy on round_level_cluster_bootstrap to verify reuse
    with patch("src.experiment.sample_size.round_level_cluster_bootstrap", wraps=round_level_cluster_bootstrap) as spy_boot:
        size_res = estimate_sample_size(file_path, config=cfg)
        assert spy_boot.called
        assert size_res["R_recommended"] >= 5
        assert size_res["projected_hw_bs"] <= 0.025
        assert size_res["projected_hw_ece"] <= 0.025
