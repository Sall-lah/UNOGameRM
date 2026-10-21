"""Tests for main runner, manifest provenance, and CLI entry point (Tasks 8.8, 8.9, 8.10)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch
import pytest

from src.config import AppConfig
from src.experiment.main_run import get_completed_round_ids, run_main_experiment
from src.experiment.manifest import write_manifest
from src.experiment.run import main as cli_main


def test_main_runner_alternates_starting_player_and_resumes(tmp_path: Path, fake_llm):
    """Verify starting player alternates by round parity (even=AGENT, odd=CPU) and resume skips finished rounds."""
    main_dir = tmp_path / "main_run"

    cfg = AppConfig.from_yaml(f"""
    audit:
      strict_invariant: true
    mcts:
      max_simulations: 5
      max_rollout_depth: 2
    experiment:
      config_version: v1.0-frozen
      seed_base: 4242
      main_data_dir: {str(main_dir)}
    """)

    # Part 1: Run rounds 1 and 2
    records_pass1 = run_main_experiment(rounds_to_run=2, config=cfg, llm_client=fake_llm)
    assert len(records_pass1) > 0

    # Read records from round files
    rounds_dir = main_dir / "rounds"
    r1_records = [json.loads(line) for line in (rounds_dir / "round_1.jsonl").read_text().splitlines() if line.strip()]
    r2_records = [json.loads(line) for line in (rounds_dir / "round_2.jsonl").read_text().splitlines() if line.strip()]

    # Round 1 is odd -> starting_player CPU
    assert r1_records[0]["identity"]["starting_player"] == "CPU"
    # Round 2 is even -> starting_player AGENT
    assert r2_records[0]["identity"]["starting_player"] == "AGENT"

    # Part 2: Resume with rounds_to_run=3 (should skip 1 & 2, and only run round 3)
    completed = get_completed_round_ids(main_dir)
    assert 1 in completed and 2 in completed

    records_pass2 = run_main_experiment(rounds_to_run=3, config=cfg, llm_client=fake_llm)

    # Only round 3 was run in pass 2
    assert (rounds_dir / "round_3.jsonl").exists()
    assert all(r.identity.round_id == 3 for r in records_pass2)


def test_run_manifest_provenance_fields(tmp_path: Path):
    """Verify manifest records all required provenance fields."""
    cfg = AppConfig.from_yaml("""
    experiment:
      config_version: v1.0-frozen
      seed_base: 7777
    opponent:
      intellect: 2
    llm:
      model: test-llm-model
    """)

    manifest = write_manifest(
        output_dir=tmp_path,
        config=cfg,
        subcommand="test_cmd",
        start_time_iso="2026-10-21T00:00:00Z",
        end_time_iso="2026-10-21T00:01:00Z",
        git_commit="abcdef1234567890",
    )

    manifest_file = tmp_path / "run_manifest.json"
    assert manifest_file.exists()

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["subcommand"] == "test_cmd"
    assert data["config_version"] == "v1.0-frozen"
    assert data["seed_base"] == 7777
    assert data["opponent_intellect"] == 2
    assert data["llm_model"] == "test-llm-model"
    assert data["git_commit"] == "abcdef1234567890"
    assert "config_hash" in data
    assert "start_time" in data
    assert "end_time" in data


def test_cli_subcommands_dispatch(monkeypatch):
    """Verify each known subcommand dispatches to its respective handler."""
    calls = []

    def mock_validate(config=None):
        calls.append("validate")
        return True

    def mock_pilot(config=None):
        calls.append("pilot")
        return []

    def mock_calibrate(config=None):
        calls.append("calibrate")
        return {}

    def mock_main(rounds_to_run=None, config=None):
        calls.append("main")
        return []

    monkeypatch.setattr("src.experiment.run.run_all_validation_checks", mock_validate)
    monkeypatch.setattr("src.experiment.run.run_pilot", mock_pilot)
    monkeypatch.setattr("src.experiment.run.run_calibration", mock_calibrate)
    monkeypatch.setattr("src.experiment.run.run_main_experiment", mock_main)

    for subcmd in ("validate", "pilot", "calibrate", "main"):
        cli_main([subcmd, "--config", "config.yaml"])
        assert calls[-1] == subcmd


def test_cli_unknown_subcommand_rejected():
    """Verify unknown subcommand exits non-zero."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["frobnicate"])
    assert exc_info.value.code != 0
