"""Main experiment runner executing R frozen experimental trial rounds.

Why: Coordinates the locked main experimental protocol across R rounds with strictly
alternating starting players, per-round PlugMem reset, round-level resumability,
atomic record persistence, and manifest generation under data/main_experiment/.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Optional

from src.audit.dataset import DatasetWriter, DecisionRecord
from src.config import AppConfig, is_frozen, load_config
from src.experiment.manifest import write_manifest
from src.harness.round_runner import run_round
from src.llm.client import FakeLLMClient, LLMClient


class FreezeViolationError(RuntimeError):
    """Raised when an attempt is made to execute the main experiment with an unfrozen config."""


def get_completed_round_ids(main_dir: Path) -> set[int]:
    """Scans main experiment directory to determine already completed round IDs.

    Why: Enables crash-resilient resumption by skipping already verified, persisted rounds.
    """
    rounds_dir = main_dir / "rounds"
    completed = set()

    # Check atomic round files if present
    if rounds_dir.exists():
        for r_file in rounds_dir.glob("round_*.jsonl"):
            stem = r_file.stem  # e.g. "round_5"
            try:
                r_id = int(stem.split("_")[1])
                completed.add(r_id)
            except (IndexError, ValueError):
                continue

    # Also inspect consolidated dataset_drt.jsonl if present
    dataset_file = main_dir / "dataset_drt.jsonl"
    if dataset_file.exists():
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    try:
                        data = json.loads(stripped)
                        r_id = int(data.get("identity", {}).get("round_id", 0))
                        if r_id > 0:
                            completed.add(r_id)
                    except Exception:
                        continue

    return completed


def run_main_experiment(
    rounds_to_run: Optional[int] = None,
    config: Optional[AppConfig] = None,
    llm_client: Optional[LLMClient] = None,
) -> list[DecisionRecord]:
    """Executes the main experiment under frozen protocol with resume capability.

    Args:
        rounds_to_run: Total number of rounds R to execute; defaults to sample-size projection or 100.
        config: Resolved AppConfig; MUST satisfy is_frozen(config) == True.
        llm_client: Optional LLMClient test double.

    Returns:
        List of all DecisionRecord instances produced during the main experiment.
    """
    cfg = config or load_config()

    # Strict Freeze Gate: MUST refuse to run with unfrozen config
    if not is_frozen(cfg):
        msg = (
            f"MAIN EXPERIMENT REFUSED: Configuration version is '{cfg.experiment.config_version}', "
            f"expected 'v1.0-frozen'. Set experiment.config_version = 'v1.0-frozen' in config.yaml "
            f"after reviewing calibration before running the main experiment."
        )
        print(f"[ERROR] {msg}")
        raise FreezeViolationError(msg)

    main_dir = Path(cfg.experiment.main_data_dir)
    rounds_dir = main_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)

    start_time_iso = datetime.now(timezone.utc).isoformat()

    writer = DatasetWriter(target_dir=main_dir)
    client = llm_client or FakeLLMClient()

    total_r = rounds_to_run or 100
    seed_base = cfg.experiment.seed_base

    completed_rounds = get_completed_round_ids(main_dir)
    all_main_records: list[DecisionRecord] = []

    print(f"=== Starting Main Experiment: Target R={total_r} (seed_base={seed_base}) ===")
    print(f"Detected {len(completed_rounds)} already completed rounds. Resuming from next unfinished round.")

    for r in range(1, total_r + 1):
        if r in completed_rounds:
            print(f"Round {r}/{total_r} already completed. Skipping.")
            continue

        # Starting player alternates: even r -> AGENT, odd r -> CPU (doc 08 §3.2)
        starting_player = "AGENT" if r % 2 == 0 else "CPU"
        print(f"Executing Round {r}/{total_r} (starting_player={starting_player})...", flush=True)

        round_records = run_round(
            r=r,
            base_seed=seed_base,
            config=cfg,
            llm_client=client,
            dataset_writer=writer,
        )

        # Atomic per-round file write (temp file + rename)
        atomic_round_file = rounds_dir / f"round_{r}.jsonl"
        temp_round_file = rounds_dir / f"round_{r}.tmp"
        with open(temp_round_file, "w", encoding="utf-8") as f:
            for rec in round_records:
                f.write(rec.model_dump_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

        temp_round_file.replace(atomic_round_file)
        completed_rounds.add(r)
        all_main_records.extend(round_records)

    end_time_iso = datetime.now(timezone.utc).isoformat()

    # Write run manifest into main experiment directory
    write_manifest(
        output_dir=main_dir,
        config=cfg,
        subcommand="main",
        start_time_iso=start_time_iso,
        end_time_iso=end_time_iso,
    )

    print(f"=== Main Experiment Completed: {len(all_main_records)} new records collected across R={total_r} ===")
    return all_main_records
