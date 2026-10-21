"""Pilot runner executing R_pilot exploratory rounds for calibration and sizing.

Why: Collects an initial clustered decision dataset under isolated pilot storage (data/pilot/)
to calibrate LLM turn budgets and estimate main experimental round sample sizes without
contaminating the main experiment partition.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.audit.dataset import DatasetWriter, DecisionRecord
from src.config import AppConfig, load_config
from src.experiment.manifest import write_manifest
from src.harness.round_runner import run_round
from src.llm.client import FakeLLMClient, LLMClient


def run_pilot(
    config: Optional[AppConfig] = None,
    llm_client: Optional[LLMClient] = None,
) -> list[DecisionRecord]:
    """Runs r_pilot complete rounds and writes records strictly to pilot_data_dir.

    Args:
        config: Experimental AppConfig; if None, loads from config.yaml.
        llm_client: Optional LLMClient test double (defaults to FakeLLMClient).

    Returns:
        List of all DecisionRecord instances produced during the pilot.
    """
    cfg = config or load_config()

    pilot_dir = Path(cfg.experiment.pilot_data_dir)
    main_dir = Path(cfg.experiment.main_data_dir)
    pilot_dir.mkdir(parents=True, exist_ok=True)

    # Invariant: Record counts or existing state in main directory must remain untouched
    main_files_before = set(main_dir.rglob("*")) if main_dir.exists() else set()

    start_time_iso = datetime.now(timezone.utc).isoformat()

    writer = DatasetWriter(target_dir=pilot_dir)
    client = llm_client or FakeLLMClient()

    r_pilot = cfg.experiment.r_pilot
    seed_base = cfg.experiment.seed_base

    all_pilot_records: list[DecisionRecord] = []

    print(f"=== Starting Pilot Run: {r_pilot} rounds (seed_base={seed_base}) ===")
    for r in range(1, r_pilot + 1):
        round_records = run_round(
            r=r,
            base_seed=seed_base,
            config=cfg,
            llm_client=client,
            dataset_writer=writer,
        )
        all_pilot_records.extend(round_records)

    end_time_iso = datetime.now(timezone.utc).isoformat()

    # Write run manifest into pilot directory
    write_manifest(
        output_dir=pilot_dir,
        config=cfg,
        subcommand="pilot",
        start_time_iso=start_time_iso,
        end_time_iso=end_time_iso,
    )

    # Assert nothing was written under main_data_dir
    main_files_after = set(main_dir.rglob("*")) if main_dir.exists() else set()
    new_main_files = main_files_after - main_files_before
    if len(new_main_files) > 0:
        raise AssertionError(
            f"Pilot run violated directory isolation! Files written to main_data_dir: {new_main_files}"
        )

    print(f"=== Pilot Run Completed: {len(all_pilot_records)} total decision records written to {pilot_dir} ===")
    return all_pilot_records
