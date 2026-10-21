"""CLI entry point for the UNO research experiment runbook.

Why: Coordinates the four experimental phases (validate -> pilot -> calibrate -> main)
via a single unified command-line interface: python -m src.experiment.run <subcommand>.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

from src.config import AppConfig, load_config, load_experiment_config
from src.experiment.calibrate import run_calibration
from src.experiment.main_run import run_main_experiment
from src.experiment.pilot import run_pilot
from src.experiment.sample_size import estimate_sample_size
from src.experiment.validate import run_all_validation_checks

VALID_SUBCOMMANDS = ("validate", "pilot", "calibrate", "main")


def print_usage_and_exit(message: Optional[str] = None) -> None:
    """Prints usage help and terminates process with exit code 1."""
    if message:
        print(f"Error: {message}", file=sys.stderr)
    print(
        "Usage: python -m src.experiment.run <subcommand> [--config <path>]\n"
        "Valid subcommands:\n"
        "  validate   Run Phase-1 pre-pilot verification checks (barrier, non-stacking, auto-UNO)\n"
        "  pilot      Run R_pilot exploratory rounds and write records to data/pilot/\n"
        "  calibrate  Process pilot records to recommend turn budgets and verify JSON compliance\n"
        "  main       Execute frozen main experiment across R rounds with resume capability",
        file=sys.stderr,
    )
    sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    """Main CLI entry point executing the requested runbook subcommand."""
    args_list = argv if argv is not None else sys.argv[1:]

    if not args_list:
        print_usage_and_exit("No subcommand provided.")

    subcommand = args_list[0].lower()

    if subcommand not in VALID_SUBCOMMANDS:
        print_usage_and_exit(f"Unknown subcommand '{subcommand}'.")

    parser = argparse.ArgumentParser(description=f"Runbook subcommand: {subcommand}")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Optional override for number of rounds to run (main only).",
    )

    parsed_args, _ = parser.parse_known_args(args_list[1:])

    # Load and validate configuration
    try:
        cfg: AppConfig = load_experiment_config(parsed_args.config)
    except Exception as exc:
        print(f"Configuration validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        if subcommand == "validate":
            run_all_validation_checks(config=cfg)

        elif subcommand == "pilot":
            run_pilot(config=cfg)

        elif subcommand == "calibrate":
            cal_report = run_calibration(config=cfg)
            # Also run sample-size estimation if pilot records exist
            pilot_dataset = Path(cfg.experiment.pilot_data_dir) / "dataset_drt.jsonl"
            if pilot_dataset.exists():
                size_res = estimate_sample_size(pilot_dataset, config=cfg)
                print(f"Projected Main Sample Size: R={size_res['R_recommended']} rounds")

        elif subcommand == "main":
            run_main_experiment(rounds_to_run=parsed_args.rounds, config=cfg)

    except Exception as err:
        print(f"Execution failed on subcommand '{subcommand}': {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
