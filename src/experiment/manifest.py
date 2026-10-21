"""Run manifest generation capturing full experimental provenance.

Why: Records exact seeds, config hash, config version, git commit, opponent intellect,
and model identifiers to ensure every pilot and main experimental run is fully auditable.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Optional

from src.config import AppConfig, compute_config_hash


def get_git_commit() -> str:
    """Attempts to retrieve current repository HEAD commit SHA.

    Why: Ties dataset provenance directly to version-controlled codebase revisions.
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unversioned_workspace"


def write_manifest(
    output_dir: str | Path,
    config: AppConfig,
    subcommand: str,
    start_time_iso: str,
    end_time_iso: str,
    git_commit: Optional[str] = None,
) -> dict[str, Any]:
    """Writes a run manifest capturing complete configuration and runtime provenance.

    Args:
        output_dir: Target data directory (e.g. data/pilot/ or data/main_experiment/).
        config: Resolved AppConfig model.
        subcommand: Executed runbook subcommand ('pilot', 'main', etc.).
        start_time_iso: ISO 8601 start timestamp.
        end_time_iso: ISO 8601 completion timestamp.
        git_commit: Optional git SHA override for test doubles.

    Returns:
        Dictionary representation of the written manifest.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    commit_sha = git_commit or get_git_commit()
    cfg_hash = compute_config_hash(config)

    manifest_data = {
        "subcommand": subcommand,
        "config_version": config.experiment.config_version,
        "config_hash": cfg_hash,
        "seed_base": config.experiment.seed_base,
        "opponent_intellect": config.opponent.intellect,
        "llm_model": config.llm.model,
        "git_commit": commit_sha,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
    }

    manifest_file = out_path / "run_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)

    return manifest_data
