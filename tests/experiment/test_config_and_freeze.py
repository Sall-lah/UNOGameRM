"""Tests for experiment configuration schema, hash derivation, and freeze gating (Tasks 8.1, 8.2, 8.3)."""

from __future__ import annotations

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.config import (
    AppConfig,
    compute_config_hash,
    is_frozen,
    load_config,
    load_experiment_config,
)
from src.experiment.main_run import FreezeViolationError, run_main_experiment


def test_valid_config_loads_all_sections():
    """Verify loading config.yaml populates all ten required program sections."""
    cfg = load_config("config.yaml", strict=True)

    assert cfg.engine is not None
    assert cfg.opponent.intellect == 2
    assert cfg.agent.d4_response == "accept"
    assert cfg.memory is not None
    assert cfg.mcts is not None
    assert cfg.llm is not None
    assert cfg.budget is not None
    assert cfg.audit is not None
    assert cfg.analysis is not None
    assert cfg.experiment.config_version == "v1.0-frozen"


def test_missing_required_section_rejected(tmp_path: Path):
    """Verify loading a config omitting a required section (e.g. budget) raises ValidationError."""
    bad_yaml = """
    engine:
      seed: null
    opponent:
      intellect: 2
    agent:
      d4_response: accept
    """
    bad_file = tmp_path / "missing_budget.yaml"
    bad_file.write_text(bad_yaml, encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_experiment_config(str(bad_file))

    err_str = str(exc_info.value)
    assert "budget" in err_str or "Missing required section" in err_str


def test_out_of_range_intellect_rejected():
    """Verify opponent.intellect = 4 is rejected with ValidationError."""
    with pytest.raises(ValidationError):
        AppConfig(opponent={"intellect": 4})  # type: ignore


def test_config_hash_determinism_and_sensitivity():
    """Verify identical configs produce identical hash, and changing a value alters the hash."""
    cfg1 = load_config("config.yaml")
    cfg2 = load_config("config.yaml")

    hash1 = compute_config_hash(cfg1)
    hash2 = compute_config_hash(cfg2)
    assert hash1 == hash2

    # Change a single value (e.g. opponent intellect)
    cfg_modified = AppConfig.from_yaml(f"""
    opponent:
      intellect: 3
    experiment:
      config_version: v1.0-frozen
    """)
    hash_modified = compute_config_hash(cfg_modified)
    assert hash1 != hash_modified


def test_freeze_flag_detection():
    """Verify is_frozen reports True only when config_version == 'v1.0-frozen'."""
    cfg_frozen = AppConfig.from_yaml("""
    experiment:
      config_version: v1.0-frozen
    """)
    assert is_frozen(cfg_frozen) is True

    cfg_draft = AppConfig.from_yaml("""
    experiment:
      config_version: draft
    """)
    assert is_frozen(cfg_draft) is False


def test_main_experiment_refuses_unfrozen_config(tmp_path: Path, fake_llm):
    """Verify main runner refuses to run with unfrozen config and writes nothing to main_data_dir."""
    main_dir = tmp_path / "main_experiment"
    cfg_unfrozen = AppConfig.from_yaml(f"""
    experiment:
      config_version: draft_unfrozen
      main_data_dir: {str(main_dir)}
    """)

    with pytest.raises(FreezeViolationError) as exc_info:
        run_main_experiment(rounds_to_run=1, config=cfg_unfrozen, llm_client=fake_llm)

    assert "MAIN EXPERIMENT REFUSED" in str(exc_info.value)
    assert not main_dir.exists() or len(list(main_dir.rglob("*"))) == 0
