"""Tests for minimal engine configuration model."""

import pytest
from pydantic import ValidationError
from src.config import AppConfig, load_config


def test_config_defaults():
    """Verify default values: opponent intellect 2, agent d4_response accept."""
    cfg = AppConfig()
    assert cfg.opponent.intellect == 2
    assert cfg.agent.d4_response == "accept"


def test_config_reject_invalid_intellect():
    """Verify validation error when opponent intellect is outside {1, 2, 3}."""
    with pytest.raises(ValidationError):
        AppConfig(opponent={"intellect": 4})  # type: ignore

    with pytest.raises(ValidationError):
        AppConfig(opponent={"intellect": 0})  # type: ignore


def test_config_reject_unexpected_keys():
    """Verify extra sections outside engine/opponent/agent are rejected."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"extra_section": {}})


def test_config_from_yaml_string():
    """Verify YAML string loading."""
    yaml_content = """
    opponent:
      intellect: 3
    agent:
      d4_response: challenge
    """
    cfg = AppConfig.from_yaml(yaml_content)
    assert cfg.opponent.intellect == 3
    assert cfg.agent.d4_response == "challenge"


def test_mcts_config_defaults_and_override():
    """Verify MCTSConfig default hyper-parameters and custom YAML parsing."""
    cfg = AppConfig()
    assert cfg.mcts.c_uct == 1.25
    assert cfg.mcts.penalty_weight_w == -0.07
    assert cfg.mcts.target_sim_latency_s == 0.002
    assert cfg.mcts.max_rollout_depth == 6
    assert cfg.mcts.max_simulations == 400
    assert cfg.mcts.alpha1 == 0.35
    assert cfg.mcts.alpha2 == 0.15
    assert cfg.mcts.alpha3 == 0.25

    custom_yaml = """
    mcts:
      c_uct: 1.5
      max_simulations: 100
    """
    cfg_custom = AppConfig.from_yaml(custom_yaml)
    assert cfg_custom.mcts.c_uct == 1.5
    assert cfg_custom.mcts.max_simulations == 100


def test_llm_config_defaults_and_override():
    """Verify LLMConfig default settings and custom YAML parsing."""
    cfg = AppConfig()
    assert cfg.llm.base_url == "http://localhost:8000/v1"
    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.llm.api_key_env == "OPENAI_API_KEY"
    assert cfg.llm.temperature == 0.2
    assert cfg.llm.max_candidates == 4
    assert cfg.llm.call1_timeout_s == 2.0
    assert cfg.llm.call2_timeout_s == 2.0

    custom_yaml = """
    llm:
      model: custom-model
      temperature: 0.7
      max_candidates: 3
    """
    cfg_custom = AppConfig.from_yaml(custom_yaml)
    assert cfg_custom.llm.model == "custom-model"
    assert cfg_custom.llm.temperature == 0.7
    assert cfg_custom.llm.max_candidates == 3


def test_audit_config_defaults_and_override():
    """Verify AuditConfig default settings and custom YAML parsing."""
    cfg = AppConfig()
    assert cfg.audit.output_root == "data"
    assert cfg.audit.dataset_split == "pilot"
    assert cfg.audit.strict_invariant is True

    custom_yaml = """
    audit:
      output_root: custom_data
      dataset_split: main
      strict_invariant: false
    """
    cfg_custom = AppConfig.from_yaml(custom_yaml)
    assert cfg_custom.audit.output_root == "custom_data"
    assert cfg_custom.audit.dataset_split == "main"
    assert cfg_custom.audit.strict_invariant is False


def test_budget_config_defaults_and_override():
    """Verify BudgetConfig default settings and custom YAML parsing."""
    cfg = AppConfig()
    assert cfg.budget.t_turn_s == 5.0
    assert cfg.budget.delta_buffer_s == 0.5
    assert cfg.budget.llm2_reserve_s == 1.0

    custom_yaml = """
    budget:
      t_turn_s: 3.0
      delta_buffer_s: 0.2
      llm2_reserve_s: 0.8
    """
    cfg_custom = AppConfig.from_yaml(custom_yaml)
    assert cfg_custom.budget.t_turn_s == 3.0
    assert cfg_custom.budget.delta_buffer_s == 0.2
    assert cfg_custom.budget.llm2_reserve_s == 0.8




