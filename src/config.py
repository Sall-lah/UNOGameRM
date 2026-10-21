"""Minimal pydantic v2 configuration models for engine, opponent, and agent.

Defines the configuration schema required to bootstrap the headless UNO engine,
strictly validating opponent intellect and agent challenge responses while forbidding
unmodeled sections until subsequent changes introduce them.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import InitErrorDetails


class AnalysisConfig(BaseModel):
    """Configuration container for dataset evaluation and statistical analysis.

    Why: Exposes all tunable parameters used in RQ1 (calibration), RQ2 (memory/search
    alignment and rationale rubric), and RQ3 (residual and resource trajectory analysis).
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    bootstrap_B: int = Field(
        default=2000,
        ge=1,
        description="Number of round-level cluster bootstrap replications.",
    )
    bootstrap_seed: int = Field(
        default=42,
        description="Deterministic seed for the cluster bootstrap random number generator.",
    )
    ece_bins: int = Field(
        default=10,
        ge=2,
        description="Number of equal-width calibration bins on [0, 1] for ECE calculation.",
    )
    tau_risk: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for classifying adverse events as anticipated (TP) vs unanticipated (FN).",
    )
    audit_sample_size: int = Field(
        default=100,
        ge=1,
        description="Target number of decisions sampled for human rationale audit.",
    )
    audit_sample_seed: int = Field(
        default=42,
        description="Deterministic seed for selecting the audit sample.",
    )
    output_dir: str = Field(
        default="reports",
        description="Target directory for generated CSV and JSON evaluation reports.",
    )


class EngineConfig(BaseModel):
    """Engine runtime configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    seed: Optional[int] = None


class OpponentConfig(BaseModel):
    """Configuration for the built-in AIPlayer opponent."""

    model_config = ConfigDict(extra="forbid")

    intellect: int = Field(default=2, description="Fixed intellect tier in {1, 2, 3}")

    @field_validator("intellect")
    @classmethod
    def validate_intellect(cls, v: int) -> int:
        """Enforces intellect within official UNO.py supported range."""
        if v not in (1, 2, 3):
            raise ValueError(f"opponent.intellect must be in {{1, 2, 3}}, got {v}")
        return v


class AgentConfig(BaseModel):
    """Agent decision and response settings."""

    model_config = ConfigDict(extra="forbid")

    d4_response: Literal["accept", "challenge"] = Field(
        default="accept",
        description="Policy for reacting to opponent Wild Draw Four cards",
    )


class BarrierConfig(BaseModel):
    """Information barrier configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    refine_unseen_pool: bool = Field(
        default=False,
        description="Whether compute_unseen_pool subtracts known discards since reshuffle",
    )


class MemoryConfig(BaseModel):
    """PlugMem memory configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    draw_decay_gamma: float = Field(default=0.75, description="Confidence decay factor per drawn card")
    prune_threshold: float = Field(default=0.35, description="Prune/soft-deactivate threshold for decayed beliefs")
    top_k_semantic: int = Field(default=5, description="Max semantic proposition nodes retrieved")
    top_k_procedural: int = Field(default=2, description="Max procedural prescription nodes retrieved")
    embedder: str = Field(default="local_fallback", description="Embedding model or local fallback identifier")
    seed_procedural_priors: bool = Field(default=True, description="Whether to seed turn 1 procedural priors")
    enable_llm_refiner: bool = Field(default=False, description="Whether optional post-horizon LLM refiner runs")
    w_sim: float = Field(default=0.40, description="Embedding similarity weight in retrieval reranking")
    w_tag: float = Field(default=0.35, description="Concept tag overlap weight in retrieval reranking")
    w_conf: float = Field(default=0.15, description="Confidence weight in retrieval reranking")
    w_rec: float = Field(default=0.10, description="Recency exponential weight in retrieval reranking")
    recency_lambda: float = Field(default=0.15, description="Recency decay parameter")
    max_hops: int = Field(default=2, description="Max bipartite expansion hops")


class MCTSConfig(BaseModel):
    """MARS budget-aware imperfect-information MCTS configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    c_uct: float = Field(default=1.25, description="UCT exploration constant")
    penalty_weight_w: float = Field(default=-0.07, description="MARS latency penalty exponent w")
    target_sim_latency_s: float = Field(default=0.002, description="Target simulation latency L in seconds")
    max_rollout_depth: int = Field(default=6, description="Maximum rollout depth (half-turns)")
    max_simulations: int = Field(default=400, description="Maximum simulation count K_max per search")
    alpha1: float = Field(default=0.35, description="Hand differential weight in raw utility")
    alpha2: float = Field(default=0.15, description="Action/wild retention weight in raw utility")
    alpha3: float = Field(default=0.25, description="A_pen penalty weight in raw utility")


class LLMConfig(BaseModel):
    """LLM client and decision call configuration parameters."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(default="http://localhost:8000/v1", description="OpenAI-compatible server base URL")
    model: str = Field(default="gpt-4o-mini", description="Target LLM model name")
    api_key_env: str = Field(default="OPENAI_API_KEY", description="Environment variable holding API credential")
    temperature: float = Field(default=0.2, description="Generation sampling temperature")
    max_candidates: int = Field(default=4, description="Maximum actions retained after Call #1 pruning")
    call1_timeout_s: float = Field(default=2.0, description="Timeout budget for Call #1 prior in seconds")
    call2_timeout_s: float = Field(default=2.0, description="Timeout budget for Call #2 risk elicitation in seconds")


class AuditConfig(BaseModel):
    """Two-stage audit logging and dataset storage configuration."""

    model_config = ConfigDict(extra="forbid")

    output_root: str = Field(default="data", description="Audit logs root directory")
    dataset_split: Literal["pilot", "main"] = Field(
        default="pilot",
        description="Dataset partition subfolder (pilot or main)",
    )
    strict_invariant: bool = Field(
        default=True,
        description="Enforces strict freeze_timestamp < dispatch_timestamp <= eval_timestamp ordering",
    )


class BudgetConfig(BaseModel):
    """Turn time budget allocation parameters."""

    model_config = ConfigDict(extra="forbid")

    t_turn_s: float = Field(default=5.0, description="Total turn wall-clock budget in seconds")
    delta_buffer_s: float = Field(default=0.5, description="Safety buffer subtracted from turn budget")
    llm2_reserve_s: float = Field(default=1.0, description="Time reserved for LLM Call #2 risk elicitation")


class ExperimentConfig(BaseModel):
    """Experiment runbook execution parameters, dataset paths, and precision targets."""

    model_config = ConfigDict(extra="forbid")

    config_version: str = Field(
        default="v1.0-frozen",
        description="Configuration version tag ('v1.0-frozen' required for main run)",
    )
    r_pilot: int = Field(default=15, ge=1, description="Number of rounds in calibration pilot")
    seed_base: int = Field(default=10000, description="Base seed for round-level seed derivation")
    pilot_data_dir: str = Field(
        default="data/pilot/",
        description="Directory for pilot datasets and calibration reports",
    )
    main_data_dir: str = Field(
        default="data/main_experiment/",
        description="Directory for main experimental trial datasets",
    )
    ci_half_width_target: float = Field(
        default=0.025,
        gt=0.0,
        description="Target 95% CI half-width for main round sample-size projection",
    )


REQUIRED_PROGRAM_SECTIONS = (
    "engine",
    "opponent",
    "agent",
    "memory",
    "mcts",
    "llm",
    "budget",
    "audit",
    "analysis",
    "experiment",
)


class AppConfig(BaseModel):
    """Root configuration holding all ten program sections."""

    model_config = ConfigDict(extra="forbid")

    engine: EngineConfig = Field(default_factory=EngineConfig)
    opponent: OpponentConfig = Field(default_factory=OpponentConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    barrier: BarrierConfig = Field(default_factory=BarrierConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, path_or_content: str, strict: bool = False) -> AppConfig:
        """Loads and parses configuration from a YAML file path or YAML string.

        When strict=True, validates that all required program sections are present.
        """
        if os.path.exists(path_or_content):
            with open(path_or_content, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = yaml.safe_load(path_or_content) or {}

        if strict and isinstance(data, dict):
            missing_errors: list[InitErrorDetails] = []
            for sec in REQUIRED_PROGRAM_SECTIONS:
                if sec not in data:
                    missing_errors.append(
                        {
                            "type": "missing",
                            "loc": (sec,),
                            "msg": f"Missing required section: '{sec}'",
                            "input": data,
                        }
                    )
            if missing_errors:
                raise ValidationError.from_exception_data(
                    title="AppConfig",
                    line_errors=missing_errors,
                )

        return cls.model_validate(data)


def compute_config_hash(config: AppConfig) -> str:
    """Computes deterministic SHA-256 hex digest over canonical JSON of the configuration.

    Why: Guarantees verifiable provenance tracking across pilot and main experiments.
    """
    config_dict = config.model_dump(mode="json")
    canonical_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def is_frozen(config: AppConfig) -> bool:
    """Returns True iff the configuration is locked with version 'v1.0-frozen'.

    Why: Ensures the main experimental runner cannot proceed with draft or uncalibrated parameters.
    """
    return config.experiment.config_version == "v1.0-frozen"


def load_config(path: str = "config.yaml", strict: bool = False) -> AppConfig:
    """Loads application configuration, validating all sections when strict=True.

    Ensures headless engine components can instantiate with sensible defaults
    while enforcing strict validation when loading full experiment configurations.
    """
    if os.path.exists(path):
        return AppConfig.from_yaml(path, strict=strict)
    return AppConfig()


def load_experiment_config(path: str = "config.yaml") -> AppConfig:
    """Authoritative loader for experiment runbooks enforcing all 10 required sections."""
    return load_config(path, strict=True)
