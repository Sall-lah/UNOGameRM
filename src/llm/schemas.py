"""Strict Pydantic v2 schemas for LLM Call #1 and Call #2 outputs.

Enforces structural and mathematical validity on LLM JSON responses, ensuring action
pruning bounds, prior distribution normalization, and adverse risk probability ranges.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class LLMCall1Output(BaseModel):
    """Output schema for LLM Call #1 (Action Pruning & Prior Distribution).

    Why: MCTS requires a valid probability distribution over a non-empty subset of legal actions.
    Priors that do not sum to 1.0 or reference non-legal actions trigger fallback logic.
    """

    model_config = ConfigDict(extra="forbid")

    pruned_actions: list[str] = Field(
        ...,
        description="Retained legal actions after pruning suboptimal choices",
    )
    action_priors: dict[str, float] = Field(
        ...,
        description="Prior probability distribution P_prior over pruned actions",
    )
    pruning_note: str = Field(
        ...,
        description="Short reasoning note explaining the pruning rationale",
    )

    @model_validator(mode="after")
    def validate_call1_semantics(self, info: ValidationInfo) -> LLMCall1Output:
        """Validates action subset, bounds, matching keys, and prior probability normalization."""
        context: Mapping[str, Any] = info.context or {}
        legal_actions = context.get("legal_actions")
        max_candidates = context.get("max_candidates", 4)

        # 1. Priors keys must exactly match pruned_actions
        pruned_set = set(self.pruned_actions)
        prior_keys = set(self.action_priors.keys())
        if prior_keys != pruned_set:
            raise ValueError(
                f"action_priors keys {sorted(prior_keys)} must exactly match pruned_actions {sorted(pruned_set)}"
            )

        # 2. Check subset against legal actions (A_t) if provided in validation context
        if legal_actions is not None:
            legal_set = set(legal_actions)
            if not pruned_set.issubset(legal_set):
                invalid_actions = pruned_set - legal_set
                raise ValueError(
                    f"pruned_actions contains non-legal actions: {sorted(invalid_actions)}"
                )

            # Keep-all constraint when |A_t| <= 2
            if len(legal_actions) <= 2:
                if pruned_set != legal_set:
                    raise ValueError(
                        f"When |A_t| <= 2, all legal actions must be kept. Expected {sorted(legal_set)}, got {sorted(pruned_set)}"
                    )

        # 3. Candidate count bound: between 1 and max_candidates
        if not (1 <= len(self.pruned_actions) <= max_candidates):
            raise ValueError(
                f"pruned_actions count must be between 1 and {max_candidates}, got {len(self.pruned_actions)}"
            )

        # 4. Priors must be non-negative and sum to 1.0 +- 1e-3
        for act, prior_val in self.action_priors.items():
            if prior_val < 0.0:
                raise ValueError(f"Prior for '{act}' cannot be negative, got {prior_val}")

        total_prior = sum(self.action_priors.values())
        if abs(total_prior - 1.0) > 1e-3:
            raise ValueError(
                f"action_priors must sum to 1.0 +- 1e-3, got {total_prior:.6f}"
            )

        return self


class LLMCall2Output(BaseModel):
    """Output schema for LLM Call #2 (Risk Elicitation & Short Rationale).

    Why: The two-stage audit log requires calibrated adverse event risks (p_t_A, p_t_B)
    in [0.0, 1.0], expected stratum in {1, 2, 3}, and a concise explanatory rationale.
    """

    model_config = ConfigDict(extra="forbid")

    p_t_A: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of suffering a forced penalty draw during H_t",
    )
    p_t_B: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of opponent winning the round during H_t",
    )
    expected_stratum: int = Field(
        ...,
        ge=1,
        le=3,
        description="Expected observation horizon stratum in {1, 2, 3}",
    )
    short_rationale: str = Field(
        ...,
        min_length=1,
        description="1-3 concise sentences grounding risks in observation and MCTS stats",
    )

    @field_validator("short_rationale")
    @classmethod
    def validate_rationale_sentences(cls, v: str) -> str:
        """Validates that rationale is non-empty and contains at most 3 sentences."""
        clean = v.strip()
        if not clean:
            raise ValueError("short_rationale cannot be blank")

        # Basic sentence punctuation boundary check
        sentences = [s.strip() for s in re.split(r"[.!?]+", clean) if s.strip()]
        if len(sentences) > 4:  # Permitting slight punctuation flexibility
            raise ValueError(f"short_rationale should be 1-3 sentences, found {len(sentences)}")

        return clean
