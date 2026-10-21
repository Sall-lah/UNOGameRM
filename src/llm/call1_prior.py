"""LLM Call #1 execution: action pruning, prior distribution, and fallback ladder.

Executes prompt rendering, LLM client invocation, Pydantic schema validation, and
graceful fallback (positive weight renormalization or uniform distribution) with
system reliability flag logging and latency measurement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from src.env.barrier import LegalObservation
from src.llm.client import LLMClient
from src.llm.prompts import render_call1_prompts
from src.llm.schemas import LLMCall1Output


@dataclass(frozen=True, slots=True)
class Call1Result:
    """Output container for LLM Call #1 prior distribution and diagnostics.

    Why: Supplies MARS MCTS with pruned root actions and prior probabilities alongside
    latency and reliability audit flags.
    """

    pruned_actions: list[str]
    action_priors: dict[str, float]
    pruning_note: str
    invalid_json_call1: bool
    latency_ms: float


def run_call1(
    obs: LegalObservation,
    distilled_context: str,
    client: LLMClient,
    max_candidates: int = 4,
    timeout_s: float = 2.0,
) -> Call1Result:
    """Executes LLM Call #1 with schema validation and multi-tiered fallback ladder.

    Why: MCTS requires a strictly valid probability distribution over legal actions.
    If the model generates invalid JSON, unnormalized weights, or hallucinated actions,
    the fallback ladder recovers a legal prior without crashing the decision loop.
    """
    start_time = time.perf_counter()
    legal_actions = list(obs.legal_actions)

    # Fast-path: When |A_t| == 0 (emergency failsafe)
    if not legal_actions:
        return Call1Result(
            pruned_actions=["draw"],
            action_priors={"draw": 1.0},
            pruning_note="Failsafe: No legal actions present in observation.",
            invalid_json_call1=True,
            latency_ms=0.0,
        )

    # Fast-path: When |A_t| == 1
    if len(legal_actions) == 1:
        act = legal_actions[0]
        return Call1Result(
            pruned_actions=[act],
            action_priors={act: 1.0},
            pruning_note="Single legal action available; pruning bypassed.",
            invalid_json_call1=False,
            latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    # Render prompt and invoke LLM client
    system_prompt, user_prompt = render_call1_prompts(
        obs=obs,
        distilled_context=distilled_context,
        max_candidates=max_candidates,
    )

    raw_response = ""
    try:
        raw_response = client.complete(
            system=system_prompt,
            user=user_prompt,
            timeout_s=timeout_s,
        )
    except Exception:
        # Network or timeout failure triggers uniform fallback
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return _build_uniform_fallback(
            legal_actions=legal_actions,
            note="Fallback: LLM client call raised an exception or timed out.",
            latency_ms=elapsed_ms,
        )

    # Attempt JSON extraction and schema validation
    try:
        data = _extract_json_object(raw_response)
        parsed = LLMCall1Output.model_validate(
            data,
            context={"legal_actions": legal_actions, "max_candidates": max_candidates},
        )

        pruned = list(parsed.pruned_actions)
        priors = dict(parsed.action_priors)

        # Enforce rule: if |A_t| <= 2, keep all legal actions
        if len(legal_actions) <= 2 and set(pruned) != set(legal_actions):
            pruned = list(legal_actions)
            uniform_p = 1.0 / len(legal_actions)
            priors = {a: uniform_p for a in legal_actions}
            return Call1Result(
                pruned_actions=pruned,
                action_priors=priors,
                pruning_note="Forced keep-all: |A_t| <= 2 rule applied.",
                invalid_json_call1=True,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return Call1Result(
            pruned_actions=pruned,
            action_priors=priors,
            pruning_note=parsed.pruning_note,
            invalid_json_call1=False,
            latency_ms=elapsed_ms,
        )

    except Exception:
        # Validation or JSON parsing failed: attempt weight renormalization fallback
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        renorm_result = _attempt_renormalization_fallback(
            raw_response=raw_response,
            legal_actions=legal_actions,
            max_candidates=max_candidates,
            latency_ms=elapsed_ms,
        )
        if renorm_result is not None:
            return renorm_result

        # Final fallback: Uniform prior across all legal actions
        return _build_uniform_fallback(
            legal_actions=legal_actions,
            note="Fallback: Response JSON invalid or unrecoverable.",
            latency_ms=elapsed_ms,
        )


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extracts and parses a JSON object from raw LLM text output."""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()

    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        clean = clean[first_brace : last_brace + 1]

    return json.loads(clean)


def _attempt_renormalization_fallback(
    raw_response: str,
    legal_actions: list[str],
    max_candidates: int,
    latency_ms: float,
) -> Optional[Call1Result]:
    """Attempts to recover strictly positive priors that failed exact 1.0 sum validation."""
    try:
        data = _extract_json_object(raw_response)
        if not isinstance(data, dict):
            return None

        raw_pruned = data.get("pruned_actions")
        raw_priors = data.get("action_priors")
        note = str(data.get("pruning_note", "Renormalized unnormalized prior weights."))

        if not isinstance(raw_pruned, list) or not isinstance(raw_priors, dict):
            return None

        legal_set = set(legal_actions)
        valid_actions = [a for a in raw_pruned if a in legal_set]

        # Handle |A_t| <= 2 constraint
        if len(legal_actions) <= 2:
            valid_actions = list(legal_actions)

        if not valid_actions:
            return None

        # Check that weights for valid actions are strictly positive
        weights: dict[str, float] = {}
        for a in valid_actions:
            val = raw_priors.get(a)
            if val is None or float(val) <= 0.0:
                return None
            weights[a] = float(val)

        total_weight = sum(weights.values())
        if total_weight <= 0.0:
            return None

        # Renormalize weights to sum exactly to 1.0
        normalized = {a: w / total_weight for a, w in weights.items()}
        return Call1Result(
            pruned_actions=list(valid_actions[:max_candidates]),
            action_priors={a: normalized[a] for a in valid_actions[:max_candidates]},
            pruning_note=note,
            invalid_json_call1=True,
            latency_ms=latency_ms,
        )
    except Exception:
        return None


def _build_uniform_fallback(
    legal_actions: list[str],
    note: str,
    latency_ms: float,
) -> Call1Result:
    """Builds a uniform prior distribution across all legal actions."""
    uniform_p = 1.0 / len(legal_actions)
    return Call1Result(
        pruned_actions=list(legal_actions),
        action_priors={a: uniform_p for a in legal_actions},
        pruning_note=note,
        invalid_json_call1=True,
        latency_ms=latency_ms,
    )
