"""LLM Call #2 execution: conditional adverse-event risk elicitation and rationale.

Invoked strictly AFTER action commitment via Most-Visited Action Commitment. Estimates
conditional probabilities (p_{t,A}, p_{t,B}) and expected stratum given that the locked
action is about to be dispatched, with deterministic timeout and parse fallback handling.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from src.core.cards import (
    DRAW_TWO,
    REVERSE,
    SKIP,
    WILD_DRAW_FOUR,
    UNOAction,
    parse_card_string,
)
from src.env.barrier import LegalObservation
from src.llm.client import LLMClient
from src.llm.prompts import render_call2_prompts
from src.llm.schemas import LLMCall2Output


FALLBACK_RATIONALE: str = (
    "Deterministic fallback triggered due to timeout or response parsing failure."
)


@dataclass(frozen=True, slots=True)
class Call2Result:
    """Output container for LLM Call #2 risk elicitation and rationale.

    Why: Provides the Stage 1 freeze log with conditional adverse probabilities (p_t_A, p_t_B)
    and expected stratum to calibrate risk prediction before execution.
    """

    p_t_A: float
    p_t_B: float
    expected_stratum: int
    short_rationale: str
    fallback_triggered_call2: bool
    latency_ms: float


def derive_action_stratum(locked_action_id: str, obs: LegalObservation) -> int:
    """Derives expected stratum (1 or 2) deterministically from locked action semantics.

    Why: In 2-player UNO, playing a turn-retention card (Skip, Reverse, +2, Wild +4)
    or emptying the hand gives the opponent 0 active turns (Stratum 1). All other
    actions pass control to the opponent (Stratum 2).
    """
    if len(obs.agent_hand) == 1:
        # Playing the agent's final card immediately wins and closes the horizon
        return 1

    if locked_action_id == "draw":
        return 2

    # Check for turn-retention action card
    if (
        "skip" in locked_action_id
        or "reverse" in locked_action_id
        or "+ 2" in locked_action_id
        or "+ 4" in locked_action_id
    ):
        return 1

    return 2


def run_call2(
    locked_action: Union[UNOAction, str],
    obs: LegalObservation,
    distilled_context: str,
    mcts_stats: Any,
    client: LLMClient,
    timeout_s: float = 2.0,
) -> Call2Result:
    """Executes LLM Call #2 risk elicitation, enforcing locked action prerequisite.

    Why: API-level assertion that locked_action is non-null guarantees the pipeline ordering
    invariant (Commitment -> Call #2 -> Dispatch) cannot be bypassed.
    """
    if locked_action is None:
        raise ValueError("locked_action must not be None; action must be committed before Call #2")

    locked_action_id = (
        locked_action.action_id if hasattr(locked_action, "action_id") else str(locked_action)
    )

    start_time = time.perf_counter()

    # Render prompt and invoke LLM client
    system_prompt, user_prompt = render_call2_prompts(
        obs=obs,
        distilled_context=distilled_context,
        locked_action=locked_action_id,
        mcts_stats=mcts_stats,
    )

    try:
        raw_response = client.complete(
            system=system_prompt,
            user=user_prompt,
            timeout_s=timeout_s,
        )

        data = _extract_json_object(raw_response)
        parsed = LLMCall2Output.model_validate(data)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return Call2Result(
            p_t_A=parsed.p_t_A,
            p_t_B=parsed.p_t_B,
            expected_stratum=parsed.expected_stratum,
            short_rationale=parsed.short_rationale,
            fallback_triggered_call2=False,
            latency_ms=elapsed_ms,
        )

    except Exception:
        # Timeout, network error, or schema validation failure: deterministic fallback
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        fallback_stratum = derive_action_stratum(locked_action_id, obs)

        return Call2Result(
            p_t_A=0.5,
            p_t_B=0.5,
            expected_stratum=fallback_stratum,
            short_rationale=FALLBACK_RATIONALE,
            fallback_triggered_call2=True,
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
