"""LLM decision calls module for two-call action pruning and risk calibration.

Exports clients, prompt formatters, Pydantic schemas, and execution runners for
LLM Call #1 (prior/pruning) and Call #2 (conditional risk elicitation).
"""

from __future__ import annotations

from src.llm.call1_prior import (
    Call1Result,
    run_call1,
)
from src.llm.call2_risk import (
    Call2Result,
    derive_action_stratum,
    run_call2,
)
from src.llm.client import (
    FakeLLMClient,
    LLMClient,
    LLMTimeoutError,
    OpenAICompatibleClient,
)
from src.llm.prompts import (
    format_mcts_stats_summary,
    render_call1_prompts,
    render_call2_prompts,
)
from src.llm.schemas import (
    LLMCall1Output,
    LLMCall2Output,
)

__all__ = [
    "LLMClient",
    "OpenAICompatibleClient",
    "FakeLLMClient",
    "LLMTimeoutError",
    "LLMCall1Output",
    "LLMCall2Output",
    "Call1Result",
    "run_call1",
    "Call2Result",
    "run_call2",
    "derive_action_stratum",
    "render_call1_prompts",
    "render_call2_prompts",
    "format_mcts_stats_summary",
]
