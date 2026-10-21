"""LLM client protocols, transport adapters, and deterministic test doubles.

This module provides the core LLMClient protocol along with an OpenAI-compatible
HTTP client and a deterministic FakeLLMClient. All sensitive API credentials are
resolved at runtime exclusively from environment variable names, ensuring zero secret
leakage into serialized configurations, string representations, or logs.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Protocol, Sequence, runtime_checkable
import httpx


class LLMTimeoutError(TimeoutError):
    """Raised when an LLM completion request exceeds its configured timeout budget."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the minimal text completion interface for decision calls.

    Why: Abstracting the LLM transport behind a single text completion method enables
    strict offline deterministic testing (FakeLLMClient) without mocking network sockets.
    """

    def complete(self, system: str, user: str, timeout_s: float) -> str:
        """Sends system and user prompts to the LLM backend and returns raw response text.

        Args:
            system: Instructions defining the agent persona and output constraints.
            user: Dynamic turn observation, PlugMem context, and task requirements.
            timeout_s: Maximum wall-clock seconds permitted before raising LLMTimeoutError.
        """
        ...


class OpenAICompatibleClient:
    """HTTP client targeting OpenAI-compatible /chat/completions endpoints.

    Why: Connects to local (vLLM, Ollama, llama.cpp) or remote inference providers
    adhering to OpenAI chat schema without pulling heavy third-party SDK dependencies.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.2,
    ) -> None:
        """Initializes client parameters without inspecting or caching secrets.

        Args:
            base_url: API endpoint root containing chat/completions.
            model: Model identifier passed to inference server.
            api_key_env: Name of the environment variable containing the bearer token.
            temperature: Sampling temperature for stochastic generation.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature

    def __repr__(self) -> str:
        """Safe string representation strictly redacting sensitive credentials."""
        return (
            f"OpenAICompatibleClient(base_url='{self.base_url}', model='{self.model}', "
            f"api_key_env='{self.api_key_env}', temperature={self.temperature})"
        )

    def complete(self, system: str, user: str, timeout_s: float) -> str:
        """Dispatches an HTTP POST request to the chat completions endpoint."""
        token = os.environ.get(self.api_key_env, "")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
        }

        url = f"{self.base_url}/chat/completions"

        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"LLM request timed out after {timeout_s}s on {self.model}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


class FakeLLMClient:
    """Deterministic test double providing queued, keyed, or default text completions.

    Why: Guarantees zero network I/O in CI and offline unit test environments while
    simulating realistic response variation and timeout triggers.
    """

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        responses_by_prompt: Optional[dict[str, str]] = None,
        default_response: str = "{}",
        simulate_timeout: bool = False,
    ) -> None:
        """Configures canned output sequences or simulated timeout behavior."""
        self._queue: list[str] = list(responses) if responses else []
        self._keyed: dict[str, str] = dict(responses_by_prompt) if responses_by_prompt else {}
        self.default_response = default_response
        self.simulate_timeout = simulate_timeout
        self.call_history: list[dict[str, Any]] = []

    def complete(self, system: str, user: str, timeout_s: float) -> str:
        """Returns canned text without performing network operations."""
        self.call_history.append({"system": system, "user": user, "timeout_s": timeout_s})

        if self.simulate_timeout:
            raise LLMTimeoutError(f"Simulated timeout triggered after {timeout_s}s")

        for key, canned in self._keyed.items():
            if key in user or key in system:
                return canned

        if self._queue:
            return self._queue.pop(0)

        return self.default_response
