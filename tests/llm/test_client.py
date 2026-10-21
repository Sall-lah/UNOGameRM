"""Tests for LLM clients, credential isolation, and timeout handling (Task 7.1).

Verifies environment variable resolution without config leakage, FakeLLMClient
offline determinism, and timeout error raising.
"""

from __future__ import annotations

import os
import pytest

from src.llm.client import (
    FakeLLMClient,
    LLMTimeoutError,
    OpenAICompatibleClient,
)


def test_http_client_reads_key_from_env_never_exposing_it(monkeypatch):
    """Verify OpenAI-compatible client reads key from env and redacts it from string repr."""
    test_key = "sk-secret-test-key-12345"
    monkeypatch.setenv("CUSTOM_LLM_KEY", test_key)

    client = OpenAICompatibleClient(
        base_url="http://test-server:8000/v1",
        model="test-model",
        api_key_env="CUSTOM_LLM_KEY",
        temperature=0.5,
    )

    # Key must NOT appear in repr or str
    client_repr = repr(client)
    assert test_key not in client_repr
    assert "CUSTOM_LLM_KEY" in client_repr
    assert "test-secret" not in str(client)


def test_fake_llm_client_returns_canned_output_without_network():
    """Verify FakeLLMClient operates offline and returns exact canned responses."""
    canned_1 = '{"result": 1}'
    canned_2 = '{"result": 2}'
    client = FakeLLMClient(responses=[canned_1, canned_2])

    resp1 = client.complete(system="sys", user="user1", timeout_s=1.0)
    resp2 = client.complete(system="sys", user="user2", timeout_s=1.0)

    assert resp1 == canned_1
    assert resp2 == canned_2
    assert len(client.call_history) == 2


def test_fake_llm_client_simulates_timeout():
    """Verify FakeLLMClient raises LLMTimeoutError when timeout simulation is enabled."""
    client = FakeLLMClient(simulate_timeout=True)

    with pytest.raises(LLMTimeoutError):
        client.complete(system="sys", user="user", timeout_s=0.5)
