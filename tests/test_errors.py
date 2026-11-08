"""Erros do provedor, timeouts e configuração ausente."""

from __future__ import annotations

import json

import pytest

from app.llm.base import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMUnavailableError,
)
from tests.conftest import FakeProvider, parse_sse, user

PAYLOAD = {"messages": [user("Oi")]}


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMRateLimitError(), 429, "rate_limited"),
        (LLMConfigurationError(), 503, "not_configured"),
        (LLMUnavailableError(), 502, "upstream_unavailable"),
    ],
)
def test_provider_failure_before_first_token_becomes_http_status(client_factory, error, status, code):
    provider = FakeProvider(error_before=error)
    response = client_factory(provider).post("/chat", json=PAYLOAD)
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert provider.closed


def test_failure_mid_stream_emits_error_event_after_partial_tokens(client_factory):
    provider = FakeProvider(chunks=["parte 1", "parte 2"], error_after=LLMUnavailableError())
    response = client_factory(provider).post("/chat", json=PAYLOAD)

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[:2] == [("token", {"text": "parte 1"}), ("token", {"text": "parte 2"})]
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "upstream_unavailable"
    assert ("done", {}) not in events


def test_first_token_timeout_returns_504(client_factory):
    provider = FakeProvider(delay_before_first=1.0)
    client = client_factory(provider, llm_first_token_timeout=0.05)
    response = client.post("/chat", json=PAYLOAD)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "timeout"
    assert provider.closed


def test_idle_timeout_mid_stream_emits_error_event(client_factory):
    provider = FakeProvider(chunks=["a", "b"], delay_between=1.0)
    client = client_factory(provider, llm_idle_timeout=0.05)
    events = parse_sse(client.post("/chat", json=PAYLOAD).text)
    assert events[0] == ("token", {"text": "a"})
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "timeout"


def test_total_timeout_caps_long_streams(client_factory):
    provider = FakeProvider(chunks=["a"] * 50, delay_between=0.02)
    client = client_factory(provider, llm_idle_timeout=5, llm_total_timeout=0.1)
    events = parse_sse(client.post("/chat", json=PAYLOAD).text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "timeout"
    assert len(events) < 50


def test_missing_api_key_starts_app_but_chat_returns_503(client_factory):
    client = client_factory(provider=None, gemini_api_key=None)
    assert client.get("/health").status_code == 200
    response = client.post("/chat", json=PAYLOAD)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "not_configured"


def test_invalid_input_wins_over_missing_configuration(client_factory):
    client = client_factory(provider=None, gemini_api_key=None)
    payload = {"messages": [{"role": "system", "content": "x"}]}
    assert client.post("/chat", json=payload).status_code == 422
    too_long = {"messages": [user("x" * 5000)]}
    assert client.post("/chat", json=too_long).json()["error"]["code"] == "message_too_long"


def test_logs_are_json_and_never_contain_message_text(client_factory, capsys):
    secret_text = "meu-cpf-e-123.456.789-00"
    provider = FakeProvider(chunks=["resposta-sensivel"])
    client = client_factory(provider)
    client.post("/chat", json={"messages": [user(secret_text)]}, headers={"X-Request-ID": "req-1"})

    output = capsys.readouterr().out
    assert secret_text not in output
    assert "resposta-sensivel" not in output
    assert "Nimbus Software" not in output  # o prompt de sistema também não vai para o log

    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    finished = [r for r in records if r["event"] == "chat_finished"]
    assert finished, output
    assert finished[0]["outcome"] == "completed"
    assert finished[0]["request_id"] == "req-1"
    assert finished[0]["input_chars"] == len(secret_text)
