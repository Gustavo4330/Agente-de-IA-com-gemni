"""Testes unitários das peças isoladas (sem HTTP e sem rede)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors

from app.chat_service import format_sse, stream_with_timeouts
from app.config import DEFAULT_PROMPT_PATH
from app.llm import build_provider
from app.llm.base import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.gemini import GeminiProvider, map_api_error, to_gemini_contents
from app.prompts import load_system_prompt
from app.rate_limit import RateLimiter
from app.schemas import Message
from tests.conftest import make_settings

# --- Configuração --------------------------------------------------------------


def test_allowed_origins_accepts_comma_separated_string(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.example, https://b.example")
    settings = make_settings()
    assert settings.allowed_origins == ["https://a.example", "https://b.example"]


def test_api_key_is_never_printed_in_settings_repr():
    settings = make_settings(gemini_api_key="chave-super-secreta")
    assert "chave-super-secreta" not in repr(settings)


def test_build_provider_requires_api_key():
    with pytest.raises(LLMConfigurationError):
        build_provider(make_settings(gemini_api_key=None))


def test_build_provider_creates_gemini_without_network():
    provider = build_provider(make_settings(gemini_api_key="fake-key-para-teste"))
    assert isinstance(provider, GeminiProvider)


def test_system_prompt_file_loads_and_is_fictional():
    text = load_system_prompt(DEFAULT_PROMPT_PATH)
    assert "Nimbus Software" in text


# --- SSE ----------------------------------------------------------------------


def test_format_sse_keeps_payload_on_single_data_line():
    frame = format_sse("token", {"text": "a\nb"})
    assert frame == 'event: token\ndata: {"text":"a\\nb"}\n\n'


# --- Timeouts -----------------------------------------------------------------


async def _gen(chunks, delay=0.0):
    for chunk in chunks:
        await asyncio.sleep(delay)
        yield chunk


async def _collect(iterator):
    return [chunk async for chunk in iterator]


def test_stream_with_timeouts_passes_chunks_through():
    result = asyncio.run(
        _collect(stream_with_timeouts(_gen(["a", "b"]), first_token_timeout=1, idle_timeout=1, total_timeout=1))
    )
    assert result == ["a", "b"]


def test_stream_with_timeouts_raises_on_slow_first_token():
    with pytest.raises(LLMTimeoutError):
        asyncio.run(
            _collect(
                stream_with_timeouts(_gen(["a"], delay=0.5), first_token_timeout=0.02, idle_timeout=1, total_timeout=1)
            )
        )


# --- Rate limiter -------------------------------------------------------------


def test_rate_limiter_resets_after_window():
    now = [0.0]
    limiter = RateLimiter(max_requests=2, window_seconds=10, clock=lambda: now[0])
    assert limiter.allow("ip") and limiter.allow("ip")
    assert not limiter.allow("ip")
    assert limiter.allow("outro-ip")
    now[0] = 10.0
    assert limiter.allow("ip")


def test_rate_limiter_disabled_with_zero():
    limiter = RateLimiter(max_requests=0, window_seconds=10)
    assert all(limiter.allow("ip") for _ in range(100))


# --- Gemini (sem rede) --------------------------------------------------------


def test_assistant_role_maps_to_gemini_model_role():
    contents = to_gemini_contents(
        [Message(role="user", content="oi"), Message(role="assistant", content="olá")]
    )
    assert [c.role for c in contents] == ["user", "model"]
    assert contents[1].parts[0].text == "olá"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, LLMRateLimitError),
        (401, LLMConfigurationError),
        (403, LLMConfigurationError),
        (404, LLMConfigurationError),
        (500, LLMUnavailableError),
        (503, LLMUnavailableError),
    ],
)
def test_map_api_error(status, expected):
    exc = errors.APIError(status, {"error": {"code": status, "message": "x", "status": "X"}})
    assert isinstance(map_api_error(exc), expected)


class _FakeAsyncModels:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.last_kwargs = None

    async def generate_content_stream(self, **kwargs):
        self.last_kwargs = kwargs
        return self._behaviour()


def _fake_client(behaviour):
    models = _FakeAsyncModels(behaviour)
    return SimpleNamespace(aio=SimpleNamespace(models=models)), models


def _provider(client):
    return GeminiProvider(api_key="unused", model="m", temperature=0.1, max_output_tokens=10, client=client)


def test_gemini_provider_streams_text_and_skips_empty_chunks():
    async def chunks():
        for text in ["Olá", None, "!"]:
            yield SimpleNamespace(text=text)

    client, models = _fake_client(chunks)
    messages = [Message(role="user", content="oi")]
    result = asyncio.run(_collect(_provider(client).stream("prompt", messages)))

    assert result == ["Olá", "!"]
    assert models.last_kwargs["config"].system_instruction == "prompt"
    assert models.last_kwargs["model"] == "m"


def test_gemini_provider_maps_sdk_errors():
    async def failing():
        raise errors.APIError(429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}})
        yield  # pragma: no cover

    client, _ = _fake_client(failing)
    with pytest.raises(LLMRateLimitError):
        asyncio.run(_collect(_provider(client).stream("p", [Message(role="user", content="oi")])))


def test_gemini_provider_maps_network_errors():
    async def failing():
        raise ConnectionError("rede caiu")
        yield  # pragma: no cover

    client, _ = _fake_client(failing)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(_collect(_provider(client).stream("p", [Message(role="user", content="oi")])))
