"""Fixtures compartilhadas. Nenhum teste acessa a internet nem precisa de chave."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm.base import LLMError, LLMProvider
from app.main import create_app
from app.schemas import Message


@dataclass
class FakeProvider(LLMProvider):
    """Provedor roteirizado: devolve ``chunks`` e, opcionalmente, falha ou atrasa."""

    chunks: list[str] = field(default_factory=lambda: ["Olá", ", ", "tudo bem?"])
    error_before: LLMError | None = None
    error_after: LLMError | None = None  # levantado depois de todos os chunks
    delay_before_first: float = 0.0
    delay_between: float = 0.0
    calls: list[tuple[str, list[Message]]] = field(default_factory=list)
    closed: bool = False
    name: str = "fake"

    async def stream(self, system_prompt: str, messages: Sequence[Message]) -> AsyncIterator[str]:
        self.calls.append((system_prompt, list(messages)))
        try:
            if self.delay_before_first:
                await asyncio.sleep(self.delay_before_first)
            if self.error_before:
                raise self.error_before
            for index, chunk in enumerate(self.chunks):
                if index and self.delay_between:
                    await asyncio.sleep(self.delay_between)
                yield chunk
            if self.error_after:
                raise self.error_after
        finally:
            self.closed = True


def make_settings(**overrides: object) -> Settings:
    """Settings isoladas: ignora o .env e define valores explícitos para os testes."""
    base: dict[str, object] = {
        "gemini_api_key": None,
        "rate_limit_requests": 0,
        "log_level": "INFO",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def client_factory() -> Callable[..., TestClient]:
    clients: list[TestClient] = []

    def _make(provider: LLMProvider | None = None, retriever=None, **settings_overrides: object) -> TestClient:
        app = create_app(make_settings(**settings_overrides), provider=provider, retriever=retriever)
        client = TestClient(app)
        client.__enter__()  # executa o lifespan
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(client_factory, fake_provider) -> TestClient:
    return client_factory(fake_provider)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """Converte o corpo SSE em uma lista de (evento, dados)."""
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        name, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data += line[len("data:"):].strip()
        events.append((name, json.loads(data) if data else {}))
    return events


def user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def assistant(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}
