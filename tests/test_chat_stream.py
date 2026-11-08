"""Rotas e streaming SSE de ponta a ponta, com o LLM falso."""

from __future__ import annotations

from tests.conftest import FakeProvider, assistant, parse_sse, user


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["x-request-id"]


def test_index_serves_test_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/chat" in response.text


def test_chat_streams_tokens_in_order_then_done(client, fake_provider):
    with client.stream("POST", "/chat", json={"messages": [user("Oi")]}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        body = response.read().decode()

    events = parse_sse(body)
    assert events == [
        ("token", {"text": "Olá"}),
        ("token", {"text": ", "}),
        ("token", {"text": "tudo bem?"}),
        ("done", {}),
    ]
    assert fake_provider.closed


def test_tokens_with_newlines_and_unicode_survive_sse_framing(client_factory):
    provider = FakeProvider(chunks=["linha 1\n\nlinha 2", " — ação ✓"])
    client = client_factory(provider)
    body = client.post("/chat", json={"messages": [user("Oi")]}).text
    tokens = [data["text"] for name, data in parse_sse(body) if name == "token"]
    assert "".join(tokens) == "linha 1\n\nlinha 2 — ação ✓"


def test_history_and_server_side_system_prompt_are_sent_to_provider(client, fake_provider):
    history = [user("Oi"), assistant("Olá! Como ajudo?"), user("Quais serviços?")]
    client.post("/chat", json={"messages": history})

    system_prompt, messages = fake_provider.calls[0]
    assert "Nimbus Software" in system_prompt
    assert [(m.role, m.content) for m in messages] == [(h["role"], h["content"]) for h in history]


def test_empty_model_response_still_finishes_with_done(client_factory):
    client = client_factory(FakeProvider(chunks=[]))
    events = parse_sse(client.post("/chat", json={"messages": [user("Oi")]}).text)
    assert events == [("done", {})]


def test_request_id_is_propagated_when_safe(client):
    response = client.post("/chat", json={"messages": [user("Oi")]}, headers={"X-Request-ID": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"


def test_unsafe_request_id_is_replaced(client):
    unsafe = "id with spaces <script>"
    response = client.get("/health", headers={"X-Request-ID": unsafe})
    generated = response.headers["x-request-id"]
    assert generated != unsafe
    assert len(generated) == 32 and generated.isalnum()
