"""Regras do chat que não dependem de HTTP: limites, timeouts e formato SSE."""

import asyncio
import json
from collections.abc import AsyncIterator

from app.config import Settings
from app.llm import LLMTimeoutError
from app.schemas import Message


class ConversationLimitError(ValueError):
    """A conversa é válida no formato, mas passou de algum limite."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_conversation(messages: list[Message], settings: Settings) -> None:
    """Confere os limites antes de chamar o modelo, para não gastar tokens à toa."""
    if len(messages) > settings.max_turns:
        raise ConversationLimitError(
            "too_many_turns", f"A conversa pode ter no máximo {settings.max_turns} mensagens."
        )

    if any(len(m.content) > settings.max_message_chars for m in messages):
        raise ConversationLimitError(
            "message_too_long", f"Cada mensagem pode ter no máximo {settings.max_message_chars} caracteres."
        )

    if sum(len(m.content) for m in messages) > settings.max_total_chars:
        raise ConversationLimitError(
            "conversation_too_long", f"A conversa passou do limite de {settings.max_total_chars} caracteres."
        )

    if messages[-1].role != "user":
        raise ConversationLimitError("last_message_not_user", "A última mensagem deve ser do usuário.")


async def stream_with_timeouts(
    chunks: AsyncIterator[str],
    first_token_timeout: float,
    idle_timeout: float,
    total_timeout: float,
) -> AsyncIterator[str]:
    """Repassa os pedaços da resposta, mas desiste se o modelo demorar demais.

    Se o 1º pedaço não chegar em `first_token_timeout`, se o modelo ficar
    calado por mais de `idle_timeout` no meio da resposta, ou se tudo passar
    de `total_timeout`, levanta LLMTimeoutError.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout
    wait = first_token_timeout

    try:
        while True:
            time_left = deadline - loop.time()
            if time_left <= 0:
                raise LLMTimeoutError()
            try:
                chunk = await asyncio.wait_for(anext(chunks), timeout=min(wait, time_left))
            except StopAsyncIteration:
                return
            except TimeoutError:
                raise LLMTimeoutError() from None
            yield chunk
            wait = idle_timeout  # depois do 1º pedaço, vale o timeout entre pedaços
    finally:
        # Fecha a conexão com o provedor mesmo se o cliente sair no meio.
        await chunks.aclose()


def format_sse(event: str, data: dict) -> str:
    """Monta um evento SSE. O JSON fica numa linha só, então "\\n" no texto não quebra o protocolo."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
