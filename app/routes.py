"""Rotas da API: GET / (front de teste), GET /health e POST /chat."""

import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse

from app import __version__
from app.chat_service import ConversationLimitError, format_sse, stream_with_timeouts, validate_conversation
from app.errors import error_response
from app.llm import LLMConfigurationError, LLMError
from app.rag import add_context_to_prompt
from app.schemas import ChatRequest, HealthResponse

logger = logging.getLogger("app.chat")
router = APIRouter()

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # evita que o nginx segure o stream em buffer
}


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    return HealthResponse(version=__version__, provider=request.app.state.settings.llm_provider)


@router.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    """Recebe o histórico da conversa e devolve a resposta do assistente em SSE.

    Os eventos são:
        event: token  data: {"text": "..."}             (um pedaço da resposta)
        event: error  data: {"code": "...", "message": "..."}
        event: done   data: {}
    """
    state = request.app.state
    settings = state.settings

    client_ip = request.client.host if request.client else "unknown"
    if not state.rate_limiter.allow(client_ip):
        logger.info("rate_limited")
        return error_response(429, "rate_limited", "Muitas requisições. Aguarde um instante e tente novamente.")

    try:
        validate_conversation(payload.messages, settings)
    except ConversationLimitError as exc:
        return error_response(422, exc.code, exc.message)

    # Só depois de validar a entrada: erro do usuário (422) vem antes de erro de configuração (503).
    if state.provider is None:
        raise LLMConfigurationError()

    system_prompt, sources = await prompt_with_documents(request, payload.messages[-1].content)

    stream = stream_with_timeouts(
        state.provider.stream(system_prompt, payload.messages),
        first_token_timeout=settings.llm_first_token_timeout,
        idle_timeout=settings.llm_idle_timeout,
        total_timeout=settings.llm_total_timeout,
    )
    # Para os logs: só tamanhos, nunca o texto.
    log_info = {
        "turns": len(payload.messages),
        "input_chars": sum(len(m.content) for m in payload.messages),
        "provider": state.provider.name,
        "sources": sources,
    }
    started = time.perf_counter()

    # Esperamos o primeiro pedaço antes de responder. Se o provedor falhar logo
    # de cara (limite, chave inválida, demora), o cliente recebe um status HTTP
    # de erro de verdade, e não um 200 seguido de um evento de erro.
    try:
        first_chunk = await anext(stream, None)
    except LLMError as exc:
        await stream.aclose()
        logger.warning("chat_failed", extra={**log_info, "error_code": exc.code, "duration_ms": ms_since(started)})
        return error_response(exc.status_code, exc.code, exc.public_message)

    async def events():
        chunks = output_chars = 0
        outcome = "completed"
        try:
            if first_chunk is not None:
                chunks, output_chars = 1, len(first_chunk)
                yield format_sse("token", {"text": first_chunk})
            async for chunk in stream:
                chunks += 1
                output_chars += len(chunk)
                yield format_sse("token", {"text": chunk})
            yield format_sse("done", {})
        except LLMError as exc:
            # O 200 já foi enviado, então o erro vai como evento.
            outcome = exc.code
            yield format_sse("error", {"code": exc.code, "message": exc.public_message})
        except BaseException:
            # O cliente fechou a conexão (ou algo inesperado aconteceu).
            outcome = "aborted"
            raise
        finally:
            await stream.aclose()
            logger.info(
                "chat_finished",
                extra={
                    **log_info,
                    "outcome": outcome,
                    "chunks": chunks,
                    "output_chars": output_chars,
                    "duration_ms": ms_since(started),
                },
            )

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


async def prompt_with_documents(request: Request, question: str) -> tuple[str, list[str]]:
    """Busca os trechos mais relevantes para a pergunta e os junta ao prompt de sistema.

    Devolve o prompt final e a lista de trechos usados (só nomes, para o log).
    Se a busca falhar ou demorar, o assistente responde só com o prompt base.
    """
    state = request.app.state
    if state.retriever is None:
        return state.system_prompt, []
    try:
        results = await asyncio.wait_for(
            state.retriever.search(question, state.settings.rag_top_k),
            timeout=state.settings.rag_timeout,
        )
    except Exception as error:
        logger.warning("rag_search_failed", extra={"error_type": type(error).__name__})
        return state.system_prompt, []
    sources = [f"{r.chunk.source}#{r.chunk.section}" for r in results]
    return add_context_to_prompt(state.system_prompt, results), sources


def ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
