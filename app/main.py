"""Monta a aplicação FastAPI.

Para rodar:  uvicorn app.main:get_app --factory
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.errors import register_error_handlers
from app.llm import LLMConfigurationError, LLMProvider, build_provider
from app.logging_config import configure_logging
from app.middleware import add_request_context
from app.prompts import load_system_prompt
from app.rag import GeminiEmbedder, Retriever, load_chunks
from app.rate_limit import RateLimiter
from app.routes import router

logger = logging.getLogger("app")


async def build_retriever(settings: Settings) -> Retriever | None:
    """Monta o índice dos documentos. Se algo falhar, o chat segue funcionando sem RAG."""
    if not settings.rag_enabled or not settings.gemini_api_key:
        return None
    try:
        embedder = GeminiEmbedder(settings.gemini_api_key.get_secret_value(), settings.embedding_model)
        retriever = Retriever(embedder, load_chunks(settings.rag_docs_path))
        await retriever.build_index()
    except Exception as error:
        logger.warning("rag_unavailable", extra={"error_type": type(error).__name__})
        return None
    logger.info("rag_ready", extra={"chunks": len(retriever.chunks)})
    return retriever


def create_app(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """Cria a aplicação. Nos testes, passamos configurações próprias e versões falsas do provedor e da busca."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.provider is None:
            try:
                app.state.provider = build_provider(settings)
            except LLMConfigurationError:
                # Sem chave a API sobe mesmo assim: /health funciona e /chat responde 503.
                logger.warning("llm_not_configured")
        if app.state.retriever is None:
            app.state.retriever = await build_retriever(settings)
        logger.info("startup", extra={"version": __version__, "provider": settings.llm_provider})
        yield

    app = FastAPI(title="Nimbus Assistant API", version=__version__, lifespan=lifespan)

    app.state.settings = settings
    app.state.provider = provider
    app.state.retriever = retriever
    # Lido na inicialização: se o arquivo faltar, a API nem sobe.
    app.state.system_prompt = load_system_prompt(settings.system_prompt_path)
    app.state.rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    add_request_context(app, max_body_bytes=settings.max_body_bytes)
    register_error_handlers(app)
    app.include_router(router)
    return app


def get_app() -> FastAPI:
    return create_app()
