"""Camada do modelo de linguagem."""

from app.config import Settings
from app.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnavailableError,
)


def build_provider(settings: Settings) -> LLMProvider:
    """Cria o provedor a partir das configurações. Sem chave, levanta LLMConfigurationError."""
    api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    if not api_key:
        raise LLMConfigurationError()

    from app.llm.gemini import GeminiProvider  # importa o SDK só quando precisa

    return GeminiProvider(
        api_key=api_key,
        model=settings.gemini_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
    )


__all__ = [
    "LLMConfigurationError",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "build_provider",
]
