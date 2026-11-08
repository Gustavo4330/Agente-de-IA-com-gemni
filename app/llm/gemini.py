"""Provedor Gemini, usando o SDK oficial `google-genai`."""

import logging
from collections.abc import AsyncIterator

from google import genai
from google.genai import errors, types

from app.llm.base import (
    LLMConfigurationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.schemas import Message

logger = logging.getLogger(__name__)


def to_gemini_contents(messages: list[Message]) -> list[types.Content]:
    """Converte o histórico para o formato do Gemini, que chama o assistente de "model"."""
    return [
        types.Content(
            role="model" if message.role == "assistant" else "user",
            parts=[types.Part(text=message.content)],
        )
        for message in messages
    ]


def map_api_error(error: errors.APIError) -> LLMError:
    """Traduz o status HTTP do Gemini para um erro nosso."""
    if error.code == 429:
        return LLMRateLimitError()
    if error.code in (400, 401, 403, 404):  # chave inválida, sem permissão, modelo inexistente
        return LLMConfigurationError()
    return LLMUnavailableError()


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float, max_output_tokens: int, client=None):
        # `client` existe para os testes poderem passar um cliente falso.
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)

    async def stream(self, system_prompt: str, messages: list[Message]) -> AsyncIterator[str]:
        config = self.config.model_copy(update={"system_instruction": system_prompt})
        try:
            response = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=to_gemini_contents(messages),
                config=config,
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
        except errors.APIError as error:
            # Logamos só o status: o corpo do erro pode trazer trechos da conversa.
            logger.warning("gemini_api_error", extra={"upstream_status": error.code})
            raise map_api_error(error) from error
        except Exception as error:
            # Rede caiu, resposta malformada etc.
            logger.warning("gemini_unexpected_error", extra={"error_type": type(error).__name__})
            raise LLMUnavailableError() from error
