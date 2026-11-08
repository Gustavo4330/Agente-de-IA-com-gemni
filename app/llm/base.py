"""O que um provedor de LLM precisa oferecer, e os erros que ele pode levantar.

As rotas só conversam com esta interface. Para trocar o Gemini por outro
modelo, basta criar uma classe nova com o método `stream` e traduzir os erros
do SDK para os erros abaixo.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas import Message


class LLMError(Exception):
    """Erro do provedor. `code` e `public_message` podem ser mostrados ao usuário."""

    code = "upstream_unavailable"
    status_code = 502
    public_message = "O assistente está indisponível no momento. Tente novamente em instantes."


class LLMUnavailableError(LLMError):
    """Provedor fora do ar, sobrecarregado ou sem rede."""


class LLMRateLimitError(LLMError):
    code = "rate_limited"
    status_code = 429
    public_message = "Muitas requisições no momento. Tente novamente em instantes."


class LLMTimeoutError(LLMError):
    code = "timeout"
    status_code = 504
    public_message = "O assistente demorou demais para responder. Tente novamente."


class LLMConfigurationError(LLMError):
    """Chave ausente ou inválida, modelo inexistente. É problema do servidor, não do usuário."""

    code = "not_configured"
    status_code = 503
    public_message = "O assistente não está configurado corretamente."


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def stream(self, system_prompt: str, messages: list[Message]) -> AsyncIterator[str]:
        """Gera a resposta em pedaços de texto, conforme o modelo vai escrevendo."""
