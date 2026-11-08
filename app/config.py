"""Configurações da aplicação.

Tudo vem de variáveis de ambiente (ou do arquivo .env). O nome da variável é
o nome do campo em maiúsculas: `max_turns` -> MAX_TURNS. Veja o .env.example.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.md"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Modelo
    llm_provider: Literal["gemini"] = "gemini"
    gemini_api_key: SecretStr | None = None  # SecretStr: não aparece em print nem em log
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_temperature: Annotated[float, Field(ge=0, le=2)] = 0.4
    llm_max_output_tokens: PositiveInt = 1024

    # Timeouts, em segundos
    llm_first_token_timeout: PositiveFloat = 20  # espera máxima pelo 1º pedaço da resposta
    llm_idle_timeout: PositiveFloat = 15  # silêncio máximo entre um pedaço e outro
    llm_total_timeout: PositiveFloat = 90  # duração máxima da resposta inteira

    # Limites da conversa
    max_message_chars: PositiveInt = 4000
    max_turns: PositiveInt = 20
    max_total_chars: PositiveInt = 24000
    max_body_bytes: PositiveInt = 200_000

    # RAG: busca nos documentos de app/rag/docs antes de responder
    rag_enabled: bool = True
    rag_top_k: PositiveInt = 3  # quantos trechos vão para o modelo
    rag_timeout: PositiveFloat = 5  # se a busca demorar mais que isso, responde sem documentos
    embedding_model: str = "gemini-embedding-001"
    rag_docs_path: Path = Path(__file__).parent / "rag" / "docs"

    # Limite de requisições por IP (0 desliga)
    rate_limit_requests: NonNegativeInt = 20
    rate_limit_window_seconds: PositiveFloat = 60

    # Outros
    allowed_origins: Annotated[list[str], NoDecode] = ["http://localhost:8000", "http://127.0.0.1:8000"]
    log_level: str = "INFO"
    system_prompt_path: Path = DEFAULT_PROMPT_PATH

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def split_origins(cls, value):
        """Permite escrever ALLOWED_ORIGINS=https://a.com,https://b.com no .env."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def upper_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_settings() -> Settings:
    return Settings()
