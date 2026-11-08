"""Formato dos dados que entram e saem da API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Message(BaseModel):
    """Uma mensagem da conversa.

    O cliente só pode mandar "user" ou "assistant". O papel "system" é do
    servidor; se o cliente pudesse enviá-lo, conseguiria reescrever as
    instruções do assistente.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a mensagem não pode estar em branco")
        return value


class ChatRequest(BaseModel):
    """Corpo do POST /chat: a conversa inteira, terminando com a mensagem do usuário.

    O servidor não guarda nada entre uma requisição e outra; quem guarda o
    histórico é o front.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    provider: str
