"""Transforma texto em vetores (embeddings), para comparar significado em vez de palavras."""

from abc import ABC, abstractmethod
from typing import Literal

from google import genai
from google.genai import types

Kind = Literal["query", "document"]


class Embedder(ABC):
    @abstractmethod
    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        """Devolve um vetor por texto, na mesma ordem."""


class GeminiEmbedder(Embedder):
    def __init__(self, api_key: str, model: str, dimensions: int = 768, client=None):
        self.client = client or genai.Client(api_key=api_key)
        self.model = model
        self.dimensions = dimensions  # 768 já basta e deixa a busca mais leve que os 3072 padrão

    async def embed(self, texts: list[str], kind: Kind) -> list[list[float]]:
        # O Gemini gera vetores um pouco diferentes para perguntas e para documentos,
        # o que melhora a busca.
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        response = await self.client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dimensions),
        )
        return [embedding.values for embedding in response.embeddings]
