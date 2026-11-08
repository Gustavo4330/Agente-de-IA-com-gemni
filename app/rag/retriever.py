"""Busca os trechos mais parecidos com a pergunta.

Como são poucos documentos, os vetores ficam em memória e a comparação é
feita em Python puro. Com milhares de trechos, valeria usar um banco vetorial.
"""

import math
from dataclasses import dataclass

from app.rag.documents import Chunk
from app.rag.embeddings import Embedder


@dataclass
class SearchResult:
    chunk: Chunk
    score: float  # similaridade de cosseno: quanto mais perto de 1, mais parecido


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class Retriever:
    def __init__(self, embedder: Embedder, chunks: list[Chunk]):
        self.embedder = embedder
        self.chunks = chunks
        self.vectors: list[list[float]] = []

    async def build_index(self) -> None:
        """Gera os vetores de todos os trechos. Feito uma vez, na inicialização."""
        self.vectors = await self.embedder.embed([chunk.text for chunk in self.chunks], kind="document")

    async def search(self, question: str, top_k: int = 3) -> list[SearchResult]:
        [query_vector] = await self.embedder.embed([question], kind="query")
        results = [
            SearchResult(chunk, cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]


def add_context_to_prompt(system_prompt: str, results: list[SearchResult]) -> str:
    """Anexa os trechos encontrados ao prompt de sistema, cada um com o nome do arquivo."""
    if not results:
        return system_prompt
    excerpts = "\n\n".join(f"[{r.chunk.source}]\n{r.chunk.text}" for r in results)
    return f"{system_prompt}\n\n## Documentos de referência\n\n{excerpts}"
