"""RAG: o assistente responde com base nos documentos de app/rag/docs, e não só no prompt."""

from app.rag.documents import Chunk, load_chunks
from app.rag.embeddings import Embedder, GeminiEmbedder
from app.rag.retriever import Retriever, SearchResult, add_context_to_prompt

__all__ = [
    "Chunk",
    "Embedder",
    "GeminiEmbedder",
    "Retriever",
    "SearchResult",
    "add_context_to_prompt",
    "load_chunks",
]
