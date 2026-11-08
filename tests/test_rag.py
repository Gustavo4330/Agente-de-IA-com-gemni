"""Testes do RAG com um embedder falso (sem internet e sem chave)."""

import asyncio
import json
import unicodedata
import zlib

import pytest

from app.config import Settings
from app.main import build_retriever
from app.rag import Chunk, Embedder, Retriever, SearchResult, add_context_to_prompt, load_chunks
from app.rag.documents import split_sections
from app.rag.retriever import cosine_similarity
from tests.conftest import FakeProvider, make_settings, user


class WordEmbedder(Embedder):
    """Embedder de mentira: cada palavra (de 4+ letras) vira uma posição do vetor.

    Textos que compartilham palavras ficam parecidos, o que basta para testar a busca.
    """

    def __init__(self, fail: bool = False, delay: float = 0.0):
        self.fail = fail
        self.delay = delay
        self.calls = 0

    async def embed(self, texts, kind):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("embedding fora do ar")
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        plain = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
        vector = [0.0] * 512
        for word in "".join(c if c.isalnum() else " " for c in plain).split():
            if len(word) >= 4:
                vector[zlib.crc32(word.encode()) % 512] += 1
        return vector


def make_retriever(**embedder_options) -> Retriever:
    retriever = Retriever(WordEmbedder(**embedder_options), load_chunks())
    asyncio.run(retriever.build_index())
    return retriever


# --- Documentos -----------------------------------------------------------------


def test_split_sections_ignores_text_before_first_section_and_empty_sections():
    markdown = "# Título\nintro solta\n## A\nconteúdo A\n## Vazia\n\n## B\nconteúdo B"
    assert split_sections(markdown) == [("A", "conteúdo A"), ("B", "conteúdo B")]


def test_load_chunks_reads_every_section_of_the_real_docs():
    chunks = load_chunks()
    sources = {chunk.source for chunk in chunks}
    assert sources == {"servicos.md", "como_trabalhamos.md", "perguntas_frequentes.md"}
    assert len(chunks) == 15
    # O título do documento vai junto, para o trecho fazer sentido sozinho.
    assert all(" - " in chunk.text.splitlines()[0] for chunk in chunks)


def test_load_chunks_fails_on_empty_folder(tmp_path):
    with pytest.raises(ValueError):
        load_chunks(tmp_path)


# --- Busca ----------------------------------------------------------------------


def test_cosine_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


@pytest.mark.parametrize(
    ("question", "expected_section"),
    [
        ("Qual a garantia para correção de defeitos?", "Garantia e propriedade do código"),
        ("Qual o horário de atendimento de vocês?", "Qual o horário de atendimento?"),
        ("Vocês desenvolvem jogos?", "O que não fazemos"),
    ],
)
def test_search_finds_the_right_section(question, expected_section):
    results = asyncio.run(make_retriever().search(question, top_k=3))
    assert results[0].chunk.section == expected_section
    assert results[0].score >= results[-1].score


def test_search_respects_top_k():
    assert len(asyncio.run(make_retriever().search("projeto", top_k=2))) == 2


def test_add_context_to_prompt_lists_each_excerpt_with_its_file():
    chunk = Chunk(source="faq.md", section="Horário", text="FAQ - Horário\nDas 9h às 18h.")
    prompt = add_context_to_prompt("PROMPT BASE", [SearchResult(chunk, 0.9)])
    assert prompt.startswith("PROMPT BASE")
    assert "## Documentos de referência" in prompt
    assert "[faq.md]\nFAQ - Horário\nDas 9h às 18h." in prompt


def test_add_context_without_results_keeps_prompt_unchanged():
    assert add_context_to_prompt("PROMPT BASE", []) == "PROMPT BASE"


# --- Integração com o /chat -----------------------------------------------------


def test_chat_sends_relevant_documents_to_the_model(client_factory):
    provider = FakeProvider()
    client = client_factory(provider, retriever=make_retriever())
    client.post("/chat", json={"messages": [user("Qual a garantia depois da entrega?")]})

    system_prompt, _ = provider.calls[0]
    assert "## Documentos de referência" in system_prompt
    assert "[como_trabalhamos.md]" in system_prompt
    assert "90 dias" in system_prompt


def test_chat_works_without_documents_when_search_fails(client_factory):
    provider = FakeProvider()
    client = client_factory(provider, retriever=make_retriever())
    client.app.state.retriever.embedder.fail = True  # índice pronto, mas a busca falha

    response = client.post("/chat", json={"messages": [user("Qual a garantia?")]})
    assert response.status_code == 200
    assert "## Documentos de referência" not in provider.calls[0][0]


def test_chat_skips_documents_when_search_is_too_slow(client_factory):
    provider = FakeProvider()
    client = client_factory(provider, retriever=make_retriever(), rag_timeout=0.05)
    client.app.state.retriever.embedder.delay = 1.0

    assert client.post("/chat", json={"messages": [user("Qual a garantia?")]}).status_code == 200
    assert "## Documentos de referência" not in provider.calls[0][0]


def test_logs_show_which_sections_were_used_but_not_their_text(client_factory, capsys):
    client = client_factory(FakeProvider(), retriever=make_retriever())
    client.post("/chat", json={"messages": [user("Qual a garantia depois da entrega?")]})

    output = capsys.readouterr().out
    finished = [json.loads(line) for line in output.splitlines() if '"chat_finished"' in line][0]
    assert "como_trabalhamos.md#Garantia e propriedade do código" in finished["sources"]
    assert "90 dias" not in output


# --- Montagem do índice ---------------------------------------------------------


def test_build_retriever_is_skipped_without_api_key():
    assert asyncio.run(build_retriever(make_settings(gemini_api_key=None))) is None


def test_build_retriever_is_skipped_when_disabled():
    assert asyncio.run(build_retriever(make_settings(gemini_api_key="x", rag_enabled=False))) is None


def test_build_retriever_returns_none_if_indexing_fails(monkeypatch):
    monkeypatch.setattr("app.main.GeminiEmbedder", lambda *args, **kwargs: WordEmbedder(fail=True))
    settings: Settings = make_settings(gemini_api_key="x")
    assert asyncio.run(build_retriever(settings)) is None
