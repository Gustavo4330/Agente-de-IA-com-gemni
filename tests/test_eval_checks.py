"""Testa as regras de correção da avaliação (sem chamar o modelo)."""

from evals.run_eval import cited_sources, grade, normalize

ANSWERABLE = {"question": "Qual a garantia?", "source": "como_trabalhamos.md", "must_include": ["90 dias"]}
UNANSWERABLE = {"question": "Qual o CNPJ?", "source": None, "must_not_match": r"\d{2}\.?\d{3}\.?\d{3}/?\d{4}"}


def test_normalize_ignores_case_and_accents():
    assert normalize("Não É Possível") == "nao e possivel"


def test_cited_sources_reads_the_source_line():
    assert cited_sources("Texto.\n\nFonte: a.md, b.md") == ["a.md", "b.md"]
    assert cited_sources("Texto.\n**Fonte:** `a.md`.") == ["a.md"]
    assert cited_sources("Texto sem fonte.") == []


def test_good_answer_passes_all_checks():
    answer = "A garantia é de 90 dias.\n\nFonte: como_trabalhamos.md"
    result = grade(ANSWERABLE, answer, found_sources=["como_trabalhamos.md", "servicos.md"])
    assert result.passed


def test_answer_without_source_line_fails_only_the_source_check():
    result = grade(ANSWERABLE, "A garantia é de 90 dias.", found_sources=["como_trabalhamos.md"])
    assert result.checks == {"busca": True, "conteudo": True, "fonte": False}


def test_wrong_retrieval_fails_the_search_check():
    answer = "A garantia é de 90 dias.\nFonte: como_trabalhamos.md"
    assert not grade(ANSWERABLE, answer, found_sources=["servicos.md"]).checks["busca"]


def test_unanswerable_passes_when_bot_admits_it_does_not_know():
    answer = "Não tenho essa informação. Fale com a equipe pelo contato do site."
    assert grade(UNANSWERABLE, answer, found_sources=["servicos.md"]).passed


def test_unanswerable_may_cite_a_source_for_the_contact_info():
    answer = "Não tenho o CNPJ. Fale com contato@nimbus.example.\nFonte: perguntas_frequentes.md"
    assert grade(UNANSWERABLE, answer, found_sources=["perguntas_frequentes.md"]).passed


def test_unanswerable_fails_when_bot_invents_the_data():
    result = grade(UNANSWERABLE, "O CNPJ é 12.345.678/0001-90.", found_sources=[])
    assert result.checks == {"admite_nao_saber": False, "nao_inventa": False}
