"""Avaliação do assistente: faz as perguntas de dataset.json ao bot real e confere as respostas.

Para cada pergunta com resposta nos documentos, checamos três coisas:
  1. busca:    o documento certo apareceu entre os trechos encontrados?
  2. conteúdo: a resposta tem a informação esperada (ex.: "90 dias")?
  3. fonte:    a resposta termina citando o arquivo certo?

Para perguntas SEM resposta nos documentos, o certo é o bot admitir que não
sabe e não inventar o dado pedido (cada pergunta traz, em "must_not_match",
o padrão do dado inventado: um CNPJ, um número de funcionários...). Citar uma
fonte aqui é permitido: o bot pode, por exemplo, indicar o contato da empresa.

Uso (precisa da GEMINI_API_KEY no .env):
    python -m evals.run_eval
"""

import asyncio
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.llm import LLMError, build_provider
from app.main import build_retriever
from app.prompts import load_system_prompt
from app.rag import add_context_to_prompt
from app.schemas import Message

EVAL_DIR = Path(__file__).parent
DATASET = EVAL_DIR / "dataset.json"
REPORT = EVAL_DIR / "results.md"

# Jeitos comuns de o modelo dizer que não tem a informação.
I_DONT_KNOW = ["nao tenho", "nao possuo", "nao encontrei", "nao ha informac", "nao consta", "nao sei", "nao disponho"]


def normalize(text: str) -> str:
    """Minúsculas e sem acento, para "Não" e "nao" contarem igual."""
    return unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()


def cited_sources(answer: str) -> list[str]:
    """Lê a linha "Fonte: a.md, b.md" da resposta."""
    for line in reversed(answer.strip().splitlines()):
        clean = line.strip().strip("*_` ")
        if normalize(clean).startswith("fonte"):
            names = clean.split(":", 1)[-1]
            return [name.strip(" *_`.") for name in names.split(",") if name.strip(" *_`.")]
    return []


@dataclass
class Result:
    question: str
    answer: str
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


def grade(item: dict, answer: str, found_sources: list[str]) -> Result:
    result = Result(item["question"], answer)
    text = normalize(answer)
    cited = cited_sources(answer)

    if item["source"]:
        result.checks["busca"] = item["source"] in found_sources
        result.checks["conteudo"] = all(normalize(term) in text for term in item["must_include"])
        result.checks["fonte"] = item["source"] in cited
    else:
        result.checks["admite_nao_saber"] = any(phrase in text for phrase in I_DONT_KNOW)
        result.checks["nao_inventa"] = not re.search(item["must_not_match"], answer, re.IGNORECASE)
    return result


async def ask(provider, system_prompt: str, question: str, attempts: int = 3) -> str | None:
    """Pergunta ao modelo e junta a resposta inteira.

    O provedor às vezes oscila (erro 503), então tentamos de novo algumas vezes.
    Se não der, devolve None: a pergunta conta como erro do provedor, não como
    resposta errada, e a avaliação segue para a próxima.
    """
    for attempt in range(1, attempts + 1):
        try:
            chunks = [chunk async for chunk in provider.stream(system_prompt, [Message(role="user", content=question)])]
            return "".join(chunks)
        except LLMError as error:
            print(f"    provedor falhou ({error.code}), tentativa {attempt}/{attempts}")
            await asyncio.sleep(10 * attempt)  # o limite gratuito é por minuto, então esperamos de verdade
    return None


def write_report(results: list[Result], model: str) -> None:
    answerable = [r for r in results if "busca" in r.checks]
    unanswerable = [r for r in results if "admite_nao_saber" in r.checks]
    provider_errors = sum("provedor_respondeu" in r.checks for r in results)

    def rate(items, check):
        return f"{sum(r.checks[check] for r in items)}/{len(items)}"

    lines = [
        "# Resultado da avaliação",
        "",
        f"- Data: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Modelo: `{model}`",
        f"- **Aprovadas: {sum(r.passed for r in results)}/{len(results) - provider_errors}** perguntas respondidas",
        f"- Busca encontrou o documento certo: {rate(answerable, 'busca')}",
        f"- Resposta com a informação esperada: {rate(answerable, 'conteudo')}",
        f"- Citou a fonte certa: {rate(answerable, 'fonte')}",
        f"- Admitiu não saber (perguntas sem resposta): {rate(unanswerable, 'admite_nao_saber')}",
        f"- Não inventou o dado pedido: {rate(unanswerable, 'nao_inventa')}",
        f"- Perguntas sem resposta por falha do provedor: {provider_errors}",
        "",
        "| | Pergunta | Checagens |",
        "|---|---|---|",
    ]
    for r in results:
        checks = ", ".join(f"{name} {'✅' if ok else '❌'}" for name, ok in r.checks.items())
        lines.append(f"| {'✅' if r.passed else '❌'} | {r.question} | {checks} |")

    failed = [r for r in results if not r.passed]
    if failed:
        lines += ["", "## Respostas reprovadas", ""]
        for r in failed:
            lines += [f"**{r.question}**", "", "> " + r.answer.strip().replace("\n", "\n> "), ""]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main() -> int:
    logging.getLogger("app").setLevel(logging.ERROR)  # o script já mostra as falhas do provedor
    settings = get_settings()
    provider = build_provider(settings)
    retriever = await build_retriever(settings)
    if retriever is None:
        print("Não consegui montar o índice dos documentos. Confira a GEMINI_API_KEY.")
        return 1
    system_prompt = load_system_prompt(settings.system_prompt_path)
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))

    results = []
    for number, item in enumerate(dataset, start=1):
        found = await retriever.search(item["question"], settings.rag_top_k)
        prompt = add_context_to_prompt(system_prompt, found)
        answer = await ask(provider, prompt, item["question"])
        if answer is None:
            result = Result(item["question"], "(sem resposta: o provedor falhou em todas as tentativas)")
            result.checks["provedor_respondeu"] = False
        else:
            result = grade(item, answer, [r.chunk.source for r in found])
        results.append(result)
        print(f"{number:2d}. {'OK   ' if result.passed else 'FALHA'} {item['question']}")
        await asyncio.sleep(4)  # ~15 perguntas por minuto, dentro da cota gratuita do Gemini

    write_report(results, settings.gemini_model)
    passed = sum(r.passed for r in results)
    answered = sum("provedor_respondeu" not in r.checks for r in results)
    print(f"\nAprovadas: {passed}/{answered} respondidas ({len(results) - answered} sem resposta do provedor).")
    print(f"Relatório em {REPORT.relative_to(EVAL_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
