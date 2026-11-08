"""Lê os documentos da empresa e quebra cada um em trechos (um por seção "## ")."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DOCS_PATH = Path(__file__).parent / "docs"


@dataclass(frozen=True)
class Chunk:
    source: str  # nome do arquivo, usado na citação ("servicos.md")
    section: str  # título da seção
    text: str  # texto que vai para o modelo


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Divide um markdown nas seções "## ". Devolve pares (título, conteúdo)."""
    sections = []
    title, lines = None, []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if title and "".join(lines).strip():
                sections.append((title, "\n".join(lines).strip()))
            title, lines = line[3:].strip(), []
        elif title:
            lines.append(line)
    if title and "".join(lines).strip():
        sections.append((title, "\n".join(lines).strip()))
    return sections


def load_chunks(folder: Path = DEFAULT_DOCS_PATH) -> list[Chunk]:
    chunks = []
    for path in sorted(folder.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        doc_title = markdown.splitlines()[0].lstrip("# ").strip() if markdown.strip() else path.stem
        for section, content in split_sections(markdown):
            # O título do documento entra no texto para o trecho fazer sentido sozinho.
            chunks.append(Chunk(source=path.name, section=section, text=f"{doc_title} - {section}\n{content}"))
    if not chunks:
        raise ValueError(f"Nenhum trecho encontrado em {folder}")
    return chunks
