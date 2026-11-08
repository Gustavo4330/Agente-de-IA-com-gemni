"""Leitura do prompt de sistema (o texto fica em system_prompt.md, fácil de revisar)."""

from functools import lru_cache
from pathlib import Path


@lru_cache
def load_system_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"O prompt de sistema está vazio: {path}")
    return text
