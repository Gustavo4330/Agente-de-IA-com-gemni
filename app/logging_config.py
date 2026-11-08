"""Logs em JSON, uma linha por evento.

Regra do projeto: nunca logar o texto das mensagens, o prompt ou chaves.
Só metadados, como quantidade de turnos, tamanhos, duração e código de erro.

Uso:
    logger.info("chat_finished", extra={"turns": 3, "duration_ms": 812.4})
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

# Preenchido pelo middleware; aparece em todos os logs da mesma requisição.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Campos que todo LogRecord já tem. O que sobrar veio do "extra=" e entra no JSON.
STANDARD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if request_id := request_id_var.get():
            entry["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in STANDARD_FIELDS:
                entry[key] = value
        if record.exc_info:
            entry["error_type"] = record.exc_info[0].__name__
        return json.dumps(entry, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    # Remove só o handler que nós mesmos criamos antes (evita logs duplicados
    # se a app for criada mais de uma vez, como nos testes).
    root.handlers = [h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    logging.getLogger("uvicorn.access").disabled = True  # nosso middleware já faz o access log
    logging.getLogger("httpx").setLevel(logging.WARNING)  # em DEBUG ele loga URLs e cabeçalhos
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
