"""Middleware HTTP: dá um ID a cada requisição, registra o acesso e barra corpos grandes demais."""

import logging
import re
import time
import uuid

from fastapi import FastAPI, Request

from app.errors import error_response
from app.logging_config import request_id_var

logger = logging.getLogger("app.access")

# Só aceitamos o X-Request-ID do cliente se ele for "comportado"; senão geramos um novo.
VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def add_request_context(app: FastAPI, max_body_bytes: int) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming_id = request.headers.get("x-request-id", "")
        request_id = incoming_id if VALID_REQUEST_ID.match(incoming_id) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            content_length = request.headers.get("content-length", "")
            if content_length.isdigit() and int(content_length) > max_body_bytes:
                response = error_response(413, "payload_too_large", "Corpo da requisição muito grande.")
            else:
                response = await call_next(request)

            response.headers["X-Request-ID"] = request_id
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
            return response
        finally:
            request_id_var.reset(token)
