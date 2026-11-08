"""Respostas de erro da API.

Todo erro sai no mesmo formato, para o front só precisar tratar um caso:

    {"error": {"code": "rate_limited", "message": "Muitas requisições..."}}
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.llm import LLMError


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


async def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # O FastAPI devolve por padrão o valor recebido em cada erro.
    # Aqui mandamos só o campo e o motivo, para não ecoar o texto do usuário.
    problems = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"][1:]) or "body"
        problems.append(f"{field}: {error['msg']}")
    return error_response(422, "invalid_request", "Requisição inválida. " + "; ".join(problems))


async def _on_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    # Erros do próprio framework: JSON ilegível, rota inexistente, método errado...
    if exc.status_code == 400:
        return error_response(400, "bad_request", "Não consegui ler o corpo da requisição (JSON UTF-8 inválido).")
    if exc.status_code == 404:
        return error_response(404, "not_found", "Rota não encontrada.")
    if exc.status_code == 405:
        return error_response(405, "method_not_allowed", "Método não permitido.")
    return error_response(exc.status_code, "http_error", str(exc.detail))


async def _on_llm_error(request: Request, exc: LLMError) -> JSONResponse:
    return error_response(exc.status_code, exc.code, exc.public_message)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(HTTPException, _on_http_error)
    app.add_exception_handler(LLMError, _on_llm_error)
