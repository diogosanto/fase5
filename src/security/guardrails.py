import re
from typing import Any

from src.security.input_guardrails import validate_user_input
from src.security.output_guardrails import validate_model_output


MAX_TEXT_LENGTH = 1000
ALLOWED_TEXT_FIELDS = {"message", "query", "prompt", "question", "text"}


def _normalize_cep(value: Any) -> str:
    return "".join(char for char in str(value) if char.isdigit())


def _text_values(data: Any) -> list[str]:
    if isinstance(data, str):
        return [data]
    if not isinstance(data, dict):
        return []
    return [
        str(value)
        for key, value in data.items()
        if key in ALLOWED_TEXT_FIELDS and value is not None
    ]


def validate_text_policy(data: Any) -> None:
    for text in _text_values(data):
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
        result = validate_user_input(cleaned)
        reason = result.reason or ""
        if reason == "input_too_long":
            raise ValueError("Texto acima do limite permitido")
        if reason == "prompt_injection_detected":
            raise ValueError("Entrada bloqueada por politica de prompt injection")
        if reason.startswith("pii_detected"):
            raise ValueError("Entrada bloqueada por politica de PII")
        if reason == "unsafe_secret_or_off_domain_request":
            raise ValueError("Entrada fora do escopo seguro do sistema")
        if not result.allowed:
            raise ValueError("Entrada bloqueada por politica de seguranca")


def validate_input(data):
    validate_text_policy(data)

    cep = _normalize_cep(data.get("cep", ""))
    if not cep:
        raise ValueError("CEP invalido")

    if float(data["area_do_terreno_m2"]) <= 0:
        raise ValueError("Area invalida")

    if "ano" in data and not 1900 <= int(data["ano"]) <= 2999:
        raise ValueError("Ano invalido")

    if "mes" in data and not 1 <= int(data["mes"]) <= 12:
        raise ValueError("Mes invalido")


def validate_output(value):
    if value < 0:
        raise ValueError("Saida invalida do modelo")

    return value
