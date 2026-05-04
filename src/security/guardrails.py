import re
from typing import Any


MAX_TEXT_LENGTH = 1000
ALLOWED_TEXT_FIELDS = {"message", "query", "prompt", "question", "text"}

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+(as\s+)?instru[c\u00e7][o\u00f5]es\s+(anteriores|acima)",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+(the\s+)?prompt",
    r"mostre\s+(o\s+)?prompt",
    r"exfiltrat",
    r"jailbreak",
]

PII_PATTERNS = [
    r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",  # CPF
    r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",  # CNPJ
    r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    r"\b(?:telefone|celular|whatsapp|phone)\s*:?\s*(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}-?\d{4}\b",
]

OFF_DOMAIN_PATTERNS = [
    r"\b(senha|password|token|api[_ -]?key|secret)\b",
    r"\b(cart[a\u00e3]o\s+de\s+cr[e\u00e9]dito|credit\s+card)\b",
]


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


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_text_policy(data: Any) -> None:
    for text in _text_values(data):
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()
        if len(cleaned) > MAX_TEXT_LENGTH:
            raise ValueError("Texto acima do limite permitido")
        if _matches_any(cleaned, PROMPT_INJECTION_PATTERNS):
            raise ValueError("Entrada bloqueada por politica de prompt injection")
        if _matches_any(cleaned, PII_PATTERNS):
            raise ValueError("Entrada bloqueada por politica de PII")
        if _matches_any(cleaned, OFF_DOMAIN_PATTERNS):
            raise ValueError("Entrada fora do escopo seguro do sistema")


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
