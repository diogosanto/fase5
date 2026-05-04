import os
import re
from dataclasses import dataclass

from src.security.pii_detection import detect_pii
from src.security.prompt_injection import detect_prompt_injection


DEFAULT_MAX_INPUT_LENGTH = 1000
OFF_DOMAIN_PATTERNS = [
    r"\b(senha|password|token|api[_ -]?key|secret|segredo)\b",
    r"\b(cart[aã]o\s+de\s+cr[eé]dito|credit\s+card)\b",
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    sanitized_text: str | None = None


def _max_input_length() -> int:
    try:
        return int(os.getenv("MAX_CHAT_MESSAGE_LENGTH", str(DEFAULT_MAX_INPUT_LENGTH)))
    except ValueError:
        return DEFAULT_MAX_INPUT_LENGTH


def _sanitize_text(text: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text or "")
    return cleaned.strip()


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_user_input(text: str) -> GuardrailResult:
    """Validate user text before it reaches the Agent, RAG or LLM provider."""
    sanitized = _sanitize_text(text)
    if not sanitized:
        return GuardrailResult(False, "input_empty", sanitized)

    if len(sanitized) > _max_input_length():
        return GuardrailResult(False, "input_too_long", sanitized)

    if detect_prompt_injection(sanitized):
        return GuardrailResult(False, "prompt_injection_detected", sanitized)

    pii_types = detect_pii(sanitized)
    if pii_types:
        return GuardrailResult(False, f"pii_detected:{','.join(pii_types)}", sanitized)

    if _matches_any(sanitized, OFF_DOMAIN_PATTERNS):
        return GuardrailResult(False, "unsafe_secret_or_off_domain_request", sanitized)

    return GuardrailResult(True, sanitized_text=sanitized)

