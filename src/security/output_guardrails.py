import re

from src.security.input_guardrails import GuardrailResult
from src.security.pii_detection import detect_pii
from src.security.prompt_injection import detect_prompt_injection


UNSAFE_CERTAINTY_PATTERNS = [
    r"\bpre[cç]o\s+garantido\b",
    r"\bvalor\s+garantido\b",
    r"\bgarantia\s+de\s+valor\b",
    r"\bcerteza\s+(jur[ií]dica|financeira)\b",
    r"\bdecis[aã]o\s+financeira\s+definitiva\b",
    r"\brentabilidade\s+garantida\b",
]

PROMPT_LEAKAGE_PATTERNS = [
    r"\bsystem\s+prompt\b",
    r"\bdeveloper\s+message\b",
    r"\bprompt\s+(oculto|interno|do\s+sistema)\b",
    r"\binstru[cç][oõ]es\s+(ocultas|internas|do\s+sistema)\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_model_output(text: str) -> GuardrailResult:
    """Validate final text before returning it to the user."""
    sanitized = (text or "").strip()
    if not sanitized:
        return GuardrailResult(False, "output_empty", sanitized)

    pii_types = detect_pii(sanitized)
    if pii_types:
        return GuardrailResult(False, f"output_pii_detected:{','.join(pii_types)}", sanitized)

    if _matches_any(sanitized, PROMPT_LEAKAGE_PATTERNS) or detect_prompt_injection(sanitized):
        return GuardrailResult(False, "output_prompt_or_instruction_leakage", sanitized)

    if _matches_any(sanitized, UNSAFE_CERTAINTY_PATTERNS):
        return GuardrailResult(False, "unsupported_financial_or_legal_certainty", sanitized)

    return GuardrailResult(True, sanitized_text=sanitized)
