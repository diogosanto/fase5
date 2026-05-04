import re


PII_PATTERNS: dict[str, str] = {
    "cpf": r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
    "cnpj": r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
    "email": r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
    "telefone": r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?9?\d{4}-?\d{4}\b",
}


def detect_pii(text: str) -> list[str]:
    """Return the PII types detected in text without exposing the matched values."""
    if not text:
        return []

    detected = [
        pii_type
        for pii_type, pattern in PII_PATTERNS.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    return sorted(set(detected))

