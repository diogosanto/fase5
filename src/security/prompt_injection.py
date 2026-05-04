import re


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+(as\s+)?instru[cç][oõ]es\s+(anteriores|acima)",
    r"desconsidere\s+(as\s+)?instru[cç][oõ]es",
    r"system\s+prompt",
    r"prompt\s+(do\s+)?sistema",
    r"developer\s+message",
    r"mensagem\s+do\s+desenvolvedor",
    r"reveal\s+(the\s+)?prompt",
    r"mostre\s+(o\s+)?prompt",
    r"exfiltrat",
    r"jailbreak",
    r"fa[cç]a\s+de\s+conta\s+que\s+voc[eê]\s+agora\s+[eé]",
    r"force\s+(the\s+)?tool",
    r"chame\s+(a\s+)?tool",
    r"execute\s+(a\s+)?tool",
]


def detect_prompt_injection(text: str) -> bool:
    """Detect obvious prompt-injection and unsafe tool-manipulation attempts."""
    if not text:
        return False

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS)

