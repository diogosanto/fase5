import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()

logger = logging.getLogger("precificador.agent.llm")
_llm_call_count: ContextVar[int] = ContextVar("llm_call_count", default=0)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    max_tokens: int
    temperature: float
    timeout_seconds: float


@dataclass
class LLMResponse:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallResult:
    content: str
    provider: str
    model: str
    latency_ms: int
    token_usage: dict[str, Any] | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
        }


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    max_tokens = _get_int_env("LLM_MAX_TOKENS", 300)
    temperature = _get_float_env("LLM_TEMPERATURE", 0.2)
    timeout_seconds = _get_float_env("LLM_TIMEOUT_SECONDS", 30.0)

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY nao esta definida no ambiente. "
                "Defina a chave em variavel de ambiente ou arquivo .env local, sem versionar secrets."
            )

        return LLMConfig(
            provider=provider,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY ou GOOGLE_API_KEY nao esta definida no ambiente.")

        return LLMConfig(
            provider=provider,
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"LLM_PROVIDER nao suportado: {provider}")


def call_llm(prompt: str, system_instruction: str | None = None) -> str:
    return call_llm_with_metadata(prompt=prompt, system_instruction=system_instruction).content


def call_llm_with_metadata(prompt: str, system_instruction: str | None = None) -> LLMCallResult:
    config = get_llm_config()
    _llm_call_count.set(_llm_call_count.get() + 1)
    approx_prompt_tokens = estimate_tokens(prompt)
    if system_instruction:
        approx_prompt_tokens += estimate_tokens(system_instruction)

    logger.info(
        (
            "Chamando LLM provider=%s model=%s approx_prompt_tokens=%s "
            "max_tokens=%s temperature=%s timeout_seconds=%s call_count=%s"
        ),
        config.provider,
        config.model,
        approx_prompt_tokens,
        config.max_tokens,
        config.temperature,
        config.timeout_seconds,
        _llm_call_count.get(),
    )

    started_at = perf_counter()
    try:
        if config.provider == "groq":
            result = _call_groq(prompt=prompt, system_instruction=system_instruction, config=config)
        elif config.provider == "gemini":
            result = _call_gemini(prompt=prompt, system_instruction=system_instruction, config=config)
        else:
            raise ValueError(f"LLM_PROVIDER nao suportado: {config.provider}")

        logger.info(
            "LLM respondida provider=%s model=%s latency_ms=%s token_usage=%s",
            result.provider,
            result.model,
            result.latency_ms,
            result.token_usage,
        )
        return result
    except Exception:
        elapsed_ms = int((perf_counter() - started_at) * 1000)
        logger.exception(
            "Erro ao chamar LLM provider=%s model=%s latency_ms=%s",
            config.provider,
            config.model,
            elapsed_ms,
        )
        raise


def _call_groq(prompt: str, system_instruction: str | None, config: LLMConfig) -> LLMCallResult:
    from groq import Groq

    started_at = perf_counter()
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    client = Groq(api_key=config.api_key, timeout=config.timeout_seconds)
    completion = client.chat.completions.create(
        messages=messages,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return LLMCallResult(
        content=completion.choices[0].message.content or "",
        provider=config.provider,
        model=config.model,
        latency_ms=int((perf_counter() - started_at) * 1000),
        token_usage=_extract_usage(getattr(completion, "usage", None)),
    )


def _call_gemini(prompt: str, system_instruction: str | None, config: LLMConfig) -> LLMCallResult:
    from google import genai
    from google.genai import types

    started_at = perf_counter()
    contents = prompt if not system_instruction else f"{system_instruction}\n\n{prompt}"
    client = genai.Client(api_key=config.api_key)
    response = client.models.generate_content(
        model=config.model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return LLMCallResult(
        content=response.text or "",
        provider=config.provider,
        model=config.model,
        latency_ms=int((perf_counter() - started_at) * 1000),
        token_usage=_extract_usage(getattr(response, "usage_metadata", None)),
    )


def _extract_usage(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage

    usage_fields = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "total_token_count",
        "prompt_token_count",
        "candidates_token_count",
    ]
    extracted = {field: getattr(usage, field) for field in usage_fields if getattr(usage, field, None) is not None}
    return extracted or None


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um numero inteiro valido.") from exc


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} deve ser um numero decimal valido.") from exc


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def reset_llm_call_count() -> None:
    _llm_call_count.set(0)


def get_llm_call_count() -> int:
    return _llm_call_count.get()


class ProjectLLM:
    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return call_llm(prompt=prompt, system_instruction=system_instruction)

    def invoke(self, messages: list[Any]) -> LLMResponse:
        prompt_parts: list[str] = []
        for message in messages:
            role = message.__class__.__name__.replace("Message", "").upper()
            prompt_parts.append(f"{role}:\n{message.content}")

        result = call_llm_with_metadata("\n\n".join(prompt_parts))
        return LLMResponse(content=result.content, metadata=result.metadata())


def get_llm() -> ProjectLLM:
    return ProjectLLM()
