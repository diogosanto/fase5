import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass
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


@dataclass
class LLMResponse:
    content: str


def get_llm_config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "300"))

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY nao esta definida no ambiente.")

        return LLMConfig(
            provider=provider,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=api_key,
            max_tokens=max_tokens,
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
        )

    raise ValueError(f"LLM_PROVIDER nao suportado: {provider}")


def call_llm(prompt: str, system_instruction: str | None = None) -> str:
    config = get_llm_config()
    _llm_call_count.set(_llm_call_count.get() + 1)
    approx_prompt_tokens = estimate_tokens(prompt)
    if system_instruction:
        approx_prompt_tokens += estimate_tokens(system_instruction)

    logger.info(
        "Chamando LLM provider=%s model=%s approx_prompt_tokens=%s max_tokens=%s call_count=%s",
        config.provider,
        config.model,
        approx_prompt_tokens,
        config.max_tokens,
        _llm_call_count.get(),
    )

    try:
        if config.provider == "groq":
            return _call_groq(prompt=prompt, system_instruction=system_instruction, config=config)

        if config.provider == "gemini":
            return _call_gemini(prompt=prompt, system_instruction=system_instruction, config=config)
    except Exception:
        logger.exception("Erro ao chamar LLM provider=%s model=%s", config.provider, config.model)
        raise

    raise ValueError(f"LLM_PROVIDER nao suportado: {config.provider}")


def _call_groq(prompt: str, system_instruction: str | None, config: LLMConfig) -> str:
    from groq import Groq

    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    client = Groq(api_key=config.api_key)
    completion = client.chat.completions.create(
        messages=messages,
        model=config.model,
        temperature=0,
        max_tokens=config.max_tokens,
    )
    return completion.choices[0].message.content or ""


def _call_gemini(prompt: str, system_instruction: str | None, config: LLMConfig) -> str:
    from google import genai
    from google.genai import types

    contents = prompt if not system_instruction else f"{system_instruction}\n\n{prompt}"
    client = genai.Client(api_key=config.api_key)
    response = client.models.generate_content(
        model=config.model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=config.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text or ""


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

        return LLMResponse(content=call_llm("\n\n".join(prompt_parts)))


def get_llm() -> ProjectLLM:
    return ProjectLLM()
