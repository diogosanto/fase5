import os
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


@dataclass
class LLMResponse:
    content: str


class GeminiLLM:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Defina GEMINI_API_KEY ou GOOGLE_API_KEY no ambiente.")

        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=api_key)
        self.config = types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        contents = prompt if not system_instruction else f"{system_instruction}\n\n{prompt}"
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.config,
        )
        return response.text or ""

    def invoke(self, messages) -> LLMResponse:
        prompt_parts: list[str] = []
        for message in messages:
            role = message.__class__.__name__.replace("Message", "").upper()
            prompt_parts.append(f"{role}:\n{message.content}")

        content = self.generate("\n\n".join(prompt_parts))
        return LLMResponse(content=content)


def get_llm() -> GeminiLLM:
    return GeminiLLM()
