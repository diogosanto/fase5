import os
import unittest
from unittest.mock import patch

from src.agent.llm import get_llm_config


class LLMConfigTests(unittest.TestCase):
    def test_reads_groq_environment_variables(self) -> None:
        env = {
            "LLM_PROVIDER": "groq",
            "GROQ_API_KEY": "test-key",
            "GROQ_MODEL": "llama-3.1-8b-instant",
            "LLM_MAX_TOKENS": "300",
        }

        with patch.dict(os.environ, env, clear=True):
            config = get_llm_config()

        self.assertEqual(config.provider, "groq")
        self.assertEqual(config.model, "llama-3.1-8b-instant")
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.max_tokens, 300)

    def test_requires_groq_api_key(self) -> None:
        env = {
            "LLM_PROVIDER": "groq",
            "GROQ_MODEL": "llama-3.1-8b-instant",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                get_llm_config()


if __name__ == "__main__":
    unittest.main()
