import importlib
import sys
import unittest
from unittest.mock import patch

import httpx

from src.agent.orchestrator import AgentOrchestrator
from src.agent.tools import ToolResult


"""
Testes de integracao do endpoint /chat.

Este arquivo valida o contrato HTTP do chat usado na camada de RAG + Agent + LLM:
- entrada obrigatoria `message` e entrada opcional `property_data`;
- resposta estruturada com `answer`, `tools_used` e `metadata`;
- tratamento padronizado de erros internos e erros do provedor LLM;
- roteamento de perguntas com dados de imovel para a tool `price_estimator`.

Os testes usam mocks para nao carregar um modelo MLflow real e para nao chamar Groq/LLM.
Assim, a validacao e rapida, deterministica e nao consome tokens.
"""


class FakeModel:
    """Modelo fake usado para impedir carregamento real do modelo MLflow durante os testes."""

    def predict(self, dataframe):
        return [123.45]


class FakeAgent:
    """Agent fake para simular respostas e falhas do AgentOrchestrator sem chamar LLM."""

    def __init__(self, response=None, error=None):
        self.response = response or {
            "answer": "Resposta estruturada do agent.",
            "tools_used": ["rag_search"],
            "chunks_retrieved": 2,
            "llm_calls": 1,
        }
        self.error = error
        self.last_message = None

    def chat(self, message: str, property_data=None):
        self.last_message = message
        self.last_property_data = property_data
        if self.error:
            raise self.error
        return self.response


def load_api_module():
    """Importa api.main com dependencias externas mockadas para isolar os testes do /chat."""

    sys.modules.pop("api.main", None)
    with patch("os.listdir", return_value=["model_2026.04.28.0119"]):
        with patch("mlflow.pyfunc.load_model", return_value=FakeModel()):
            return importlib.import_module("api.main")


class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Valida o comportamento HTTP do endpoint /chat exposto pelo FastAPI."""

    def setUp(self) -> None:
        self.api_main = load_api_module()

    async def post_chat(self, payload: dict) -> httpx.Response:
        """Executa chamadas ASGI in-memory, sem subir uvicorn e sem depender de rede local."""

        transport = httpx.ASGITransport(app=self.api_main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post("/chat", json=payload)

    async def test_chat_with_valid_message_returns_structured_response(self) -> None:
        self.api_main.agent_orchestrator = FakeAgent()

        response = await self.post_chat({"message": "Quais fatores influenciam o preco?"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Resposta estruturada do agent.")
        self.assertEqual(payload["tools_used"], ["rag_search"])
        self.assertEqual(payload["metadata"]["chunks_retrieved"], 2)
        self.assertIsNotNone(payload["metadata"]["request_id"])

    async def test_chat_accepts_optional_property_data(self) -> None:
        fake_agent = FakeAgent()
        self.api_main.agent_orchestrator = fake_agent

        response = await self.post_chat(
            {
                "message": "Avalie este imovel",
                "property_data": {
                    "area": 80,
                    "quartos": 2,
                    "bairro": "Moema",
                    "preco": 950000,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Dados do imovel informados pelo usuario", fake_agent.last_message)
        self.assertEqual(fake_agent.last_property_data["area"], 80)
        self.assertIn("Moema", fake_agent.last_message)

    async def test_chat_rejects_empty_message(self) -> None:
        response = await self.post_chat({"message": "   "})

        self.assertEqual(response.status_code, 422)

    async def test_chat_rejects_missing_message(self) -> None:
        response = await self.post_chat({})

        self.assertEqual(response.status_code, 422)

    async def test_chat_maps_agent_error_to_500(self) -> None:
        self.api_main.agent_orchestrator = FakeAgent(error=RuntimeError("falha controlada"))

        response = await self.post_chat({"message": "Teste de erro"})

        self.assertEqual(response.status_code, 500)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "internal_error")
        self.assertIn("request_id", payload["error"])

    async def test_chat_maps_llm_error_to_503(self) -> None:
        self.api_main.agent_orchestrator = FakeAgent(error=RuntimeError("groq rate limit"))

        response = await self.post_chat({"message": "Teste de erro LLM"})

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "llm_provider_error")
        self.assertIn("request_id", payload["error"])


class AgentOrchestratorTests(unittest.TestCase):
    """Valida regras de roteamento do Agent que afetam diretamente o /chat."""

    def test_property_data_price_question_uses_price_estimator_directly(self) -> None:
        """Garante que perguntas de preco com property_data acionem o modelo, nao o RAG."""

        orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)

        with patch("src.agent.orchestrator.price_estimator") as fake_price_estimator:
            fake_price_estimator.return_value = ToolResult(
                tool_name="price_estimator",
                content='{"valor_estimado": 450000, "unidade": "R$", "versao_modelo": "model_test"}',
                metadata={"model_version": "model_test"},
            )

            result = orchestrator.chat(
                "Qual o preco desse apartamento?",
                property_data={
                    "area": 60,
                    "bairro": "MOOCA - SP",
                    "valor_m2": 1500,
                    "ano_mes": 202401,
                    "media_valor_cep": 2000,
                },
            )

        self.assertEqual(result["tools_used"], ["price_estimator"])
        self.assertIn("Estimativa gerada pelo modelo de predicao", result["answer"])
        self.assertEqual(result["steps"][0]["action_input"]["area_do_terreno_m2"], 60)


if __name__ == "__main__":
    unittest.main()
