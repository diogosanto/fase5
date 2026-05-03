import importlib
import sys
import unittest
from unittest.mock import patch

import httpx


class FakeModel:
    def predict(self, dataframe):
        assert list(dataframe.columns) == ["cep", "area_do_terreno_m2", "ano", "mes"]
        assert dataframe.iloc[0]["cep"] == "04001000"
        return [987654.32]


class FakeAgent:
    def chat(self, message: str, property_data=None):
        return {
            "answer": "Resposta integrada do agent.",
            "tools_used": ["rag_search"],
            "chunks_retrieved": 1,
            "llm_calls": 0,
        }


def load_api_module():
    sys.modules.pop("api.main", None)
    with patch("os.listdir", return_value=["model_2026.05.02.1934"]):
        with patch("mlflow.pyfunc.load_model", return_value=FakeModel()):
            module = importlib.import_module("api.main")
    module.model = FakeModel()
    module.agent_orchestrator = FakeAgent()
    return module


class ApiEndpointIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.api_main = load_api_module()

    async def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        transport = httpx.ASGITransport(app=self.api_main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    async def test_health_endpoint_returns_model_metadata(self) -> None:
        response = await self.request("GET", "/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["modelo"], "prod")
        self.assertEqual(payload["versao_modelo"], "2026.05.02.1934")

    async def test_predict_endpoint_returns_estimated_price(self) -> None:
        response = await self.request(
            "POST",
            "/predict",
            json={
                "cep": "04001000",
                "area_do_terreno_m2": 120,
                "ano": 2024,
                "mes": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["valor_estimado"], 987654.32)
        self.assertEqual(payload["unidade"], "R$")
        self.assertEqual(payload["versao_modelo"], "2026.05.02.1934")

    async def test_predict_endpoint_rejects_incomplete_payload(self) -> None:
        response = await self.request(
            "POST",
            "/predict",
            json={
                "cep": "04001000",
                "area_do_terreno_m2": 120,
            },
        )

        self.assertEqual(response.status_code, 422)
        payload = response.json()
        self.assertEqual(payload["error"]["type"], "validation_error")
        self.assertEqual(
            payload["error"]["message"],
            "Preencha todos os campos obrigatorios com valores validos.",
        )

    async def test_chat_endpoint_returns_structured_response(self) -> None:
        response = await self.request("POST", "/chat", json={"message": "Quais fatores influenciam o preco?"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "Resposta integrada do agent.")
        self.assertEqual(payload["tools_used"], ["rag_search"])
        self.assertEqual(payload["metadata"]["chunks_retrieved"], 1)
        self.assertIsNotNone(payload["metadata"]["request_id"])

    async def test_metrics_endpoint_is_available_for_prometheus(self) -> None:
        await self.request(
            "POST",
            "/predict",
            json={
                "cep": "04001000",
                "area_do_terreno_m2": 120,
                "ano": 2024,
                "mes": 1,
            },
        )
        await self.request("POST", "/chat", json={"message": "Quais fatores influenciam o preco?"})

        response = await self.request("GET", "/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertTrue("http_requests_total" in response.text or "http_request" in response.text)
        self.assertIn("business_prediction_requests_total", response.text)
        self.assertIn("business_prediction_value_brl_bucket", response.text)
        self.assertIn("business_chat_requests_total", response.text)
        self.assertIn("business_chat_latency_seconds_bucket", response.text)
        self.assertIn("business_chat_tools_used_total", response.text)


if __name__ == "__main__":
    unittest.main()
