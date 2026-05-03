import unittest

import httpx

import api.main as api_main
from api.main import PredictRequest, health, predict


class FakeModel:
    def predict(self, dataframe):
        assert list(dataframe.columns) == ["cep", "area_do_terreno_m2", "ano", "mes"]
        return [123456.0]


def test_health():
    response = health()
    assert response["status"] == "ok"


def test_predict():
    api_main.model = FakeModel()
    response = predict(
        PredictRequest(
            cep="01001000",
            area_do_terreno_m2=100,
            ano=2024,
            mes=1,
        )
    )

    assert "valor_estimado" in response


class PredictEndpointValidationTests(unittest.IsolatedAsyncioTestCase):
    async def test_predict_rejects_missing_period_with_simple_message(self):
        api_main.model = FakeModel()
        transport = httpx.ASGITransport(app=api_main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/predict",
                json={
                    "cep": "04001000",
                    "area_do_terreno_m2": 120,
                },
            )

        assert response.status_code == 422
        payload = response.json()
        assert payload["error"]["type"] == "validation_error"
        assert payload["error"]["message"] == "Preencha todos os campos obrigatorios com valores validos."

    async def test_predict_rejects_invalid_types_with_simple_message(self):
        api_main.model = FakeModel()
        transport = httpx.ASGITransport(app=api_main.app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/predict",
                json={
                    "cep": "04001000",
                    "area_do_terreno_m2": "abc",
                    "ano": 2024,
                    "mes": 1,
                },
            )

        assert response.status_code == 422
        assert response.json()["error"]["type"] == "validation_error"
