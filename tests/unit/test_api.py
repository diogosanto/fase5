import api.main as api_main
from api.main import PredictRequest, health, predict


class FakeModel:
    def predict(self, dataframe):
        assert list(dataframe.columns) == ["bairro", "cep_prefixo", "area_do_terreno_m2", "ano", "mes"]
        return [123456.0]


def test_health():
    response = health()
    assert response["status"] == "ok"


def test_predict():
    api_main.model = FakeModel()
    response = predict(
        PredictRequest(
            bairro="CENTRO",
            cep_prefixo="01001",
            area_do_terreno_m2=100,
            ano=2024,
            mes=1,
        )
    )

    assert "valor_estimado" in response
