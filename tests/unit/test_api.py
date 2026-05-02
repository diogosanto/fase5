from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    response = client.post("/predict", params={
        "bairro": "CENTRO",
        "area_do_terreno_m2": 100,
        "valor_m2": 2000,
        "ano_mes": 202401,
        "media_valor_cep": 300000
    })

    assert response.status_code == 200
    assert "valor_estimado" in response.json()