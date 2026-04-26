from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
import os
import re
from prometheus_fastapi_instrumentator import Instrumentator


# -----------------------------
# Versão da API (manual)
# -----------------------------
API_VERSION = "1.0.0"

# -----------------------------
# Função para extrair versão do modelo
# -----------------------------
def get_model_version(path):
    """
    Extrai a versão do modelo a partir do nome da pasta.
    Exemplo: models/prod/model_2026.04.25.1430
    """
    match = re.search(r"model_(\d+\.\d+\.\d+\.\d+)", path)
    return match.group(1) if match else "unknown"


# -----------------------------
# Carregar modelo de produção
# -----------------------------
MODEL_PATH = "models/prod"

# Encontrar a pasta do modelo dentro de prod
folders = [f for f in os.listdir(MODEL_PATH) if f.startswith("model_")]
if not folders:
    raise RuntimeError("Nenhum modelo versionado encontrado em models/prod")

# Assumir que existe apenas 1 modelo em prod (boa prática)
MODEL_FOLDER = folders[0]
FULL_MODEL_PATH = os.path.join(MODEL_PATH, MODEL_FOLDER)

# Extrair versão do modelo
MODEL_VERSION = get_model_version(FULL_MODEL_PATH)

# Carregar modelo
model = mlflow.pyfunc.load_model(FULL_MODEL_PATH)


# -----------------------------
# Inicializar API
# -----------------------------
app = FastAPI(
    title="API de Precificação Imobiliária",
    description="Serviço de inferência do modelo de valor venal de terrenos",
    version=API_VERSION
)

# 🔥 Instrumentação Prometheus
Instrumentator().instrument(app).expose(app)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "API funcionando",
        "api_version": API_VERSION,
        "modelo": "prod",
        "versao_modelo": MODEL_VERSION
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "online",
        "api_version": API_VERSION,
        "modelo": "prod",
        "versao_modelo": MODEL_VERSION
    }


@app.post("/predict")
def predict(
    bairro: str,
    area_do_terreno_m2: float,
    valor_m2: float,
    ano_mes: int,
    media_valor_cep: float
):
    df = pd.DataFrame([{
        "bairro": bairro,
        "area_do_terreno_m2": area_do_terreno_m2,
        "valor_m2": valor_m2,
        "ano_mes": ano_mes,
        "media_valor_cep": media_valor_cep
    }])

    pred = model.predict(df)[0]

    return {
        "valor_estimado": float(pred),
        "unidade": "R$",
        "versao_modelo": MODEL_VERSION
    }
