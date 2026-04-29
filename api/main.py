import logging
import os
import re

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

from src.agent.orchestrator import AgentOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("precificador.api")

API_VERSION = "1.0.0"


def get_model_version(path):
    match = re.search(r"model_(\d+\.\d+\.\d+\.\d+)", path)
    return match.group(1) if match else "unknown"


MODEL_PATH = "models/prod"
folders = [f for f in os.listdir(MODEL_PATH) if f.startswith("model_")]
if not folders:
    raise RuntimeError("Nenhum modelo versionado encontrado em models/prod")

MODEL_FOLDER = folders[0]
FULL_MODEL_PATH = os.path.join(MODEL_PATH, MODEL_FOLDER)
MODEL_VERSION = get_model_version(FULL_MODEL_PATH)
model = mlflow.pyfunc.load_model(FULL_MODEL_PATH)
agent_orchestrator = AgentOrchestrator()

app = FastAPI(
    title="API de Precificacao Imobiliaria",
    description="Servico de inferencia do modelo de valor venal de terrenos",
    version=API_VERSION,
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def root():
    return {
        "status": "API funcionando",
        "api_version": API_VERSION,
        "modelo": "prod",
        "versao_modelo": MODEL_VERSION,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": "online",
        "api_version": API_VERSION,
        "modelo": "prod",
        "versao_modelo": MODEL_VERSION,
    }


@app.post("/predict")
def predict(
    bairro: str,
    area_do_terreno_m2: float,
    valor_m2: float,
    ano_mes: int,
    media_valor_cep: float,
):
    df = pd.DataFrame(
        [
            {
                "bairro": bairro,
                "area_do_terreno_m2": area_do_terreno_m2,
                "valor_m2": valor_m2,
                "ano_mes": ano_mes,
                "media_valor_cep": media_valor_cep,
            }
        ]
    )
    pred = model.predict(df)[0]

    return {
        "valor_estimado": float(pred),
        "unidade": "R$",
        "versao_modelo": MODEL_VERSION,
    }


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info("Pergunta recebida no endpoint /chat")
    try:
        return agent_orchestrator.chat(request.message)
    except Exception as exc:
        logger.exception("Falha nao tratada no endpoint /chat")
        return {
            "answer": "O endpoint /chat encontrou uma falha interna.",
            "tools_used": [],
            "response_time_seconds": 0.0,
            "chunks_retrieved": 0,
            "steps": [],
            "error": str(exc),
        }
