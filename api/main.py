import asyncio
import logging
import os
import re
import uuid
from time import perf_counter

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from src.agent.orchestrator import AgentOrchestrator
from src.security.guardrails import validate_input, validate_output


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("precificador.api")

API_VERSION = "1.0.0"


def _get_or_create_counter(name: str, documentation: str, labelnames: list[str]):
    collector = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
    return collector or Counter(name, documentation, labelnames)


def _get_or_create_histogram(name: str, documentation: str, labelnames: list[str], buckets: tuple[float, ...]):
    collector = getattr(REGISTRY, "_names_to_collectors", {}).get(name)
    return collector or Histogram(name, documentation, labelnames, buckets=buckets)


PREDICTION_REQUESTS = _get_or_create_counter(
    "business_prediction_requests",
    "Total de requisicoes de predicao processadas pela API.",
    ["status", "model_version"],
)
PREDICTION_VALUE = _get_or_create_histogram(
    "business_prediction_value_brl",
    "Distribuicao dos valores estimados pelo modelo em BRL.",
    ["model_version"],
    (100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000),
)
CHAT_REQUESTS = _get_or_create_counter(
    "business_chat_requests",
    "Total de requisicoes de chat processadas pela API.",
    ["status", "provider", "model"],
)
CHAT_TOOLS_USED = _get_or_create_counter(
    "business_chat_tools_used",
    "Total de tools usadas pelo Agent durante requisicoes de chat.",
    ["tool_name"],
)
CHAT_LATENCY = _get_or_create_histogram(
    "business_chat_latency_seconds",
    "Latencia de ponta a ponta do endpoint de chat.",
    ["status", "provider", "model"],
    (0.1, 0.5, 1, 2, 5, 10, 30, 60),
)


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


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "message": "Preencha todos os campos obrigatorios com valores validos.",
                "details": _serialize_validation_errors(exc.errors()),
            }
        },
    )


def _serialize_validation_errors(errors: list[dict]) -> list[dict]:
    serialized = []
    for error in errors:
        clean_error = dict(error)
        if "ctx" in clean_error:
            clean_error["ctx"] = {
                key: str(value)
                for key, value in dict(clean_error["ctx"]).items()
            }
        serialized.append(clean_error)
    return serialized


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


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "bairro": "CENTRO",
                "cep_prefixo": "01001",
                "area_do_terreno_m2": 300,
                "ano": 2025,
                "mes": 3,
            }
        }
    )

    bairro: str = Field(..., min_length=1, max_length=120)
    cep_prefixo: str = Field(..., min_length=1, max_length=8)
    area_do_terreno_m2: float = Field(..., gt=0)
    ano: int | None = Field(default=None, ge=1900, le=2999)
    mes: int | None = Field(default=None, ge=1, le=12)
    ano_mes: int | None = Field(default=None, ge=190001, le=299912)

    @field_validator("bairro", "cep_prefixo")
    @classmethod
    def sanitize_text_fields(cls, value: str) -> str:
        sanitized = _sanitize_text(value)
        if not sanitized:
            raise ValueError("campo textual nao pode ser vazio")
        return sanitized

    @model_validator(mode="after")
    def validate_period_fields(self):
        if (self.ano is None or self.mes is None) and self.ano_mes is None:
            raise ValueError("Preencha ano e mes, ou informe ano_mes.")
        if self.ano_mes is not None and self.ano_mes % 100 not in range(1, 13):
            raise ValueError("ano_mes deve terminar com mes entre 01 e 12.")
        return self

    def to_model_frame(self) -> pd.DataFrame:
        ano = self.ano
        mes = self.mes
        if (ano is None or mes is None) and self.ano_mes is not None:
            ano = self.ano_mes // 100
            mes = self.ano_mes % 100

        payload = {
            "bairro": self.bairro.strip().upper(),
            "cep_prefixo": str(self.cep_prefixo).strip()[:5],
            "area_do_terreno_m2": float(self.area_do_terreno_m2),
            "ano": int(ano),
            "mes": int(mes),
        }
        validate_input(payload)
        return pd.DataFrame([payload])


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        df = request.to_model_frame()
        pred = model.predict(df)[0]
        pred = validate_output(float(pred))
    except Exception:
        PREDICTION_REQUESTS.labels(status="error", model_version=MODEL_VERSION).inc()
        raise

    PREDICTION_REQUESTS.labels(status="success", model_version=MODEL_VERSION).inc()
    PREDICTION_VALUE.labels(model_version=MODEL_VERSION).observe(pred)

    return {
        "valor_estimado": pred,
        "unidade": "R$",
        "versao_modelo": MODEL_VERSION,
    }


MAX_CHAT_MESSAGE_LENGTH = 1000
CHAT_TIMEOUT_SECONDS = int(os.getenv("CHAT_TIMEOUT_SECONDS", "30"))


class ChatPropertyData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "area": 60,
                "bairro": "MOOCA - SP",
                "cep_prefixo": "03110",
                "ano_mes": 202401,
            }
        }
    )

    area: float | None = Field(default=None, ge=0)
    area_do_terreno_m2: float | None = Field(default=None, ge=0)
    quartos: int | None = Field(default=None, ge=0, le=50)
    bairro: str | None = Field(default=None, min_length=1, max_length=120)
    cep_prefixo: str | None = Field(default=None, min_length=1, max_length=8)
    preco: float | None = Field(default=None, ge=0)
    ano: int | None = Field(default=None, ge=1900, le=2999)
    mes: int | None = Field(default=None, ge=1, le=12)
    ano_mes: int | None = Field(default=None, ge=190001, le=299912)

    @field_validator("bairro")
    @classmethod
    def sanitize_bairro(cls, value: str | None) -> str | None:
        if value is None:
            return value
        sanitized = _sanitize_text(value)
        if not sanitized:
            raise ValueError("bairro nao pode ser vazio")
        return sanitized


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Qual o preco desse apartamento?",
                "property_data": {
                    "area": 60,
                    "bairro": "MOOCA - SP",
                    "cep_prefixo": "03110",
                    "ano_mes": 202401,
                },
            }
        }
    )

    message: str = Field(..., min_length=1, max_length=MAX_CHAT_MESSAGE_LENGTH)
    property_data: ChatPropertyData | None = None

    @field_validator("message")
    @classmethod
    def sanitize_message(cls, value: str) -> str:
        sanitized = _sanitize_text(value)
        if not sanitized:
            raise ValueError("message nao pode ser vazio")
        if len(sanitized) > MAX_CHAT_MESSAGE_LENGTH:
            raise ValueError(f"message deve ter no maximo {MAX_CHAT_MESSAGE_LENGTH} caracteres")
        return sanitized


class ChatMetadata(BaseModel):
    provider: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    chunks_retrieved: int | None = None
    request_id: str | None = None
    llm_calls: int | None = None


class ChatResponse(BaseModel):
    answer: str | None
    tools_used: list[str] = Field(default_factory=list)
    metadata: ChatMetadata


def _sanitize_text(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return cleaned.strip()


def _truncate_for_log(value: str, max_length: int = 160) -> str:
    return value if len(value) <= max_length else f"{value[:max_length]}..."


def _provider_metadata() -> tuple[str | None, str | None]:
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider == "groq":
        return provider, os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    if provider == "gemini":
        return provider, os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    return provider, None


def _build_agent_message(request: ChatRequest) -> str:
    if request.property_data is None:
        return request.message

    if hasattr(request.property_data, "model_dump"):
        property_payload = request.property_data.model_dump(exclude_none=True)
    else:
        property_payload = request.property_data.dict(exclude_none=True)
    if not property_payload:
        return request.message

    return f"{request.message}\n\nDados do imovel informados pelo usuario: {property_payload}"


def _property_payload(request: ChatRequest) -> dict:
    if request.property_data is None:
        return {}
    if hasattr(request.property_data, "model_dump"):
        return request.property_data.model_dump(exclude_none=True)
    return request.property_data.dict(exclude_none=True)


def _error_response(status_code: int, error_type: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": error_type,
                "message": message,
                "request_id": request_id,
            }
        },
    )


def _looks_like_llm_provider_error(error_message: str) -> bool:
    lowered = error_message.lower()
    markers = ["groq", "llm", "provider", "api key", "rate limit", "quota", "timeout"]
    return any(marker in lowered for marker in markers)


def _observe_chat_metrics(
    *,
    status: str,
    provider: str | None,
    model_name: str | None,
    started_at: float,
    tools_used: list[str] | None = None,
) -> None:
    provider_label = provider or "unknown"
    model_label = model_name or "unknown"
    CHAT_REQUESTS.labels(status=status, provider=provider_label, model=model_label).inc()
    CHAT_LATENCY.labels(status=status, provider=provider_label, model=model_label).observe(perf_counter() - started_at)
    for tool_name in tools_used or []:
        CHAT_TOOLS_USED.labels(tool_name=str(tool_name)).inc()


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat com Agent/RAG",
    description=(
        "Executa o Agent ReAct com RAG e tools. "
        "Para perguntas conceituais, envie apenas `message`. "
        "Para estimar preco com o modelo de predicao, envie `property_data` com "
        "`bairro`, `cep_prefixo`, `area` ou `area_do_terreno_m2`, e `ano`/`mes` ou `ano_mes`."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "estimativa_preco": {
                            "summary": "Estimativa de preco pelo modelo",
                            "description": "Use este formato quando quiser que o Agent acione a tool price_estimator.",
                            "value": {
                                "message": "Qual o preco desse apartamento?",
                                "property_data": {
                                    "area": 60,
                                    "bairro": "MOOCA - SP",
                                    "cep_prefixo": "03110",
                                    "ano_mes": 202401,
                                },
                            },
                        },
                        "pergunta_rag": {
                            "summary": "Pergunta conceitual via RAG",
                            "description": "Use este formato para perguntas respondidas com base documental.",
                            "value": {
                                "message": "Quais fatores influenciam o preco de um imovel?",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def chat(request: ChatRequest):
    request_id = str(uuid.uuid4())
    started_at = perf_counter()
    provider, model_name = _provider_metadata()
    logger.info(
        "Inicio /chat request_id=%s provider=%s model=%s message=%s",
        request_id,
        provider,
        model_name,
        _truncate_for_log(request.message),
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent_orchestrator.chat,
                _build_agent_message(request),
                property_data=_property_payload(request),
            ),
            timeout=CHAT_TIMEOUT_SECONDS,
        )
        latency_ms = int((perf_counter() - started_at) * 1000)

        if result.get("error"):
            error_message = str(result.get("error", ""))
            if _looks_like_llm_provider_error(error_message):
                logger.warning(
                    "Erro de provedor LLM request_id=%s provider=%s model=%s",
                    request_id,
                    provider,
                    model_name,
                )
                _observe_chat_metrics(
                    status="llm_provider_error",
                    provider=provider,
                    model_name=model_name,
                    started_at=started_at,
                    tools_used=result.get("tools_used") or [],
                )
                return _error_response(
                    status_code=503,
                    error_type="llm_provider_error",
                message="Falha temporaria ao consultar o provedor LLM.",
                request_id=request_id,
            )
            logger.error("Erro interno do agent request_id=%s", request_id)
            _observe_chat_metrics(
                status="agent_internal_error",
                provider=provider,
                model_name=model_name,
                started_at=started_at,
                tools_used=result.get("tools_used") or [],
            )
            return _error_response(
                status_code=500,
                error_type="agent_internal_error",
                message="Falha interna ao executar o agent.",
                request_id=request_id,
            )

        answer = result.get("answer")
        if not answer:
            logger.error("Resposta vazia do agent request_id=%s", request_id)
            _observe_chat_metrics(
                status="empty_agent_response",
                provider=provider,
                model_name=model_name,
                started_at=started_at,
                tools_used=result.get("tools_used") or [],
            )
            return _error_response(
                status_code=500,
                error_type="empty_agent_response",
                message="O agent retornou uma resposta vazia.",
                request_id=request_id,
            )

        response = ChatResponse(
            answer=answer,
            tools_used=result.get("tools_used") or [],
            metadata=ChatMetadata(
                provider=provider,
                model=model_name,
                latency_ms=latency_ms,
                chunks_retrieved=result.get("chunks_retrieved"),
                request_id=request_id,
                llm_calls=result.get("llm_calls"),
            ),
        )
        logger.info(
            "Fim /chat request_id=%s latency_ms=%s tools=%s chunks=%s",
            request_id,
            latency_ms,
            response.tools_used,
            response.metadata.chunks_retrieved,
        )
        _observe_chat_metrics(
            status="success",
            provider=provider,
            model_name=model_name,
            started_at=started_at,
            tools_used=response.tools_used,
        )
        return response
    except TimeoutError:
        logger.warning("Timeout no /chat request_id=%s provider=%s model=%s", request_id, provider, model_name)
        _observe_chat_metrics(status="timeout", provider=provider, model_name=model_name, started_at=started_at)
        return _error_response(
            status_code=503,
            error_type="timeout",
            message="Tempo esgotado ao executar o agent.",
            request_id=request_id,
        )
    except Exception as exc:
        logger.exception("Falha nao tratada no endpoint /chat")
        if _looks_like_llm_provider_error(str(exc)):
            _observe_chat_metrics(
                status="llm_provider_error",
                provider=provider,
                model_name=model_name,
                started_at=started_at,
            )
            return _error_response(
                status_code=503,
                error_type="llm_provider_error",
                message="Falha temporaria ao consultar o provedor LLM.",
                request_id=request_id,
            )
        _observe_chat_metrics(status="internal_error", provider=provider, model_name=model_name, started_at=started_at)
        return _error_response(
            status_code=500,
            error_type="internal_error",
            message="Falha interna inesperada no endpoint /chat.",
            request_id=request_id,
        )
