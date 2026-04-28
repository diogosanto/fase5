import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from src.rag.rag_pipeline import rag_pipeline


logger = logging.getLogger("precificador.agent.tools")
PROCESSED_DATA_PATH = Path("data/processed/itbi_features_minimal.csv")
MODEL_ROOT = Path("models/prod")


@dataclass
class ToolResult:
    tool_name: str
    content: str
    metadata: dict[str, Any]


def _extract_json_payload(raw_input: Any) -> Any:
    if isinstance(raw_input, (dict, list)):
        return raw_input

    if raw_input is None:
        return {}

    text = str(raw_input).strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _normalize_region_name(region_name: str) -> str:
    return str(region_name).strip().upper()


def rag_search(action_input: Any) -> ToolResult:
    payload = _extract_json_payload(action_input)
    query = payload.get("query") if isinstance(payload, dict) else str(payload)
    if not query or not str(query).strip():
        return ToolResult(
            tool_name="rag_search",
            content="Nenhuma pergunta valida foi informada para a busca documental.",
            metadata={"chunks_retrieved": 0, "sources": []},
        )

    result = rag_pipeline(query=query, k=3)

    content = {
        "answer": result.answer,
        "sources": result.sources,
        "chunks": [
            {
                "source": chunk.source,
                "content": chunk.content,
            }
            for chunk in result.chunks
        ],
    }
    logger.info("Tool rag_search usada com %s chunks", result.chunks_retrieved)
    return ToolResult(
        tool_name="rag_search",
        content=json.dumps(content, ensure_ascii=False),
        metadata={
            "chunks_retrieved": result.chunks_retrieved,
            "sources": result.sources,
        },
    )


@lru_cache(maxsize=1)
def _load_prediction_model():
    model_dirs = sorted(MODEL_ROOT.glob("model_*"), reverse=True)
    if not model_dirs:
        raise FileNotFoundError("Nenhum modelo encontrado em models/prod")
    return mlflow.pyfunc.load_model(str(model_dirs[0])), model_dirs[0].name


def price_estimator(action_input: Any) -> ToolResult:
    payload = _extract_json_payload(action_input)
    if not isinstance(payload, dict):
        return ToolResult(
            tool_name="price_estimator",
            content="Entrada invalida. Envie um objeto JSON com os campos do modelo.",
            metadata={"missing_fields": ["bairro", "area_do_terreno_m2", "valor_m2", "ano_mes", "media_valor_cep"]},
        )

    required_fields = [
        "bairro",
        "area_do_terreno_m2",
        "valor_m2",
        "ano_mes",
        "media_valor_cep",
    ]
    missing_fields = [field for field in required_fields if field not in payload or payload[field] in (None, "")]
    if missing_fields:
        return ToolResult(
            tool_name="price_estimator",
            content=f"Campos obrigatorios ausentes para estimativa: {', '.join(missing_fields)}.",
            metadata={"missing_fields": missing_fields},
        )

    try:
        model, model_version = _load_prediction_model()
    except FileNotFoundError as exc:
        return ToolResult(
            tool_name="price_estimator",
            content=str(exc),
            metadata={"error": "model_not_found"},
        )

    frame = pd.DataFrame(
        [
            {
                "bairro": str(payload["bairro"]).strip().upper(),
                "area_do_terreno_m2": float(payload["area_do_terreno_m2"]),
                "valor_m2": float(payload["valor_m2"]),
                "ano_mes": int(payload["ano_mes"]),
                "media_valor_cep": float(payload["media_valor_cep"]),
            }
        ]
    )
    prediction = float(model.predict(frame)[0])
    content = {
        "valor_estimado": prediction,
        "unidade": "R$",
        "versao_modelo": model_version,
    }
    logger.info("Tool price_estimator usada com modelo %s", model_version)
    return ToolResult(
        tool_name="price_estimator",
        content=json.dumps(content, ensure_ascii=False),
        metadata={"model_version": model_version},
    )


@lru_cache(maxsize=1)
def _load_region_dataframe() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Base processada nao encontrada: {PROCESSED_DATA_PATH}")

    dataframe = pd.read_csv(PROCESSED_DATA_PATH, sep=";")
    dataframe["bairro"] = dataframe["bairro"].astype(str).str.strip().str.upper()
    return dataframe


def region_comparer(action_input: Any) -> ToolResult:
    payload = _extract_json_payload(action_input)
    if not isinstance(payload, dict):
        text = str(payload)
        matches = re.findall(r"[A-Za-zÀ-ÿ]+", text)
        if len(matches) >= 2:
            payload = {"region_a": matches[0], "region_b": matches[1]}
        else:
            payload = {}

    region_a = payload.get("region_a")
    region_b = payload.get("region_b")
    metric = payload.get("metric", "valor_m2")

    if not region_a or not region_b:
        return ToolResult(
            tool_name="region_comparer",
            content="Para comparar regioes, informe `region_a` e `region_b`.",
            metadata={"missing_fields": ["region_a", "region_b"]},
        )

    try:
        dataframe = _load_region_dataframe()
    except FileNotFoundError as exc:
        return ToolResult(
            tool_name="region_comparer",
            content=str(exc),
            metadata={"error": "dataset_not_found"},
        )
    region_a_normalized = _normalize_region_name(region_a)
    region_b_normalized = _normalize_region_name(region_b)

    if metric not in dataframe.columns:
        metric = "valor_m2"

    stats = (
        dataframe[dataframe["bairro"].isin([region_a_normalized, region_b_normalized])]
        .groupby("bairro")
        .agg(
            media_metric=(metric, "mean"),
            mediana_metric=(metric, "median"),
            quantidade_imoveis=(metric, "count"),
        )
        .reset_index()
    )

    available_regions = stats["bairro"].tolist()
    missing_regions = [region for region in [region_a_normalized, region_b_normalized] if region not in available_regions]
    if missing_regions:
        return ToolResult(
            tool_name="region_comparer",
            content=f"Regioes nao encontradas na base: {', '.join(missing_regions)}.",
            metadata={"missing_regions": missing_regions},
        )

    region_a_stats = stats[stats["bairro"] == region_a_normalized].iloc[0]
    region_b_stats = stats[stats["bairro"] == region_b_normalized].iloc[0]
    difference = float(region_a_stats["media_metric"] - region_b_stats["media_metric"])
    higher_region = region_a_normalized if difference >= 0 else region_b_normalized

    content = {
        "metric": metric,
        "region_a": {
            "bairro": region_a_normalized,
            "media": float(region_a_stats["media_metric"]),
            "mediana": float(region_a_stats["mediana_metric"]),
            "amostra": int(region_a_stats["quantidade_imoveis"]),
        },
        "region_b": {
            "bairro": region_b_normalized,
            "media": float(region_b_stats["media_metric"]),
            "mediana": float(region_b_stats["mediana_metric"]),
            "amostra": int(region_b_stats["quantidade_imoveis"]),
        },
        "higher_region": higher_region,
        "absolute_difference": abs(difference),
    }
    logger.info("Tool region_comparer usada para %s vs %s", region_a_normalized, region_b_normalized)
    return ToolResult(
        tool_name="region_comparer",
        content=json.dumps(content, ensure_ascii=False),
        metadata={"metric": metric},
    )


TOOLS = {
    "rag_search": rag_search,
    "price_estimator": price_estimator,
    "region_comparer": region_comparer,
}
