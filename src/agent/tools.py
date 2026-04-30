import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

logger = logging.getLogger("precificador.agent.tools")
PROCESSED_DATA_PATH = Path("data/processed/itbi_features_minimal.csv")
MODEL_CANDIDATE_DIRS = [
    Path("models/prod"),
    Path("models/dev"),
]
REQUIRED_MODEL_FIELDS = [
    "bairro",
    "area_do_terreno_m2",
    "valor_m2",
    "ano_mes",
    "media_valor_cep",
]


@dataclass
class ToolResult:
    tool_name: str
    content: str
    metadata: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "success"))

    def to_dict(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.content)
        except json.JSONDecodeError:
            payload = {"content": self.content}
        return {
            "tool": self.tool_name,
            "status": self.status,
            "content": payload,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    function: Any


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


def _json_tool_result(tool_name: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> ToolResult:
    payload = {"tool": tool_name, **payload}
    status = str(payload.get("status", "success"))
    result_metadata = {"status": status, **(metadata or {})}
    return ToolResult(
        tool_name=tool_name,
        content=json.dumps(payload, ensure_ascii=False),
        metadata=result_metadata,
    )


def _safe_float(value: Any) -> float:
    return float(str(value).replace(",", "."))


def _safe_int(value: Any) -> int:
    return int(float(str(value).replace(",", ".")))


def _normalize_prediction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "area_do_terreno_m2" not in normalized and "area" in normalized:
        normalized["area_do_terreno_m2"] = normalized["area"]
    return normalized


def _retrieve_context(query: str):
    from src.rag.rag_pipeline import retrieve_context

    return retrieve_context(query=query)


def _parse_prediction_payload_from_text(text: str) -> dict[str, Any]:
    bairro_match = re.search(
        r"bairro\s+([a-zA-ZÀ-ÿ\s]+?)(?:,| com| area| área| valor_m2| ano_mes| media_valor_cep|$)",
        text,
        flags=re.IGNORECASE,
    )
    area_match = re.search(r"area(?:_do_terreno_m2)?\s+(\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)
    valor_match = re.search(r"valor_m2\s+(\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)
    ano_mes_match = re.search(r"ano_mes\s+(\d{6})", text, flags=re.IGNORECASE)
    media_match = re.search(r"media_valor_cep\s+(\d+(?:[.,]\d+)?)", text, flags=re.IGNORECASE)

    payload: dict[str, Any] = {}
    if bairro_match:
        payload["bairro"] = bairro_match.group(1).strip(" ,.")
    if area_match:
        payload["area_do_terreno_m2"] = area_match.group(1).replace(",", ".")
    if valor_match:
        payload["valor_m2"] = valor_match.group(1).replace(",", ".")
    if ano_mes_match:
        payload["ano_mes"] = ano_mes_match.group(1)
    if media_match:
        payload["media_valor_cep"] = media_match.group(1).replace(",", ".")
    return payload


def _parse_region_payload_from_text(text: str) -> dict[str, Any]:
    compare_match = re.search(
        r"compare\s+([a-zA-ZÀ-ÿ\s]+?)\s+e\s+([a-zA-ZÀ-ÿ\s]+?)(?:\s+usando\s+([a-zA-Z0-9_]+)|[.?!,]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not compare_match:
        return {}

    payload = {
        "region_a": compare_match.group(1).strip(),
        "region_b": compare_match.group(2).strip(),
    }
    if compare_match.group(3):
        payload["metric"] = compare_match.group(3).strip()
    return payload


def rag_search(action_input: Any) -> ToolResult:
    started_at = perf_counter()
    payload = _extract_json_payload(action_input)
    query = payload.get("query") if isinstance(payload, dict) else str(payload)
    logger.info("Tool rag_search chamada query=%s", str(query)[:160])

    if not query or not str(query).strip():
        result = _json_tool_result(
            "rag_search",
            {
                "query": "",
                "context": "",
                "sources": [],
                "chunks_retrieved": 0,
                "status": "error",
                "error": "empty_query",
                "message": "Nenhuma pergunta valida foi informada para a busca documental.",
            },
            {"chunks_retrieved": 0, "sources": [], "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool rag_search status=error duration_ms=%s", result.metadata["duration_ms"])
        return result

    chunks = _retrieve_context(query=str(query).strip())
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))
    context = "\n\n".join(chunk.content for chunk in chunks)
    if not chunks:
        result = _json_tool_result(
            "rag_search",
            {
                "query": str(query).strip(),
                "context": "",
                "sources": [],
                "chunks_retrieved": 0,
                "status": "no_context",
                "message": "Nenhum contexto relevante foi encontrado nos documentos.",
            },
            {"chunks_retrieved": 0, "sources": [], "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.info("Tool rag_search status=no_context duration_ms=%s", result.metadata["duration_ms"])
        return result

    duration_ms = int((perf_counter() - started_at) * 1000)
    content = {
        "query": str(query).strip(),
        "context": context,
        "answer": "Contexto recuperado com sucesso. Use os chunks e fontes retornados para responder sem inventar dados.",
        "sources": sources,
        "chunks_retrieved": len(chunks),
        "chunks": [
            {
                "source": chunk.source,
                "content": chunk.content,
                "chunk_index": getattr(chunk, "chunk_index", None),
            }
            for chunk in chunks
        ],
        "status": "success",
    }
    result = _json_tool_result(
        "rag_search",
        content,
        {"chunks_retrieved": len(chunks), "sources": sources, "duration_ms": duration_ms},
    )
    logger.info("Tool rag_search status=success chunks=%s duration_ms=%s", len(chunks), duration_ms)
    return result


@lru_cache(maxsize=1)
def _load_prediction_model():
    import mlflow.pyfunc

    for model_root in MODEL_CANDIDATE_DIRS:
        if not model_root.exists():
            continue
        model_dirs = sorted(model_root.glob("model_*"), reverse=True)
        if not model_dirs:
            continue

        selected_model = model_dirs[0]
        return mlflow.pyfunc.load_model(str(selected_model)), selected_model.name

    searched_paths = ", ".join(str(path) for path in MODEL_CANDIDATE_DIRS)
    raise FileNotFoundError(f"Nenhum modelo encontrado em: {searched_paths}")


def price_estimator(action_input: Any) -> ToolResult:
    started_at = perf_counter()
    payload = _extract_json_payload(action_input)
    if isinstance(payload, str):
        payload = _parse_prediction_payload_from_text(payload)
    if not isinstance(payload, dict):
        result = _json_tool_result(
            "price_estimator",
            {
                "input": {},
                "status": "error",
                "error": "invalid_input",
                "message": "Entrada invalida. Envie um objeto JSON com os campos do modelo.",
                "required_fields": REQUIRED_MODEL_FIELDS,
            },
            {"missing_fields": REQUIRED_MODEL_FIELDS, "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool price_estimator status=error reason=invalid_input")
        return result

    payload = _normalize_prediction_payload(payload)
    logger.info("Tool price_estimator chamada campos=%s", sorted(payload.keys()))

    missing_fields = [field for field in REQUIRED_MODEL_FIELDS if field not in payload or payload[field] in (None, "")]
    if missing_fields:
        result = _json_tool_result(
            "price_estimator",
            {
                "input": payload,
                "status": "error",
                "error": "missing_fields",
                "message": f"Campos obrigatorios ausentes para estimativa: {', '.join(missing_fields)}.",
                "required_fields": REQUIRED_MODEL_FIELDS,
                "missing_fields": missing_fields,
            },
            {"missing_fields": missing_fields, "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool price_estimator status=error missing_fields=%s", missing_fields)
        return result

    try:
        model, model_version = _load_prediction_model()
    except FileNotFoundError as exc:
        result = _json_tool_result(
            "price_estimator",
            {
                "input": payload,
                "status": "error",
                "error": "model_not_found",
                "message": str(exc),
            },
            {"error": "model_not_found", "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool price_estimator status=error reason=model_not_found")
        return result

    try:
        model_input = {
            "bairro": str(payload["bairro"]).strip().upper(),
            "area_do_terreno_m2": _safe_float(payload["area_do_terreno_m2"]),
            "valor_m2": _safe_float(payload["valor_m2"]),
            "ano_mes": _safe_int(payload["ano_mes"]),
            "media_valor_cep": _safe_float(payload["media_valor_cep"]),
        }
        frame = pd.DataFrame([model_input])
        prediction = float(model.predict(frame)[0])
    except Exception as exc:
        result = _json_tool_result(
            "price_estimator",
            {
                "input": payload,
                "status": "error",
                "error": "prediction_error",
                "message": f"Falha controlada ao executar a predicao: {exc}",
            },
            {"error": "prediction_error", "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.exception("Tool price_estimator status=error reason=prediction_error")
        return result

    duration_ms = int((perf_counter() - started_at) * 1000)
    content = {
        "input": model_input,
        "estimated_price": prediction,
        "currency": "BRL",
        "valor_estimado": prediction,
        "unidade": "R$",
        "versao_modelo": model_version,
        "status": "success",
    }
    result = _json_tool_result(
        "price_estimator",
        content,
        {"model_version": model_version, "duration_ms": duration_ms},
    )
    logger.info("Tool price_estimator status=success model_version=%s duration_ms=%s", model_version, duration_ms)
    return result


@lru_cache(maxsize=1)
def _load_region_dataframe() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"Base processada nao encontrada: {PROCESSED_DATA_PATH}")

    dataframe = pd.read_csv(PROCESSED_DATA_PATH, sep=";")
    dataframe["bairro"] = dataframe["bairro"].astype(str).str.strip().str.upper()
    return dataframe


def region_comparer(action_input: Any) -> ToolResult:
    started_at = perf_counter()
    payload = _extract_json_payload(action_input)
    if isinstance(payload, str):
        payload = _parse_region_payload_from_text(payload)
    if not isinstance(payload, dict):
        payload = {}

    region_a = payload.get("region_a")
    region_b = payload.get("region_b")
    metric = payload.get("metric", "valor_m2")
    logger.info("Tool region_comparer chamada region_a=%s region_b=%s metric=%s", region_a, region_b, metric)

    if not region_a or not region_b:
        result = _json_tool_result(
            "region_comparer",
            {
                "regions": [],
                "status": "error",
                "error": "missing_fields",
                "message": "Para comparar regioes, informe `region_a` e `region_b`.",
                "missing_fields": ["region_a", "region_b"],
            },
            {"missing_fields": ["region_a", "region_b"], "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool region_comparer status=error missing_fields=%s", ["region_a", "region_b"])
        return result

    try:
        dataframe = _load_region_dataframe()
    except FileNotFoundError as exc:
        result = _json_tool_result(
            "region_comparer",
            {
                "regions": [region_a, region_b],
                "status": "error",
                "error": "dataset_not_found",
                "message": str(exc),
            },
            {"error": "dataset_not_found", "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool region_comparer status=error reason=dataset_not_found")
        return result

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
        result = _json_tool_result(
            "region_comparer",
            {
                "regions": [region_a_normalized, region_b_normalized],
                "status": "error",
                "error": "regions_not_found",
                "message": f"Regioes nao encontradas na base: {', '.join(missing_regions)}.",
                "missing_regions": missing_regions,
            },
            {"missing_regions": missing_regions, "duration_ms": int((perf_counter() - started_at) * 1000)},
        )
        logger.warning("Tool region_comparer status=error missing_regions=%s", missing_regions)
        return result

    region_a_stats = stats[stats["bairro"] == region_a_normalized].iloc[0]
    region_b_stats = stats[stats["bairro"] == region_b_normalized].iloc[0]
    difference = float(region_a_stats["media_metric"] - region_b_stats["media_metric"])
    higher_region = region_a_normalized if difference >= 0 else region_b_normalized
    metrics_payload = {
        region_a_normalized: _region_metrics(dataframe, region_a_normalized),
        region_b_normalized: _region_metrics(dataframe, region_b_normalized),
    }
    duration_ms = int((perf_counter() - started_at) * 1000)

    content = {
        "regions": [region_a_normalized, region_b_normalized],
        "metric": metric,
        "metrics": metrics_payload,
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
        "status": "success",
    }
    result = _json_tool_result(
        "region_comparer",
        content,
        {"metric": metric, "duration_ms": duration_ms},
    )
    logger.info("Tool region_comparer status=success regions=%s duration_ms=%s", content["regions"], duration_ms)
    return result


def _region_metrics(dataframe: pd.DataFrame, region_name: str) -> dict[str, Any]:
    region_df = dataframe[dataframe["bairro"] == region_name]
    price_column = "valor_venal_de_referencia" if "valor_venal_de_referencia" in region_df.columns else None

    metrics = {
        "count": int(len(region_df)),
        "avg_price_m2": float(region_df["valor_m2"].mean()) if "valor_m2" in region_df.columns else None,
        "median_price_m2": float(region_df["valor_m2"].median()) if "valor_m2" in region_df.columns else None,
    }
    if price_column:
        metrics.update(
            {
                "avg_price": float(region_df[price_column].mean()),
                "median_price": float(region_df[price_column].median()),
                "min_price": float(region_df[price_column].min()),
                "max_price": float(region_df[price_column].max()),
            }
        )
    return metrics


TOOLS = {
    "rag_search": rag_search,
    "price_estimator": price_estimator,
    "region_comparer": region_comparer,
}

TOOL_REGISTRY = {
    "rag_search": ToolSpec(
        name="rag_search",
        description="Busca contexto e fontes nos documentos indexados pelo RAG, sem chamar LLM diretamente.",
        function=rag_search,
    ),
    "price_estimator": ToolSpec(
        name="price_estimator",
        description="Executa inferencia no modelo existente usando bairro, area_do_terreno_m2, valor_m2, ano_mes e media_valor_cep.",
        function=price_estimator,
    ),
    "region_comparer": ToolSpec(
        name="region_comparer",
        description="Compara bairros/regioes usando metricas calculadas do dataset processado.",
        function=region_comparer,
    ),
}
