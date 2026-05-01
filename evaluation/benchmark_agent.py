import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

from src.agent.orchestrator import AgentOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("evaluation.benchmark_agent")

GOLDEN_SET_JSON = PROJECT_ROOT / "data" / "golden_set" / "golden_set.json"
GOLDEN_SET_JSONL = PROJECT_ROOT / "data" / "golden_set" / "real_estate_chat_golden_set.jsonl"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_agent_results.json"
RESULTS_CSV = RESULTS_DIR / "benchmark_agent_results.csv"

BENCHMARK_CONFIGS = [
    {"name": "top_k_1", "rag_top_k": 1},
    {"name": "top_k_3", "rag_top_k": 3},
    {"name": "top_k_5", "rag_top_k": 5},
]


def load_golden_set() -> list[dict[str, Any]]:
    if GOLDEN_SET_JSON.exists():
        return json.loads(GOLDEN_SET_JSON.read_text(encoding="utf-8"))

    if GOLDEN_SET_JSONL.exists():
        rows = []
        for line in GOLDEN_SET_JSONL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    raise FileNotFoundError(
        "Golden set nao encontrado. Esperado data/golden_set/golden_set.json "
        "ou data/golden_set/real_estate_chat_golden_set.jsonl."
    )


def limit_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_questions = int(os.getenv("BENCHMARK_MAX_QUESTIONS", "20"))
    if max_questions <= 0:
        return questions
    return questions[:max_questions]


def extract_sources(value: Any) -> list[str]:
    sources: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            raw_sources = item.get("sources")
            if isinstance(raw_sources, list):
                sources.extend(str(source) for source in raw_sources)
            for nested_value in item.values():
                visit(nested_value)
        elif isinstance(item, list):
            for nested_item in item:
                visit(nested_item)
        elif isinstance(item, str):
            try:
                visit(json.loads(item))
            except json.JSONDecodeError:
                return

    visit(value)
    return list(dict.fromkeys(sources))


def run_question(agent: AgentOrchestrator, item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    question_id = item.get("id")
    question = item["question"]
    logger.info("Executando pergunta config=%s id=%s", config["name"], question_id)

    started_at = perf_counter()
    try:
        response = agent.chat(question)
        latency_ms = int((perf_counter() - started_at) * 1000)
        error = response.get("error")
        answer = response.get("answer") or ""
        tools_used = response.get("tools_used") or []
        chunks_retrieved = int(response.get("chunks_retrieved") or 0)
        sources = extract_sources(response)

        return {
            "config_name": config["name"],
            "rag_top_k": config["rag_top_k"],
            "id": question_id,
            "category": item.get("category"),
            "question": question,
            "expected_tool": item.get("expected_tool"),
            "answer": answer,
            "tools_used": tools_used,
            "sources": sources,
            "chunks_retrieved": chunks_retrieved,
            "latency_ms": latency_ms,
            "response_length": len(answer),
            "success": bool(answer) and not error,
            "error": str(error) if error else None,
            "tool_match": item.get("expected_tool") in tools_used if item.get("expected_tool") else None,
        }
    except Exception as exc:
        latency_ms = int((perf_counter() - started_at) * 1000)
        logger.exception("Erro no benchmark config=%s id=%s", config["name"], question_id)
        return {
            "config_name": config["name"],
            "rag_top_k": config["rag_top_k"],
            "id": question_id,
            "category": item.get("category"),
            "question": question,
            "expected_tool": item.get("expected_tool"),
            "answer": "",
            "tools_used": [],
            "sources": [],
            "chunks_retrieved": 0,
            "latency_ms": latency_ms,
            "response_length": 0,
            "success": False,
            "error": str(exc),
            "tool_match": False if item.get("expected_tool") else None,
        }


def calculate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(runs)
    if total == 0:
        return {
            "total_questions": 0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "avg_tools_used": 0.0,
            "avg_chunks_retrieved": 0.0,
            "avg_response_length": 0.0,
            "source_presence_rate": 0.0,
            "tool_match_rate": 0.0,
        }

    successes = [run for run in runs if run["success"]]
    tool_match_values = [run["tool_match"] for run in runs if run["tool_match"] is not None]
    return {
        "total_questions": total,
        "success_rate": len(successes) / total,
        "error_rate": (total - len(successes)) / total,
        "avg_latency_ms": mean(run["latency_ms"] for run in runs),
        "avg_tools_used": mean(len(run["tools_used"]) for run in runs),
        "avg_chunks_retrieved": mean(run["chunks_retrieved"] for run in runs),
        "avg_response_length": mean(run["response_length"] for run in runs),
        "source_presence_rate": mean(1 if run["sources"] else 0 for run in runs),
        "tool_match_rate": mean(1 if value else 0 for value in tool_match_values) if tool_match_values else 0.0,
    }


def save_results(payload: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "config_name",
        "rag_top_k",
        "id",
        "category",
        "expected_tool",
        "tools_used",
        "chunks_retrieved",
        "latency_ms",
        "response_length",
        "success",
        "error",
        "sources",
        "question",
        "answer",
    ]
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for run in payload["runs"]:
            row = {field: run.get(field) for field in fieldnames}
            row["tools_used"] = "|".join(run.get("tools_used") or [])
            row["sources"] = "|".join(run.get("sources") or [])
            writer.writerow(row)


def run_benchmark() -> dict[str, Any]:
    os.environ.setdefault("LLM_MAX_TOKENS", "200")

    questions = limit_questions(load_golden_set())
    logger.info("Benchmark iniciado total_questions=%s configs=%s", len(questions), BENCHMARK_CONFIGS)

    all_runs: list[dict[str, Any]] = []
    config_summaries: list[dict[str, Any]] = []

    for config in BENCHMARK_CONFIGS:
        os.environ["RAG_TOP_K"] = str(config["rag_top_k"])
        logger.info("Executando configuracao %s RAG_TOP_K=%s", config["name"], config["rag_top_k"])

        agent = AgentOrchestrator()
        config_runs = [run_question(agent=agent, item=item, config=config) for item in questions]
        metrics = calculate_metrics(config_runs)
        config_summaries.append({**config, "metrics": metrics})
        all_runs.extend(config_runs)

        logger.info("Resumo config=%s metrics=%s", config["name"], metrics)

    payload = {
        "benchmark_name": "agent_rag_top_k_comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_set_path": str(GOLDEN_SET_JSON if GOLDEN_SET_JSON.exists() else GOLDEN_SET_JSONL),
        "max_questions": len(questions),
        "configs": config_summaries,
        "runs": all_runs,
    }
    save_results(payload)
    logger.info("Benchmark finalizado json=%s csv=%s", RESULTS_JSON, RESULTS_CSV)
    return payload


if __name__ == "__main__":
    run_benchmark()
