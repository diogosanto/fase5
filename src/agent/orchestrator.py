import logging
import json
import os
from dataclasses import asdict
from time import perf_counter
from typing import Any

from src.agent.llm import get_llm_call_count, reset_llm_call_count
from src.agent.react_agent import AgentResponse, ReActAgent
from src.agent.tools import price_estimator
from src.rag.rag_pipeline import rag_pipeline


logger = logging.getLogger("precificador.agent.orchestrator")


def _truncate_for_log(value: str, max_length: int = 160) -> str:
    return value if len(value) <= max_length else f"{value[:max_length]}..."


class AgentOrchestrator:
    def __init__(self) -> None:
        self.agent = ReActAgent()

    def chat(self, message: str, property_data: dict[str, Any] | None = None) -> dict:
        logger.info("Pergunta recebida no agent: %s", _truncate_for_log(message))
        reset_llm_call_count()
        try:
            if property_data and self._should_use_price_estimator(message):
                return self._run_price_estimator(message=message, property_data=property_data)

            if self._should_use_direct_rag(message):
                started_at = perf_counter()
                result = rag_pipeline(message)
                elapsed = perf_counter() - started_at
                logger.info(
                    "Resposta direta RAG tools=%s tempo=%.3fs chunks=%s llm_calls=%s",
                    ["rag_search"],
                    elapsed,
                    result.chunks_retrieved,
                    get_llm_call_count(),
                )
                return {
                    "answer": result.answer,
                    "tools_used": ["rag_search"],
                    "response_time_seconds": elapsed,
                    "chunks_retrieved": result.chunks_retrieved,
                    "llm_calls": get_llm_call_count(),
                    "observations": [result.to_dict()],
                    "metadata": self._agent_metadata(
                        response_time_seconds=elapsed,
                        chunks_retrieved=result.chunks_retrieved,
                        steps_executed=1,
                    ),
                    "steps": [
                        {
                            "action": "rag_search",
                            "action_input": {"query": message},
                            "observation": result.to_dict(),
                        }
                    ],
                }

            response: AgentResponse = self.agent.run(message)
            logger.info(
                "Resposta gerada com tools=%s tempo=%.3fs chunks=%s llm_calls=%s",
                response.tools_used,
                response.response_time_seconds,
                response.chunks_retrieved,
                get_llm_call_count(),
            )
            return {
                "answer": response.answer,
                "tools_used": response.tools_used,
                "response_time_seconds": response.response_time_seconds,
                "chunks_retrieved": response.chunks_retrieved,
                "llm_calls": get_llm_call_count(),
                "observations": response.observations,
                "metadata": response.metadata,
                "steps": [asdict(step) for step in response.steps],
            }
        except Exception as exc:
            logger.exception("Falha nao tratada no fluxo do agent")
            return {
                "answer": "Ocorreu uma falha interna no fluxo do agent.",
                "tools_used": [],
                "response_time_seconds": 0.0,
                "chunks_retrieved": 0,
                "llm_calls": get_llm_call_count(),
                "steps": [],
                "error": str(exc),
            }

    def _should_use_direct_rag(self, message: str) -> bool:
        lowered = message.lower()
        tool_keywords = [
            "estime",
            "estimar",
            "quanto vale",
            "quanto custa",
            "qual o preco",
            "qual o preço",
            "preco desse",
            "preço desse",
            "preco para bairro",
            "valor_m2",
            "compare",
            "comparar",
        ]
        return not any(keyword in lowered for keyword in tool_keywords)

    def _should_use_price_estimator(self, message: str) -> bool:
        lowered = message.lower()
        price_keywords = [
            "estime",
            "estimar",
            "quanto vale",
            "quanto custa",
            "qual o preco",
            "qual o preço",
            "preco desse",
            "preço desse",
            "valor estimado",
            "precificar",
        ]
        return any(keyword in lowered for keyword in price_keywords)

    def _run_price_estimator(self, message: str, property_data: dict[str, Any]) -> dict:
        started_at = perf_counter()
        estimator_payload = self._normalize_property_data_for_model(property_data)
        tool_result = price_estimator(estimator_payload)
        elapsed = perf_counter() - started_at
        answer = self._format_price_estimator_answer(tool_result.content, tool_result.metadata)

        logger.info(
            "Resposta direta price_estimator tempo=%.3fs llm_calls=%s missing_fields=%s",
            elapsed,
            get_llm_call_count(),
            tool_result.metadata.get("missing_fields"),
        )
        return {
            "answer": answer,
            "tools_used": ["price_estimator"],
            "response_time_seconds": elapsed,
            "chunks_retrieved": 0,
            "llm_calls": get_llm_call_count(),
            "observations": [tool_result.content],
            "metadata": self._agent_metadata(
                response_time_seconds=elapsed,
                chunks_retrieved=0,
                steps_executed=1,
            ),
            "steps": [
                {
                    "action": "price_estimator",
                    "action_input": estimator_payload,
                    "observation": tool_result.content,
                }
            ],
        }

    def _normalize_property_data_for_model(self, property_data: dict[str, Any]) -> dict[str, Any]:
        payload = dict(property_data)
        if "area_do_terreno_m2" not in payload and "area" in payload:
            payload["area_do_terreno_m2"] = payload["area"]
        return {
            key: payload.get(key)
            for key in [
                "bairro",
                "area_do_terreno_m2",
                "valor_m2",
                "ano_mes",
                "media_valor_cep",
            ]
            if payload.get(key) not in (None, "")
        }

    def _format_price_estimator_answer(self, tool_content: str, metadata: dict[str, Any]) -> str:
        missing_fields = metadata.get("missing_fields") or []
        if missing_fields:
            return (
                "Acionei o modelo de predicao, mas ainda nao e possivel estimar o preco porque faltam "
                f"campos obrigatorios: {', '.join(missing_fields)}. "
                "Envie esses campos em property_data para obter a estimativa pelo modelo."
            )

        try:
            payload = json.loads(tool_content)
        except json.JSONDecodeError:
            return (
                "Nao foi possivel gerar uma estimativa com o modelo de predicao. "
                f"Retorno da tool: {tool_content}"
            )

        if "valor_estimado" in payload:
            return (
                "Estimativa gerada pelo modelo de predicao: "
                f"{payload.get('unidade', 'R$')} {float(payload['valor_estimado']):,.2f}. "
                f"Versao do modelo: {payload.get('versao_modelo', 'desconhecida')}."
            )

        return str(tool_content)

    def _agent_metadata(
        self,
        response_time_seconds: float,
        chunks_retrieved: int,
        steps_executed: int,
    ) -> dict[str, Any]:
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        if provider == "groq":
            model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        elif provider == "gemini":
            model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        else:
            model = None

        return {
            "agent_type": "react",
            "max_steps": getattr(self.agent, "max_steps", self._default_max_steps()),
            "steps_executed": steps_executed,
            "provider": provider,
            "model": model,
            "response_time_seconds": response_time_seconds,
            "chunks_retrieved": chunks_retrieved,
        }

    def _default_max_steps(self) -> int:
        try:
            return int(os.getenv("AGENT_MAX_STEPS", "3"))
        except ValueError:
            return 3
