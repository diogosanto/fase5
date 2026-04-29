import logging
from dataclasses import asdict
from time import perf_counter

from src.agent.llm import get_llm_call_count, reset_llm_call_count
from src.agent.react_agent import AgentResponse, ReActAgent
from src.rag.rag_pipeline import rag_pipeline


logger = logging.getLogger("precificador.agent.orchestrator")


class AgentOrchestrator:
    def __init__(self) -> None:
        self.agent = ReActAgent()

    def chat(self, message: str) -> dict:
        logger.info("Pergunta recebida no agent: %s", message)
        reset_llm_call_count()
        try:
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
            "preco para bairro",
            "valor_m2",
            "compare",
            "comparar",
        ]
        return not any(keyword in lowered for keyword in tool_keywords)
