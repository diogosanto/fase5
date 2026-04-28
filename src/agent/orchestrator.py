import logging
from dataclasses import asdict

from src.agent.react_agent import AgentResponse, ReActAgent


logger = logging.getLogger("precificador.agent.orchestrator")


class AgentOrchestrator:
    def __init__(self) -> None:
        self.agent = ReActAgent()

    def chat(self, message: str) -> dict:
        logger.info("Pergunta recebida no agent: %s", message)
        try:
            response: AgentResponse = self.agent.run(message)
            logger.info(
                "Resposta gerada com tools=%s tempo=%.3fs chunks=%s",
                response.tools_used,
                response.response_time_seconds,
                response.chunks_retrieved,
            )
            return {
                "answer": response.answer,
                "tools_used": response.tools_used,
                "response_time_seconds": response.response_time_seconds,
                "chunks_retrieved": response.chunks_retrieved,
                "steps": [asdict(step) for step in response.steps],
            }
        except Exception as exc:
            logger.exception("Falha nao tratada no fluxo do agent")
            return {
                "answer": "Ocorreu uma falha interna no fluxo do agent.",
                "tools_used": [],
                "response_time_seconds": 0.0,
                "chunks_retrieved": 0,
                "steps": [],
                "error": str(exc),
            }
