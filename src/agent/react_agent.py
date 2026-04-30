import json
import logging
import os
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm import get_llm
from src.agent.prompts import REACT_SYSTEM_PROMPT, build_react_user_prompt
from src.agent.tools import TOOLS, ToolResult


logger = logging.getLogger("precificador.agent.react")


@dataclass
class AgentStep:
    action: str
    action_input: Any
    observation: str
    duration_ms: int | None = None
    error: str | None = None


@dataclass
class AgentResponse:
    answer: str
    tools_used: list[str]
    steps: list[AgentStep] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    response_time_seconds: float = 0.0
    chunks_retrieved: int = 0


class ReActAgent:
    def __init__(self, max_steps: int | None = None, llm: Any | None = None) -> None:
        self.max_steps = max_steps or self._read_max_steps()
        self.llm = llm or get_llm()

    def run(self, message: str) -> AgentResponse:
        started_at = perf_counter()
        scratchpad = "Nenhuma observacao ainda."
        steps: list[AgentStep] = []
        chunks_retrieved = 0

        logger.info(
            "Agent ReAct iniciado max_steps=%s message=%s",
            self.max_steps,
            self._truncate_for_log(message),
        )

        for step_number in range(1, self.max_steps + 1):
            try:
                decision = self._choose_next_action(message=message, scratchpad=scratchpad, steps=steps)
            except Exception as exc:
                logger.exception("Falha na decisao do agente; ativando fallback heuristico.")
                decision = self._fallback_decision(message=message, steps=steps, error=exc)

            action = decision.get("action", "final")
            action_input = decision.get("action_input", "")
            final_answer = decision.get("final_answer", "")

            if action == "final":
                answer = final_answer or self._fallback_answer(steps)
                elapsed = perf_counter() - started_at
                logger.info(
                    "Agent ReAct finalizado steps=%s tools=%s tempo=%.3fs",
                    len(steps),
                    [step.action for step in steps],
                    elapsed,
                )
                return self._build_response(
                    answer=answer,
                    steps=steps,
                    elapsed=elapsed,
                    chunks_retrieved=chunks_retrieved,
                )

            tool = TOOLS.get(action)
            if tool is None:
                logger.warning("Tool desconhecida selecionada action=%s step=%s", action, step_number)
                steps.append(
                    AgentStep(
                        action=action,
                        action_input=action_input,
                        observation=f"Tool desconhecida: {action}",
                        error="unknown_tool",
                    )
                )
                scratchpad = self._render_scratchpad(steps)
                continue

            try:
                tool_started_at = perf_counter()
                logger.info("Executando tool action=%s step=%s", action, step_number)
                tool_result = tool(action_input)
                tool_duration_ms = int((perf_counter() - tool_started_at) * 1000)
            except Exception as exc:
                logger.exception("Falha na execucao da tool %s", action)
                steps.append(
                    AgentStep(
                        action=action,
                        action_input=action_input,
                        observation=f"Erro ao executar a tool {action}: {exc}",
                        error=str(exc),
                    )
                )
                scratchpad = self._render_scratchpad(steps)
                continue

            if not tool_result.content:
                logger.warning("Tool retornou conteudo vazio action=%s step=%s", action, step_number)
                tool_result = ToolResult(
                    tool_name=action,
                    content=f"A tool {action} nao retornou conteudo suficiente para responder.",
                    metadata=tool_result.metadata,
                )

            chunks_retrieved += int(tool_result.metadata.get("chunks_retrieved", 0))
            logger.info(
                "Tool concluida action=%s step=%s duration_ms=%s chunks=%s",
                action,
                step_number,
                tool_duration_ms,
                tool_result.metadata.get("chunks_retrieved", 0),
            )
            steps.append(
                AgentStep(
                    action=action,
                    action_input=action_input,
                    observation=tool_result.content,
                    duration_ms=tool_duration_ms,
                )
            )

            final_answer = self._final_answer_from_tool_result(action=action, tool_result=tool_result)
            if final_answer:
                elapsed = perf_counter() - started_at
                logger.info(
                    "Agent ReAct gerou resposta final steps=%s tools=%s tempo=%.3fs",
                    len(steps),
                    [step.action for step in steps],
                    elapsed,
                )
                return self._build_response(
                    answer=final_answer,
                    steps=steps,
                    elapsed=elapsed,
                    chunks_retrieved=chunks_retrieved,
                )

            scratchpad = self._render_scratchpad(steps)

        elapsed = perf_counter() - started_at
        logger.warning("Agent ReAct atingiu max_steps=%s tools=%s", self.max_steps, [step.action for step in steps])
        return self._build_response(
            answer=(
                "Atingi o limite de passos do agente antes de concluir com seguranca. "
                f"{self._fallback_answer(steps)}"
            ),
            steps=steps,
            elapsed=elapsed,
            chunks_retrieved=chunks_retrieved,
        )

    def _choose_next_action(self, message: str, scratchpad: str, steps: list[AgentStep]) -> dict[str, Any]:
        if not steps:
            heuristic_decision = self._intent_based_decision(message)
            if heuristic_decision is not None:
                logger.info("Tool selecionada por heuristica action=%s", heuristic_decision["action"])
                return heuristic_decision

        return self._decide_next_action(message=message, scratchpad=scratchpad)

    def _decide_next_action(self, message: str, scratchpad: str) -> dict[str, Any]:
        prompt = build_react_user_prompt(message=message, scratchpad=scratchpad)
        response = self.llm.invoke(
            [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return self._parse_json_response(response.content)

    def _intent_based_decision(self, message: str) -> dict[str, Any] | None:
        lowered = message.lower()

        if any(keyword in lowered for keyword in ["compare", "comparar", "comparacao", "comparação"]):
            return {
                "thought": "internal_intent_routing",
                "action": "region_comparer",
                "action_input": message,
                "final_answer": "",
            }

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
            "valor_m2",
        ]
        if any(keyword in lowered for keyword in price_keywords):
            return {
                "thought": "internal_intent_routing",
                "action": "price_estimator",
                "action_input": message,
                "final_answer": "",
            }

        rag_keywords = [
            "fatores",
            "influenciam",
            "explique",
            "explicar",
            "como funciona",
            "documento",
            "contexto",
            "rag",
        ]
        if any(keyword in lowered for keyword in rag_keywords):
            return {
                "thought": "internal_intent_routing",
                "action": "rag_search",
                "action_input": {"query": message},
                "final_answer": "",
            }

        return None

    def _parse_json_response(self, raw_response: str) -> dict[str, Any]:
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        logger.warning("Resposta do agente nao veio em JSON valido; finalizando em modo fallback.")
        return {
            "thought": "fallback",
            "action": "final",
            "action_input": "",
            "final_answer": raw_response.strip(),
        }

    def _render_scratchpad(self, steps: list[AgentStep]) -> str:
        rendered_steps = []
        for index, step in enumerate(steps, start=1):
            rendered_steps.append(
                f"Passo {index}\n"
                f"Action: {step.action}\n"
                f"Action Input: {step.action_input}\n"
                f"Observation: {step.observation}\n"
            )
        return "\n".join(rendered_steps) if rendered_steps else "Nenhuma observacao ainda."

    def _fallback_answer(self, steps: list[AgentStep]) -> str:
        if not steps:
            return "Nao consegui reunir contexto suficiente para responder."

        last_observation = steps[-1].observation
        if last_observation:
            return (
                "Consegui executar parte do fluxo, mas nao obtive uma finalizacao estruturada do agente. "
                f"Ultima observacao: {last_observation}"
            )

        return "O agente nao conseguiu concluir a resposta com seguranca."

    def _final_answer_from_tool_result(self, action: str, tool_result: ToolResult) -> str:
        if action == "rag_search":
            try:
                payload = json.loads(tool_result.content)
                answer = payload.get("answer")
                if answer:
                    return str(answer)
            except json.JSONDecodeError:
                pass
            return tool_result.content

        if action == "price_estimator":
            missing_fields = tool_result.metadata.get("missing_fields") or []
            if missing_fields:
                return (
                    "Ainda nao consigo estimar o preco com seguranca porque faltam campos obrigatorios "
                    f"para o modelo: {', '.join(missing_fields)}."
                )

            try:
                payload = json.loads(tool_result.content)
            except json.JSONDecodeError:
                return tool_result.content

            if "valor_estimado" in payload:
                return (
                    "Estimativa gerada pelo modelo de predicao: "
                    f"{payload.get('unidade', 'R$')} {float(payload['valor_estimado']):,.2f}. "
                    f"Versao do modelo: {payload.get('versao_modelo', 'desconhecida')}."
                )
            return tool_result.content

        if action == "region_comparer":
            try:
                payload = json.loads(tool_result.content)
            except json.JSONDecodeError:
                return tool_result.content

            if "region_a" in payload and "region_b" in payload:
                region_a = payload["region_a"]
                region_b = payload["region_b"]
                metric = payload.get("metric", "valor_m2")
                return (
                    f"Comparacao por {metric}: {region_a['bairro']} tem media "
                    f"{region_a['media']:.2f} com {region_a['amostra']} registros; "
                    f"{region_b['bairro']} tem media {region_b['media']:.2f} "
                    f"com {region_b['amostra']} registros. "
                    f"Maior media: {payload.get('higher_region')}."
                )
            return tool_result.content

        return tool_result.content

    def _build_response(
        self,
        answer: str,
        steps: list[AgentStep],
        elapsed: float,
        chunks_retrieved: int,
    ) -> AgentResponse:
        return AgentResponse(
            answer=answer,
            tools_used=[step.action for step in steps],
            steps=steps,
            observations=[step.observation for step in steps],
            metadata={
                "agent_type": "react",
                "max_steps": self.max_steps,
                "steps_executed": len(steps),
                "provider": os.getenv("LLM_PROVIDER", "groq").strip().lower(),
                "model": self._current_model_name(),
            },
            response_time_seconds=elapsed,
            chunks_retrieved=chunks_retrieved,
        )

    def _fallback_decision(self, message: str, steps: list[AgentStep], error: Exception) -> dict[str, Any]:
        lowered = message.lower()

        if steps:
            return {
                "thought": f"fallback apos erro: {error}",
                "action": "final",
                "action_input": "",
                "final_answer": self._fallback_answer(steps),
            }

        if any(
            keyword in lowered
            for keyword in [
                "estime",
                "estimar",
                "quanto vale",
                "quanto custa",
                "qual o preco",
                "qual o preço",
                "preco para bairro",
                "valor_m2",
            ]
        ):
            return {
                "thought": f"fallback apos erro: {error}",
                "action": "price_estimator",
                "action_input": message,
                "final_answer": "",
            }

        if any(keyword in lowered for keyword in ["compare", "comparar", "comparacao", "comparação"]):
            return {
                "thought": f"fallback apos erro: {error}",
                "action": "region_comparer",
                "action_input": message,
                "final_answer": "",
            }

        return {
            "thought": f"fallback apos erro: {error}",
            "action": "rag_search",
            "action_input": {"query": message},
            "final_answer": "",
        }

    def _read_max_steps(self) -> int:
        try:
            max_steps = int(os.getenv("AGENT_MAX_STEPS", "3"))
        except ValueError:
            logger.warning("AGENT_MAX_STEPS invalido; usando default 3.")
            return 3
        return max(1, min(max_steps, 10))

    def _current_model_name(self) -> str | None:
        provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        if provider == "groq":
            return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if provider == "gemini":
            return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        return None

    def _truncate_for_log(self, value: str, max_length: int = 160) -> str:
        return value if len(value) <= max_length else f"{value[:max_length]}..."
