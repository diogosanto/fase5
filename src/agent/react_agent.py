import json
import logging
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm import get_llm
from src.agent.prompts import REACT_SYSTEM_PROMPT, build_react_user_prompt
from src.agent.tools import TOOLS


logger = logging.getLogger("precificador.agent.react")


@dataclass
class AgentStep:
    action: str
    action_input: Any
    observation: str


@dataclass
class AgentResponse:
    answer: str
    tools_used: list[str]
    steps: list[AgentStep] = field(default_factory=list)
    response_time_seconds: float = 0.0
    chunks_retrieved: int = 0


class ReActAgent:
    def __init__(self, max_steps: int = 4) -> None:
        self.max_steps = max_steps
        self.llm = get_llm()

    def run(self, message: str) -> AgentResponse:
        started_at = perf_counter()
        scratchpad = "Nenhuma observacao ainda."
        steps: list[AgentStep] = []
        chunks_retrieved = 0

        for _ in range(self.max_steps):
            decision = self._decide_next_action(message=message, scratchpad=scratchpad)
            action = decision.get("action", "final")
            action_input = decision.get("action_input", "")
            final_answer = decision.get("final_answer", "")

            if action == "final":
                answer = final_answer or self._fallback_answer(steps)
                elapsed = perf_counter() - started_at
                return AgentResponse(
                    answer=answer,
                    tools_used=[step.action for step in steps],
                    steps=steps,
                    response_time_seconds=elapsed,
                    chunks_retrieved=chunks_retrieved,
                )

            tool = TOOLS.get(action)
            if tool is None:
                steps.append(
                    AgentStep(
                        action=action,
                        action_input=action_input,
                        observation=f"Tool desconhecida: {action}",
                    )
                )
                scratchpad = self._render_scratchpad(steps)
                continue

            try:
                tool_result = tool(action_input)
            except Exception as exc:
                logger.exception("Falha na execucao da tool %s", action)
                steps.append(
                    AgentStep(
                        action=action,
                        action_input=action_input,
                        observation=f"Erro ao executar a tool {action}: {exc}",
                    )
                )
                scratchpad = self._render_scratchpad(steps)
                continue

            chunks_retrieved += int(tool_result.metadata.get("chunks_retrieved", 0))
            steps.append(
                AgentStep(
                    action=action,
                    action_input=action_input,
                    observation=tool_result.content,
                )
            )
            scratchpad = self._render_scratchpad(steps)

        elapsed = perf_counter() - started_at
        return AgentResponse(
            answer=self._fallback_answer(steps),
            tools_used=[step.action for step in steps],
            steps=steps,
            response_time_seconds=elapsed,
            chunks_retrieved=chunks_retrieved,
        )

    def _decide_next_action(self, message: str, scratchpad: str) -> dict[str, Any]:
        prompt = build_react_user_prompt(message=message, scratchpad=scratchpad)
        response = self.llm.invoke(
            [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        return self._parse_json_response(response.content)

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
