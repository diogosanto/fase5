import json
import os
import unittest
from unittest.mock import patch

from src.agent.react_agent import ReActAgent
from src.agent.tools import TOOLS, ToolResult


class NeverCalledLLM:
    """Falha se o teste chamar LLM em intencoes que devem ser roteadas por heuristica."""

    def invoke(self, messages):
        raise AssertionError("A LLM nao deveria ser chamada neste teste.")


class UnknownToolLLM:
    """LLM fake que força uma tool inexistente para validar max_steps e erro controlado."""

    def invoke(self, messages):
        return type(
            "FakeLLMResponse",
            (),
            {
                "content": json.dumps(
                    {
                        "thought": "teste interno",
                        "action": "tool_inexistente",
                        "action_input": "",
                        "final_answer": "",
                    }
                )
            },
        )()


def fake_rag_search(action_input):
    return ToolResult(
        tool_name="rag_search",
        content=json.dumps({"answer": "Fatores como localizacao, area e infraestrutura influenciam o preco."}),
        metadata={"chunks_retrieved": 1, "sources": ["fake.md"]},
    )


def fake_price_estimator(action_input):
    return ToolResult(
        tool_name="price_estimator",
        content=json.dumps({"valor_estimado": 500000.0, "unidade": "R$", "versao_modelo": "model_test"}),
        metadata={"model_version": "model_test"},
    )


def fake_region_comparer(action_input):
    return ToolResult(
        tool_name="region_comparer",
        content=json.dumps(
            {
                "metric": "valor_m2",
                "region_a": {"bairro": "MOEMA", "media": 12000.0, "amostra": 10},
                "region_b": {"bairro": "PINHEIROS", "media": 14000.0, "amostra": 12},
                "higher_region": "PINHEIROS",
                "absolute_difference": 2000.0,
            }
        ),
        metadata={"metric": "valor_m2"},
    )


class ReActAgentTests(unittest.TestCase):
    def test_agent_has_at_least_three_registered_tools(self) -> None:
        self.assertGreaterEqual(len(TOOLS), 3)
        self.assertIn("rag_search", TOOLS)
        self.assertIn("price_estimator", TOOLS)
        self.assertIn("region_comparer", TOOLS)

    def test_generic_question_uses_rag_search(self) -> None:
        agent = ReActAgent(max_steps=3, llm=NeverCalledLLM())

        with patch("src.agent.react_agent.TOOLS", {"rag_search": fake_rag_search}):
            response = agent.run("Quais fatores influenciam o preco de um imovel?")

        self.assertEqual(response.tools_used, ["rag_search"])
        self.assertIn("localizacao", response.answer)
        self.assertEqual(response.metadata["agent_type"], "react")

    def test_property_price_question_uses_price_estimator(self) -> None:
        agent = ReActAgent(max_steps=3, llm=NeverCalledLLM())

        with patch("src.agent.react_agent.TOOLS", {"price_estimator": fake_price_estimator}):
            response = agent.run("Quanto vale um apartamento com bairro Moema, area 80, valor_m2 1500, ano_mes 202401 e media_valor_cep 2000?")

        self.assertEqual(response.tools_used, ["price_estimator"])
        self.assertIn("Estimativa gerada pelo modelo de predicao", response.answer)

    def test_comparison_question_uses_region_comparer(self) -> None:
        agent = ReActAgent(max_steps=3, llm=NeverCalledLLM())

        with patch("src.agent.react_agent.TOOLS", {"region_comparer": fake_region_comparer}):
            response = agent.run("Compare Moema e Pinheiros para compra de imovel")

        self.assertEqual(response.tools_used, ["region_comparer"])
        self.assertIn("PINHEIROS", response.answer)

    def test_agent_returns_answer_tools_observations_and_metadata(self) -> None:
        agent = ReActAgent(max_steps=3, llm=NeverCalledLLM())

        with patch("src.agent.react_agent.TOOLS", {"rag_search": fake_rag_search}):
            response = agent.run("Explique os fatores de preco")

        self.assertTrue(response.answer)
        self.assertEqual(response.tools_used, ["rag_search"])
        self.assertEqual(len(response.observations), 1)
        self.assertEqual(response.metadata["max_steps"], 3)
        self.assertEqual(response.metadata["steps_executed"], 1)

    def test_agent_respects_agent_max_steps(self) -> None:
        with patch.dict(os.environ, {"AGENT_MAX_STEPS": "2"}, clear=False):
            agent = ReActAgent(llm=UnknownToolLLM())

        with patch("src.agent.react_agent.TOOLS", {}):
            response = agent.run("Execute uma tarefa desconhecida")

        self.assertEqual(response.metadata["max_steps"], 2)
        self.assertEqual(response.metadata["steps_executed"], 2)
        self.assertIn("limite de passos", response.answer)


if __name__ == "__main__":
    unittest.main()
