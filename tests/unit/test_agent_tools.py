"""
Testes unitarios das tools usadas pelo Agent ReAct.

Objetivo para avaliacao/banca:
- validar que as 3 tools obrigatorias estao registradas;
- garantir que `rag_search` retorna contexto, fontes e status;
- garantir que `price_estimator` valida os campos exigidos pelo modelo sem vazamento;
- garantir que `region_comparer` calcula metricas com dados processados e trata bairros inexistentes.

Os testes usam mocks/fakes para evitar carregar MLflow, chamar LLM ou depender de arquivos grandes.
Execute com:
    python -m unittest tests.unit.test_agent_tools
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.agent import tools


class FakeModel:
    """Modelo minimo para validar o adapter de predicao sem carregar artefatos MLflow reais."""

    def predict(self, dataframe):
        return [950000.0]


class AgentToolsTests(unittest.TestCase):
    """Cobre o contrato publico das tools consumidas pelo Agent."""

    def test_required_tools_are_registered_with_descriptions(self) -> None:
        """Garante que o Agent consiga descobrir nome, descricao e funcao das tools."""

        for tool_name in ["rag_search", "price_estimator", "region_comparer"]:
            self.assertIn(tool_name, tools.TOOLS)
            self.assertIn(tool_name, tools.TOOL_REGISTRY)
            self.assertEqual(tools.TOOL_REGISTRY[tool_name].name, tool_name)
            self.assertTrue(tools.TOOL_REGISTRY[tool_name].description)
            self.assertTrue(callable(tools.TOOL_REGISTRY[tool_name].function))

    def test_rag_search_returns_context_sources_and_status(self) -> None:
        """Valida que a busca documental retorna contexto auditavel e fontes reais."""

        fake_chunks = [
            SimpleNamespace(source="doc_1.md", content="Localizacao influencia o preco."),
            SimpleNamespace(source="doc_2.md", content="Area e infraestrutura tambem influenciam."),
        ]

        with patch("src.agent.tools._retrieve_context", return_value=fake_chunks):
            result = tools.rag_search({"query": "Quais fatores influenciam o preco?"})

        payload = json.loads(result.content)
        self.assertEqual(payload["tool"], "rag_search")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["chunks_retrieved"], 2)
        self.assertEqual(payload["sources"], ["doc_1.md", "doc_2.md"])
        self.assertIn("Localizacao", payload["context"])
        self.assertEqual(result.metadata["status"], "success")

    def test_rag_search_handles_no_context(self) -> None:
        """Garante resposta controlada quando o retriever nao encontra chunks relevantes."""

        with patch("src.agent.tools._retrieve_context", return_value=[]):
            result = tools.rag_search({"query": "Pergunta sem contexto"})

        payload = json.loads(result.content)
        self.assertEqual(payload["status"], "no_context")
        self.assertEqual(payload["chunks_retrieved"], 0)
        self.assertEqual(payload["sources"], [])

    def test_price_estimator_validates_required_fields(self) -> None:
        """Confirma que a tool nao inventa predicao quando faltam campos do modelo."""

        result = tools.price_estimator({"bairro": "Moema", "area": 80})

        payload = json.loads(result.content)
        self.assertEqual(payload["tool"], "price_estimator")
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "missing_fields")
        self.assertIn("cep_prefixo", payload["missing_fields"])
        self.assertIn("ano", payload["missing_fields"])
        self.assertIn("mes", payload["missing_fields"])

    def test_price_estimator_calls_existing_model_adapter(self) -> None:
        """Valida o contrato de entrada/saida da tool de preco usando um modelo fake."""

        with patch("src.agent.tools._load_prediction_model", return_value=(FakeModel(), "model_test")):
            result = tools.price_estimator(
                {
                    "bairro": "Moema",
                    "area": 80,
                    "cep_prefixo": "04001",
                    "ano_mes": 202401,
                }
            )

        payload = json.loads(result.content)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["estimated_price"], 950000.0)
        self.assertEqual(payload["currency"], "BRL")
        self.assertEqual(payload["input"]["area_do_terreno_m2"], 80.0)
        self.assertEqual(payload["input"]["cep_prefixo"], "04001")
        self.assertEqual(payload["input"]["ano"], 2024)
        self.assertEqual(payload["input"]["mes"], 1)
        self.assertEqual(payload["versao_modelo"], "model_test")

    def test_region_comparer_returns_metrics_for_existing_regions(self) -> None:
        """Confirma que regioes validas retornam metricas derivadas do dataset processado."""

        dataframe = pd.DataFrame(
            [
                {
                    "bairro": "MOEMA",
                    "valor_m2": 12000.0,
                    "valor_venal_de_referencia": 1000000.0,
                },
                {
                    "bairro": "MOEMA",
                    "valor_m2": 10000.0,
                    "valor_venal_de_referencia": 900000.0,
                },
                {
                    "bairro": "PINHEIROS",
                    "valor_m2": 14000.0,
                    "valor_venal_de_referencia": 1200000.0,
                },
                {
                    "bairro": "PINHEIROS",
                    "valor_m2": 13000.0,
                    "valor_venal_de_referencia": 1100000.0,
                },
            ]
        )

        with patch("src.agent.tools._load_region_dataframe", return_value=dataframe):
            result = tools.region_comparer({"region_a": "Moema", "region_b": "Pinheiros"})

        payload = json.loads(result.content)
        self.assertEqual(payload["tool"], "region_comparer")
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["regions"], ["MOEMA", "PINHEIROS"])
        self.assertEqual(payload["metrics"]["MOEMA"]["count"], 2)
        self.assertEqual(payload["metrics"]["PINHEIROS"]["avg_price_m2"], 13500.0)
        self.assertEqual(result.metadata["status"], "success")

    def test_region_comparer_handles_unknown_region(self) -> None:
        """Garante erro amigavel quando uma regiao nao existe na base processada."""

        dataframe = pd.DataFrame(
            [
                {"bairro": "MOEMA", "valor_m2": 12000.0, "valor_venal_de_referencia": 1000000.0},
                {"bairro": "PINHEIROS", "valor_m2": 14000.0, "valor_venal_de_referencia": 1200000.0},
            ]
        )

        with patch("src.agent.tools._load_region_dataframe", return_value=dataframe):
            result = tools.region_comparer({"region_a": "Moema", "region_b": "Bairro Inexistente"})

        payload = json.loads(result.content)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "regions_not_found")
        self.assertIn("BAIRRO INEXISTENTE", payload["missing_regions"])


if __name__ == "__main__":
    unittest.main()
