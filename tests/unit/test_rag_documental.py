"""
Testes unitarios do RAG documental.

Objetivo para avaliacao/banca:
- validar carregamento de documentos `.md` e `.txt`;
- garantir chunking com metadata de fonte e `chunk_index`;
- validar que o retriever respeita top_k;
- garantir que `retrieve_context` retorna fonte e indice do chunk;
- validar que mudancas nos documentos alteram a assinatura do indice.

Os testes usam arquivos temporarios e mocks de embedding/vector store para serem rapidos e deterministas.
Execute com:
    python -m unittest tests.unit.test_rag_documental
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from src.rag.chunking import split_documents
from src.rag.document_loader import load_markdown_documents, raw_documents_signature
from src.rag.rag_pipeline import retrieve_context
from src.rag.retriever import retrieve_documents


class FakeEmbeddingModel:
    """Embedding fake com vetor fixo para testar ordenacao do retriever sem modelo externo."""

    def embed_query(self, text):
        return [1.0, 0.0]


class RagDocumentalTests(unittest.TestCase):
    """Cobre o fluxo documental: loader, chunking, retriever e metadata."""

    def test_loads_markdown_and_text_documents(self) -> None:
        """Garante que a base documental aceita arquivos Markdown e texto puro."""

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir)
            (raw_path / "fatores.md").write_text("Localizacao influencia preco.", encoding="utf-8")
            (raw_path / "notas.txt").write_text("Preco por metro quadrado ajuda comparacao.", encoding="utf-8")

            documents = load_markdown_documents(raw_path)

        self.assertEqual(len(documents), 2)
        self.assertTrue(all("source" in document.metadata for document in documents))

    def test_chunking_preserves_source_and_adds_chunk_index(self) -> None:
        """Valida que cada chunk mantem a fonte original e recebe indice auditavel."""

        document = Document(
            page_content="Localizacao influencia preco. " * 20,
            metadata={"source": "fatores_precificacao.md"},
        )

        chunks = split_documents([document], chunk_size=120, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source"], "fatores_precificacao.md")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[1].metadata["chunk_index"], 1)

    def test_retriever_returns_relevant_documents_and_respects_top_k(self) -> None:
        """Confirma que o retriever ordena por similaridade e limita resultados por top_k."""

        store = {
            "documents": [
                {
                    "page_content": "Documento relevante sobre localizacao.",
                    "metadata": {"source": "localizacao_bairros.md", "chunk_index": 0},
                    "embedding": [1.0, 0.0],
                },
                {
                    "page_content": "Documento menos relevante.",
                    "metadata": {"source": "outro.md", "chunk_index": 1},
                    "embedding": [0.0, 1.0],
                },
            ]
        }

        with patch("src.rag.retriever.load_vector_store", return_value=store):
            with patch("src.rag.retriever.get_embedding_model", return_value=FakeEmbeddingModel()):
                documents = retrieve_documents("localizacao", k=1)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["source"], "localizacao_bairros.md")

    def test_retrieve_context_returns_sources_and_chunk_index(self) -> None:
        """Garante que a camada de contexto usada pela tool preserva fonte e chunk_index."""

        documents = [
            Document(
                page_content="Localizacao, transporte e infraestrutura influenciam o preco.",
                metadata={"source": "localizacao_bairros.md", "chunk_index": 2},
            )
        ]

        with patch("src.rag.rag_pipeline.ensure_vector_store", return_value=None):
            with patch("src.rag.rag_pipeline.retrieve_documents", return_value=documents):
                chunks = retrieve_context("localizacao", k=1)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source, "localizacao_bairros.md")
        self.assertEqual(chunks[0].chunk_index, 2)

    def test_raw_documents_signature_changes_when_document_changes(self) -> None:
        """Valida invalidacao do indice quando documentos brutos sao alterados."""

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir)
            document_path = raw_path / "fatores.md"
            document_path.write_text("conteudo inicial", encoding="utf-8")
            first_signature = raw_documents_signature(raw_path)

            document_path.write_text("conteudo alterado", encoding="utf-8")
            second_signature = raw_documents_signature(raw_path)

        self.assertNotEqual(first_signature, second_signature)


if __name__ == "__main__":
    unittest.main()
