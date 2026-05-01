"""
Testes unitarios do script `scripts/build_rag_index.py`.

Objetivo para avaliacao/banca:
- garantir que o script usa as camadas atuais do RAG;
- validar que documentos sao carregados, quebrados em chunks e enviados ao vector store;
- garantir erros claros para pasta ausente ou ausencia de documentos.

Os testes mockam o filesystem e as funcoes pesadas para nao reconstruir o indice real.
Execute com:
    python -m unittest tests.unit.test_build_rag_index
"""

import unittest
from pathlib import Path
from unittest.mock import patch


class BuildRagIndexScriptTests(unittest.TestCase):
    """Valida o contrato do script de build do indice RAG."""

    def test_build_index_uses_current_rag_layers(self) -> None:
        """Confirma que o script chama loader, chunking, assinatura e vector store atuais."""

        from scripts import build_rag_index

        fake_documents = [object()]
        fake_chunks = [object(), object()]

        with patch.object(Path, "exists", return_value=True):
            with patch("scripts.build_rag_index._load_markdown_documents", return_value=fake_documents) as load_docs:
                with patch("scripts.build_rag_index._split_documents", return_value=fake_chunks) as split_docs:
                    with patch("scripts.build_rag_index._raw_documents_signature", return_value="signature-test"):
                        with patch("scripts.build_rag_index._build_vector_store") as build_store:
                            total_chunks = build_rag_index.build_index()

        self.assertEqual(total_chunks, 2)
        load_docs.assert_called_once_with(build_rag_index.RAW_DOCS_PATH)
        split_docs.assert_called_once_with(fake_documents)
        build_store.assert_called_once_with(
            documents=fake_chunks,
            persist_directory=build_rag_index.VECTORSTORE_PATH,
            source_signature="signature-test",
        )

    def test_build_index_fails_when_raw_folder_is_missing(self) -> None:
        """Garante erro explicito quando `data/rag/raw` nao existe."""

        from scripts import build_rag_index

        with patch.object(Path, "exists", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Pasta de documentos RAG"):
                build_rag_index.build_index()

    def test_build_index_fails_when_no_documents_are_loaded(self) -> None:
        """Garante erro explicito quando nao existem arquivos .md ou .txt para indexar."""

        from scripts import build_rag_index

        with patch.object(Path, "exists", return_value=True):
            with patch("scripts.build_rag_index._load_markdown_documents", return_value=[]):
                with self.assertRaisesRegex(ValueError, "Nenhum documento"):
                    build_rag_index.build_index()


if __name__ == "__main__":
    unittest.main()
