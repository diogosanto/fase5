import logging
import sys
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("scripts.build_rag_index")
RAW_DOCS_PATH = PROJECT_ROOT / "data" / "rag" / "raw"
VECTORSTORE_PATH = PROJECT_ROOT / "data" / "rag" / "vectorstore"


def _load_markdown_documents(raw_docs_path: Path):
    from src.rag.document_loader import load_markdown_documents

    return load_markdown_documents(raw_docs_path)


def _split_documents(documents):
    from src.rag.chunking import split_documents

    return split_documents(documents)


def _raw_documents_signature(raw_docs_path: Path) -> str:
    from src.rag.document_loader import raw_documents_signature

    return raw_documents_signature(raw_docs_path)


def _build_vector_store(documents, persist_directory: Path, source_signature: str):
    from src.rag.vector_store import build_vector_store

    return build_vector_store(
        documents=documents,
        persist_directory=persist_directory,
        source_signature=source_signature,
    )


def build_index() -> int:
    started_at = perf_counter()
    logger.info("Iniciando build do indice RAG raw_docs_path=%s", RAW_DOCS_PATH)

    if not RAW_DOCS_PATH.exists():
        raise FileNotFoundError(f"Pasta de documentos RAG nao encontrada: {RAW_DOCS_PATH}")

    documents = _load_markdown_documents(RAW_DOCS_PATH)
    logger.info("Documentos carregados: %s", len(documents))
    if not documents:
        raise ValueError(f"Nenhum documento .md ou .txt encontrado em {RAW_DOCS_PATH}")

    chunks = _split_documents(documents)
    logger.info("Chunks gerados: %s", len(chunks))
    if not chunks:
        raise ValueError("Nenhum chunk foi gerado a partir dos documentos RAG.")

    signature = _raw_documents_signature(RAW_DOCS_PATH)
    _build_vector_store(
        documents=chunks,
        persist_directory=VECTORSTORE_PATH,
        source_signature=signature,
    )

    elapsed_ms = int((perf_counter() - started_at) * 1000)
    logger.info(
        "Indice RAG salvo com sucesso path=%s chunks=%s elapsed_ms=%s",
        VECTORSTORE_PATH,
        len(chunks),
        elapsed_ms,
    )
    return len(chunks)


def main() -> int:
    try:
        total_chunks = build_index()
    except Exception as exc:
        logger.exception("Falha ao construir indice RAG: %s", exc)
        return 1

    print(f"Indice RAG criado com sucesso. Chunks indexados: {total_chunks}. Path: {VECTORSTORE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
