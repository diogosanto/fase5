import json
import shutil
from pathlib import Path

from langchain_core.documents import Document

from src.rag.embeddings import get_embedding_model


VECTORSTORE_PATH = Path("data/rag/vectorstore")
INDEX_FILE = VECTORSTORE_PATH / "index.json"
VECTORSTORE_MARKER = "embedding_backend=local_hash_v2"
MARKER_FILE = VECTORSTORE_PATH / "index_metadata.txt"


def build_vector_store(
    documents: list[Document],
    persist_directory: Path = VECTORSTORE_PATH,
) -> dict:
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    embedding_function = get_embedding_model()
    texts = [document.page_content for document in documents]
    embeddings = embedding_function.embed_documents(texts) if texts else []

    payload = {
        "marker": VECTORSTORE_MARKER,
        "documents": [
            {
                "page_content": document.page_content,
                "metadata": document.metadata,
                "embedding": embedding,
            }
            for document, embedding in zip(documents, embeddings)
        ],
    }
    INDEX_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    MARKER_FILE.write_text(VECTORSTORE_MARKER, encoding="utf-8")
    return payload


def load_vector_store(persist_directory: Path = VECTORSTORE_PATH) -> dict:
    index_file = persist_directory / "index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Indice RAG nao encontrado em {index_file}")
    return json.loads(index_file.read_text(encoding="utf-8"))


def vector_store_exists(persist_directory: Path = VECTORSTORE_PATH) -> bool:
    index_file = persist_directory / "index.json"
    marker_file = persist_directory / "index_metadata.txt"
    if not index_file.exists() or not marker_file.exists():
        return False

    return marker_file.read_text(encoding="utf-8").strip() == VECTORSTORE_MARKER
