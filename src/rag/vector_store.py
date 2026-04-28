from pathlib import Path
import shutil

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.rag.embeddings import get_embedding_model


VECTORSTORE_PATH = Path("data/rag/vectorstore")


def build_vector_store(
    documents: list[Document],
    persist_directory: Path = VECTORSTORE_PATH,
) -> Chroma:
    if persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)
    embedding_function = get_embedding_model()

    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=str(persist_directory),
    )


def load_vector_store(persist_directory: Path = VECTORSTORE_PATH) -> Chroma:
    embedding_function = get_embedding_model()
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embedding_function,
    )


def vector_store_exists(persist_directory: Path = VECTORSTORE_PATH) -> bool:
    sqlite_file = persist_directory / "chroma.sqlite3"
    return sqlite_file.exists()
