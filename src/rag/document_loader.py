from pathlib import Path
import hashlib

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


RAW_DOCS_PATH = Path("data/rag/raw")


def load_markdown_documents(raw_docs_path: Path = RAW_DOCS_PATH) -> list[Document]:
    if not raw_docs_path.exists():
        return []

    documents: list[Document] = []
    for pattern in ["**/*.md", "**/*.txt"]:
        loader = DirectoryLoader(
            str(raw_docs_path),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
        )
        documents.extend(loader.load())
    return documents


def raw_documents_signature(raw_docs_path: Path = RAW_DOCS_PATH) -> str:
    if not raw_docs_path.exists():
        return "missing"

    digest = hashlib.sha256()
    for path in sorted(raw_docs_path.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        relative_path = path.relative_to(raw_docs_path).as_posix()
        stat = path.stat()
        digest.update(f"{relative_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()
