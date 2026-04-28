from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document


RAW_DOCS_PATH = Path("data/rag/raw")


def load_markdown_documents(raw_docs_path: Path = RAW_DOCS_PATH) -> list[Document]:
    if not raw_docs_path.exists():
        return []

    loader = DirectoryLoader(
        str(raw_docs_path),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    return loader.load()
