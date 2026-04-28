import math

from langchain_core.documents import Document

from src.rag.embeddings import get_embedding_model
from src.rag.vector_store import load_vector_store


def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    store = load_vector_store()
    embedding_function = get_embedding_model()
    query_embedding = embedding_function.embed_query(query)

    ranked_documents = sorted(
        store.get("documents", []),
        key=lambda item: _cosine_similarity(query_embedding, item["embedding"]),
        reverse=True,
    )

    top_documents = ranked_documents[:k]
    return [
        Document(
            page_content=item["page_content"],
            metadata=item.get("metadata", {}),
        )
        for item in top_documents
    ]
