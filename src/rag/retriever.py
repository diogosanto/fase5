from langchain_core.documents import Document

from src.rag.vector_store import load_vector_store


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)
