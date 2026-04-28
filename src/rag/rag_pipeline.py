import logging
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.rag.chunking import split_documents
from src.rag.document_loader import load_markdown_documents
from src.rag.retriever import retrieve_documents
from src.rag.vector_store import build_vector_store, vector_store_exists


load_dotenv()

logger = logging.getLogger("precificador.rag")


@dataclass
class RetrievedChunk:
    source: str
    content: str


@dataclass
class RagResult:
    answer: str
    chunks_retrieved: int
    sources: list[str]
    chunks: list[RetrievedChunk]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "chunks_retrieved": self.chunks_retrieved,
            "sources": self.sources,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


def _get_rag_llm() -> ChatGoogleGenerativeAI:
    model_name = os.getenv("GEMINI_RAG_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
    )


def build_rag_index() -> int:
    documents = load_markdown_documents()
    chunks = split_documents(documents)
    if not chunks:
        logger.warning("Nenhum documento markdown encontrado para indexacao RAG.")
        return 0

    build_vector_store(chunks)
    logger.info("Indice RAG atualizado com %s chunks", len(chunks))
    return len(chunks)


def ensure_vector_store() -> None:
    if not vector_store_exists():
        build_rag_index()


def retrieve_context(query: str, k: int = 3) -> list[RetrievedChunk]:
    ensure_vector_store()
    docs = retrieve_documents(query=query, k=k)
    chunks = [
        RetrievedChunk(
            source=doc.metadata.get("source", "unknown"),
            content=doc.page_content,
        )
        for doc in docs
    ]
    logger.info("Consulta RAG recuperou %s chunks", len(chunks))
    return chunks


def rag_pipeline(query: str, k: int = 3) -> RagResult:
    chunks = retrieve_context(query=query, k=k)
    if not chunks:
        return RagResult(
            answer="Nao encontrei contexto suficiente nos documentos para responder com seguranca.",
            chunks_retrieved=0,
            sources=[],
            chunks=[],
        )

    context = "\n\n".join(
        f"Fonte: {chunk.source}\nTrecho:\n{chunk.content}"
        for chunk in chunks
    )
    prompt = ChatPromptTemplate.from_template(
        """
Voce e um assistente especializado em precificacao imobiliaria.

Responda em portugues do Brasil.
Use somente o contexto recuperado.
Se o contexto nao trouxer informacao suficiente, diga explicitamente que nao ha base documental suficiente.
Nao invente dados nem regras.

Contexto:
{context}

Pergunta:
{question}
"""
    )
    response = (prompt | _get_rag_llm()).invoke(
        {
            "context": context,
            "question": query,
        }
    )

    sources = list(dict.fromkeys(chunk.source for chunk in chunks))
    return RagResult(
        answer=response.content,
        chunks_retrieved=len(chunks),
        sources=sources,
        chunks=chunks,
    )


def rag_answer(query: str, k: int = 3) -> str:
    return rag_pipeline(query=query, k=k).answer
