import logging
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

from src.agent.llm import estimate_tokens, get_llm
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


def retrieve_context(query: str, k: int | None = None) -> list[RetrievedChunk]:
    k = k or int(os.getenv("RAG_TOP_K", "3"))
    max_chunk_chars = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    ensure_vector_store()
    docs = retrieve_documents(query=query, k=k)
    chunks = [
        RetrievedChunk(
            source=doc.metadata.get("source", "unknown"),
            content=doc.page_content[:max_chunk_chars],
        )
        for doc in docs
    ]
    logger.info("Consulta RAG recuperou %s chunks top_k=%s", len(chunks), k)
    return chunks


def rag_pipeline(query: str, k: int | None = None) -> RagResult:
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
    prompt = f"""
Voce e um assistente especializado em precificacao imobiliaria.

Responda em portugues do Brasil em no maximo 5 frases.
Use somente o contexto recuperado.
Se o contexto nao trouxer informacao suficiente, diga explicitamente que nao ha base documental suficiente.
Nao invente dados nem regras.

Contexto:
{context}

Pergunta:
{query}
"""
    logger.info(
        "Prompt RAG approx_tokens=%s chunks=%s",
        estimate_tokens(prompt),
        len(chunks),
    )
    try:
        response_text = get_llm().generate(prompt)
    except Exception as exc:
        logger.exception("Falha na geracao LLM do RAG; usando fallback extrativo. Erro: %s", exc)
        response_text = (
            "Nao consegui usar a LLM nesta chamada, mas encontrei contexto relevante nos documentos.\n\n"
            + "\n\n".join(
                f"Fonte: {chunk.source}\n{chunk.content}"
                for chunk in chunks
            )
        )

    sources = list(dict.fromkeys(chunk.source for chunk in chunks))
    return RagResult(
        answer=response_text,
        chunks_retrieved=len(chunks),
        sources=sources,
        chunks=chunks,
    )


def rag_answer(query: str, k: int = 3) -> str:
    return rag_pipeline(query=query, k=k).answer
