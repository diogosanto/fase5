from src.rag.rag_pipeline import build_rag_index


if __name__ == "__main__":
    total_chunks = build_rag_index()
    print(f"Indice RAG criado com sucesso. Chunks indexados: {total_chunks}")
