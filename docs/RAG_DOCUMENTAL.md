# RAG Documental

## Objetivo

O RAG documental fornece contexto textual para o Agent responder perguntas conceituais sobre precificacao imobiliaria sem inventar informacoes.

## Documentos de Apoio

Os documentos ficam em `data/rag/raw/` e cobrem:

- fatores de precificacao;
- preco por metro quadrado;
- localizacao e bairros;
- caracteristicas do imovel;
- infraestrutura urbana;
- comparacao entre regioes;
- interpretacao de previsoes;
- limitacoes do modelo.

Os arquivos sao pequenos, em Markdown, e separados por tema para facilitar recuperacao de contexto.

## Chunking

O chunking e implementado em `src/rag/chunking.py`.

Configuracoes:

```env
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
```

Cada chunk preserva metadata de origem e recebe `chunk_index`.

## Retriever

O retriever e implementado em `src/rag/retriever.py`.

Configuracao:

```env
RAG_TOP_K=3
```

Ele retorna os chunks mais relevantes para a query, preservando `source` e demais metadados. O RAG nao inventa fontes: as fontes retornadas vêm dos documentos carregados.

## Atualizacao do Indice

O indice em `data/rag/vectorstore/` guarda uma assinatura dos documentos brutos. Se arquivos em `data/rag/raw/` forem adicionados ou alterados, o indice e reconstruido automaticamente na proxima consulta.

## Integracao com Agent

A tool `rag_search` usa `retrieve_context` para retornar:

```json
{
  "context": "...",
  "sources": ["fatores_precificacao.md"],
  "chunks_retrieved": 3
}
```

Essa tool apenas recupera contexto. A decisao de resposta final fica com o Agent/RAG pipeline.
