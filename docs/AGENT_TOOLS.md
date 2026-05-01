# Tools do Agent

## Objetivo

As tools conectam o Agent ReAct aos componentes reais do projeto: RAG, modelo de precificacao e dataset processado.

Todas retornam `ToolResult`, com:

```json
{
  "tool": "nome_da_tool",
  "status": "success | error | no_context",
  "...": "campos especificos da tool"
}
```

O Agent usa `content` para montar a resposta final e `metadata` para observabilidade.

## rag_search

Responsabilidade: recuperar contexto e fontes no RAG.

Entrada:

```json
{
  "query": "Quais fatores influenciam o preco?"
}
```

Saida principal:

```json
{
  "tool": "rag_search",
  "query": "...",
  "context": "...",
  "sources": ["arquivo.md"],
  "chunks_retrieved": 3,
  "status": "success"
}
```

A tool usa `src/rag.retrieve_context` e nao chama LLM diretamente. Se nenhum contexto for encontrado, retorna `status: "no_context"` sem inventar fontes.

## price_estimator

Responsabilidade: chamar o modelo existente de precificacao, sem alterar o modelo.

Campos obrigatorios alinhados ao endpoint `/predict`:

```text
bairro
area_do_terreno_m2
valor_m2
ano_mes
media_valor_cep
```

A tool tambem aceita `area` como alias para `area_do_terreno_m2`.

Saida principal:

```json
{
  "tool": "price_estimator",
  "input": {
    "bairro": "MOEMA",
    "area_do_terreno_m2": 80,
    "valor_m2": 1500,
    "ano_mes": 202401,
    "media_valor_cep": 2000
  },
  "estimated_price": 950000,
  "currency": "BRL",
  "status": "success"
}
```

Se faltarem campos, retorna erro controlado com `missing_fields`.

## region_comparer

Responsabilidade: comparar bairros usando o dataset processado `data/processed/itbi_features_minimal.csv`.

Entrada:

```json
{
  "region_a": "Moema",
  "region_b": "Pinheiros",
  "metric": "valor_m2"
}
```

Saida principal:

```json
{
  "tool": "region_comparer",
  "regions": ["MOEMA", "PINHEIROS"],
  "metrics": {
    "MOEMA": {
      "avg_price": 1000000,
      "median_price": 950000,
      "avg_price_m2": 12000,
      "count": 150
    }
  },
  "status": "success"
}
```

Se algum bairro nao existir no dataset, retorna `status: "error"` e lista `missing_regions`.

## Registro

As tools estao registradas em:

```python
TOOLS = {
    "rag_search": rag_search,
    "price_estimator": price_estimator,
    "region_comparer": region_comparer,
}
```

As descricoes formais ficam em `TOOL_REGISTRY`, usado para documentacao e validacao.
