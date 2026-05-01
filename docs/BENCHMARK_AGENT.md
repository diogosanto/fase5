# Benchmark Agent/RAG

## Objetivo

O benchmark compara tres configuracoes de recuperacao documental do Agent/RAG:

```text
top_k_1 -> RAG_TOP_K=1
top_k_3 -> RAG_TOP_K=3
top_k_5 -> RAG_TOP_K=5
```

O script principal e:

```text
evaluation/benchmark_agent.py
```

## Golden Set

O benchmark usa o arquivo:

```text
data/golden_set/real_estate_chat_golden_set.jsonl
```

Esse arquivo contem perguntas sobre RAG, estimativa de preco, comparacao de bairros e casos multi-tool.

## Execucao

Para executar:

```powershell
python evaluation/benchmark_agent.py
```

Ou:

```powershell
python -m evaluation.benchmark_agent
```

Para limitar custo durante testes:

```powershell
$env:BENCHMARK_MAX_QUESTIONS="5"
python evaluation/benchmark_agent.py
```

## Saidas

Os resultados sao salvos em:

```text
evaluation/results/benchmark_agent_results.json
evaluation/results/benchmark_agent_results.csv
```

## Metricas

O benchmark calcula por configuracao:

- total de perguntas;
- taxa de sucesso;
- taxa de erro;
- latencia media;
- media de tools usadas;
- media de chunks recuperados;
- tamanho medio da resposta;
- presenca de fontes;
- taxa simples de match com a tool esperada.

## Seguranca e Custo

O script nao registra secrets. Para reduzir custo, use `BENCHMARK_MAX_QUESTIONS` e mantenha `LLM_MAX_TOKENS` baixo durante a execucao.
