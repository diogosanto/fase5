# Agent ReAct

## Objetivo

O Agent ReAct coordena a camada de RAG, tools e LLM para responder perguntas sobre precificacao imobiliaria.

Ele segue o fluxo:

```text
pergunta do usuario -> escolha de tool -> execucao da tool -> observation -> resposta final
```

O raciocinio interno usado para escolher tools nao e exposto ao usuario final.

## Tools Disponiveis

### rag_search

Busca contexto nos documentos carregados pelo RAG.

Use quando a pergunta for conceitual, explicativa ou depender de contexto textual.

Exemplo:

```text
Quais fatores influenciam o preco de um imovel?
```

Tool esperada:

```text
rag_search
```

### price_estimator

Chama a camada existente de previsao de preco, sem alterar o modelo de ML.

Use quando a pergunta pedir estimativa de preco e houver dados suficientes para o modelo:

```text
bairro, area_do_terreno_m2, valor_m2, ano_mes, media_valor_cep
```

Exemplo:

```text
Quanto vale um apartamento com bairro Moema, area 80, valor_m2 1500, ano_mes 202401 e media_valor_cep 2000?
```

Tool esperada:

```text
price_estimator
```

### region_comparer

Compara bairros ou regioes usando dados processados disponiveis no projeto.

Exemplo:

```text
Compare Moema e Pinheiros para compra de imovel
```

Tool esperada:

```text
region_comparer
```

## Fluxo ReAct

O Agent trabalha com quatro etapas logicas:

```text
Thought -> Action -> Observation -> Final Answer
```

No codigo:

- `Thought` representa a decisao interna do Agent ou da LLM.
- `Action` e a tool selecionada.
- `Observation` e o retorno da tool.
- `Final Answer` e a resposta final ao usuario.

O campo `thought` nao aparece na resposta final. A API retorna apenas `answer`, `tools_used`, observations/steps quando disponiveis e metadados operacionais.

## Limite de Iteracoes

O limite de passos evita loops ou chamadas repetidas.

Configuracao:

```env
AGENT_MAX_STEPS=3
```

Se a variavel nao existir, o Agent usa `3` como default seguro.

## Resposta Estruturada

O Agent retorna um objeto estruturado com:

```json
{
  "answer": "resposta final ao usuario",
  "tools_used": ["rag_search"],
  "observations": ["resultado da tool"],
  "metadata": {
    "agent_type": "react",
    "max_steps": 3,
    "steps_executed": 1,
    "provider": "groq",
    "model": "llama-3.1-8b-instant"
  }
}
```

## Tratamento de Erros

O Agent trata:

- tool inexistente;
- erro durante execucao da tool;
- retorno vazio da tool;
- falha de LLM;
- limite maximo de steps atingido.

Em todos os casos, a resposta e controlada e nao expõe stack trace, API key ou raciocinio interno completo.

## Limitacoes

- O `price_estimator` depende dos campos exigidos pelo modelo existente.
- O `region_comparer` depende da base processada disponivel no ambiente.
- O `rag_search` depende de documentos e vector store previamente preparados.
- Para reduzir custo e instabilidade, intencoes claras usam roteamento heuristico antes de chamar a LLM.
