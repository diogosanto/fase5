# Decisao de Serving da LLM

## Resumo

O projeto usa a API Groq como provedor oficial de LLM para a camada de RAG e Agent.

Configuracao principal:

```env
LLM_PROVIDER=groq
GROQ_MODEL=llama-3.1-8b-instant
```

A chave `GROQ_API_KEY` deve ser injetada por variavel de ambiente ou arquivo `.env` local e nunca deve ser versionada.

## Decisao

A LLM sera servida via API Groq, usando o modelo `llama-3.1-8b-instant`, em vez de executar uma LLM local quantizada dentro do ambiente Docker do projeto.

Essa decisao atende ao requisito de "LLM servido via API com quantizacao / decidir entre LLM local quantizado ou justificar o uso da API Groq", pois documenta explicitamente a alternativa avaliada e a justificativa tecnica para o caminho escolhido.

## Alternativa Considerada: LLM Local Quantizada

A alternativa seria empacotar uma LLM local quantizada, por exemplo em formato GGUF, servida por ferramentas como Ollama, llama.cpp ou vLLM.

Essa abordagem teria algumas vantagens:

- maior controle sobre infraestrutura e modelo;
- menor dependencia de provedor externo;
- possibilidade de execucao offline;
- controle direto sobre politicas de retencao de dados.

No entanto, ela nao foi selecionada para esta fase do projeto.

## Por Que a LLM Local Quantizada Nao Foi Selecionada

Para o escopo da FIAP Fase 5, a LLM local quantizada aumentaria complexidade operacional sem trazer ganho proporcional para o MVP.

Principais motivos:

- exigiria imagem Docker maior e mais lenta para build/deploy;
- aumentaria consumo de CPU/RAM ou exigiria GPU local;
- adicionaria uma nova camada operacional para servir, monitorar e atualizar a LLM;
- dificultaria reproducibilidade em maquinas sem hardware adequado;
- deslocaria o foco do projeto, que esta na integracao MLE de RAG, Agent, API, observabilidade e governanca.

Com Groq, o projeto ganha baixa latencia, simplicidade de deploy e menor carga operacional para demonstrar o fluxo completo de Agent/RAG.

## Impacto no Docker

O container da API nao inclui pesos de LLM nem secrets.

As variaveis da LLM sao injetadas em runtime:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.1-8b-instant
LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=30
```

Isso reduz tamanho da imagem, tempo de build e risco de vazamento de credenciais.

## Seguranca da API Key

Boas praticas adotadas:

- `GROQ_API_KEY` nao e hardcoded no codigo;
- `.env.example` usa placeholder, nao chave real;
- a chave deve ficar apenas no `.env` local, secret manager ou variaveis do ambiente de execucao;
- logs registram provider/model, mas nao registram API key;
- mensagens de erro nao retornam secrets ao usuario.

## Configuracoes Operacionais

As seguintes variaveis controlam custo, latencia e previsibilidade:

```env
LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=30
```

`LLM_MAX_TOKENS` limita o tamanho da resposta, `LLM_TEMPERATURE` reduz variabilidade e `LLM_TIMEOUT_SECONDS` evita chamadas presas indefinidamente.

## Metadados e Observabilidade

A camada `src/agent/llm.py` registra:

- provider usado;
- modelo usado;
- latencia da chamada;
- quantidade aproximada de tokens no prompt;
- token usage retornado pelo provedor, quando disponivel;
- erros de chamada sem expor secrets.

## Limitacoes e Mitigacoes

Limitacao: dependencia de servico externo.
Mitigacao: erros de provider sao tratados e retornados como falha temporaria, sem stack trace.

Limitacao: custo e limites de quota.
Mitigacao: `LLM_MAX_TOKENS`, `RAG_TOP_K`, `AGENT_MAX_STEPS` e logs de chamadas reduzem consumo desnecessario.

Limitacao: envio de dados para terceiro.
Mitigacao: nao enviar secrets, evitar payloads sensiveis completos em logs e manter escopo de dados minimo para a resposta.

## Decisao Final

Para este projeto, a API Groq e a escolha oficial de serving da LLM. A LLM local quantizada permanece como alternativa futura caso o projeto exija execucao offline, controle total de infraestrutura ou reducao de dependencia externa.
