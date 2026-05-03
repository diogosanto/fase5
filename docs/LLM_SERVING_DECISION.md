# Decisao de Serving da LLM

## Resumo

O projeto suporta dois caminhos para servir a LLM usada pelo Agent/RAG:

1. **LLM quantizada servida via API compativel com Ollama**
2. **Groq API como provider padrao/fallback operacional**

Isso permite atender ao requisito de LLM servida via API com quantizacao, sem obrigar que o Ollama rode na maquina local do desenvolvedor.

## Configuracao Para LLM Quantizada via Ollama API

Para usar uma LLM quantizada servida por Ollama, configure:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://servidor:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4
```

O `OLLAMA_BASE_URL` pode apontar para:

- Ollama rodando localmente;
- Ollama em outro computador da rede;
- Ollama em uma VM/cloud;
- Ollama em um container Docker;
- qualquer endpoint compativel com a API `/api/generate` do Ollama.

O ponto principal e: **o projeto nao precisa hospedar os pesos da LLM dentro da propria API FastAPI**. Ele precisa conseguir consumir uma LLM quantizada servida por uma API. Essa integracao esta implementada em `src/agent/llm.py`.

## Configuracao Padrao/Fallback com Groq

Por limitacao operacional do MVP, o provider padrao continua sendo Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Groq e usado como fallback por ser simples de executar na demonstracao, nao exigir GPU local e reduzir complexidade de infraestrutura.

## Decisao Arquitetural

A decisao final e:

> O projeto suporta LLM quantizada servida via API compativel com Ollama. Para execucao local ou remota, basta configurar `OLLAMA_BASE_URL` e `OLLAMA_MODEL`. Por padrao, usamos Groq por limitacao operacional, mas a camada esta preparada para LLM quantizada servida via API.

Essa abordagem e mais flexivel do que acoplar uma LLM local ao container da API.

## Por Que Nao Rodar Ollama Obrigatoriamente Local

Rodar uma LLM quantizada localmente pode exigir:

- mais memoria RAM;
- CPU/GPU mais forte;
- download de pesos do modelo;
- imagem Docker maior;
- mais tempo de build/deploy;
- maior complexidade para a banca reproduzir.

Por isso, o projeto separa responsabilidades:

```text
Agent/RAG -> src/agent/llm.py -> Ollama API -> modelo quantizado
```

Assim, a LLM pode estar local, remota ou em container, desde que exponha a API esperada.

## Impacto no Docker

O container da API nao inclui pesos de LLM e nao inclui secrets.

Variaveis relevantes:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://servidor:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4
LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=30
```

Ou, para fallback Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

## Seguranca

Boas praticas adotadas:

- `GROQ_API_KEY` nao e hardcoded;
- `.env.example` usa placeholders;
- secrets ficam apenas em `.env`, secret manager ou variaveis do ambiente;
- logs registram provider/model/latencia, mas nao registram API keys;
- Ollama nao exige API key por padrao, mas o endpoint deve ficar protegido por rede, firewall ou proxy quando remoto.

## Observabilidade

A camada `src/agent/llm.py` registra:

- provider usado;
- modelo usado;
- latencia;
- estimativa de tokens do prompt;
- token usage quando o provedor retorna essa informacao;
- erros de chamada sem expor secrets.

## Limitacoes e Mitigacoes

Limitacao: Ollama remoto precisa estar disponivel.
Mitigacao: usar `LLM_TIMEOUT_SECONDS` e fallback operacional com Groq quando necessario.

Limitacao: a quantizacao depende do modelo servido no Ollama.
Mitigacao: documentar explicitamente o modelo configurado em `OLLAMA_MODEL`, por exemplo `llama3.1:8b-instruct-q4`.

Limitacao: custo/quota no Groq.
Mitigacao: limitar `LLM_MAX_TOKENS`, `RAG_TOP_K` e `AGENT_MAX_STEPS`.

## Conclusao

O projeto atende ao requisito ao permitir LLM quantizada servida via API compativel com Ollama, mantendo Groq como fallback operacional para demonstracao e desenvolvimento.
