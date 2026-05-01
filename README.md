# Precificador Imobiliario

Projeto de Machine Learning Engineering para previsao de precos de imoveis, com API FastAPI, RAG, Agent ReAct, observabilidade e governanca.

## Estrategia de Serving da LLM

A camada de RAG e Agent usa a API Groq como provedor oficial de LLM, com o modelo `llama-3.1-8b-instant`.

Variaveis principais:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=30
```

A decisao foi usar LLM servida via API em vez de empacotar uma LLM local quantizada no Docker. Isso reduz complexidade operacional, tamanho da imagem e necessidade de hardware especializado para a demonstracao do MVP.

A justificativa completa esta em [docs/LLM_SERVING_DECISION.md](docs/LLM_SERVING_DECISION.md).
