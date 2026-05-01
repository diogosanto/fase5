# Testes do Projeto

Esta pasta concentra testes da camada RAG, Agent, LLM e endpoint `/chat`.

Os testes foram desenhados para serem deterministas e baratos: sempre que possivel, usam mocks/fakes para evitar chamadas reais ao Groq, MLflow ou leitura de bases grandes.

## Como Executar

Rodar todos os testes:

```powershell
python -m unittest discover tests
```

Rodar apenas testes unitarios:

```powershell
python -m unittest discover tests/unit
```

Rodar apenas o teste de integracao do `/chat`:

```powershell
python -m unittest tests.integration.test_chat_endpoint
```

## O Que Cada Arquivo Valida

`tests/unit/test_llm_config.py`

Valida leitura das variaveis de ambiente da LLM, provider Groq, modelo, max tokens, temperatura, timeout e erro quando `GROQ_API_KEY` nao existe.

`tests/unit/test_react_agent.py`

Valida o Agent ReAct: existencia de pelo menos 3 tools, roteamento para `rag_search`, `price_estimator` e `region_comparer`, resposta estruturada e respeito a `AGENT_MAX_STEPS`.

`tests/unit/test_agent_tools.py`

Valida as tools do Agent: registro formal, retorno estruturado, fontes do RAG, validacao de campos do modelo e comparacao de regioes com dataset processado.

`tests/unit/test_rag_documental.py`

Valida o RAG documental: carregamento de `.md`/`.txt`, chunking, metadata de fonte, `chunk_index`, retriever com top_k e assinatura dos documentos.

`tests/unit/test_build_rag_index.py`

Valida o script `scripts/build_rag_index.py`, garantindo que ele usa as camadas atuais de loader, chunking, assinatura e vector store.

`tests/integration/test_chat_endpoint.py`

Valida o endpoint `/chat` via ASGI in-memory, sem subir `uvicorn`: contrato HTTP, validacao de entrada, resposta estruturada e tratamento de erros.

## Observacoes

Warnings de bibliotecas externas, como MLflow, protobuf ou Pydantic, podem aparecer dependendo da versao instalada. O criterio principal e o resultado final `OK`.

Se algum teste falhar por dependencia ausente, ative a `.venv` do projeto e rode novamente:

```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover tests
```
