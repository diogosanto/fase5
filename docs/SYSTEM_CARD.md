# System Card

## Visao Geral

O sistema e um MVP de MLOps para precificacao imobiliaria em Sao Paulo usando dados publicos de ITBI, pipeline DVC, modelo MLflow, API FastAPI, Agent ReAct, RAG documental e monitoramento Prometheus/Grafana.

## Componentes

| Componente | Responsabilidade | Evidencia |
| --- | --- | --- |
| Pipeline DVC | Ingestao, limpeza, features, treino, validacao e reproducibilidade. | `dvc.yaml`, `params.yaml` |
| Modelo ML | Estimar valor usando `cep`, `area_do_terreno_m2`, `ano`, `mes`. | `src/training/`, `models/` |
| API FastAPI | Servir `/predict`, `/chat`, `/health`, `/metrics`. | `api/main.py` |
| Agent ReAct | Selecionar tools e coordenar resposta final. | `src/agent/react_agent.py` |
| RAG | Recuperar contexto em documentos locais. | `src/rag/`, `data/rag/raw/` |
| Guardrails | Validar input/output, PII e prompt injection. | `src/security/guardrails.py` |
| Observabilidade | Expor metricas de API, chat, tools e predicoes. | `monitoring/`, `api/main.py` |

## Entradas e Saidas

Entradas principais:

- `/predict`: `cep`, `area_do_terreno_m2`, `ano`/`mes` ou `ano_mes`.
- `/chat`: `message` e opcionalmente `property_data`.

Saidas principais:

- `/predict`: `valor_estimado`, `unidade`, `versao_modelo`.
- `/chat`: `answer`, `tools_used`, metadados de provider/model/latencia/chunks.

## Controles de Seguranca

- Validacao Pydantic nos endpoints.
- `validate_input` para CEP, area, ano e mes.
- `validate_output` para bloquear valor negativo.
- `validate_text_policy` para prompt injection, PII e segredos.
- Agent usa allowlist de tools.
- `thought` interno nao deve aparecer na resposta final.
- Erros de provedor LLM e agent retornam mensagens controladas.

## Limitacoes

- O sistema nao substitui avaliacao profissional de imovel.
- O modelo usa poucos atributos e pode errar em casos com caracteristicas nao representadas.
- RAG depende dos documentos locais indexados.
- LLM externa pode variar resposta; por isso tools e respostas controladas sao priorizadas.

## Monitoramento

Metricas expostas:

- `business_prediction_requests`
- `business_prediction_value_brl`
- `business_chat_requests`
- `business_chat_tools_used`
- `business_chat_latency_seconds`

Grafana e Prometheus podem ser iniciados com `docker compose up --build`.

## Uso Pretendido

- Demonstracao de MLOps, RAG e agent para Datathon.
- Estimativas exploratorias de valor com dados publicos.
- Explicacoes documentais sobre fatores de precificacao.

## Uso Nao Pretendido

- Decisao financeira automatizada sem revisao humana.
- Avaliacao juridica, fiscal ou crediticia.
- Processamento de CPF, CNPJ, telefone, e-mail, documentos pessoais ou secrets.
