# Governanca do Sistema

## Objetivo

Definir controles minimos de governanca para o sistema de precificacao imobiliaria com API FastAPI, modelo MLflow, RAG documental, Agent ReAct e LLM servida via API.

## Responsabilidades

| Area | Responsabilidade | Evidencia |
| --- | --- | --- |
| Dados e modelo | Reproducao do pipeline, validacao de metricas, promocao controlada de modelo. | `dvc.yaml`, `params.yaml`, `src/training/promote_model.py` |
| RAG e Agent | Documentos de apoio, indice vetorial, tools, prompts, benchmark e testes. | `src/rag/`, `src/agent/`, `evaluation/benchmark_agent.py` |
| Seguranca | Guardrails, PII, prompt injection, output seguro e mapeamento OWASP. | `src/security/`, `docs/OWASP_MAPPING.md` |
| Privacidade | Minimizacao, bloqueio de PII e uso seguro de provedores LLM. | `docs/LGPD_PLAN.md`, `.env.example` |
| Observabilidade | Logs, metricas Prometheus, dashboards Grafana e diagnostico operacional. | `api/main.py`, `monitoring/`, `README.md` |

## Controles Aplicados

- Dados e artefatos rastreados com DVC.
- Modelo promovido para `models/prod` apenas apos validacao objetiva.
- Endpoint `/chat` usa Agent/RAG centralizado e nao chama LLM diretamente na API.
- Tools do Agent sao registradas em allowlist.
- Guardrails de input bloqueiam PII, prompt injection, payload vazio e payload grande.
- Guardrails de output bloqueiam PII, vazamento de prompt e certeza financeira/juridica indevida.
- Secrets devem ser injetados por `.env` ou variaveis de ambiente, nunca hardcoded.
- Logs usam `request_id` e nao devem registrar secrets.

## Criterios de Aceite para Demonstracao

Antes da apresentacao, validar:

- `python -m unittest tests.unit.test_guardrails`
- `python -m unittest tests.unit.test_adversarial_scenarios`
- `python -m unittest tests.unit.test_react_agent`
- `python -m unittest tests.unit.test_agent_tools`
- `python scripts/build_rag_index.py`
- `python evaluation/benchmark_agent.py` com `BENCHMARK_MAX_QUESTIONS` reduzido, se necessario.
- `docker compose up --build` para API, Prometheus e Grafana.

## Gestao de Mudancas

- Alteracoes em documentos RAG exigem reconstruir o indice com `python scripts/build_rag_index.py`.
- Alteracoes em prompts/tools devem ser acompanhadas de testes de Agent e cenarios adversariais.
- Alteracoes no modelo devem passar por `dvc repro`, validacao e promocao explicita para `models/prod`.
- Alteracoes em variaveis de LLM devem atualizar `.env.example` e a documentacao aplicavel.

## Riscos Residuais e Mitigacoes

| Risco | Mitigacao |
| --- | --- |
| PII em formato incomum nao detectada por regex. | Revisao periodica dos cenarios adversariais e, em producao, classificador dedicado. |
| Resposta LLM instavel. | Preferir tools estruturadas, limites de token, RAG com fontes e output guardrail. |
| RAG com documento incorreto. | Versionamento dos documentos, build idempotente do indice e teste de fontes. |
| Uso indevido da estimativa como preco garantido. | System Card, Model Card, output guardrail e mensagens de limite de uso. |

