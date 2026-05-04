# OWASP LLM Top 10 - Mapeamento de Ameacas

## Escopo

Sistema avaliado: API FastAPI de precificacao imobiliaria, endpoint `/predict`, endpoint `/chat`, Agent ReAct, RAG documental, tools `rag_search`, `price_estimator` e `region_comparer`.

Artefatos relacionados:

- `api/main.py`
- `src/security/guardrails.py`
- `src/agent/prompts.py`
- `src/agent/react_agent.py`
- `src/agent/tools.py`
- `docs/LLM_SERVING_DECISION.md`
- `docs/AGENT_REACT.md`

## Mapeamento

| Ameaca OWASP | Risco no projeto | Mitigacoes implementadas | Evidencia |
| --- | --- | --- | --- |
| LLM01 Prompt Injection | Usuario tenta sobrescrever instrucoes do sistema, expor prompt ou forcar uso indevido de tools. | Prompt do agente limita tools e exige JSON; `validate_text_policy` bloqueia padroes de prompt injection; roteamento heuristico reduz chamadas desnecessarias a LLM. | `src/security/guardrails.py`, `src/agent/prompts.py`, `src/agent/react_agent.py` |
| LLM02 Insecure Output Handling | Resposta do modelo ou da LLM poderia retornar valor invalido, vazio ou erro bruto. | `validate_output` bloqueia predicao negativa; API transforma erros do agent/provedor em respostas controladas; resposta final nao expoe `thought`. | `src/security/guardrails.py`, `api/main.py`, `docs/AGENT_REACT.md` |
| LLM03 Training Data Poisoning | Dados publicos ou documentos RAG alterados poderiam induzir respostas e metricas ruins. | DVC rastreia pipeline; quality gates em `params.yaml`; RAG usa documentos locais versionaveis; reproducibilidade e contrato de features documentados. | `dvc.yaml`, `params.yaml`, `data/rag/raw/`, `data/metrics/reproducibility_report.json` |
| LLM05 Improper Output / Agency | Agent poderia usar tool errada, repetir passos ou inventar resultado. | Limite `AGENT_MAX_STEPS`; lista fechada de tools; fallback controlado; regra "Nao invente resultados de tools"; mensagens de campos faltantes. | `src/agent/prompts.py`, `src/agent/react_agent.py`, `src/agent/tools.py` |
| LLM06 Sensitive Information Disclosure | Usuario envia CPF, CNPJ, e-mail, telefone ou pede secrets. | `validate_text_policy` bloqueia PII comum e termos de segredo; `.env.example` usa placeholders; logs truncam mensagens e nao registram API keys. | `src/security/guardrails.py`, `.env.example`, `api/main.py`, `docs/LLM_SERVING_DECISION.md` |
| LLM07 Insecure Plugin / Tool Design | Tools executam acesso a modelo, RAG e dataset processado. | Tools sao funcoes internas registradas em allowlist; entradas sao normalizadas; campos obrigatorios sao validados; erros retornam payload controlado. | `src/agent/tools.py` |
| LLM09 Overreliance | Usuario pode interpretar estimativa como valor definitivo. | System Card e Model Card documentam limites; respostas de preco indicam que a estimativa vem do modelo; metricas de erro sao acompanhadas. | `docs/MODEL_CARD.md`, `docs/SYSTEM_CARD.md`, `data/metrics/validation_dev.json` |

## Controles Operacionais

- Limite de tamanho do chat: `MAX_CHAT_MESSAGE_LENGTH=1000`.
- Timeout do chat: `CHAT_TIMEOUT_SECONDS`, default 30s.
- Limite de passos do agent: `AGENT_MAX_STEPS`, default 3 e maximo 10 no agente.
- Observabilidade: metricas Prometheus para predicao, chat, tools, latencia e valores estimados.
- Secrets: variaveis de ambiente para Groq/Ollama; sem chave hardcoded.

## Riscos Residuais

- Padroes de prompt injection e PII sao baseados em regex e nao substituem um classificador dedicado.
- RAG depende da qualidade dos documentos versionados em `data/rag/raw/`.
- Ollama remoto deve ser protegido por rede, firewall ou proxy quando exposto fora do host local.
- O modelo estima valor venal/de referencia e nao deve ser usado como unica base para decisao financeira.
