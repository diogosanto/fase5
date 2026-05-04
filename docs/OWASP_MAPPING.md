# OWASP Mapping - LLM, RAG e Agent

## Escopo

Sistema avaliado: API FastAPI de precificacao imobiliaria, endpoint `/predict`, endpoint `/chat`, Agent ReAct, RAG documental e tools `rag_search`, `price_estimator` e `region_comparer`.

Este mapeamento usa como referencia o OWASP Top 10 for LLM Applications 2025, publicado pelo OWASP GenAI Security Project: <https://genai.owasp.org/llm-top-10/>.

## Mapeamento de Ameacas

| ID da ameaca | Nome da ameaca | Como se aplica ao projeto | Nivel de risco | Mitigacao implementada ou planejada | Arquivo/modulo relacionado | Cobertura de teste |
| --- | --- | --- | --- | --- | --- | --- |
| LLM01:2025 | Prompt Injection | Usuario tenta sobrescrever instrucoes do sistema, revelar prompt oculto ou manipular a escolha de tools do Agent. | Alto | Guardrail de input bloqueia padroes conhecidos; prompt do Agent usa allowlist de tools; resposta final nao expoe pensamento interno. | `src/security/input_guardrails.py`, `src/security/prompt_injection.py`, `src/agent/prompts.py`, `src/agent/react_agent.py` | `tests/unit/test_guardrails.py`, `tests/unit/test_adversarial_scenarios.py` |
| LLM02:2025 | Sensitive Information Disclosure | Usuario pode enviar CPF, e-mail, telefone ou tentar obter secrets/API keys; a LLM poderia repetir PII na resposta. | Alto | Deteccao de PII antes do Agent/LLM; validacao de output antes de retornar `/chat`; `.env.example` usa placeholders; logs truncados. | `src/security/pii_detection.py`, `src/security/input_guardrails.py`, `src/security/output_guardrails.py`, `api/main.py`, `.env.example` | `tests/unit/test_guardrails.py`, `tests/unit/test_adversarial_scenarios.py` |
| LLM04:2025 | Data and Model Poisoning | Documentos RAG alterados ou dataset processado manipulado podem induzir respostas incorretas, enviesadas ou inseguras. | Medio | Documentos RAG ficam versionaveis em `data/rag/raw`; script de build preserva fontes; pipeline DVC e contratos de features rastreiam dados/modelo. | `data/rag/raw/`, `scripts/build_rag_index.py`, `src/rag/`, `dvc.yaml`, `params.yaml` | `tests/unit/test_rag_documental.py`, `tests/unit/test_build_rag_index.py` |
| LLM05:2025 | Improper Output Handling | Resposta textual da LLM/Agent pode conter PII, certeza financeira indevida, vazamento de prompt ou resposta vazia. | Alto | `validate_model_output` bloqueia PII, prompt leakage e alegacoes de preco garantido; API usa fallback seguro para output bloqueado. | `src/security/output_guardrails.py`, `api/main.py` | `tests/unit/test_guardrails.py`, `tests/unit/test_adversarial_scenarios.py` |
| LLM06:2025 | Excessive Agency | Agent poderia chamar tool inadequada, repetir passos, usar parametros ruins ou agir com autonomia excessiva. | Medio | Lista fechada de tools; `AGENT_MAX_STEPS`; roteamento por intencao; tools retornam status estruturado e erros controlados. | `src/agent/react_agent.py`, `src/agent/tools.py`, `docs/AGENT_REACT.md`, `docs/AGENT_TOOLS.md` | `tests/unit/test_react_agent.py`, `tests/unit/test_agent_tools.py` |
| LLM08:2025 | Vector and Embedding Weaknesses | RAG pode recuperar contexto irrelevante, fontes inexistentes ou chunks contaminados. | Medio | Retriever limita `RAG_TOP_K`; chunks preservam metadata de fonte; `rag_search` retorna fontes reais e numero de chunks. | `src/rag/`, `src/agent/tools.py`, `data/rag/raw/` | `tests/unit/test_rag_documental.py`, `tests/unit/test_agent_tools.py` |
| LLM10:2025 | Unbounded Consumption | Input muito grande ou loops de Agent podem aumentar custo, latencia e consumo de tokens no Groq/Ollama. | Medio | Limite de tamanho de mensagem; `CHAT_TIMEOUT_SECONDS`; `LLM_MAX_TOKENS`; `AGENT_MAX_STEPS`; `RAG_TOP_K`. | `api/main.py`, `src/security/input_guardrails.py`, `src/agent/react_agent.py`, `src/agent/llm.py`, `.env.example` | `tests/unit/test_guardrails.py`, `tests/unit/test_react_agent.py` |
| OWASP A09:2025 | Security Logging and Alerting Failures | Falhas do Agent, tools ou provedor LLM podem passar despercebidas sem logs e metricas. | Medio | Logs com `request_id`; metricas Prometheus para chat, tools, latencia e predicoes; Grafana provisionado via Docker Compose. | `api/main.py`, `monitoring/`, `README.md` | `tests/integration/test_chat_endpoint.py` |

## Controles Operacionais

- Limite de tamanho do chat: `MAX_CHAT_MESSAGE_LENGTH`, default 1000 caracteres.
- Timeout do chat: `CHAT_TIMEOUT_SECONDS`, default 30 segundos.
- Limite de passos do Agent: `AGENT_MAX_STEPS`, default 3.
- Limite de tokens da LLM: `LLM_MAX_TOKENS`, default 300.
- Limite de recuperacao RAG: `RAG_TOP_K`, default 3.
- Secrets carregados por variaveis de ambiente, sem chave hardcoded.
- Logs nao devem registrar API keys, prompts internos completos ou payload sensivel.

## Riscos Residuais

- Regex de PII e prompt injection nao cobre todas as tecnicas de evasao.
- Provedores externos de LLM devem ser avaliados juridicamente antes de uso em producao real.
- Documentos RAG exigem revisao humana para evitar contexto incorreto ou desatualizado.
- Estimativas de preco nao devem ser usadas como decisao financeira, juridica ou fiscal automatizada.

