# Plano LGPD

## Objetivo

Definir como o projeto trata privacidade, minimizacao de dados e riscos LGPD no sistema de precificacao imobiliaria com dados publicos de ITBI, API de predicao e Agent/RAG.

## Dados Tratados

| Categoria | Uso no projeto | Observacao LGPD |
| --- | --- | --- |
| CEP | Feature do modelo e comparacoes regionais | Dado geolocalizavel; evitar combinar com identificadores pessoais. |
| Area do terreno | Feature do modelo | Dado do imovel, usado para estimativa. |
| Ano e mes | Feature temporal | Usado para contexto de mercado e holdout temporal. |
| Valor venal/de referencia | Target e metricas | Pode ter impacto economico; documentar limitacoes. |
| Mensagem do chat | Entrada do usuario para RAG/agent | Nao deve conter CPF, e-mail, telefone ou secrets. |

## Bases Legais e Minimizacao

- O projeto usa dados publicos de ITBI para finalidade analitica e educacional do Datathon.
- O modelo final usa conjunto minimo de features: `cep`, `area_do_terreno_m2`, `ano`, `mes`.
- Colunas com maior risco de vazamento ou dependencia indevida, como agregados diretos de target, sao proibidas no contrato de features.
- O endpoint `/chat` bloqueia padroes comuns de PII antes de acionar LLM ou tools.

## Controles Implementados

- Sanitizacao de CEP e textos em `api/main.py`.
- Politicas de PII, prompt injection e escopo seguro em `src/security/input_guardrails.py`, `src/security/pii_detection.py` e `src/security/prompt_injection.py`.
- Validacao de saida textual em `src/security/output_guardrails.py` para reduzir risco de resposta com PII ou alegacao financeira indevida.
- `.env.example` sem secrets reais.
- Logs truncados para reduzir exposicao de entrada do usuario.
- Metricas operacionais sem identificador pessoal.
- DVC e contratos de qualidade para rastrear pipeline e artefatos.

## Direitos do Titular e Retencao

Como MVP local, o projeto nao implementa cadastro de usuarios nem persistencia nominal de conversas. Caso evolua para producao:

- registrar base legal e finalidade por endpoint;
- criar politica de retencao de logs;
- anonimizar ou mascarar CEP quando granularidade fina nao for necessaria;
- permitir remocao de registros de atendimento quando houver identificador de usuario;
- manter inventario de operadores/suboperadores, incluindo provedor LLM.

## Compartilhamento com Terceiros

O endpoint `/chat` pode usar Groq ou Ollama conforme `LLM_PROVIDER`.

- Groq: mensagens podem ser enviadas ao provedor externo configurado.
- Ollama: mensagens ficam no ambiente que hospeda o endpoint Ollama.

Mitigacao: bloquear PII antes da chamada e documentar provider/model em metadados.

## Riscos Residuais

- CEP pode ser dado quase identificador quando combinado com outras fontes.
- Regex de PII nao garante cobertura total.
- Provedores externos de LLM devem ser avaliados juridicamente antes de producao.
