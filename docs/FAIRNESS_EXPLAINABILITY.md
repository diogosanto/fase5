# Fairness e Explicabilidade

## Objetivo

Documentar como o projeto avalia riscos de erro desigual, vies regional e explicabilidade para o modelo de precificacao imobiliaria.

## Riscos de Fairness

O modelo usa `cep` como feature. Mesmo sem atributos sensiveis diretos, CEP pode funcionar como proxy de condicoes socioeconomicas e gerar erro desigual por regiao.

Riscos acompanhados:

- maior erro em CEPs com baixa amostra;
- superprecificacao ou subprecificacao sistematica por regiao;
- pior desempenho em faixas de valor extremas;
- drift temporal por mudanca no mercado imobiliario.

## Evidencias no Projeto

- Holdout temporal configurado em `params.yaml`.
- Metricas segmentadas descritas no README: periodo, faixa de valor e CEPs com pior erro.
- Contrato de features em `data/metrics/feature_contract.json`.
- Metricas de treino e validacao em `data/metrics/train_metrics.json` e `data/metrics/validation_dev.json`.
- Tool `region_comparer` para comparar regioes com dados processados.

## Explicabilidade

Explicabilidade operacional adotada:

- features do modelo sao poucas e documentadas: `cep`, `area_do_terreno_m2`, `ano`, `mes`;
- resposta da API informa a versao do modelo;
- Agent/RAG responde perguntas conceituais usando documentos locais;
- Model Card explica target, dados, metricas e limites.

## Criterios Recomendados

Antes de promover novo modelo:

- comparar MAE e p95 global;
- revisar erro por periodo;
- revisar erro por CEP ou prefixo de CEP;
- revisar vies medio por segmento;
- bloquear promocao quando houver regressao material em regioes com amostra suficiente.

## Limitacoes

- O projeto nao possui atributos sensiveis para teste de paridade entre grupos protegidos.
- CEP nao deve ser interpretado como caracteristica causal isolada.
- Explicabilidade atual e documental/operacional; para producao, recomenda-se adicionar SHAP ou importancia de features por predicao.
