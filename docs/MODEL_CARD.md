# Model Card

## Modelo

Modelo de precificacao imobiliaria treinado no pipeline DVC/MLflow para estimar `valor_venal_de_referencia` com features minimizadas.

## Finalidade

Gerar estimativa exploratoria de valor para terrenos/imoveis a partir de dados publicos de ITBI e servir essa estimativa via API ou tool do Agent.

## Features

Features usadas pelo modelo:

- `cep`
- `area_do_terreno_m2`
- `ano`
- `mes`

O projeto evita features com vazamento direto de alvo, como `valor_m2` e `media_valor_cep`, conforme `params.yaml`.

## Target

Target padrao:

- `valor_venal_de_referencia`

Caso o objetivo seja preco declarado de venda, o README orienta alterar `model.target_column` e reproduzir a esteira.

## Dados

Origem: dados publicos de ITBI de Sao Paulo processados pelo pipeline do projeto.

Artefatos relevantes:

- `data/processed/itbi_clean.csv`
- `data/processed/itbi_features_minimal.csv`
- `data/metrics/feature_contract.json`
- `data/metrics/train_metrics.json`
- `data/metrics/validation_dev.json`

## Avaliacao

Metricas acompanhadas:

- MAE
- erro mediano absoluto
- erro p95
- vies
- R2
- metricas segmentadas por periodo, faixa de valor e CEP

A validacao usa holdout temporal para reduzir avaliacao otimista por divisao aleatoria.

## Fairness e Segmentos

Como o modelo usa CEP, existe risco de desempenho desigual por regiao. O projeto acompanha metricas segmentadas e recomenda avaliar:

- erro por CEP ou prefixo de CEP;
- erro por periodo;
- erro por faixa de valor;
- regioes com pior MAE/p95.

Nao ha uso intencional de atributos sensiveis como raca, religiao, genero ou saude.

## Limitacoes

- Poucas features reduzem risco de vazamento, mas tambem limitam precisao.
- CEP pode capturar diferencas socioeconomicas regionais.
- A estimativa nao considera estado de conservacao, andar, vagas, documentacao, reforma ou condicoes negociais.
- Valores extremos podem ter erro maior, acompanhado por p95.

## Riscos

- Superprecificacao ou subprecificacao em regioes pouco representadas.
- Uso indevido da estimativa como decisao final.
- Drift temporal no mercado imobiliario.

## Mitigacoes

- Holdout temporal e backtesting.
- Contrato de features e quality gates.
- Monitoramento de valor estimado e latencia.
- Guardrails de entrada e saida.
- Documentacao de uso pretendido e nao pretendido no System Card.
