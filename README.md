# Precificador Imobiliario

Projeto de MLOps para precificacao de terrenos urbanos em Sao Paulo usando dados publicos de ITBI, rastreamento com DVC, experimentos com MLflow, API FastAPI, agente RAG e monitoramento com Prometheus/Grafana.

## Requisitos

- Python 3.10 ou 3.11
- Git
- DVC 3.x
- Docker e Docker Compose, para API e monitoramento

## Setup local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Copie `.env.example` para `.env` e preencha as chaves de LLM quando for usar o endpoint `/chat`.

## Pipeline de dados e modelo

Execute a esteira completa:

```powershell
dvc repro
```

Ou rode etapas individuais:

```powershell
dvc repro clean
dvc repro features
dvc repro eda
dvc repro train
dvc repro validate
dvc repro repro_check
```

O pipeline produz:

- `data/processed/itbi_clean.csv`: base limpa.
- `data/processed/itbi_clean_profile.json`: perfil de qualidade rastreado como metrica DVC.
- `data/processed/itbi_features_minimal.csv`: dataset de features sem colunas de vazamento.
- `data/metrics/feature_contract.json`: contrato validado do dataset de features.
- `src/data/eda/features_modelagem/<nome_do_dataset>_<hash>`: artefatos de EDA rastreaveis pelo DVC.
- `models/dev/model_<versao>`: modelo treinado.
- `data/metrics/train_metrics.json`: comparacao de candidatos e melhor modelo.
- `data/metrics/validation_dev.json`: validacao em holdout temporal.
- `data/metrics/reproducibility_report.json`: checklist automatizado de reprodutibilidade.

Consulte metricas:

```powershell
dvc metrics show
```

Os limites de qualidade, contrato de features, target e parametros de treino ficam em `params.yaml`. Mudancas nesses valores invalidam as etapas correspondentes do DVC.

O target padrao do modelo fica em `model.target_column`. O projeto usa `valor_venal_de_referencia` por padrao para manter compatibilidade com a base de features atual. Se o objetivo for estimar o preco declarado de venda, altere para `valor_de_transacao_declarado_pelo_contribuinte` e rode `dvc repro` para regenerar features, EDA, treino e validacao.

## Criterios de qualidade aplicados

### Dados e limpeza

A limpeza remove linhas de cabecalho embutidas nas abas mensais, repara mojibake comum dos arquivos de origem e gera um perfil com nulos, tipos, duplicados, intervalo de datas, linhas de terreno e alvos invalidos. A etapa tambem aplica quality gates parametrizados para falhar cedo quando a base nao atende aos minimos combinados.

### Features e modelagem

As features finais evitam vazamento de alvo. Foram removidas colunas derivadas diretamente de `valor_venal_de_referencia`, como `valor_m2` e `media_valor_cep`. A etapa gera um contrato de features com colunas obrigatorias, colunas proibidas, cobertura temporal, cobertura por bairro e faixas numericas. O treino e a validacao usam holdout temporal: os registros mais recentes ficam para teste, reduzindo a chance de avaliar o modelo com uma divisao aleatoria otimista. Os relatorios incluem MAE, erro mediano, erro p95, vies, metricas segmentadas por periodo, faixa de valor e bairros com pior erro, alem de backtesting temporal em multiplas janelas.

### MLOps e reprodutibilidade

DVC controla a esteira de dados/modelo, MLflow registra candidatos e o Docker Compose sobe API, Prometheus, Alertmanager e Grafana. Artefatos grandes ficam ignorados no Git, os relatorios leves de metricas ficam versionaveis e os parametros criticos ficam centralizados em `params.yaml`. A etapa `repro_check` valida arquivos essenciais, comandos documentados, estagios DVC, dependencias principais e padroes de `.gitignore`.

## Promocao de modelo

Promova um modelo aprovado entre ambientes usando criterios objetivos de metrica. O script le `validation.json` ou `metrics.json` do modelo candidato e usa `mae` como metrica padrao:

- `--max-mae`: criterio absoluto. Bloqueia a promocao se o MAE do candidato estiver acima do limite informado.
- `--improvement-pct`: criterio relativo. Quando ja existe modelo ativo no ambiente de destino, o candidato precisa melhorar o MAE ativo pelo percentual minimo informado.
- Se nao houver modelo ativo no destino, a promocao usa apenas o criterio absoluto configurado.

Exemplos:

```powershell
python src/training/promote_model.py --from-env dev --to-env test --max-mae 50000
python src/training/promote_model.py --from-env test --to-env prod --max-mae 50000 --improvement-pct 5
```

## API e monitoramento

Depois de promover um modelo para `models/prod`, suba a stack:

```powershell
docker compose up --build
```

Endpoints principais:

- `GET /health`
- `POST /predict`
- `POST /chat`
- `GET /metrics`

Servicos:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3030`

## Testes

```powershell
pytest -q
```

## EDA de valor por m2

Gere a analise de valor medio por m2 por bairro:

```powershell
python src/features/modelagem/eda_valor_m2_bairro.py --input data/processed/itbi_features_minimal.csv
```

A saida fica em `src/data/eda/features_modelagem/<nome_do_dataset>_<hash>`, com cinco arquivos Excel. Todos possuem aba `dados` e aba `grafico`:

- `valor_m2_por_bairro.xlsx`: media e variancia do valor por m2 por bairro, incluindo resumo do valor medio do m2 de Sao Paulo.
- `top_10_bairros_maior_valor_m2.xlsx`: 10 bairros com maior preco por m2.
- `bottom_30_bairros_menor_valor_m2.xlsx`: 30 bairros com menor preco por m2.
- `frequencia_venda_por_m2.xlsx`: frequencia de venda por faixa de valor por m2.
- `frequencia_venda_por_valor.xlsx`: frequencia de venda por faixa de valor de venda.

A pasta `images` tambem recebe uma imagem PNG para cada analise, pronta para visualizacao direta:

- `valor_m2_por_bairro.png`
- `top_10_bairros_maior_valor_m2.png`
- `bottom_30_bairros_menor_valor_m2.png`
- `frequencia_venda_por_m2.png`
- `frequencia_venda_por_valor.png`

As tabelas e graficos de frequencia usam os valores ate o percentil 99 quando ha ao menos 100 observacoes, reduzindo distorcao visual por outliers extremos. O dataset bruto de features permanece inalterado.

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