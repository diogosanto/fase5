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

Copie `.env.example` para `.env` e preencha as variaveis de LLM quando for usar o endpoint `/chat`.

Exemplo usando Groq:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=sua_chave_real_da_groq
GROQ_MODEL=llama-3.1-8b-instant
```

Nesse caso, voce nao precisa preencher `OLLAMA_BASE_URL` nem `OLLAMA_MODEL`.

Exemplo usando Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://servidor:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4
```

Use Ollama quando houver uma LLM quantizada servida por API local, remota ou em container.

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

As features finais evitam vazamento de alvo. O modelo usa somente `cep`, `area_do_terreno_m2`, `ano` e `mes`; colunas como `bairro`, `cep_prefixo`, `valor_m2` e `media_valor_cep` nao entram no treinamento. A etapa gera um contrato de features com colunas obrigatorias, colunas proibidas, cobertura temporal, cobertura por CEP e faixas numericas. O treino e a validacao usam holdout temporal: os registros mais recentes ficam para teste, reduzindo a chance de avaliar o modelo com uma divisao aleatoria otimista. Os relatorios incluem MAE, erro mediano, erro p95, vies, metricas segmentadas por periodo, faixa de valor e CEPs com pior erro, alem de backtesting temporal em multiplas janelas.

O resumo de atendimento da Etapa 1 do Datathon fica em `docs/ETAPA1_DADOS_BASELINE.md`. Ele lista os artefatos esperados de EDA, as evidencias registradas no `dvc.lock`, o roteiro de reproducao com DVC/Docker e o mapeamento das metricas tecnicas para impacto de negocio.

### Metricas de negocio

As metricas salvas em `data/metrics/train_metrics.json` e `data/metrics/validation_dev.json` devem ser interpretadas tambem em linguagem de negocio:

- `mae`: erro medio absoluto em reais; mede quanto a precificacao erra em media.
- `median_absolute_error`: erro tipico em reais; descreve uma previsao comum sem ser dominado por outliers.
- `p95_absolute_error`: risco de erro extremo; mostra o erro nos piores 5% dos casos.
- `bias`: tendencia de superprecificacao ou subprecificacao; valores positivos indicam estimativa acima do real e valores negativos indicam estimativa abaixo do real.
- `r2`: capacidade explicativa do modelo; complementa as metricas em reais, mas nao substitui a leitura de erro financeiro.

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

### Observabilidade com Prometheus e Grafana

Nao e necessario instalar Prometheus ou Grafana separadamente. Ao executar `docker compose up --build`, o Docker Compose baixa e sobe os containers da API, Prometheus e Grafana conforme a configuracao do projeto.

Fluxo recomendado para validar a observabilidade:

1. Verifique se o Docker esta instalado:

```powershell
docker --version
docker compose version
```

2. Suba a stack:

```powershell
docker compose up --build
```

3. Gere trafego na API:

```powershell
Invoke-RestMethod http://localhost:8000/health

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `
  -ContentType "application/json" `
  -Body '{"message":"Quais fatores influenciam o preco de um imovel?"}'
```

4. Valide as metricas expostas pela API:

```powershell
Invoke-RestMethod http://localhost:8000/metrics
```

5. Acesse o Prometheus em `http://localhost:9090` e consulte metricas como:

```text
http_requests_total
http_request_duration_seconds_count
```

6. Acesse o Grafana em `http://localhost:3030`.

Credenciais padrao, se nao alteradas no Compose:

```text
usuario: admin
senha: admin
```

No Grafana, valide se o datasource do Prometheus esta configurado e se os dashboards conseguem consultar as metricas da API. Caso nenhum dado apareca, gere novas chamadas para `/health`, `/predict` ou `/chat` e aguarde alguns segundos para o Prometheus coletar os dados.

## Testes

```powershell
pytest -q
```

## EDA de valor por m2

Gere a analise de valor medio por m2. Quando o dataset de features nao tiver `bairro`, a analise usa `cep` como agrupamento:

```powershell
python src/features/modelagem/eda_valor_m2_bairro.py --input data/processed/itbi_features_minimal.csv
```

A saida fica em `src/data/eda/features_modelagem/<nome_do_dataset>_<hash>`, com cinco arquivos Excel. Todos possuem aba `dados` e aba `grafico`:

- `valor_m2_por_bairro.xlsx`: media e variancia do valor por m2 por agrupamento, incluindo resumo do valor medio do m2 de Sao Paulo.
- `top_10_bairros_maior_valor_m2.xlsx`: 10 agrupamentos com maior preco por m2.
- `bottom_30_bairros_menor_valor_m2.xlsx`: 30 agrupamentos com menor preco por m2.
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

A camada de RAG e Agent suporta LLM quantizada servida via API compativel com Ollama e mantem Groq como provider padrao/fallback operacional.

Variaveis principais:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4

LLM_MAX_TOKENS=300
LLM_TEMPERATURE=0.2
LLM_TIMEOUT_SECONDS=30
```

Para usar Groq, defina `LLM_PROVIDER=groq` e preencha `GROQ_API_KEY`. Para usar Ollama, defina `LLM_PROVIDER=ollama` e configure `OLLAMA_BASE_URL` e `OLLAMA_MODEL`.

A justificativa completa esta em [docs/LLM_SERVING_DECISION.md](docs/LLM_SERVING_DECISION.md).
