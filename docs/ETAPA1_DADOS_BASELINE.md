# Etapa 1 - Dados + Baseline

Este documento consolida os criterios da Etapa 1 do Datathon Fase 5 e as evidencias do projeto para dados, EDA, baseline, DVC, Docker, metricas e dependencias.

## Status dos criterios

| Criterio | Status | Evidencia |
| --- | --- | --- |
| EDA documentada | OK | A etapa `eda` esta declarada no `dvc.yaml`, gera artefatos em `src/data/eda/features_modelagem` e esta documentada no README. O `dvc.lock` registra uma execucao com 10 artefatos de EDA, totalizando 603722 bytes. |
| Baseline treinado e metricas no MLflow | OK | `src/training/train_mlflow.py` treina baseline media, regressao linear, ridge e random forest; registra parametros, tags, metricas e modelo no MLflow. |
| Pipeline DVC + Docker reproduzivel | OK | `dvc.yaml` define ingestao, extracao, limpeza, features, EDA, treino, validacao e repro_check. `Dockerfile` e `docker-compose.yml` sobem API, Prometheus, Alertmanager e Grafana. Artefatos grandes ficam fora do Git por `.gitignore` e sao regenerados pelo pipeline. |
| Metricas de negocio mapeadas para metricas tecnicas | OK | Este documento mapeia MAE, erro mediano, p95, vies e R2 para impacto de negocio. O treino e a validacao salvam essas metricas em `data/metrics`. |
| `pyproject.toml` com dependencias | OK | `pyproject.toml` lista DVC, MLflow, pandas, scikit-learn, FastAPI, pytest, openpyxl, matplotlib, Evidently, LangChain, ChromaDB e bibliotecas de API/monitoramento. |

## Artefatos esperados da EDA

A EDA e executada por:

```powershell
python src/features/modelagem/eda_valor_m2_bairro.py --input data/processed/itbi_features_minimal.csv
```

Tambem pode ser reproduzida pelo DVC:

```powershell
dvc repro eda
```

Saida esperada:

```text
src/data/eda/features_modelagem/<nome_do_dataset>_<hash>/
```

Arquivos Excel esperados:

- `valor_m2_por_bairro.xlsx`
- `top_10_bairros_maior_valor_m2.xlsx`
- `bottom_30_bairros_menor_valor_m2.xlsx`
- `frequencia_venda_por_m2.xlsx`
- `frequencia_venda_por_valor.xlsx`

Imagens PNG esperadas em `images/`:

- `valor_m2_por_bairro.png`
- `top_10_bairros_maior_valor_m2.png`
- `bottom_30_bairros_menor_valor_m2.png`
- `frequencia_venda_por_m2.png`
- `frequencia_venda_por_valor.png`

Observacao: esses arquivos sao artefatos grandes/intermediarios e estao ignorados no Git por `.gitignore`. A execucao registrada em `dvc.lock` mostra que a etapa `eda` ja produziu 10 arquivos em `src/data/eda/features_modelagem`.

## Evidencias do pipeline DVC

O pipeline completo e:

```powershell
dvc repro
```

Etapas individuais:

```powershell
dvc repro ingest
dvc repro extract
dvc repro clean
dvc repro features
dvc repro eda
dvc repro train
dvc repro validate
dvc repro repro_check
```

Artefatos rastreados pelo `dvc.lock`:

| Etapa | Artefato | Evidencia no lock |
| --- | --- | --- |
| `ingest` | `src/data/raw` | 4 arquivos, aproximadamente 114 MB |
| `extract` | `data/interim/itbi_2023_2025_raw.csv` | aproximadamente 210 MB |
| `clean` | `data/processed/itbi_clean.csv` e perfil de qualidade | dataset limpo e `itbi_clean_profile.json` |
| `features` | `data/processed/itbi_features_minimal.csv` e `feature_contract.json` | dataset de features e contrato validado |
| `eda` | `src/data/eda/features_modelagem` | 10 arquivos de EDA, aproximadamente 604 KB |
| `train` | `models/dev` e `train_metrics.json` | modelo treinado, metricas e artefatos MLflow |
| `validate` | `validation_dev.json` | validacao em holdout temporal |
| `repro_check` | `reproducibility_report.json` | checklist automatizado de reprodutibilidade |

## Docker

Validacao estrutural:

```powershell
docker compose config
```

Execucao da stack:

```powershell
docker compose up --build
```

Servicos esperados:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3030`
- Alertmanager: `http://localhost:9093`

Antes de subir a API em modo produtivo, garanta que exista um modelo promovido em `models/prod/model_<versao>`. Se o modelo ainda estiver apenas em `models/dev`, promova com:

```powershell
python src/training/promote_model.py --from-env dev --to-env prod --max-mae <limite_mae>
```

## Mapeamento de metricas tecnicas para negocio

| Metrica tecnica | Leitura de negocio | Como usar na decisao |
| --- | --- | --- |
| MAE | Erro medio absoluto em reais. | Mede quanto a estimativa erra em media. Quanto menor, mais confiavel e a precificacao operacional. |
| Erro mediano | Erro tipico em reais para metade das previsoes. | Ajuda a explicar o erro esperado em casos comuns, reduzindo o efeito de outliers. |
| p95 do erro absoluto | Risco de erro extremo. | Indica o tamanho do erro nos piores 5% dos casos; importante para limites de aprovacao e revisao manual. |
| Vies | Tendencia de superprecificar ou subprecificar. | Vies positivo indica estimativas acima do real; vies negativo indica estimativas abaixo do real. |
| R2 | Capacidade explicativa do modelo. | Mostra quanto da variacao do alvo e explicada pelas features; complementa, mas nao substitui metricas em reais. |

## Confirmacao das dependencias

O `pyproject.toml` cobre as principais necessidades da Etapa 1:

- Dados e EDA: `pandas`, `numpy`, `openpyxl`, `matplotlib`, `beautifulsoup4`, `requests`
- Modelo e baseline: `scikit-learn`, `lightgbm`, `xgboost`
- Tracking e versionamento: `mlflow`, `dvc`, `PyYAML`
- API e testes: `fastapi`, `uvicorn`, `pytest`
- Monitoramento e extensoes posteriores: `prometheus-fastapi-instrumentator`, `evidently`

## Checklist rapido para banca

Antes da demonstracao, execute:

```powershell
python -m pytest -q
dvc repro
dvc metrics show
docker compose config
```

Se o ambiente local estiver sem dependencias, instale antes:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```
