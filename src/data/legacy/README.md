# Scripts legados

Esta pasta guarda scripts antigos que nao fazem parte do pipeline oficial DVC.

- `1normalize.py`: normalizacao exploratoria de XLSX em `data/raw/`. A normalizacao oficial acontece em `src/data/2clean_all.py`, depois da consolidacao em `data/interim/itbi_2023_2025_raw.csv`.
- `merge_raw.py`: consolidacao simples dos XLSX em `data/interim/itbi_raw.csv`. Foi substituido por `src/data/1extract_all_itbi.py`, que extrai apenas abas mensais, padroniza colunas por posicao e registra arquivo/aba de origem.

Eles foram isolados para reduzir ambiguidade operacional. Nao referencie estes scripts em DVC, testes ou documentacao principal sem reavaliar o fluxo de dados.
