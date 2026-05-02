# Dados ITBI

Esta pasta concentra o fluxo oficial de ingestao, extracao e limpeza dos dados publicos de ITBI usados no modelo de precificacao de terrenos.

## Origem dos dados

- Fonte primaria: arquivos publicos de ITBI do municipio de Sao Paulo.
- Pagina de origem: `https://prefeitura.sp.gov.br/fazenda/w/acesso_a_informacao/31501`.
- Entrada bruta oficial: arquivos XLSX em `src/data/raw/`.
- Manifesto da origem: `src/data/raw/itbi_sources.json`, gerado pela ingestao com a URL de cada ano baixado.

## Fluxo oficial

1. `0ingest.py`
   - Baixa os XLSX publicos de ITBI para `src/data/raw/`.
   - Gera `src/data/raw/itbi_sources.json`.
   - Stage DVC: `ingest`.

2. `1extract_all_itbi.py`
   - Le todos os XLSX de `src/data/raw/`.
   - Processa apenas abas mensais no padrao `JAN-2023`, `FEV-2023` etc.
   - Padroniza colunas por posicao, preservando `arquivo` e `aba` como lineage.
   - Gera `data/interim/itbi_2023_2025_raw.csv`.
   - Stage DVC: `extract`.

3. `2clean_all.py`
   - Padroniza nomes de colunas.
   - Corrige mojibake comum dos arquivos de origem.
   - Remove linhas de cabecalho embutidas.
   - Remove duplicadas.
   - Converte numericos e datas em formato brasileiro.
   - Normaliza textos importantes.
   - Remove linhas com target nao positivo.
   - Gera `data/processed/itbi_clean.csv`.
   - Gera `data/processed/itbi_clean_profile.json`.
   - Stage DVC: `clean`.

## Artefatos gerados

- `src/data/raw/`: XLSX brutos e manifesto de origem.
- `data/interim/itbi_2023_2025_raw.csv`: consolidado intermediario extraido dos XLSX.
- `data/processed/itbi_clean.csv`: dataset limpo usado pela etapa de features.
- `data/processed/itbi_clean_profile.json`: perfil de qualidade com colunas, tipos, nulos, amostra, duplicadas, periodo, linhas de terreno e alvos invalidos.

## Contrato minimo para modelagem

O dataset limpo precisa conter:

- `data_de_transacao`
- `bairro`
- `cep`
- `descricao_do_uso_iptu`
- `area_do_terreno_m2`
- `valor_venal_de_referencia`

As features de treino nao devem ser derivadas diretamente de `valor_venal_de_referencia`, para evitar vazamento de alvo.

## Scripts legados

Scripts antigos ou duplicados ficam em `src/data/legacy/`. Eles nao fazem parte do pipeline oficial e nao devem ser usados por DVC sem nova revisao.
