import argparse
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = "src/data/raw"
INTERIM_DIR = "data/interim"
OUTPUT_FILENAME = "itbi_2023_2025_raw.csv"
MONTH_PATTERN = re.compile(
    r"^(JAN|FEV|MAR|ABR|MAI|JUN|JUL|AGO|SET|OUT|NOV|DEZ)-20\d{2}$",
    flags=re.IGNORECASE,
)
ITBI_COLUMNS = [
    "n_do_cadastro",
    "nome_do_logradouro",
    "numero",
    "complemento",
    "bairro",
    "nome_do_condominio",
    "cep",
    "natureza_de_transacao",
    "valor_de_transacao_declarado_pelo_contribuinte",
    "data_de_transacao",
    "valor_venal_de_referencia",
    "proporcao_transmitida",
    "valor_venal_de_referencia_proporcional",
    "base_de_calculo_adotada",
    "tipo_de_financiamento",
    "valor_financiado",
    "cartorio_de_registro",
    "matricula_do_imovel",
    "situacao_do_sql",
    "area_do_terreno_m2",
    "testada_m",
    "fracao_ideal",
    "area_construida_m2",
    "uso_iptu",
    "descricao_do_uso_iptu",
    "padrao_iptu",
    "descricao_do_padrao_iptu",
    "ano_da_construcao",
]


def normalize_text(text):
    if pd.isna(text):
        return None
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.strip()


def resolve_dir(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def is_month_sheet(sheet_name):
    return bool(MONTH_PATTERN.match(str(sheet_name).strip()))


def normalize_columns_by_position(df):
    columns = ITBI_COLUMNS[: df.shape[1]]
    if df.shape[1] > len(ITBI_COLUMNS):
        extra_columns = [f"coluna_extra_{index}" for index in range(len(ITBI_COLUMNS), df.shape[1])]
        columns.extend(extra_columns)

    df = df.copy()
    df.columns = columns
    return df


def read_month_sheet(filepath, sheet):
    df = pd.read_excel(filepath, sheet_name=sheet, header=None, dtype=str)
    df = df.dropna(how="all")

    if df.empty:
        raise ValueError(f"Aba mensal vazia: {sheet}")
    if df.shape[1] < 20:
        raise ValueError(f"Aba mensal com poucas colunas ({df.shape[1]}): {sheet}")

    df = normalize_columns_by_position(df)
    df = df.dropna(how="all")
    df["arquivo"] = os.path.basename(filepath)
    df["aba"] = sheet
    return df


def read_excel_all_sheets(filepath):
    print(f"\nLendo arquivo: {filepath}")
    excel_file = pd.ExcelFile(filepath)
    valid_dataframes = []
    ignored_sheets = []

    for sheet in excel_file.sheet_names:
        print(f"Lendo aba: {sheet}")
        if not is_month_sheet(sheet):
            ignored_sheets.append(sheet)
            print(f"Aba ignorada por nao ser mensal: {sheet}")
            continue

        try:
            df = read_month_sheet(filepath, sheet)
        except Exception as exc:
            raise RuntimeError(f"Falha ao extrair aba mensal {sheet}: {exc}") from exc

        valid_dataframes.append(df)
        print(f"Aba mensal extraida: {sheet} ({len(df)} linhas)")

    if not valid_dataframes:
        raise ValueError(f"Nenhuma aba mensal valida encontrada em {filepath}")

    print(f"Abas nao mensais ignoradas: {ignored_sheets}")
    return pd.concat(valid_dataframes, ignore_index=True)


def extract_all_itbi(raw_dir=RAW_DIR, interim_dir=INTERIM_DIR):
    print("Iniciando extracao completa dos arquivos ITBI...")

    raw_path = resolve_dir(raw_dir)
    interim_path = resolve_dir(interim_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Diretorio de dados brutos nao encontrado: {raw_path}")

    files = sorted(file for file in os.listdir(raw_path) if file.lower().endswith(".xlsx"))
    if not files:
        raise FileNotFoundError(f"Nenhum arquivo XLSX encontrado em {raw_path}")

    dataframes = []
    for file in files:
        filepath = raw_path / file
        dataframes.append(read_excel_all_sheets(filepath))

    final_df = pd.concat(dataframes, ignore_index=True)
    interim_path.mkdir(parents=True, exist_ok=True)
    out_path = interim_path / OUTPUT_FILENAME
    final_df.to_csv(out_path, index=False, sep=";", encoding="utf-8")

    print("Extracao completa.")
    print(f"Arquivo final salvo em: {out_path}")
    print(f"Total de linhas: {len(final_df)}")


def main():
    parser = argparse.ArgumentParser(description="Extracao dos arquivos XLSX de ITBI.")
    parser.add_argument("--raw-dir", default=RAW_DIR)
    parser.add_argument("--interim-dir", default=INTERIM_DIR)
    args = parser.parse_args()

    extract_all_itbi(raw_dir=args.raw_dir, interim_dir=args.interim_dir)


if __name__ == "__main__":
    main()
