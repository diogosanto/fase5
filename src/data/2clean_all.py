import json
import os
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import load_params


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "itbi_2023_2025_raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROCESSED_DIR / "itbi_clean_profile.json"
REQUIRED_COLS = [
    "data_de_transacao",
    "bairro",
    "cep",
    "descricao_do_uso_iptu",
    "area_do_terreno_m2",
    "valor_venal_de_referencia",
]
QUALITY_PARAMS = load_params().get("data_quality", {})
MIN_ROWS = int(QUALITY_PARAMS.get("min_rows", 1))
MIN_TERRAIN_ROWS = int(QUALITY_PARAMS.get("min_terrain_rows", 1))
MAX_REQUIRED_NULL_RATE = QUALITY_PARAMS.get("max_required_null_rate", {})
MAX_NON_POSITIVE_TARGET_ROWS = int(QUALITY_PARAMS.get("max_non_positive_target_rows", 0))


def repair_mojibake(text):
    if pd.isna(text):
        return None

    value = str(text)
    mojibake_markers = (chr(195), chr(194))
    if not any(marker in value for marker in mojibake_markers):
        return value

    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def normalize_text(text):
    if pd.isna(text):
        return None
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.upper().strip()


def snake_case(col):
    col = normalize_text(col)
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def convert_numeric(series):
    return (
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
        .astype(float)
    )


def convert_date(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def fix_object_encoding(df):
    df = df.copy()
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].map(repair_mojibake)
    return df


def drop_embedded_header_rows(df):
    header_like = pd.Series(False, index=df.index)

    if "n_do_cadastro" in df.columns:
        header_like = header_like | df["n_do_cadastro"].map(normalize_text).fillna("").str.contains(
            "N DO CADASTRO",
            regex=False,
        )

    if "data_de_transacao" in df.columns:
        header_like = header_like | df["data_de_transacao"].map(normalize_text).fillna("").eq("DATA DE TRANSACAO")

    return df.loc[~header_like].copy(), int(header_like.sum())


def drop_non_positive_target_rows(df):
    if "valor_venal_de_referencia" not in df.columns:
        return df.copy(), 0

    non_positive_target = df["valor_venal_de_referencia"] <= 0
    return df.loc[~non_positive_target].copy(), int(non_positive_target.sum())


def build_profile(
    df,
    dropped_header_rows=0,
    dropped_duplicate_rows=0,
    dropped_non_positive_target_rows=0,
):
    missing_required_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    critical_null_rate = {
        col: round(float(df[col].isna().mean()), 4)
        for col in REQUIRED_COLS
        if col in df.columns
    }
    quality_checks = {
        "duplicated_rows": int(df.duplicated().sum()),
        "embedded_header_rows_removed": int(dropped_header_rows),
        "duplicate_rows_removed": int(dropped_duplicate_rows),
        "non_positive_target_rows_removed": int(dropped_non_positive_target_rows),
        "critical_null_rate": critical_null_rate,
    }

    if "data_de_transacao" in df.columns:
        valid_dates = df["data_de_transacao"].dropna()
        quality_checks["date_range"] = {
            "min": str(valid_dates.min().date()) if not valid_dates.empty else None,
            "max": str(valid_dates.max().date()) if not valid_dates.empty else None,
        }

    if "descricao_do_uso_iptu" in df.columns:
        quality_checks["terrain_rows"] = int((df["descricao_do_uso_iptu"] == "TERRENO").sum())

    if "valor_venal_de_referencia" in df.columns:
        quality_checks["non_positive_target_rows"] = int((df["valor_venal_de_referencia"] <= 0).sum())

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_required_cols": missing_required_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "nulls": {col: int(value) for col, value in df.isna().sum().items()},
        "quality_checks": quality_checks,
        "sample": df.head(5).where(pd.notna(df.head(5)), None).to_dict(orient="records"),
    }


def validate_quality_profile(profile):
    failures = []
    rows = profile["rows"]
    checks = profile["quality_checks"]

    if rows < MIN_ROWS:
        failures.append(f"rows={rows} abaixo do minimo {MIN_ROWS}")

    terrain_rows = checks.get("terrain_rows", 0)
    if terrain_rows < MIN_TERRAIN_ROWS:
        failures.append(f"terrain_rows={terrain_rows} abaixo do minimo {MIN_TERRAIN_ROWS}")

    for column, max_rate in MAX_REQUIRED_NULL_RATE.items():
        actual_rate = checks["critical_null_rate"].get(column)
        if actual_rate is not None and actual_rate > float(max_rate):
            failures.append(f"taxa de nulos em {column}={actual_rate} acima de {max_rate}")

    non_positive_rows = checks.get("non_positive_target_rows", 0)
    if non_positive_rows > MAX_NON_POSITIVE_TARGET_ROWS:
        failures.append(
            f"non_positive_target_rows={non_positive_rows} acima de {MAX_NON_POSITIVE_TARGET_ROWS}"
        )

    if failures:
        raise ValueError("Falhas nos criterios de qualidade dos dados: " + "; ".join(failures))


def clean_all():
    print(f"\nLendo arquivo consolidado: {INTERIM_PATH}")
    if not INTERIM_PATH.exists():
        raise FileNotFoundError(f"Arquivo intermediario nao encontrado: {INTERIM_PATH}")

    df = pd.read_csv(INTERIM_PATH, sep=";", dtype=str, low_memory=False)
    if df.empty:
        raise ValueError(f"Arquivo intermediario sem linhas: {INTERIM_PATH}")

    print("Padronizando nomes de colunas...")
    df.columns = [snake_case(c) for c in df.columns]
    df = fix_object_encoding(df)
    df, dropped_header_rows = drop_embedded_header_rows(df)
    print(f"Linhas de cabecalho embutidas removidas: {dropped_header_rows}")

    rows_before_duplicates = len(df)
    df = df.drop_duplicates().copy()
    dropped_duplicate_rows = rows_before_duplicates - len(df)
    print(f"Linhas duplicadas removidas: {dropped_duplicate_rows}")

    print("Convertendo numeros...")
    numeric_cols = [
        "valor_de_transacao_declarado_pelo_contribuinte",
        "valor_venal_de_referencia",
        "proporcao_transmitida",
        "valor_venal_de_referencia_proporcional",
        "base_de_calculo_adotada",
        "valor_financiado",
        "area_do_terreno_m2",
        "testada_m",
        "fracao_ideal",
        "area_construida_m2",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = convert_numeric(df[col])

    print("Convertendo datas...")
    if "data_de_transacao" in df.columns:
        df["data_de_transacao"] = convert_date(df["data_de_transacao"])

    print("Normalizando textos importantes...")
    text_cols = [
        "descricao_do_padrao_iptu",
        "nome_do_logradouro",
        "bairro",
        "natureza_de_transacao",
        "tipo_de_financiamento",
        "descricao_do_uso_iptu",
        "padrao_iptu",
        "uso_iptu",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(normalize_text)

    df, dropped_non_positive_target_rows = drop_non_positive_target_rows(df)
    print(f"Linhas com alvo nao positivo removidas: {dropped_non_positive_target_rows}")

    print("Salvando dataset final e perfil de qualidade...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = PROCESSED_DIR / "itbi_clean.csv"
    df.to_csv(out_path, index=False, sep=";", encoding="utf-8")

    profile = build_profile(
        df,
        dropped_header_rows=dropped_header_rows,
        dropped_duplicate_rows=dropped_duplicate_rows,
        dropped_non_positive_target_rows=dropped_non_positive_target_rows,
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2, default=str, ensure_ascii=False)

    if profile["missing_required_cols"]:
        raise ValueError(f"Colunas obrigatorias ausentes: {profile['missing_required_cols']}")

    critical_nulls = {
        column: profile["nulls"][column]
        for column in [
            "bairro",
            "descricao_do_uso_iptu",
            "area_do_terreno_m2",
            "valor_venal_de_referencia",
        ]
        if profile["nulls"].get(column, 0) == len(df)
    }
    if critical_nulls:
        raise ValueError(f"Colunas criticas totalmente nulas: {critical_nulls}")

    validate_quality_profile(profile)

    print("Limpeza completa.")
    print(f"Arquivo final salvo em: {out_path}")
    print(f"Perfil salvo em: {REPORT_PATH}")
    print(f"Total de linhas: {len(df)}")


if __name__ == "__main__":
    clean_all()
