import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_params  # noqa: E402


INPUT = PROJECT_ROOT / "data" / "processed" / "itbi_clean.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "itbi_features_minimal.csv"
REPORT_PATH = PROJECT_ROOT / "data" / "metrics" / "feature_contract.json"
PARAMS = load_params()
FEATURE_PARAMS = PARAMS.get("feature_quality", {})
MODEL_PARAMS = PARAMS.get("model", {})
TARGET = str(MODEL_PARAMS.get("target_column", "valor_venal_de_referencia"))
MIN_ROWS = int(FEATURE_PARAMS.get("min_rows", 1))
MIN_UNIQUE_CEPS = int(FEATURE_PARAMS.get("min_unique_ceps", 1))
MIN_UNIQUE_PERIODS = int(FEATURE_PARAMS.get("min_unique_periods", 1))
AREA_MIN = float(FEATURE_PARAMS.get("area_min", 1))
AREA_MAX = float(FEATURE_PARAMS.get("area_max", 50000))
FORBIDDEN_COLUMNS = set(FEATURE_PARAMS.get("forbidden_columns", []))


def normalize_cep(value):
    if pd.isna(value):
        return None
    digits = "".join(char for char in str(value) if char.isdigit())
    return digits or None


def build_feature_contract(df):
    required_feature_cols = ["cep", "area_do_terreno_m2", TARGET, "ano", "mes"]
    forbidden_present = sorted(FORBIDDEN_COLUMNS.intersection(df.columns))
    period_count = int(df[["ano", "mes"]].drop_duplicates().shape[0]) if {"ano", "mes"}.issubset(df.columns) else 0
    profile = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_column": TARGET,
        "required_columns": required_feature_cols,
        "missing_required_columns": [col for col in required_feature_cols if col not in df.columns],
        "forbidden_columns_present": forbidden_present,
        "unique_ceps": int(df["cep"].nunique()) if "cep" in df.columns else 0,
        "unique_periods": period_count,
        "nulls": {col: int(value) for col, value in df.isna().sum().items()},
        "area_range": {
            "min": float(df["area_do_terreno_m2"].min()) if "area_do_terreno_m2" in df.columns else None,
            "max": float(df["area_do_terreno_m2"].max()) if "area_do_terreno_m2" in df.columns else None,
        },
        "target_range": {
            "min": float(df[TARGET].min()) if TARGET in df.columns else None,
            "median": float(df[TARGET].median()) if TARGET in df.columns else None,
            "max": float(df[TARGET].max()) if TARGET in df.columns else None,
        },
    }
    profile["passed"] = not _feature_contract_failures(profile)
    return profile


def _feature_contract_failures(profile):
    failures = []
    if profile["rows"] < MIN_ROWS:
        failures.append(f"rows={profile['rows']} abaixo do minimo {MIN_ROWS}")
    if profile["unique_ceps"] < MIN_UNIQUE_CEPS:
        failures.append(
            f"unique_ceps={profile['unique_ceps']} abaixo do minimo {MIN_UNIQUE_CEPS}"
        )
    if profile["unique_periods"] < MIN_UNIQUE_PERIODS:
        failures.append(
            f"unique_periods={profile['unique_periods']} abaixo do minimo {MIN_UNIQUE_PERIODS}"
        )
    if profile["missing_required_columns"]:
        failures.append(f"colunas obrigatorias ausentes: {profile['missing_required_columns']}")
    if profile["forbidden_columns_present"]:
        failures.append(f"colunas proibidas presentes: {profile['forbidden_columns_present']}")
    if profile["area_range"]["min"] is not None and profile["area_range"]["min"] < AREA_MIN:
        failures.append(f"area minima abaixo de {AREA_MIN}")
    if profile["area_range"]["max"] is not None and profile["area_range"]["max"] > AREA_MAX:
        failures.append(f"area maxima acima de {AREA_MAX}")
    return failures


def validate_feature_contract(profile):
    failures = _feature_contract_failures(profile)
    if failures:
        raise ValueError("Falhas no contrato de features: " + "; ".join(failures))


def build_features():
    print("Lendo base limpa...")
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset limpo nao encontrado: {INPUT}")

    df = pd.read_csv(INPUT, sep=";", low_memory=False)
    if df.empty:
        raise ValueError(f"Dataset limpo sem linhas: {INPUT}")

    required_cols = [
        "descricao_do_uso_iptu",
        "cep",
        "area_do_terreno_m2",
        TARGET,
        "data_de_transacao",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    df = df[df["descricao_do_uso_iptu"] == "TERRENO"].copy()

    df["cep"] = df["cep"].apply(normalize_cep)

    df["data_de_transacao"] = pd.to_datetime(df["data_de_transacao"], errors="coerce")
    df["ano"] = df["data_de_transacao"].dt.year
    df["mes"] = df["data_de_transacao"].dt.month
    df["ano_mes"] = (df["ano"] * 100 + df["mes"]).astype("Int64")

    df = df[
        df["area_do_terreno_m2"].between(AREA_MIN, AREA_MAX, inclusive="both")
        & df[TARGET].notna()
        & (df[TARGET] > 0)
    ].copy()

    cols = [
        "cep",
        "area_do_terreno_m2",
        TARGET,
        "ano",
        "mes",
    ]
    df_final = df[cols].copy()
    required_feature_cols = ["cep", "area_do_terreno_m2", TARGET, "ano", "mes"]
    rows_before_required_filter = len(df_final)
    df_final = df_final.dropna(subset=required_feature_cols).copy()
    dropped_required_rows = rows_before_required_filter - len(df_final)

    if df_final.empty:
        raise ValueError("Nenhuma linha valida restou apos filtros de terrenos e qualidade.")
    if df_final[required_feature_cols].isna().any().any():
        raise ValueError("Features finais contem nulos em colunas obrigatorias.")
    if not df_final["mes"].between(1, 12).all():
        raise ValueError("Coluna mes contem valores fora do intervalo 1..12.")

    os.makedirs(OUTPUT.parent, exist_ok=True)
    df_final.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8")
    os.makedirs(REPORT_PATH.parent, exist_ok=True)
    contract = build_feature_contract(df_final)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(contract, file, indent=2, ensure_ascii=False)
    validate_feature_contract(contract)

    print(f"Features minimas salvas em: {OUTPUT}")
    print(f"Contrato de features salvo em: {REPORT_PATH}")
    print(f"Linhas: {len(df_final)}")
    print(f"Linhas descartadas por nulos obrigatorios: {dropped_required_rows}")
    print(f"Target configurado: {TARGET}")
    print("Features do modelo: cep, area_do_terreno_m2, ano, mes")
    print("Colunas removidas por vazamento: valor_m2, media_valor_cep")


if __name__ == "__main__":
    build_features()
