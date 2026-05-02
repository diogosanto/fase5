import argparse
import json
import os
import re
import sys
from pathlib import Path

import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_params  # noqa: E402
from src.training.evaluation import regression_metrics, segment_metrics  # noqa: E402
from src.training.splits import get_period_range, split_temporal_holdout  # noqa: E402


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "itbi_features_minimal.csv"
MODEL_PARAMS = load_params().get("model", {})
TARGET = str(MODEL_PARAMS.get("target_column", "valor_venal_de_referencia"))
FEATURES = ["bairro", "cep_prefixo", "area_do_terreno_m2", "ano", "mes"]
TEST_SIZE = float(MODEL_PARAMS.get("test_size", 0.2))
SPLIT_STRATEGY = str(MODEL_PARAMS.get("split_strategy", "temporal_holdout"))
MIN_GROUP_SIZE_FOR_SEGMENTS = int(MODEL_PARAMS.get("min_group_size_for_segments", 30))


def get_model_version(path):
    match = re.search(r"model_(\d+\.\d+\.\d+\.\d+)", str(path))
    return match.group(1) if match else "unknown"


def resolve_model_path(env, version=None):
    base_path = PROJECT_ROOT / "models" / env
    if not base_path.exists():
        raise FileNotFoundError(f"Ambiente '{env}' nao existe em {base_path}")

    if version:
        full_model_path = base_path / f"model_{version}"
        if not full_model_path.exists():
            raise FileNotFoundError(f"Versao {version} nao encontrada em {base_path}")
        return full_model_path

    folders = sorted(
        [path for path in base_path.iterdir() if path.name.startswith("model_")],
        reverse=True,
    )
    if not folders:
        raise FileNotFoundError(f"Nenhum modelo encontrado em {base_path}")

    return folders[0]


def load_holdout_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset de features nao encontrado: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";")
    required_cols = FEATURES + [TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    df = df.dropna(subset=required_cols).copy()
    counts = df["bairro"].value_counts()
    bairros_validos = counts[counts >= 2].index
    df = df[df["bairro"].isin(bairros_validos)].copy()
    if len(df) < 10:
        raise ValueError("Dados insuficientes para validacao apos filtros minimos.")

    _, holdout_df = split_temporal_holdout(df, TEST_SIZE)
    return holdout_df[FEATURES], holdout_df[TARGET], holdout_df


def save_report(path, report):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def validate(env="test", version=None):
    full_model_path = resolve_model_path(env, version)
    version = get_model_version(full_model_path)

    print(f"[INFO] Carregando modelo versao {version} de {full_model_path}...")
    model = mlflow.sklearn.load_model(full_model_path)

    X_test, y_test, holdout_df = load_holdout_data()
    predictions = model.predict(X_test)
    holdout_period_start, holdout_period_end = get_period_range(holdout_df)
    metrics = regression_metrics(y_test, predictions)

    report = {
        "env": env,
        "version": version,
        "model_path": str(full_model_path),
        "split_strategy": SPLIT_STRATEGY,
        "holdout_period_start": holdout_period_start,
        "holdout_period_end": holdout_period_end,
        "n_holdout": metrics["n"],
        "mae": metrics["mae"],
        "median_absolute_error": metrics["median_absolute_error"],
        "p95_absolute_error": metrics["p95_absolute_error"],
        "bias": metrics["bias"],
        "r2": metrics["r2"],
        "prediction_range": {
            "min": float(pd.Series(predictions).min()),
            "p05": float(pd.Series(predictions).quantile(0.05)),
            "median": float(pd.Series(predictions).median()),
            "p95": float(pd.Series(predictions).quantile(0.95)),
            "max": float(pd.Series(predictions).max()),
        },
        "segment_metrics": segment_metrics(
            X_test,
            y_test,
            predictions,
            min_group_size=MIN_GROUP_SIZE_FOR_SEGMENTS,
        ),
        "features": FEATURES,
        "target": TARGET,
    }

    save_report(PROJECT_ROOT / "data" / "metrics" / f"validation_{env}.json", report)

    print(f"[OK] MAE holdout: {report['mae']:,.2f}")
    print(f"[OK] R2 holdout: {report['r2']:.3f}")
    print(
        "[OK] Faixa de predicao: "
        f"{report['prediction_range']['p05']:,.2f} a "
        f"{report['prediction_range']['p95']:,.2f} (p05-p95)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validacao de modelos")
    parser.add_argument("--env", type=str, default="dev", choices=["dev", "test", "prod"])
    parser.add_argument("--version", type=str, default=None)

    args = parser.parse_args()
    validate(env=args.env, version=args.version)
