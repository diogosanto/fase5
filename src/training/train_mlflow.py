import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_params  # noqa: E402
from src.training.evaluation import regression_metrics, segment_metrics  # noqa: E402
from src.training.splits import (  # noqa: E402
    get_period_range,
    iter_temporal_backtest_splits,
    split_temporal_holdout,
)


INPUT = PROJECT_ROOT / "data" / "processed" / "itbi_features_minimal.csv"
MODEL_PARAMS = load_params().get("model", {})
TARGET = str(MODEL_PARAMS.get("target_column", "valor_venal_de_referencia"))
FEATURES = ["bairro", "cep_prefixo", "area_do_terreno_m2", "ano", "mes"]
CATEGORICAL_FEATURES = ["bairro", "cep_prefixo"]
NUMERIC_FEATURES = ["area_do_terreno_m2", "ano", "mes"]
TEST_SIZE = float(MODEL_PARAMS.get("test_size", 0.2))
RANDOM_STATE = int(MODEL_PARAMS.get("random_state", 42))
N_ESTIMATORS = int(MODEL_PARAMS.get("n_estimators", 200))
SPLIT_STRATEGY = str(MODEL_PARAMS.get("split_strategy", "temporal_holdout"))
MIN_GROUP_SIZE_FOR_SEGMENTS = int(MODEL_PARAMS.get("min_group_size_for_segments", 30))
BACKTEST_WINDOWS = int(MODEL_PARAMS.get("backtest_windows", 3))
MODEL_SPECS = [
    ("baseline_media", DummyRegressor(strategy="mean")),
    ("regressao_linear", LinearRegression()),
    ("ridge", Ridge(alpha=1.0, solver="lsqr")),
    (
        "random_forest",
        RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
]


def generate_version():
    return datetime.datetime.now().strftime("%Y.%m.%d.%H%M")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_training_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset de features nao encontrado: {INPUT}")

    df = pd.read_csv(INPUT, sep=";")
    required_cols = FEATURES + [TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    df = df.dropna(subset=required_cols).copy()

    counts = df["bairro"].value_counts()
    bairros_validos = counts[counts >= 2].index
    df = df[df["bairro"].isin(bairros_validos)].copy()

    if len(df) < 10:
        raise ValueError("Dados insuficientes para treino apos filtros minimos.")

    return df


def save_metrics(path, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def get_model_params(model):
    if not hasattr(model, "get_params"):
        return {}

    params = {}
    for key, value in model.get_params().items():
        if isinstance(value, (str, int, float, bool, type(None))):
            params[f"model_param_{key}"] = value
    return params


def train_candidate(model_name, model, X_train, X_test, y_train, y_test, version, dataset_info):
    pipeline = build_pipeline(model)

    with mlflow.start_run(run_name=f"{model_name}_{version}") as run:
        mlflow.set_tags(
            {
                "model_name": model_name,
                "model_version": version,
                "target": TARGET,
                "leakage_removed": "valor_m2,media_valor_cep",
                "stage": "dev",
                "model_type": "regression",
                "training_data_version": dataset_info["training_data_version"],
                "git_sha": dataset_info["git_sha"],
                "split_strategy": dataset_info["split_strategy"],
            }
        )
        mlflow.log_params(
            {
                "model_version": version,
                "model_name": model_name,
                "model_class": model.__class__.__name__,
                "target": TARGET,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "features": ",".join(FEATURES),
                **dataset_info,
                **get_model_params(model),
            }
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metric_values = regression_metrics(y_test, y_pred)
        metrics = {
            "model_name": model_name,
            "run_id": run.info.run_id,
            **metric_values,
        }

        mlflow.log_metrics(
            {
                "mae": metrics["mae"],
                "median_absolute_error": metrics["median_absolute_error"],
                "p95_absolute_error": metrics["p95_absolute_error"],
                "bias": metrics["bias"],
                "r2": metrics["r2"],
            }
        )
        mlflow.sklearn.log_model(pipeline, "model")

    return pipeline, metrics, y_pred


def temporal_backtest(df, model):
    windows = []
    for period, train_df, test_df in iter_temporal_backtest_splits(df, windows=BACKTEST_WINDOWS):
        pipeline = build_pipeline(clone(model))
        pipeline.fit(train_df[FEATURES], train_df[TARGET])
        predictions = pipeline.predict(test_df[FEATURES])
        metrics = regression_metrics(test_df[TARGET], predictions)
        windows.append({"period": period, **metrics})

    if not windows:
        return {"windows": [], "summary": {}}

    mae_values = pd.Series([window["mae"] for window in windows], dtype="float64")
    return {
        "windows": windows,
        "summary": {
            "windows": int(len(windows)),
            "mean_mae": float(mae_values.mean()),
            "max_mae": float(mae_values.max()),
            "min_mae": float(mae_values.min()),
            "std_mae": float(mae_values.std(ddof=0)),
        },
    }


def train_mlflow():
    version = generate_version()
    output = PROJECT_ROOT / "models" / "dev" / f"model_{version}"

    mlflow.set_experiment("itbi-terrenos")

    df = load_training_data()
    train_df, test_df = split_temporal_holdout(df, TEST_SIZE)
    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    train_period_start, train_period_end = get_period_range(train_df)
    test_period_start, test_period_end = get_period_range(test_df)

    dataset_info = {
        "n_rows": int(len(df)),
        "training_data_version": file_sha256(INPUT),
        "git_sha": git_sha(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "split_strategy": SPLIT_STRATEGY,
        "train_period_start": train_period_start,
        "train_period_end": train_period_end,
        "test_period_start": test_period_start,
        "test_period_end": test_period_end,
    }

    results = []
    pipelines = {}
    for model_name, model in MODEL_SPECS:
        pipeline, metrics, predictions = train_candidate(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            version,
            dataset_info,
        )
        results.append(metrics)
        pipelines[model_name] = pipeline
        pipelines[f"{model_name}__predictions"] = predictions
        print(f"{model_name}: MAE={metrics['mae']:,.2f} | R2={metrics['r2']:.3f}")

    best_metrics = min(results, key=lambda item: item["mae"])
    best_model_name = best_metrics["model_name"]
    best_pipeline = pipelines[best_model_name]
    best_predictions = pipelines[f"{best_model_name}__predictions"]
    best_model_spec = dict(MODEL_SPECS)[best_model_name]

    summary = {
        "version": version,
        "run_id": best_metrics["run_id"],
        "best_model": best_model_name,
        "mae": best_metrics["mae"],
        "r2": best_metrics["r2"],
        "target": TARGET,
        "features": FEATURES,
        **dataset_info,
        "segment_metrics": segment_metrics(
            X_test,
            y_test,
            best_predictions,
            min_group_size=MIN_GROUP_SIZE_FOR_SEGMENTS,
        ),
        "temporal_backtest": temporal_backtest(df, best_model_spec),
        "candidates": sorted(results, key=lambda item: item["mae"]),
    }

    os.makedirs(output.parent, exist_ok=True)
    if os.path.exists(output):
        shutil.rmtree(output)
    mlflow.sklearn.save_model(best_pipeline, output)

    save_metrics(os.path.join(output, "metrics.json"), summary)
    save_metrics(PROJECT_ROOT / "data" / "metrics" / "train_metrics.json", summary)

    print(f"Melhor modelo: {best_model_name}")
    print(f"MAE: {summary['mae']:,.2f}")
    print(f"R2: {summary['r2']:.3f}")
    print(f"Modelo salvo em: {output}")
    print("Candidatos registrados no MLflow")


if __name__ == "__main__":
    train_mlflow()
