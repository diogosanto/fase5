from typing import Iterable

import pandas as pd


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    truth = pd.Series(y_true, dtype="float64").reset_index(drop=True)
    pred = pd.Series(y_pred, dtype="float64").reset_index(drop=True)
    residual = truth - pred
    absolute_error = residual.abs()
    denominator = ((truth - truth.mean()) ** 2).sum()
    r2 = 0.0 if denominator == 0 else 1 - float((residual**2).sum() / denominator)

    return {
        "n": int(len(truth)),
        "mae": float(absolute_error.mean()),
        "median_absolute_error": float(absolute_error.median()),
        "p95_absolute_error": float(absolute_error.quantile(0.95)),
        "bias": float((pred - truth).mean()),
        "r2": float(r2),
    }


def segment_metrics(
    frame: pd.DataFrame,
    y_true: Iterable[float],
    y_pred: Iterable[float],
    min_group_size: int = 30,
) -> dict[str, list[dict[str, float | int | str]]]:
    eval_frame = frame.copy().reset_index(drop=True)
    eval_frame["y_true"] = pd.Series(y_true, dtype="float64").reset_index(drop=True)
    eval_frame["y_pred"] = pd.Series(y_pred, dtype="float64").reset_index(drop=True)
    eval_frame["absolute_error"] = (eval_frame["y_true"] - eval_frame["y_pred"]).abs()
    eval_frame["period"] = eval_frame["ano"].astype(int).astype(str) + "-" + eval_frame["mes"].astype(int).astype(str).str.zfill(2)
    eval_frame["target_band"] = pd.qcut(
        eval_frame["y_true"],
        q=4,
        labels=["baixo", "medio_baixo", "medio_alto", "alto"],
        duplicates="drop",
    ).astype(str)

    return {
        "by_period": _group_metrics(eval_frame, "period", min_group_size),
        "by_target_band": _group_metrics(eval_frame, "target_band", min_group_size),
        "worst_bairros_by_mae": _group_metrics(eval_frame, "bairro", min_group_size)[:10],
    }


def _group_metrics(frame: pd.DataFrame, column: str, min_group_size: int) -> list[dict[str, float | int | str]]:
    rows = []
    for group_value, group in frame.groupby(column, dropna=False):
        if len(group) < min_group_size:
            continue

        rows.append(
            {
                column: str(group_value),
                "n": int(len(group)),
                "mae": float(group["absolute_error"].mean()),
                "median_absolute_error": float(group["absolute_error"].median()),
                "bias": float((group["y_pred"] - group["y_true"]).mean()),
            }
        )

    return sorted(rows, key=lambda item: item["mae"], reverse=True)
