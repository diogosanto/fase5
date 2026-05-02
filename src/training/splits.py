import pandas as pd


def split_temporal_holdout(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_df = df.sort_values(["ano", "mes"]).reset_index(drop=True)
    test_rows = max(1, int(len(ordered_df) * test_size))
    split_index = len(ordered_df) - test_rows

    train_df = ordered_df.iloc[:split_index].copy()
    test_df = ordered_df.iloc[split_index:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Split temporal gerou treino ou teste vazio.")

    return train_df, test_df


def get_period_range(df: pd.DataFrame) -> tuple[str, str]:
    periods = (df["ano"].astype(int) * 100 + df["mes"].astype(int)).sort_values()
    start_period = int(periods.iloc[0])
    end_period = int(periods.iloc[-1])
    return f"{start_period // 100}-{start_period % 100:02d}", f"{end_period // 100}-{end_period % 100:02d}"


def iter_temporal_backtest_splits(
    df: pd.DataFrame,
    windows: int = 3,
    min_train_periods: int = 3,
) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    ordered_df = df.sort_values(["ano", "mes"]).reset_index(drop=True)
    periods = (
        ordered_df[["ano", "mes"]]
        .drop_duplicates()
        .assign(period=lambda frame: frame["ano"].astype(int) * 100 + frame["mes"].astype(int))
        .sort_values("period")["period"]
        .tolist()
    )
    validation_periods = periods[min_train_periods:][-windows:]
    splits = []

    for period in validation_periods:
        train_df = ordered_df[(ordered_df["ano"].astype(int) * 100 + ordered_df["mes"].astype(int)) < period].copy()
        test_df = ordered_df[(ordered_df["ano"].astype(int) * 100 + ordered_df["mes"].astype(int)) == period].copy()
        if train_df.empty or test_df.empty:
            continue
        splits.append((f"{period // 100}-{period % 100:02d}", train_df, test_df))

    return splits
