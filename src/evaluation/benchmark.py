import pandas as pd
from sklearn.metrics import mean_absolute_error


def run():
    df = pd.read_csv("data/processed/itbi_features_minimal.csv", sep=";")

    # baseline: média
    baseline = df["valor_venal_de_referencia"].mean()

    df["baseline_pred"] = baseline

    mae_baseline = mean_absolute_error(
        df["valor_venal_de_referencia"],
        df["baseline_pred"]
    )

    print("📊 Benchmark:")
    print(f"Baseline MAE: {mae_baseline:.2f}")

    df.to_csv("evaluation/benchmark_results.csv", index=False)

    print("✔ Benchmark salvo")


if __name__ == "__main__":
    run()