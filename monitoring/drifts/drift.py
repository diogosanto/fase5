import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric
)
import os

INPUT = "data/processed/itbi_features_minimal.csv"
OUTPUT = "monitoring/drift_advanced.html"


def load_data():
    df = pd.read_csv(INPUT, sep=";")
    df = df.dropna()
    return df


def split_reference_current(df):
    ref = df.sample(frac=0.7, random_state=42)
    cur = df.drop(ref.index)
    return ref, cur


def run():

    print("📊 Carregando dados...")
    df = load_data()

    print("🔀 Separando dados...")
    ref, cur = split_reference_current(df)

    print("🧠 Gerando relatório avançado...")

    report = Report(metrics=[

        # 🔥 Presets
        DataDriftPreset(),
        TargetDriftPreset(),
        DataQualityPreset(),

        # 🔥 Drift geral do dataset (com threshold)
        DatasetDriftMetric(drift_share=0.3),

        # 🔥 Drift por coluna crítica
        ColumnDriftMetric(column_name="area_do_terreno_m2"),
        ColumnDriftMetric(column_name="ano"),
        ColumnDriftMetric(column_name="mes"),

        # 🔥 Estatísticas detalhadas
        ColumnSummaryMetric(column_name="area_do_terreno_m2"),
        ColumnSummaryMetric(column_name="valor_venal_de_referencia"),
    ])

    report.run(
        reference_data=ref,
        current_data=cur
    )

    os.makedirs("monitoring", exist_ok=True)
    report.save_html(OUTPUT)

    print(f"📄 Relatório salvo em: {OUTPUT}")

    # 🔥 ALERTA automático (nível produção)
    result = report.as_dict()

    dataset_drift = result["metrics"][3]["result"]["dataset_drift"]

    if dataset_drift:
        print("🚨 ALERTA: DRIFT DETECTADO NO DATASET!")
    else:
        print("✅ Dataset está estável")


if __name__ == "__main__":
    run()
