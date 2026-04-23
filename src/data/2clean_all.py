import os
import pandas as pd
import unicodedata
import re


INTERIM_PATH = "data/interim/itbi_2023_2025_raw.csv"
PROCESSED_DIR = "data/processed"


def normalize_text(text):
    if pd.isna(text):
        return None
    text = str(text)
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
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


def clean_all():
    print(f"\n📄 Lendo arquivo consolidado: {INTERIM_PATH}")
    df = pd.read_csv(INTERIM_PATH, sep=";", dtype=str, low_memory=False)

    print("🔧 Padronizando nomes de colunas...")
    df.columns = [snake_case(c) for c in df.columns]

    print("🔢 Convertendo números...")
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

    print("📅 Convertendo datas...")
    if "data_de_transacao" in df.columns:
        df["data_de_transacao"] = convert_date(df["data_de_transacao"])

    print("🧹 Normalizando textos importantes...")
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

    print("💾 Salvando dataset final...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "itbi_clean.csv")
    df.to_csv(out_path, index=False, sep=";", encoding="utf-8")

    print(f"\n🎉 LIMPEZA COMPLETA!")
    print(f"✔ Arquivo final salvo em: {out_path}")
    print(f"📊 Total de linhas: {len(df)}")


if __name__ == "__main__":
    clean_all()
