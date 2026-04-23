import os
import pandas as pd
import unicodedata


INTERIM_PATH = "data/interim/itbi_raw.csv"
PROCESSED_DIR = "data/processed"


def normalize_text(text):
    """Remove acentos, converte para ASCII e deixa maiúsculo."""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.upper().strip()


def load_data(path=INTERIM_PATH):
    print(f"📄 Lendo arquivo consolidado: {path}")
    return pd.read_csv(
        path,
        sep=";",
        decimal=",",
        thousands=".",
        encoding="utf-8",
        low_memory=False
    )


def clean_and_filter(df):
    print("🧹 Limpando e filtrando dados...")

    # Normaliza coluna de uso
    if "Descrição do uso (IPTU)" in df.columns:
        df["Descrição do uso (IPTU)"] = df["Descrição do uso (IPTU)"].apply(normalize_text)

    # Converte ano
    if "Ano" in df.columns:
        df["Ano"] = pd.to_numeric(df["Ano"], errors="coerce")

    # Filtra terrenos com bairro válido
    df = df[
        (df["Descrição do uso (IPTU)"] == "TERRENO") &
        (df["Bairro"].notna()) &
        (df["Bairro"].astype(str).str.strip().ne(""))
    ].copy()

    return df


def select_and_transform(df):
    print("🔧 Selecionando e transformando colunas...")

    cols = [
        "Bairro",
        "Área do Terreno (m2)",
        "Valor Venal de Referência",
        "Ano"
    ]

    df = df[cols].copy()

    # Converte números
    df["Área do Terreno (m2)"] = pd.to_numeric(df["Área do Terreno (m2)"], errors="coerce")
    df["Valor Venal de Referência"] = pd.to_numeric(df["Valor Venal de Referência"], errors="coerce")

    # Remove inválidos
    df = df[
        df["Área do Terreno (m2)"].notna() &
        df["Valor Venal de Referência"].notna() &
        df["Ano"].notna()
    ]

    # Remove zeros
    df = df[df["Valor Venal de Referência"] > 0]

    # Arredonda
    df["Área do Terreno (m2)"] = df["Área do Terreno (m2)"].round(0).astype("int64")
    df["Valor Venal de Referência"] = df["Valor Venal de Referência"].round(0).astype("int64")
    df["Ano"] = df["Ano"].astype("int64")

    return df


def save_output(df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    ano_min = df["Ano"].min()
    ano_max = df["Ano"].max()

    output_path = f"{PROCESSED_DIR}/itbi_features_{ano_min}_{ano_max}.csv"

    df.to_csv(output_path, index=False, sep=";", encoding="utf-8")

    print(f"\n🎉 Dataset final salvo em: {output_path}")
    print(f"📊 Total de linhas: {len(df)}")


def build_features():
    df = load_data()
    df = clean_and_filter(df)
    df = select_and_transform(df)
    save_output(df)


if __name__ == "__main__":
    build_features()
