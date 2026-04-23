import pandas as pd
import os
import unicodedata


INPUT = "data/processed/itbi_clean.csv"
OUTPUT = "data/processed/itbi_features_minimal.csv"


def normalize_text(text):
    if pd.isna(text):
        return None
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.upper().strip()


def build_features():
    print("📄 Lendo base limpa...")
    df = pd.read_csv(INPUT, sep=";", low_memory=False)

    # -------------------------
    # 1. Filtrar apenas terrenos
    # -------------------------
    df = df[df["descricao_do_uso_iptu"] == "TERRENO"].copy()

    # -------------------------
    # 2. Limpar e padronizar bairro
    # -------------------------
    df["bairro"] = df["bairro"].apply(normalize_text)

    # -------------------------
    # 3. Remover áreas absurdas
    # (ex: terrenos > 50.000 m2)
    # -------------------------
    df = df[df["area_do_terreno_m2"] < 50000]

    # -------------------------
    # 4. Criar valor_m2
    # -------------------------
    df["valor_m2"] = (
        df["valor_venal_de_referencia"] / df["area_do_terreno_m2"]
    )

    # -------------------------
    # 5. Criar ano_mes
    # -------------------------
    df["ano_mes"] = (
        df["data_de_transacao"].str.slice(0, 7).str.replace("-", "")
    )

    # -------------------------
    # 6. Agrupar por CEP (opcional)
    # Criar feature: média do valor venal por CEP
    # -------------------------
    if "cep" in df.columns:
        df["cep"] = df["cep"].astype(str).str.extract(r"(\d+)", expand=False)
        media_cep = (
            df.groupby("cep")["valor_venal_de_referencia"]
            .mean()
            .rename("media_valor_cep")
        )
        df = df.merge(media_cep, on="cep", how="left")

    # -------------------------
    # 7. Selecionar colunas finais
    # -------------------------
    cols = [
        "bairro",
        "cep",
        "area_do_terreno_m2",
        "valor_venal_de_referencia",
        "valor_m2",
        "ano_mes",
        "media_valor_cep",
        "data_de_transacao",
    ]

    df_final = df[cols].copy()

    # -------------------------
    # 8. Exportar
    # -------------------------
    os.makedirs("data/processed", exist_ok=True)
    df_final.to_csv(OUTPUT, sep=";", index=False, encoding="utf-8")

    print(f"🎉 Features mínimas salvas em: {OUTPUT}")
    print(f"📊 Linhas: {len(df_final)}")


if __name__ == "__main__":
    build_features()
