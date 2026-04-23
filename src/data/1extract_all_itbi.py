import os
import pandas as pd
import unicodedata


RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"


def normalize_text(text):
    if pd.isna(text):
        return None
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.strip()


def is_valid_sheet(df):
    """
    Detecta se a aba contém dados de guias.
    Critérios:
    - Deve ter pelo menos 10 colunas
    - Deve conter colunas típicas como 'Bairro', 'Valor', 'Data'
    - Deve ter pelo menos 5 linhas de dados
    """
    if df is None or df.empty:
        return False

    if df.shape[1] < 10:
        return False

    colunas = [normalize_text(c).lower() for c in df.columns]

    sinais = [
        "bairro",
        "valor",
        "data",
        "transacao",
        "matricula",
        "cartorio",
        "uso",
        "padrao",
    ]

    score = sum(1 for s in sinais if any(s in c for c in colunas))

    return score >= 3 and len(df) > 5


def read_excel_all_sheets(filepath):
    print(f"\n📄 Lendo arquivo: {filepath}")

    xls = pd.ExcelFile(filepath)
    sheets = xls.sheet_names

    print(f"📑 Abas encontradas: {sheets}")

    dfs_validos = []

    for sheet in sheets:
        print(f"➡ Lendo aba: {sheet}")

        try:
            df = pd.read_excel(filepath, sheet_name=sheet, dtype=str)
        except Exception:
            print(f"⚠ Erro ao ler aba {sheet}, ignorando.")
            continue

        # Remove linhas totalmente vazias
        df = df.dropna(how="all")

        # Se a aba não tem dados de guias, ignorar
        if not is_valid_sheet(df):
            print(f"⛔ Aba ignorada (não contém guias): {sheet}")
            continue

        # Normaliza nomes das colunas
        df.columns = [normalize_text(c) for c in df.columns]

        # Adiciona metadados
        df["arquivo"] = os.path.basename(filepath)
        df["aba"] = sheet

        dfs_validos.append(df)

    if not dfs_validos:
        print("⚠ Nenhuma aba válida encontrada neste arquivo.")
        return None

    return pd.concat(dfs_validos, ignore_index=True)


def extract_all_itbi():
    print("\n🚀 Iniciando extração completa dos arquivos ITBI...")

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".xlsx")]

    if not files:
        print("⚠ Nenhum arquivo XLSX encontrado em data/raw/")
        return

    dfs = []

    for file in files:
        filepath = os.path.join(RAW_DIR, file)
        df = read_excel_all_sheets(filepath)
        if df is not None:
            dfs.append(df)

    if not dfs:
        print("❌ Nenhum dado válido encontrado em nenhum arquivo.")
        return

    final_df = pd.concat(dfs, ignore_index=True)

    os.makedirs(INTERIM_DIR, exist_ok=True)
    out_path = os.path.join(INTERIM_DIR, "itbi_2023_2025_raw.csv")

    final_df.to_csv(out_path, index=False, sep=";", encoding="utf-8")

    print(f"\n🎉 EXTRAÇÃO COMPLETA!")
    print(f"✔ Arquivo final salvo em: {out_path}")
    print(f"📊 Total de linhas: {len(final_df)}")


if __name__ == "__main__":
    extract_all_itbi()
