import os
import pandas as pd


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def load_raw_files(raw_dir=RAW_DIR):
    """Carrega todos os arquivos XLSX da pasta raw."""
    files = [f for f in os.listdir(raw_dir) if f.endswith(".xlsx")]
    dfs = []

    for file in files:
        path = os.path.join(raw_dir, file)
        print(f"📄 Lendo arquivo: {path}")
        df = pd.read_excel(path)
        df["ano_referencia"] = int(file.split("_")[-1].split(".")[0])
        dfs.append(df)

    return dfs


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas para snake_case."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("-", "_")
    )
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e padroniza um DataFrame individual."""
    df = standardize_columns(df)

    # Remove colunas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # Remove colunas sem nome (ex: 'Unnamed: 0')
    df = df.loc[:, ~df.columns.str.contains("unnamed")]

    # Converte números quando possível
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Converte datas quando possível
    for col in df.columns:
        col_str = str(col).lower()
        if "data" in col_str or "dt" in col_str:
            df[col] = pd.to_datetime(df[col], errors="ignore")

    return df




def process_all_files():
    """Pipeline completo de limpeza e consolidação."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    dfs = load_raw_files()

    cleaned = []
    for df in dfs:
        cleaned.append(clean_dataframe(df))

    final_df = pd.concat(cleaned, ignore_index=True)

    output_path = os.path.join(PROCESSED_DIR, "itbi_clean.parquet")
    final_df.to_parquet(output_path, index=False)

    print(f"\n🎉 Dataset limpo salvo em: {output_path}")
    print(f"📊 Total de linhas: {len(final_df)}")


if __name__ == "__main__":
    process_all_files()
