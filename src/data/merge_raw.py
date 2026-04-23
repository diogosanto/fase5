import os
import pandas as pd

RAW_DIR = "data/raw"
INTERIM_DIR = "data/interim"


def merge_raw_files():
    os.makedirs(INTERIM_DIR, exist_ok=True)

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".xlsx")]
    dfs = []

    for file in files:
        path = os.path.join(RAW_DIR, file)
        print(f"📄 Lendo: {path}")
        df = pd.read_excel(path)
        df["ano_referencia"] = int(file.split("_")[-1].split(".")[0])
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    output_path = os.path.join(INTERIM_DIR, "itbi_raw.csv")
    final_df.to_csv(output_path, index=False, sep=";", encoding="utf-8")

    print(f"\n🎉 Arquivo consolidado salvo em: {output_path}")
    print(f"📊 Total de linhas: {len(final_df)}")


if __name__ == "__main__":
    merge_raw_files()
