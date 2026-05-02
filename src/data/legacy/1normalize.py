import os
import re
import pandas as pd
import unicodedata


RAW_DIR = "data/raw"
NORMALIZED_DIR = "data/normalized"


# ---------------------------------------------------------
# 1. Funções utilitárias
# ---------------------------------------------------------

def normalize_text(text):
    """Remove acentos, normaliza e deixa maiúsculo."""
    if pd.isna(text):
        return None
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text.upper().strip()


def snake_case(col):
    """Padroniza nomes de colunas."""
    col = normalize_text(col)
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_")


def detect_header(df):
    """Encontra a linha onde está o cabeçalho real."""
    for i, row in df.iterrows():
        if "N° do Cadastro" in str(row.values):
            return i
    return 0  # fallback


def convert_numeric(series):
    """Converte números no padrão BR → float."""
    return (
        series.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False)
        .astype(float)
    )


def convert_date(series):
    """Converte datas mm/dd/yyyy."""
    return pd.to_datetime(series, errors="coerce", dayfirst=False)


# ---------------------------------------------------------
# 2. Normalização principal
# ---------------------------------------------------------

def normalize_file(filepath):
    print(f"\n📄 Normalizando arquivo: {filepath}")

    # Lê sem header
    df_raw = pd.read_excel(filepath, header=None, dtype=str)

    # Detecta cabeçalho real
    header_row = detect_header(df_raw)
    df = pd.read_excel(filepath, header=header_row, dtype=str)

    # Padroniza nomes de colunas
    df.columns = [snake_case(c) for c in df.columns]

    # Remove duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # Converte tipos
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

    # Datas
    if "data_de_transacao" in df.columns:
        df["data_de_transacao"] = convert_date(df["data_de_transacao"])

    # Normaliza textos importantes
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

    # Extrai ano e mês do nome do arquivo
    filename = os.path.basename(filepath)
    match = re.search(r"(\d{4}).*?(\d{2})", filename)
    if match:
        ano, mes = match.group(1), match.group(2)
    else:
        ano, mes = None, None

    df["ano"] = ano
    df["mes"] = mes

    # Salva arquivo normalizado
    os.makedirs(NORMALIZED_DIR, exist_ok=True)
    out_path = os.path.join(NORMALIZED_DIR, f"normalized_{filename.replace('.xlsx', '.csv')}")
    df.to_csv(out_path, index=False, sep=";", encoding="utf-8")

    print(f"✔ Arquivo normalizado salvo em: {out_path}")
    print(f"📊 Linhas: {len(df)}")

    return df


# ---------------------------------------------------------
# 3. Pipeline para normalizar todos os arquivos
# ---------------------------------------------------------

def normalize_all():
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".xlsx")]

    if not files:
        print("⚠ Nenhum arquivo XLSX encontrado em data/raw/")
        return

    for file in files:
        normalize_file(os.path.join(RAW_DIR, file))


if __name__ == "__main__":
    normalize_all()
