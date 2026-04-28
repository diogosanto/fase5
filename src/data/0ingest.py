import os
import re
import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "https://prefeitura.sp.gov.br/fazenda/w/acesso_a_informacao/31501"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_html(url: str) -> str:
    """Baixa o HTML da página principal."""
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text


def extract_year_links(html: str, min_year=2023, max_year=2025) -> dict:
    """Extrai links de planilhas ITBI para os anos desejados."""
    soup = BeautifulSoup(html, "html.parser")
    year_links = {}

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if ".xlsx" in href.lower() and "itbi" in href.lower():
            strong = a.find_previous("strong")

            if strong:
                text = strong.get_text(strip=True)
                match = re.search(r"(20\d{2})", text)

                if match:
                    year = int(match.group(1))
                    if min_year <= year <= max_year:
                        year_links[year] = href

    return dict(sorted(year_links.items(), reverse=True))


def download_file(url: str, output_path: str):
    """Baixa um arquivo XLSX para o caminho especificado."""
    with requests.get(url, headers=HEADERS, stream=True) as r:
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_itbi_data(min_year=2023, max_year=2025, output_dir="data/raw"):
    """Pipeline completo: baixa HTML, extrai links e baixa arquivos."""
    print("🔍 Baixando HTML da página...")
    html = fetch_html(BASE_URL)

    print("🔎 Extraindo links dos anos desejados...")
    links = extract_year_links(html, min_year, max_year)

    if not links:
        print("⚠ Nenhum link encontrado para os anos especificados.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for year, link in links.items():
        filename = f"itbi_{year}.xlsx"
        filepath = os.path.join(output_dir, filename)

        print(f"⬇ Baixando {year}...")
        download_file(link, filepath)
        print(f"✔ Arquivo salvo em: {filepath}")

    print("\n🎉 Download concluído com sucesso!")


if __name__ == "__main__":
    download_itbi_data()
