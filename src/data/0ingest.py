import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://prefeitura.sp.gov.br/fazenda/w/acesso_a_informacao/31501"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT_SECONDS = 30
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "raw"


def resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def fetch_html(url: str) -> str:
    """Baixa o HTML da pagina principal."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def extract_year_links(html: str, min_year=2023, max_year=2025) -> dict[int, str]:
    """Extrai links de planilhas ITBI para os anos desejados."""
    soup = BeautifulSoup(html, "html.parser")
    year_links = {}

    for link_tag in soup.find_all("a", href=True):
        href = link_tag["href"]
        if ".xlsx" not in href.lower() or "itbi" not in href.lower():
            continue

        surrounding_text = " ".join(
            part.get_text(" ", strip=True)
            for part in [link_tag.find_previous("strong"), link_tag.parent]
            if part is not None
        )
        match = re.search(r"(20\d{2})", surrounding_text)
        if not match:
            match = re.search(r"(20\d{2})", href)
        if not match:
            continue

        year = int(match.group(1))
        if min_year <= year <= max_year:
            year_links[year] = urljoin(BASE_URL, href)

    return dict(sorted(year_links.items()))


def download_file(url: str, output_path: Path):
    """Baixa um arquivo XLSX para o caminho especificado."""
    with requests.get(url, headers=HEADERS, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)


def save_manifest(links: dict[int, str], output_dir: Path):
    manifest_path = output_dir / "itbi_sources.json"
    manifest = {
        "source_page": BASE_URL,
        "years": [{"year": year, "url": url} for year, url in links.items()],
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    return manifest_path


def download_itbi_data(min_year=2023, max_year=2025, output_dir=DEFAULT_OUTPUT_DIR):
    """Baixa os arquivos ITBI e registra a origem dos dados."""
    output_path = resolve_output_dir(output_dir)

    print("Baixando HTML da pagina de origem...")
    html = fetch_html(BASE_URL)

    print("Extraindo links dos anos desejados...")
    links = extract_year_links(html, min_year, max_year)
    if not links:
        raise RuntimeError(f"Nenhum link ITBI encontrado entre {min_year} e {max_year}.")

    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = save_manifest(links, output_path)

    for year, link in links.items():
        filepath = output_path / f"itbi_{year}.xlsx"
        if filepath.exists() and filepath.stat().st_size > 0:
            print(f"Arquivo ja existe, mantendo: {filepath}")
            continue

        print(f"Baixando ITBI {year}...")
        download_file(link, filepath)
        print(f"Arquivo salvo em: {filepath}")

    print("Download concluido.")
    print(f"Manifesto de origem salvo em: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingestao dos arquivos ITBI publicos.")
    parser.add_argument("--min-year", type=int, default=2023)
    parser.add_argument("--max-year", type=int, default=2025)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    download_itbi_data(args.min_year, args.max_year, args.output_dir)


if __name__ == "__main__":
    main()
