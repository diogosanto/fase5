import mlflow.pyfunc
import pandas as pd
import os
import re


def get_model_version(path):
    """
    Extrai a versão do modelo a partir do nome da pasta.
    Exemplo: model_2026.04.25.1430
    """
    match = re.search(r"model_(\d+\.\d+\.\d+\.\d+)", path)
    return match.group(1) if match else "unknown"


def validate(env="test", version=None):
    """
    Valida um modelo em um ambiente específico.
    
    env: "dev", "test" ou "prod"
    version: string opcional no formato YYYY.MM.DD.HHMM
             Se None → usa a única versão presente no ambiente.
    """
    base_path = f"models/{env}"

    if not os.path.exists(base_path):
        print(f"[ERRO] Ambiente '{env}' não existe em {base_path}")
        return

    # Se o usuário especificou uma versão, usa ela
    if version:
        model_folder = f"model_{version}"
        full_model_path = os.path.join(base_path, model_folder)

        if not os.path.exists(full_model_path):
            print(f"[ERRO] Versão {version} não encontrada em {base_path}")
            return
    else:
        # Descobrir automaticamente a versão ativa do ambiente
        folders = [f for f in os.listdir(base_path) if f.startswith("model_")]
        if not folders:
            print(f"[ERRO] Nenhum modelo encontrado em {base_path}")
            return

        # Assumir que existe apenas 1 modelo por ambiente
        model_folder = folders[0]
        full_model_path = os.path.join(base_path, model_folder)
        version = get_model_version(model_folder)

    print(f"[INFO] Carregando modelo versão {version} de {full_model_path}...")
    model = mlflow.pyfunc.load_model(full_model_path)

    # Amostra de teste
    sample = pd.DataFrame([{
        "bairro": "CENTRO",
        "area_do_terreno_m2": 300,
        "valor_m2": 1500,
        "ano_mes": 202401,
        "media_valor_cep": 2000
    }])

    print("[INFO] Rodando previsão de teste...")
    pred = model.predict(sample)[0]

    print(f"[OK] Previsão gerada (versão {version}): {pred:,.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validação de modelos")
    parser.add_argument("--env", type=str, default="dev", help="Ambiente: dev, test ou prod")
    parser.add_argument("--version", type=str, default=None, help="Versão específica do modelo (YYYY.MM.DD.HHMM)")

    args = parser.parse_args()

    validate(env=args.env, version=args.version)
