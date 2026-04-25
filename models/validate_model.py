import mlflow.pyfunc
import pandas as pd
import os


def validate(env="test"):
    model_path = f"models/{env}/model.pkl"  # É uma pasta, não um arquivo

    if not os.path.exists(model_path):
        print(f"[ERRO] Modelo não encontrado em {model_path}")
        return

    print(f"[INFO] Carregando modelo de {model_path}...")
    model = mlflow.pyfunc.load_model(model_path)

    sample = pd.DataFrame([{
        "bairro": "CENTRO",
        "area_do_terreno_m2": 300,
        "valor_m2": 1500,
        "ano_mes": 202401,
        "media_valor_cep": 2000
    }])

    print("[INFO] Rodando previsão de teste...")
    pred = model.predict(sample)[0]

    print(f"[OK] Previsão gerada: {pred:,.2f}")


if __name__ == "__main__":
    validate("test")
