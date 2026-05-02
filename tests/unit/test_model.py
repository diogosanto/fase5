import pandas as pd
import mlflow.pyfunc


def test_model_prediction():
    model = mlflow.pyfunc.load_model("models/dev")

    df = pd.DataFrame([{
        "bairro": "CENTRO",
        "area_do_terreno_m2": 100,
        "valor_m2": 2000,
        "ano_mes": 202401,
        "media_valor_cep": 300000
    }])

    pred = model.predict(df)

    assert pred is not None
    assert pred[0] > 0