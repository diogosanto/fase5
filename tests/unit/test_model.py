import pandas as pd
from sklearn.dummy import DummyRegressor

from src.training.train_mlflow import FEATURES, TARGET, build_pipeline


def test_model_pipeline_uses_current_feature_contract():
    frame = pd.DataFrame(
        [
            {
                "bairro": "CENTRO",
                "cep_prefixo": "01001",
                "area_do_terreno_m2": 100.0,
                "ano": 2024,
                "mes": 1,
                TARGET: 300000.0,
            },
            {
                "bairro": "MOOCA",
                "cep_prefixo": "03110",
                "area_do_terreno_m2": 120.0,
                "ano": 2024,
                "mes": 2,
                TARGET: 350000.0,
            },
        ]
    )

    model = build_pipeline(DummyRegressor(strategy="mean"))
    model.fit(frame[FEATURES], frame[TARGET])
    pred = model.predict(frame[FEATURES])

    assert pred is not None
    assert len(pred) == len(frame)
    assert set(FEATURES) == {"bairro", "cep_prefixo", "area_do_terreno_m2", "ano", "mes"}
