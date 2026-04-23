import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib


INPUT = "data/processed/itbi_features_minimal.csv"
OUTPUT = "models/dev/model.pkl"


def train():
    print("📄 Carregando dataset de features...")
    df = pd.read_csv(INPUT, sep=";")

    # -------------------------
    # 1. Separar X e y
    # -------------------------
    X = df[["bairro", "area_do_terreno_m2", "ano_mes", "media_valor_cep"]]
    y = df["valor_venal_de_referencia"]

    # -------------------------
    # 2. Split estratificado por bairro
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=X["bairro"]
    )

    # -------------------------
    # 3. Pré-processamento
    # -------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["bairro"]),
            ("num", "passthrough", ["area_do_terreno_m2", "ano_mes", "media_valor_cep"])
        ]
    )

    # -------------------------
    # 4. Modelo
    # -------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # -------------------------
    # 5. Treinar
    # -------------------------
    print("🚀 Treinando modelo...")
    pipeline.fit(X_train, y_train)

    # -------------------------
    # 6. Avaliação
    # -------------------------
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")

    # -------------------------
    # 7. Salvar modelo
    # -------------------------
    os.makedirs("models/dev", exist_ok=True)
    joblib.dump(pipeline, OUTPUT)

    print(f"✔ Modelo salvo em: {OUTPUT}")


if __name__ == "__main__":
    train()
