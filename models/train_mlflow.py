import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import datetime
import shutil


INPUT = "data/processed/itbi_features_minimal.csv"


def generate_version():
    return datetime.datetime.now().strftime("%Y.%m.%d.%H%M")


def train_mlflow():

    # gerar versão automática
    version = generate_version()

    # definir caminho de saída AGORA (depois de gerar a versão)
    OUTPUT = f"models/dev/model_{version}"

    mlflow.set_experiment("itbi-terrenos")

    with mlflow.start_run():

        mlflow.log_param("model_version", version)

        df = pd.read_csv(INPUT, sep=";")

        df = df.dropna(subset=[
            "bairro",
            "area_do_terreno_m2",
            "ano_mes",
            "media_valor_cep"
        ])

        counts = df["bairro"].value_counts()
        bairros_validos = counts[counts >= 2].index
        df = df[df["bairro"].isin(bairros_validos)]

        X = df[[
            "bairro",
            "area_do_terreno_m2",
            "valor_m2",
            "ano_mes",
            "media_valor_cep"
        ]]

        y = df["valor_venal_de_referencia"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=X["bairro"]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), ["bairro"]),
                ("num", "passthrough", [
                    "area_do_terreno_m2",
                    "valor_m2",
                    "ano_mes",
                    "media_valor_cep"
                ])
            ]
        )

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("features", X.columns.tolist())

        # log no MLflow
        mlflow.sklearn.log_model(pipeline, "model")

        # salvar localmente com versionamento
        os.makedirs("models/dev", exist_ok=True)

        if os.path.exists(OUTPUT):
            shutil.rmtree(OUTPUT)

        mlflow.sklearn.save_model(pipeline, OUTPUT)

        print(f"MAE: {mae:,.2f}")
        print(f"R²: {r2:.3f}")
        print(f"✔ Modelo salvo em: {OUTPUT}")
        print("✔ Experimento registrado no MLflow")


if __name__ == "__main__":
    train_mlflow()
