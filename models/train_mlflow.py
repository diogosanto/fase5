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


INPUT = "data/processed/itbi_features_minimal.csv"
OUTPUT = "models/dev/model.pkl"


def train_mlflow():

    df = pd.read_csv(INPUT, sep=";")

    # Remover linhas com NaN nas features essenciais
    df = df.dropna(subset=[
        "bairro",
        "area_do_terreno_m2",
        "ano_mes",
        "media_valor_cep"
    ])
    # Remover bairros com apenas 1 ocorrência (não podem ser estratificados)
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

    mlflow.set_experiment("itbi-terrenos")

    with mlflow.start_run():

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("features", X.columns.tolist())

        mlflow.sklearn.log_model(pipeline, "model")

        os.makedirs("models/dev", exist_ok=True)
        import shutil

        # Remover modelo antigo, se existir
        if os.path.exists(OUTPUT):
            shutil.rmtree(OUTPUT)
        mlflow.sklearn.save_model(pipeline, OUTPUT)

        print(f"MAE: {mae:,.2f}")
        print(f"R²: {r2:.3f}")
        print(f"✔ Modelo salvo em: {OUTPUT}")
        print("✔ Experimento registrado no MLflow")


if __name__ == "__main__":
    train_mlflow()
