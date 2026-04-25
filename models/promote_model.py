import argparse
import mlflow
import shutil
from pathlib import Path


def get_last_run_metric(experiment_name: str, metric: str = "mae"):
    """Retorna o valor da métrica do último run do experimento."""
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        return None

    run = runs[0]
    return run.data.metrics.get(metric)


def copy_model(model_path, to_env):
    dst_env = Path("models") / to_env
    dst = dst_env / model_path.name  # cria a pasta model_XXXX dentro do ambiente

    # limpar ambiente
    if dst_env.exists():
        shutil.rmtree(dst_env)

    # recriar ambiente
    dst_env.mkdir(parents=True, exist_ok=True)

    # copiar a pasta inteira
    shutil.copytree(model_path, dst)
    print(f"[OK] Modelo {model_path.name} copiado para {dst}")



def promote(from_env, to_env, improvement_pct, version=None):
    """Promove um modelo entre ambientes."""
    src_dir = Path("models") / from_env

    if version:
        # promover versão específica
        model_path = src_dir / f"model_{version}"
        if not model_path.exists():
            raise ValueError(f"Versão {version} não encontrada em {src_dir}")
        print(f"[INFO] Promovendo versão específica: {version}")

    else:
        # promover versão mais recente
        versions = sorted(src_dir.glob("model_*"), reverse=True)
        if not versions:
            raise ValueError(f"Nenhum modelo encontrado em {src_dir}")
        model_path = versions[0]
        print(f"[INFO] Promovendo versão mais recente: {model_path.name}")

    # copiar modelo
    copy_model(model_path, to_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False, help="Versão específica do modelo a promover")
    parser.add_argument("--from-env", required=True, choices=["dev", "test"])
    parser.add_argument("--to-env", required=True, choices=["test", "prod"])
    parser.add_argument("--improvement-pct", type=float, default=5.0)
    args = parser.parse_args()

    promote(args.from_env, args.to_env, args.improvement_pct, args.version)


if __name__ == "__main__":
    main()
