import argparse
import json
import shutil
from pathlib import Path


METRIC_NAME = "mae"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_metric(model_path: Path, metric: str = METRIC_NAME):
    for filename in ("validation.json", "metrics.json"):
        path = model_path / filename
        if path.exists():
            with path.open(encoding="utf-8") as file:
                data = json.load(file)
            if metric in data:
                return float(data[metric]), path
    return None, None


def resolve_model_path(env: str, version: str | None = None):
    src_dir = PROJECT_ROOT / "models" / env
    if not src_dir.exists():
        raise ValueError(f"Ambiente {env} nao encontrado em {src_dir}")

    if version:
        model_path = src_dir / f"model_{version}"
        if not model_path.exists():
            raise ValueError(f"Versao {version} nao encontrada em {src_dir}")
        return model_path

    versions = sorted(src_dir.glob("model_*"), reverse=True)
    if not versions:
        raise ValueError(f"Nenhum modelo encontrado em {src_dir}")
    return versions[0]


def get_active_model(env: str):
    env_dir = PROJECT_ROOT / "models" / env
    if not env_dir.exists():
        return None

    versions = sorted(env_dir.glob("model_*"), reverse=True)
    return versions[0] if versions else None


def assert_promotion_criteria(candidate_path, to_env, improvement_pct, max_mae):
    candidate_metric, candidate_source = load_metric(candidate_path)
    if candidate_metric is None:
        raise ValueError(
            f"Modelo {candidate_path} nao tem metrics.json ou validation.json com {METRIC_NAME}"
        )

    if max_mae is not None and candidate_metric > max_mae:
        raise ValueError(
            f"Promocao bloqueada: {METRIC_NAME}={candidate_metric:,.2f} acima "
            f"do limite {max_mae:,.2f}"
        )

    active_model = get_active_model(to_env)
    if active_model is None:
        print(f"[INFO] Nenhum modelo ativo em {to_env}; usando criterio absoluto.")
        print(f"[INFO] Metrica candidata lida de {candidate_source}: {candidate_metric:,.2f}")
        return

    active_metric, active_source = load_metric(active_model)
    if active_metric is None:
        raise ValueError(
            f"Modelo ativo {active_model} nao tem metrics.json ou validation.json com {METRIC_NAME}"
        )

    required_metric = active_metric * (1 - improvement_pct / 100)
    if candidate_metric > required_metric:
        raise ValueError(
            f"Promocao bloqueada: candidato {METRIC_NAME}={candidate_metric:,.2f}; "
            f"ativo {METRIC_NAME}={active_metric:,.2f}; necessario <= {required_metric:,.2f}"
        )

    print(f"[INFO] Candidato aprovado por {METRIC_NAME}: {candidate_metric:,.2f}")
    print(f"[INFO] Modelo ativo comparado: {active_model} ({active_source})")


def copy_model(model_path, to_env):
    dst_env = PROJECT_ROOT / "models" / to_env
    dst = dst_env / model_path.name

    if dst_env.exists():
        shutil.rmtree(dst_env)
    dst_env.mkdir(parents=True, exist_ok=True)

    shutil.copytree(model_path, dst)
    print(f"[OK] Modelo {model_path.name} copiado para {dst}")


def promote(from_env, to_env, improvement_pct, version=None, max_mae=None):
    model_path = resolve_model_path(from_env, version)
    print(f"[INFO] Avaliando promocao de {model_path}")

    assert_promotion_criteria(model_path, to_env, improvement_pct, max_mae)
    copy_model(model_path, to_env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=False)
    parser.add_argument("--from-env", required=True, choices=["dev", "test"])
    parser.add_argument("--to-env", required=True, choices=["test", "prod"])
    parser.add_argument("--improvement-pct", type=float, default=5.0)
    parser.add_argument("--max-mae", type=float, default=None)
    args = parser.parse_args()

    promote(args.from_env, args.to_env, args.improvement_pct, args.version, args.max_mae)


if __name__ == "__main__":
    main()
