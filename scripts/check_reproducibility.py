import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "data" / "metrics" / "reproducibility_report.json"


REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "params.yaml",
    "dvc.yaml",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    "src/config.py",
    "src/data/README.md",
    "src/data/0ingest.py",
    "src/data/1extract_all_itbi.py",
    "src/data/2clean_all.py",
    "src/features/build_features_minimal.py",
    "src/features/modelagem/eda_valor_m2_bairro.py",
    "src/training/train_mlflow.py",
    "src/training/validate_model.py",
    "src/training/evaluation.py",
    "src/training/splits.py",
    "tests/unit/test_data_cleaning.py",
    "tests/unit/test_feature_contract.py",
    "tests/unit/test_temporal_holdout.py",
    "tests/unit/test_evaluation.py",
    "tests/unit/test_eda_valor_m2_bairro.py",
]

REQUIRED_DVC_STAGES = ["ingest", "extract", "clean", "features", "eda", "train", "validate", "repro_check"]
REQUIRED_COMMANDS_IN_README = [
    "dvc repro",
    "dvc metrics show",
    "pytest -q",
    "docker compose up --build",
    "python src/features/modelagem/eda_valor_m2_bairro.py",
]
REQUIRED_IMPORTS = ["pandas", "sklearn", "mlflow", "dvc", "yaml", "openpyxl", "matplotlib"]
SMOKE_TESTS = [
    "tests/unit/test_data_cleaning.py",
    "tests/unit/test_feature_contract.py",
    "tests/unit/test_temporal_holdout.py",
    "tests/unit/test_evaluation.py",
    "tests/unit/test_eda_valor_m2_bairro.py",
]


def main() -> None:
    checks = {
        "python_version": check_python_version(),
        "required_files": check_required_files(),
        "readme_commands": check_readme_commands(),
        "dvc_stages": check_dvc_stages(),
        "pyproject_dependencies": check_pyproject_dependencies(),
        "gitignore_artifacts": check_gitignore_artifacts(),
        "critical_imports": check_critical_imports(),
        "unit_smoke_tests": check_unit_smoke_tests(),
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    report = {
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    if failed:
        raise SystemExit(f"Falhas de reprodutibilidade: {', '.join(failed)}")

    print(f"Relatorio de reprodutibilidade salvo em: {REPORT_PATH}")


def check_python_version() -> dict:
    version = sys.version_info
    passed = version.major == 3 and 10 <= version.minor < 12
    return {
        "passed": passed,
        "value": f"{version.major}.{version.minor}.{version.micro}",
        "expected": ">=3.10,<3.12",
    }


def check_required_files() -> dict:
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).exists()]
    return {"passed": not missing, "missing": missing}


def check_readme_commands() -> dict:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8") if (PROJECT_ROOT / "README.md").exists() else ""
    missing = [command for command in REQUIRED_COMMANDS_IN_README if command not in readme]
    return {"passed": not missing, "missing": missing}


def check_dvc_stages() -> dict:
    dvc_yaml = (PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8") if (PROJECT_ROOT / "dvc.yaml").exists() else ""
    missing = [stage for stage in REQUIRED_DVC_STAGES if f"  {stage}:" not in dvc_yaml]
    return {"passed": not missing, "missing": missing}


def check_pyproject_dependencies() -> dict:
    pyproject = (
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        if (PROJECT_ROOT / "pyproject.toml").exists()
        else ""
    )
    required = ["dvc", "mlflow", "fastapi", "pytest", "PyYAML", "pandas", "openpyxl", "matplotlib"]
    missing = [dependency for dependency in required if dependency not in pyproject]
    return {"passed": not missing, "missing": missing}


def check_gitignore_artifacts() -> dict:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8") if (PROJECT_ROOT / ".gitignore").exists() else ""
    required = ["data/interim/", "data/processed/", "mlruns/", "models/dev/", "models/test/", "models/prod/"]
    missing = [pattern for pattern in required if pattern not in gitignore]
    return {"passed": not missing, "missing": missing}


def check_critical_imports() -> dict:
    missing = []
    for module in REQUIRED_IMPORTS:
        try:
            __import__(module)
        except ModuleNotFoundError:
            missing.append(module)

    return {"passed": not missing, "missing": missing}


def check_unit_smoke_tests() -> dict:
    command = [sys.executable, "-m", "pytest", *SMOKE_TESTS, "-q"]
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": None, "summary": "timeout"}

    output = (result.stdout + "\n" + result.stderr).strip()
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "summary": output[-2000:],
    }


if __name__ == "__main__":
    main()
