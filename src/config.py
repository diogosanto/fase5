from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params() -> dict[str, Any]:
    if not PARAMS_PATH.exists():
        return {}

    with PARAMS_PATH.open(encoding="utf-8") as file:
        if yaml is not None:
            return yaml.safe_load(file) or {}

        content = file.read()
        parsed = _load_simple_yaml(content)
        _patch_simple_lists(content, parsed)
        return parsed


def _load_simple_yaml(content: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, params)]

    for raw_line in content.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, _, raw_value = raw_line.strip().partition(":")
        if not key:
            continue

        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        value = raw_value.strip()
        if not value:
            nested: dict[str, Any] = {}
            parent[key] = nested
            stack.append((indent, nested))
        else:
            parent[key] = _parse_scalar(value)

    return params


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _patch_simple_lists(content: str, parsed: dict[str, Any]) -> None:
    lines = content.splitlines()
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped.endswith(":"):
            continue

        key = stripped[:-1]
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        values = []
        saw_child = False
        for next_line in lines[index + 1 :]:
            if not next_line.strip():
                continue

            next_indent = len(next_line) - len(next_line.lstrip(" "))
            next_stripped = next_line.strip()
            if next_indent <= indent:
                break
            if not saw_child:
                saw_child = True
                if not next_stripped.startswith("- "):
                    break
            if next_stripped.startswith("- "):
                values.append(_parse_scalar(next_stripped[2:].strip()))

        if values:
            _replace_key(parsed, key, values)


def _replace_key(node: dict[str, Any], target_key: str, value: Any) -> bool:
    for key, current_value in node.items():
        if key == target_key:
            node[key] = value
            return True
        if isinstance(current_value, dict) and _replace_key(current_value, target_key, value):
            return True
    return False
