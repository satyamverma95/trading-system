from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(path_value: Optional[Union[str, Path]], *, root: Optional[Path] = None) -> Path:
    if path_value is None:
        return root or PROJECT_ROOT

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    base_root = root or PROJECT_ROOT
    return base_root / candidate


def ensure_dir(path_value: Optional[Union[str, Path]]) -> Path:
    resolved = resolve_path(path_value)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
