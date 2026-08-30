from __future__ import annotations

from pathlib import Path
import yaml

from source_code.common.path_resolver import get_project_root, resolve_path


def load_yaml_file(path: str | Path) -> dict:
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}
    with open(resolved, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config() -> dict:
    root = get_project_root()
    config_path = root / "config" / "settings.yaml"
    secrets_path = root / "config" / "secrets.yaml"

    config = load_yaml_file(config_path)
    secrets = load_yaml_file(secrets_path)

    if "zerodha" in secrets and isinstance(secrets["zerodha"], dict):
        config.setdefault("zerodha", {})
        config["zerodha"].update(secrets["zerodha"])

    return config


def get_paths_config() -> dict:
    config = load_config()
    return config.get("paths", {})
