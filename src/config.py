from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_arg(value, config: dict[str, Any], key: str, default):
    if value is not None:
        return value
    if key in config:
        return config[key]
    return default
