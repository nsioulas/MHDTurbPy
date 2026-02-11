"""Centralized user-configurable paths for MHDTurbPy.

Configure once by creating ``user_paths.json`` in the repository root (copy from
``user_paths.example.json``), or via environment variables:
- MHDTURBPY_DATA_PATH
- MHDTURBPY_SAVE_DESTINATION
- MHDTURBPY_CDF_LIB_PATH

Any additional keys in ``user_paths.json`` are preserved (for notebook-specific
optional paths such as ``analysis_data_path`` or ``solo_dist_path``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_paths() -> Dict[str, Any]:
    root = repo_root()
    return {
        "repo_root": str(root),
        "Data_path": str(root),
        "save_destination": str(root / "examples"),
        "cdf_lib_path": None,
    }


def load_user_paths(config_path: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    paths = _default_paths()

    cfg_path = Path(config_path) if config_path else repo_root() / "user_paths.json"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            for key, value in raw.items():
                if value is not None:
                    paths[key] = str(value) if isinstance(value, (str, os.PathLike)) else value

    env_map = {
        "MHDTURBPY_DATA_PATH": "Data_path",
        "MHDTURBPY_SAVE_DESTINATION": "save_destination",
        "MHDTURBPY_CDF_LIB_PATH": "cdf_lib_path",
    }
    for env_key, out_key in env_map.items():
        val = os.environ.get(env_key)
        if val:
            paths[out_key] = val

    return paths


def merge_user_paths_into_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(settings or {})
    user_paths = load_user_paths()

    for key in ("Data_path", "save_destination", "cdf_lib_path"):
        if out.get(key) is None:
            out[key] = user_paths.get(key)

    return out


def repo_data_file(*relative_parts: str) -> Path:
    return repo_root().joinpath(*relative_parts)
