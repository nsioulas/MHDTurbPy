from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _prepend(path: Path) -> None:
    path = path.resolve()
    s = str(path)
    if path.exists() and s not in sys.path:
        sys.path.insert(0, s)


def _find_functions_dir(start: Path) -> Optional[Path]:
    start = start.resolve()
    candidates = [start.parent] + list(start.parents)
    for base in candidates:
        if base.name == 'functions' and (base / 'general_functions.py').exists():
            return base
        maybe = base / 'functions'
        if maybe.is_dir() and (maybe / 'general_functions.py').exists():
            return maybe
    return None


def ensure_project_paths(start: Optional[Path] = None,
                         include_downloading_helpers: bool = False,
                         include_anisotropy_toolbox: bool = False):
    if start is None:
        start = Path.cwd()
    start_path = Path(start).resolve()
    local_dir = start_path if start_path.is_dir() else start_path.parent
    if (local_dir / 'three_D_funcs.py').exists():
        _prepend(local_dir)

    functions_dir = _find_functions_dir(start_path)
    if functions_dir is None:
        return None

    _prepend(functions_dir)

    if include_downloading_helpers:
        helper_dir = functions_dir / 'downloading_helpers'
        if helper_dir.is_dir():
            _prepend(helper_dir)

    if include_anisotropy_toolbox:
        tool_dir = functions_dir / '3d_anis_analysis_toolbox'
        if tool_dir.is_dir():
            _prepend(tool_dir)

    return functions_dir
