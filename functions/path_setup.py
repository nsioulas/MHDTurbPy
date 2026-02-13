"""Centralized import-path bootstrap for MHDTurbPy."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__).resolve()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "functions").is_dir() and (candidate / "pyspedas").is_dir():
            return candidate
    raise RuntimeError("Could not locate MHDTurbPy root (missing functions/ and pyspedas/).")


def ensure_project_paths(
    *,
    start: Path | None = None,
    include_downloading_helpers: bool = False,
    include_anisotropy_toolbox: bool = False,
    include_sc_pos: bool = False,
    include_paths: Iterable[Path] | None = None,
) -> Path:
    repo_root = find_repo_root(start)
    functions_dir = repo_root / "functions"
    ordered = [repo_root, repo_root / "pyspedas", functions_dir]
    if include_downloading_helpers:
        ordered.append(functions_dir / "downloading_helpers")
    if include_anisotropy_toolbox:
        ordered.append(functions_dir / "3d_anis_analysis_toolboox")
    if include_sc_pos:
        ordered.append(functions_dir / "sc_pos")
    if include_paths:
        ordered.extend(include_paths)
    for path in ordered:
        p = str(path.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root
