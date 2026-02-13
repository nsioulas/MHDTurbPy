"""MHDTurbPy functions package."""

from pathlib import Path

from .path_setup import ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
