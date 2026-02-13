"""MHDTurbPy functions package.

This package keeps backward compatibility with legacy intra-package imports
that use top-level module names (e.g. ``import general_functions``).
"""

from pathlib import Path
import sys

_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent

_PATHS = (
    _REPO_ROOT,
    _REPO_ROOT / "pyspedas",
    _FUNCTIONS_DIR,
    _FUNCTIONS_DIR / "downloading_helpers",
)

for _path in _PATHS:
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
