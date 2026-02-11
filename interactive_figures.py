"""Top-level compatibility shim for legacy ``interactive_figures`` imports.

Notebooks often run from the repository root and call ``import interactive_figures``.
The implementation now lives in ``functions/interactive_figs.py``. This module
adds the expected path and re-exports the implementation to preserve old imports.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
FUNCTIONS_DIR = REPO_ROOT / "functions"
if str(FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(FUNCTIONS_DIR))

from interactive_figs import *  # noqa: F401,F403
