"""MHDTurbPy functions package.

This package keeps backward compatibility with legacy intra-package imports
that use top-level module names (e.g. ``import general_functions``).
"""

from pathlib import Path
import sys

_FUNCTIONS_DIR = Path(__file__).resolve().parent
if str(_FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_FUNCTIONS_DIR))
