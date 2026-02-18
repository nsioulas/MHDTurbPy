"""MHDTurbPy source-surface backmapping.

Public API
----------
- ``backmap_interval`` (preferred)
- ``run_backmapping_interval`` (legacy shim)
- ``VAR_SPECS`` (plot configuration)

The implementation is intentionally minimal and auditable.
"""

from .pipeline import backmap_interval, run_backmapping_interval
from .plotting import VAR_SPECS

__all__ = ["backmap_interval", "run_backmapping_interval", "VAR_SPECS"]
