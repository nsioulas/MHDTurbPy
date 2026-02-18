"""
Backwards-compatible shim.

Old imports:
    from sc_pos.ballistic_backmap import run_backmapping_interval

New implementation lives in:
    sc_pos.backmap.pipeline.run_backmapping_interval
"""
from __future__ import annotations

from .backmap.pipeline import run_backmapping_interval  # noqa: F401
from .backmap.plotting import VAR_SPECS  # noqa: F401
