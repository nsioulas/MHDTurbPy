"""Backward-compatible alias for :mod:`interactive_figs`.

Several notebooks import ``interactive_figures`` from the ``functions`` path.
After the module rename to ``interactive_figs.py``, that import started failing.
This shim keeps legacy imports working by re-exporting the same public symbols.
"""

from interactive_figs import *  # noqa: F401,F403
