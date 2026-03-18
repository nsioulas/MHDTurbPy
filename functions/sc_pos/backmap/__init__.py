"""MHDTurbPy source-surface backmapping (+ optional PFSS context).

This package must remain **safe to import** in interactive workflows.
In particular, importing ``sc_pos.backmap`` should not eagerly import heavy
optional dependencies (e.g. Plotly/Kaleido, PFSS-related stack) unless the
corresponding functionality is actually used.

Public API (lazy-exported)
-------------------------
- backmap_interval
- run_backmapping_interval
- run_backmapping_range
- VAR_SPECS
- merge_dicts, deep_merge
- run_pfss_backmap_pipeline
- PFSSConfig, pfss_maps_cached
- make_pfss_backmap_movie, MovieConfig
- make_pfss_backmap_movie_3d, Movie3DConfig
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "backmap_interval",
    "run_backmapping_interval",
    "run_backmapping_range",
    "VAR_SPECS",
    "merge_dicts",
    "deep_merge",
    "run_pfss_backmap_pipeline",
    "PFSSConfig",
    "pfss_maps_cached",
    "MovieConfig",
    "make_pfss_backmap_movie",
    "Movie3DConfig",
    "make_pfss_backmap_movie_3d",
]


if TYPE_CHECKING:
    # Import for type checkers only; keep runtime import side-effects minimal.
    from .config import deep_merge, merge_dicts  # noqa: F401
    from .pipeline import backmap_interval, run_backmapping_interval, run_backmapping_range  # noqa: F401
    from .plotting import VAR_SPECS  # noqa: F401
    from .runner import run_pfss_backmap_pipeline  # noqa: F401
    from .pfss import PFSSConfig, pfss_maps_cached  # noqa: F401
    from .movie import (  # noqa: F401
        MovieConfig,
        make_pfss_backmap_movie,
        Movie3DConfig,
        make_pfss_backmap_movie_3d,
    )


def __getattr__(name: str) -> Any:
    # Lazy attribute resolution (PEP 562).
    if name in {"backmap_interval", "run_backmapping_interval", "run_backmapping_range"}:
        from . import pipeline as _pipeline
        return getattr(_pipeline, name)

    if name == "VAR_SPECS":
        from . import plotting as _plotting
        return _plotting.VAR_SPECS

    if name in {"merge_dicts", "deep_merge"}:
        from . import config as _config
        return getattr(_config, name)

    if name == "run_pfss_backmap_pipeline":
        from . import runner as _runner
        return _runner.run_pfss_backmap_pipeline

    if name in {"PFSSConfig", "pfss_maps_cached"}:
        from . import pfss as _pfss
        return getattr(_pfss, name)

    if name in {
        "MovieConfig",
        "make_pfss_backmap_movie",
        "Movie3DConfig",
        "make_pfss_backmap_movie_3d",
    }:
        from . import movie as _movie
        return getattr(_movie, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
