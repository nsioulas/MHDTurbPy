"""sc_pos.backmap.units

Small utilities to keep units explicit and machine-checkable.

Design choice
-------------
Pandas DataFrames do not reliably store ``astropy.units.Quantity`` columns.
We therefore:

1) Perform *all* physics calculations with ``Quantity``.
2) Store the *numerical* values in the returned DataFrame.
3) Store a per-column unit map in ``df.attrs['units']``.

This keeps the output lightweight and compatible with MHDTurbPy's existing
pickle workflows, while still making unit assumptions explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd

try:
    import astropy.units as u
except Exception as e:  # pragma: no cover
    raise RuntimeError("MHDTurbPy backmapping requires astropy. Install with: pip install astropy") from e


UnitMap = Dict[str, u.Unit]


def default_input_units() -> UnitMap:
    """Default unit contract for MHDTurbPy interval pickles.

    These are the only assumptions made *unless* the caller overrides them.

    - ``Br`` : nT
    - ``Vr`` : km/s
    - ``Np`` : 1/cm^3

    If your loader returns different units, pass ``input_units={...}``.
    """

    return {
        "Br": u.nT,
        "Vr": u.km / u.s,
        "Np": u.cm ** -3,
    }


def attach_units(df: pd.DataFrame, units: Mapping[str, u.Unit]) -> None:
    """Attach/merge a unit map into ``df.attrs['units']``."""
    um = dict(df.attrs.get("units", {}))
    for k, v in units.items():
        um[str(k)] = u.Unit(v)
    df.attrs["units"] = um


def require_units(df: pd.DataFrame, cols: list[str]) -> UnitMap:
    """Return the unit map for columns, raising if any are missing."""
    um = df.attrs.get("units", {})
    missing = [c for c in cols if c not in um]
    if missing:
        raise KeyError(f"Missing unit metadata for columns: {missing}.")
    return {c: u.Unit(um[c]) for c in cols}


def q_from_df(df: pd.DataFrame, col: str) -> u.Quantity:
    """Get a Quantity from a DataFrame column using ``df.attrs['units']``."""
    um = require_units(df, [col])
    arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return arr * um[col]


def to_value(q: u.Quantity, unit: u.Unit) -> np.ndarray:
    """Safe float conversion with NaN preservation."""
    qq = u.Quantity(q)
    out = qq.to_value(unit)
    return np.asarray(out, dtype=float)


def ensure_finite_positive(q: u.Quantity, *, name: str) -> None:
    arr = np.asarray(q.to_value(q.unit), dtype=float)
    if not np.isfinite(arr).any():
        raise ValueError(f"{name} contains no finite values.")
    if np.nanmin(arr) <= 0:
        raise ValueError(f"{name} must be > 0 everywhere it is used; found min={float(np.nanmin(arr))}.")
