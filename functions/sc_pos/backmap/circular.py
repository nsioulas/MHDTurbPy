"""sc_pos.backmap.circular

Minimal circular-statistics helpers for longitudes.

Conventions
-----------
- All angles in degrees unless stated.
- Longitudes are wrapped to [0, 360).
- Signed angular differences are returned in (-180, 180].

We keep these utilities small and dependency-free.
"""

from __future__ import annotations

import numpy as np


def wrap_0_360(phi_deg: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(phi_deg, dtype=float), 360.0)


def delta_deg(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Return the signed minimal angular difference a-b in (-180, 180]."""
    a = np.asarray(a_deg, dtype=float)
    b = np.asarray(b_deg, dtype=float)
    return ((a - b + 180.0) % 360.0) - 180.0


def halfwidth_deg(lo_deg: np.ndarray, hi_deg: np.ndarray) -> np.ndarray:
    """Half-width of an interval on the circle, using the minimal arc."""
    lo = np.asarray(lo_deg, dtype=float)
    hi = np.asarray(hi_deg, dtype=float)
    return 0.5 * np.abs(delta_deg(hi, lo))


from typing import Optional


def circ_mean_deg(phi_deg: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    c = np.nanmean(np.cos(phi), axis=axis)
    s = np.nanmean(np.sin(phi), axis=axis)
    return wrap_0_360(np.rad2deg(np.arctan2(s, c)))


def circ_std_deg(phi_deg: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Circular standard deviation (Fisher 1993) in degrees."""
    phi = np.deg2rad(np.asarray(phi_deg, dtype=float))
    c = np.nanmean(np.cos(phi), axis=axis)
    s = np.nanmean(np.sin(phi), axis=axis)
    R = np.sqrt(c * c + s * s)
    # Guard: R can be 0 for uniform distributions
    R = np.clip(R, 1e-15, 1.0)
    std = np.sqrt(-2.0 * np.log(R))
    return np.rad2deg(std)


def circ_percentile_deg(phi_deg: np.ndarray, q: float, axis: Optional[int] = None) -> np.ndarray:
    """Circular percentile via centering on the circular mean.

    This is robust to wrap-around at 0/360 as long as the distribution does not
    cover most of the circle (in which case any percentile is ill-defined).
    """
    phi = np.asarray(phi_deg, dtype=float)
    mu = circ_mean_deg(phi, axis=axis)
    d = delta_deg(phi, mu)
    dp = np.nanpercentile(d, q, axis=axis)
    return wrap_0_360(mu + dp)
