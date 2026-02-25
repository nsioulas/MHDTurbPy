"""Azimuthal (v_phi) correction for Carrington backmapping (Appendix A of aa44327-22).

This module implements the optional correction

    \Delta\phi = \int_{r_{ss}}^{r_{sc}} \frac{\Omega - v_\phi(r)/r}{v_r(r)}\,dr,

where v_r(r) is the radial wind profile used for the travel-time model.

The intent is to keep the implementation:
- unit-safe (astropy.units),
- numerically stable near the Alfv\'en point r=r_A (removable 0/0),
- consistent with the backmapping sign convention applied upstream via phi_sign.

Notes
-----
1) For a strictly ballistic constant-speed model (v_r=const), the Weber--Davis
   expression used here yields v_\phi=0 (because v_A=v_r everywhere by construction),
   so \Delta\phi reduces exactly to \Omega\tau.

2) The correction depends on the assumed Alfv\'en radius r_A. There is no universal
   value; users should treat r_A as a physical input (or a model output) and
   document the choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import astropy.units as u


@dataclass
class AzimuthalResult:
    """Container for Appendix-A azimuthal correction outputs."""

    delta_phi: u.Quantity  # unsigned, in angle units (deg)
    meta: Dict[str, Any]


def _coerce_length(x: Any) -> u.Quantity:
    q = u.Quantity(x)
    if not q.unit.is_equivalent(u.m):
        raise u.UnitConversionError(f"Expected a length quantity, got {q}")
    return q.to(u.m)


def _coerce_speed(x: Any) -> u.Quantity:
    q = u.Quantity(x)
    if not q.unit.is_equivalent(u.m / u.s):
        raise u.UnitConversionError(f"Expected a speed quantity, got {q}")
    return q.to(u.m / u.s)


def _coerce_omega(x: Any) -> u.Quantity:
    q = u.Quantity(x)
    if not q.unit.is_equivalent(u.rad / u.s):
        # allow deg/day etc.
        q = q.to(u.rad / u.s)
    return q.to(u.rad / u.s)


def weber_davis_vphi(
    *,
    r: u.Quantity,
    v_r: u.Quantity,
    omega: u.Quantity,
    r_A: u.Quantity,
    v_A: u.Quantity,
    ma2_tol: float = 1e-3,
) -> u.Quantity:
    """Weber--Davis azimuthal speed (paper Appendix A).

    Implements (in the paper's notation):

        v_\phi(r) = \Omega\,r\,\frac{1 - v_r/v_A}{1 - (v_r/v_A)\,(r/r_A)^2},

    with a removable 0/0 at r=r_A handled by an explicit limit v_\phi(r_A)=\Omega r_A.

    Parameters
    ----------
    r : length Quantity (array-like)
    v_r : speed Quantity (array-like, same shape as r)
    omega : angular speed Quantity (rad/s)
    r_A : Alfv\'en radius (length)
    v_A : v_r(r_A) (speed)
    ma2_tol : tolerance used to detect the removable singularity near M_A^2=1.

    Returns
    -------
    v_phi : speed Quantity (same shape as r)
    """

    rr = _coerce_length(r)
    vr = _coerce_speed(v_r)
    om = _coerce_omega(omega)
    rA = _coerce_length(r_A)
    vA = _coerce_speed(v_A)

    if not np.isfinite(rA.to_value(u.m)) or (rA <= 0 * u.m):
        raise ValueError("r_A must be finite and > 0")
    if not np.isfinite(vA.to_value(u.m / u.s)) or (vA <= 0 * (u.m / u.s)):
        raise ValueError("v_A must be finite and > 0")

    x = (vr / vA).to_value(u.dimensionless_unscaled)
    y = (rr / rA).to_value(u.dimensionless_unscaled) ** 2

    denom = 1.0 - x * y
    numer = 1.0 - x

    ratio = np.full_like(denom, np.nan, dtype=float)

    # Generic points
    ok = np.isfinite(denom) & np.isfinite(numer) & (np.abs(denom) > 0)
    ratio[ok] = numer[ok] / denom[ok]

    # Removable singularity near M_A^2 = x*y = 1 and x ~ 1 (r ~ r_A)
    # In that limit, ratio -> 1 and v_phi -> Omega r.
    ma2 = x * y
    sing = np.isfinite(ma2) & (np.abs(ma2 - 1.0) <= float(ma2_tol)) & np.isfinite(x) & (np.abs(x - 1.0) <= float(ma2_tol))
    ratio[sing] = 1.0

    # If denom is ~0 but numer is not, the expression diverges; we clip to finite values
    # to avoid propagating infs into the integral. This only occurs if r_A is chosen
    # inconsistently with the v_r profile.
    bad = np.isfinite(denom) & (np.abs(denom) <= 1e-12) & ~sing
    if np.any(bad):
        ratio[bad] = np.sign(numer[bad]) * 1e6

    vphi = (om * rr * ratio)  # (rad/s)*m -> m/s
    return vphi.to(u.m / u.s)


def delta_phi_appendixA(
    *,
    r_grid: u.Quantity,
    v_r_grid: u.Quantity,
    omega: u.Quantity,
    r_A: u.Quantity,
    v_A: u.Quantity,
    ma2_tol: float = 1e-3,
) -> u.Quantity:
    """Evaluate Appendix-A \Delta\phi integral on a supplied grid.

    Returns an *unsigned* angular displacement in degrees.
    """

    rr = _coerce_length(r_grid)
    vr = _coerce_speed(v_r_grid)
    om = _coerce_omega(omega)
    rA = _coerce_length(r_A)
    vA = _coerce_speed(v_A)

    if rr.ndim != 1:
        raise ValueError("r_grid must be 1D")
    if vr.shape != rr.shape:
        raise ValueError("v_r_grid must have the same shape as r_grid")

    # v_phi from Weber--Davis expression
    vphi = weber_davis_vphi(r=rr, v_r=vr, omega=om, r_A=rA, v_A=vA, ma2_tol=ma2_tol)

    # integrand: (Omega - v_phi/r)/v_r  [1/length]
    ang = (om - (vphi / rr)).to(1 / u.s)
    integrand = (ang / vr).to(1 / u.m)

    x = rr.to_value(u.m)
    y = integrand.to_value(1 / u.m)

    # trapz returns dimensionless (radians)
    dphi_rad = np.trapz(y, x) * u.rad
    return dphi_rad.to(u.deg)


def compute_delta_phi_series(
    *,
    model: Any,
    r_sc: u.Quantity,
    V_bg: u.Quantity,
    r_ss: u.Quantity,
    omega: u.Quantity,
    r_A: u.Quantity,
    n_grid: int = 2048,
    v_min: u.Quantity = 50 * u.km / u.s,
    v_fallback: u.Quantity = 400 * u.km / u.s,
    ma2_tol: float = 1e-3,
) -> AzimuthalResult:
    """Compute Appendix-A \Delta\phi for a time series.

    Parameters
    ----------
    model : travel-time model instance with a ``speed_profile(r_grid, r_sc, V_bg, r_ss, v_min, v_fallback)`` method.
            If absent, a constant-speed profile v_r(r)=V_bg is used.
    r_sc : spacecraft radius (array-like Quantity)
    V_bg : background speed at spacecraft (array-like Quantity)
    r_ss : source-surface radius (scalar Quantity)
    omega : Carrington angular speed (Quantity, e.g. deg/day)
    r_A : Alfv\'en radius (scalar Quantity)
    n_grid : number of radial grid points for the integral

    Returns
    -------
    AzimuthalResult with delta_phi (unsigned, degrees) and meta.
    """

    r_sc_q = _coerce_length(r_sc)
    V_q = _coerce_speed(V_bg)
    r_ss_q = _coerce_length(r_ss)
    rA = _coerce_length(r_A)
    om = _coerce_omega(omega)

    if r_ss_q.isscalar is False:
        # allow length-1 arrays, but keep semantics scalar
        r_ss_q = u.Quantity(np.atleast_1d(r_ss_q.to_value(u.m))[0], u.m)

    if rA < r_ss_q:
        raise ValueError("Appendix-A correction requires r_A >= r_ss")

    n = int(len(np.atleast_1d(r_sc_q.to_value(u.m))))
    out = np.full(n, np.nan, dtype=float)  # deg

    # Prepare scalar floors
    vmin = _coerce_speed(v_min)
    vfb = _coerce_speed(v_fallback)

    # Extract model callback if available
    speed_profile = getattr(model, "speed_profile", None)

    # loop over time; the r-grid depends on r_sc(t)
    r_sc_m = np.atleast_1d(r_sc_q.to_value(u.m)).astype(float)
    V_mps = np.atleast_1d(V_q.to_value(u.m / u.s)).astype(float)

    for i in range(n):
        rsc = r_sc_m[i] * u.m
        vsc = V_mps[i] * (u.m / u.s)

        if not np.isfinite(rsc.to_value(u.m)) or not np.isfinite(vsc.to_value(u.m / u.s)):
            continue
        if rsc <= r_ss_q:
            continue

        # Radial grid: log-spaced to resolve near-Sun gradients
        rr = np.geomspace(r_ss_q.to_value(u.m), rsc.to_value(u.m), num=int(max(16, n_grid))) * u.m

        # v_r(r) profile on rr
        if callable(speed_profile):
            try:
                vr = u.Quantity(speed_profile(r_grid=rr, r_sc=rsc, V_bg=vsc, r_ss=r_ss_q, v_min=vmin, v_fallback=vfb))
                vr = _coerce_speed(vr)
            except Exception:
                vr = np.full_like(rr.to_value(u.m), np.nan, dtype=float) * (u.m / u.s)
        else:
            vr = np.full_like(rr.to_value(u.m), float(vsc.to_value(u.m / u.s)), dtype=float) * (u.m / u.s)

        # enforce a strict positive floor
        vr = np.maximum(vr.to_value(u.m / u.s), vmin.to_value(u.m / u.s)) * (u.m / u.s)

        # v_A = v_r(r_A) using the same profile scaling
        if callable(speed_profile):
            try:
                vrA = u.Quantity(speed_profile(r_grid=(np.array([rA.to_value(u.m)]) * u.m), r_sc=rsc, V_bg=vsc, r_ss=r_ss_q, v_min=vmin, v_fallback=vfb))
                vrA = _coerce_speed(vrA)[0]
            except Exception:
                vrA = vsc
        else:
            vrA = vsc

        vrA = max(float(vrA.to_value(u.m / u.s)), float(vmin.to_value(u.m / u.s))) * (u.m / u.s)

        try:
            dphi = delta_phi_appendixA(r_grid=rr, v_r_grid=vr, omega=om, r_A=rA, v_A=vrA, ma2_tol=ma2_tol)
            out[i] = float(dphi.to_value(u.deg))
        except Exception:
            continue

    meta = {
        "n": int(n),
        "n_grid": int(n_grid),
        "r_ss_Rsun": float(r_ss_q.to_value(u.R_sun)),
        "r_A_Rsun": float(rA.to_value(u.R_sun)),
        "omega_deg_per_day": float(u.Quantity(omega).to_value(u.deg / u.day)),
        "ma2_tol": float(ma2_tol),
        "model_class": type(model).__name__,
    }
    return AzimuthalResult(delta_phi=(out * u.deg), meta=meta)
