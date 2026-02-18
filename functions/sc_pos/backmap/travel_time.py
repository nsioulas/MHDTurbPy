"""sc_pos.backmap.travel_time

Physics-first travel-time models from a source surface r_ss to spacecraft radius r_sc.

Key requirements enforced here
------------------------------
- ``tau`` is strictly positive whenever r_sc > r_ss.
- Models are free of hidden singularities by construction (all speeds are floored > 0).
- All computations use ``astropy.units.Quantity``.

Model taxonomy
--------------
We separate:

1) *Ballistic* models: tau = (r_sc - r_ss) / V_bg(t)
2) *Acceleration-shape* parameterizations: a monotonic shape s(r) is scaled so that
   U(r_sc,t) = V_bg(t). These are **heuristics** unless the user independently
   validates the chosen shape against a proper solar-wind solution.

The implementation is intentionally minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:
    from scipy.special import lambertw  # type: ignore
    _HAS_LAMBERTW = True
except Exception:
    lambertw = None
    _HAS_LAMBERTW = False

try:
    import astropy.units as u
except Exception as e:  # pragma: no cover
    raise RuntimeError("MHDTurbPy backmapping requires astropy. Install with: pip install astropy") from e



@dataclass(frozen=True)
class TravelTimeResult:
    tau: u.Quantity               # shape (N,)
    fallback_mask: np.ndarray     # True where fallback was used
    meta: Dict[str, Any]


def _as1d(q: u.Quantity) -> u.Quantity:
    qq = u.Quantity(q)
    return qq.reshape(-1)

def _first_scalar(x: Any) -> float:
    """Return the first element of x as a python float, safe for 0d arrays."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 0:
        return float(a)
    return float(a.ravel()[0])



def _validate_radii(r_sc: u.Quantity, r_ss: u.Quantity) -> None:
    r_sc = _as1d(r_sc)
    r_ss = u.Quantity(r_ss)
    if not r_ss.isscalar:
        raise ValueError("r_ss must be a scalar Quantity.")
    if not np.isfinite(r_ss.to_value(r_ss.unit)) or (r_ss <= 0 * r_ss.unit):
        raise ValueError("r_ss must be finite and > 0.")
    if not np.isfinite(r_sc.to_value(r_sc.unit)).any():
        raise ValueError("r_sc contains no finite values.")
    if np.nanmin(r_sc.to_value(r_sc.unit)) <= r_ss.to_value(r_sc.unit):
        raise ValueError(
            f"Found r_sc <= r_ss (min r_sc={float(np.nanmin(r_sc.to_value(r_sc.unit)))}). "
            "Travel time is ill-defined for r_sc <= r_ss."
        )


def _robust_speed(
    V: u.Quantity,
    *,
    v_min: u.Quantity,
    v_fallback: u.Quantity,
) -> tuple[u.Quantity, np.ndarray]:
    """Return (V_eff, fallback_mask) with V_eff >= v_min and finite."""
    V = _as1d(V.to(v_min.unit))
    v_min = u.Quantity(v_min).to(V.unit)
    v_fallback = u.Quantity(v_fallback).to(V.unit)

    arr = V.to_value(V.unit)
    bad = (~np.isfinite(arr)) | (arr < v_min.to_value(V.unit))

    V_eff = V.copy()
    if np.any(bad):
        V_eff = V_eff.to(v_fallback.unit)
        V_eff[bad] = v_fallback
        V_eff = V_eff.to(v_min.unit)
    return V_eff, bad


class TravelTimeModel:
    name: str = "base"

    def evaluate(
        self,
        *,
        r_sc: u.Quantity,
        V_bg: u.Quantity,
        r_ss: u.Quantity,
        v_min: u.Quantity,
        v_fallback: u.Quantity,
    ) -> TravelTimeResult:
        raise NotImplementedError

    def speed_profile(
        self,
        *,
        r_grid: u.Quantity,
        r_sc: u.Quantity,
        V_bg: u.Quantity,
        r_ss: u.Quantity,
        v_min: Optional[u.Quantity] = None,
    ) -> Optional[u.Quantity]:
        """Return U(r) along r_grid for a representative sample.

        This is used for visualization/diagnostics (e.g., showing the assumed
        velocity/acceleration profile). The default returns None.
        """
        return None


class BallisticBackground(TravelTimeModel):
    """Ballistic travel time using a background outflow speed V_bg(t)."""

    name = "ballistic_bg"

    def evaluate(self, *, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: u.Quantity, v_fallback: u.Quantity) -> TravelTimeResult:
        r_sc = _as1d(r_sc)
        V_bg = _as1d(V_bg)
        _validate_radii(r_sc, r_ss)

        V_eff, fb = _robust_speed(V_bg, v_min=v_min, v_fallback=v_fallback)
        tau = (r_sc - r_ss).to(u.m) / V_eff.to(u.m / u.s)
        # Strict positivity
        if np.nanmin(tau.to_value(u.s)) <= 0:
            raise ValueError("Internal error: non-positive tau produced in BallisticBackground.")

        meta = {
            "model": self.name,
            "assumption": "radial ballistic propagation using background speed V_bg(t)",
        }
        return TravelTimeResult(tau=tau.to(u.s), fallback_mask=fb, meta=meta)

    def speed_profile(self, *, r_grid: u.Quantity, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: Optional[u.Quantity] = None) -> Optional[u.Quantity]:
        rr = _as1d(u.Quantity(r_grid).to(u.m))
        U = np.full(rr.shape, float(u.Quantity(V_bg).to_value(u.m / u.s))) * (u.m / u.s)
        return u.Quantity(U).to(u.km / u.s)


class ConstantSpeed(TravelTimeModel):
    """Constant-speed baseline: tau = (r_sc - r_ss)/V0."""

    name = "constant"

    def __init__(self, V0: u.Quantity):
        V0 = u.Quantity(V0)
        if not V0.isscalar:
            raise ValueError("V0 must be a scalar Quantity.")
        if not np.isfinite(V0.to_value(u.m / u.s)) or (V0 <= 0 * (u.m / u.s)):
            raise ValueError("V0 must be finite and > 0.")
        self.V0 = V0.to(u.m / u.s)

    def evaluate(self, *, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: u.Quantity, v_fallback: u.Quantity) -> TravelTimeResult:
        r_sc = _as1d(r_sc)
        _validate_radii(r_sc, r_ss)
        tau = (r_sc - r_ss).to(u.m) / self.V0
        meta = {"model": self.name, "V0": self.V0.to_value(u.km / u.s)}
        fb = np.zeros(len(r_sc), dtype=bool)
        return TravelTimeResult(tau=tau.to(u.s), fallback_mask=fb, meta=meta)

    def speed_profile(self, *, r_grid: u.Quantity, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: Optional[u.Quantity] = None) -> Optional[u.Quantity]:
        rr = _as1d(u.Quantity(r_grid).to(u.m))
        U = np.full(rr.shape, float(self.V0.to_value(u.m / u.s))) * (u.m / u.s)
        return u.Quantity(U).to(u.km / u.s)


class ScaledExpAcceleration(TravelTimeModel):
    """Monotonic acceleration-shape parameterization scaled to match V_bg(t) at r_sc.

    Shape
    -----
    We define a dimensionless monotonic shape factor s(r) in (u0, 1]:

        s(r) = u0 + (1-u0) * (1 - exp(-((r-r_ss)/L)^a)).

    Then set:

        U(r,t) = V_bg(t) * s(r) / s(r_sc(t)),

    so that U(r_sc,t) = V_bg(t) exactly.

    This is a **heuristic** acceleration profile unless independently justified.

    Guarantees
    ----------
    - u0>0 ensures s(r) is bounded away from zero -> no singular travel times.
    """

    name = "exp_accel"

    def __init__(self, *, L: u.Quantity = 20 * u.R_sun, a: float = 1.0, u0: float = 0.05, n_grid: int = 4096):
        L = u.Quantity(L)
        if not L.isscalar or (L <= 0 * L.unit) or (not np.isfinite(L.to_value(L.unit))):
            raise ValueError("L must be a finite, positive scalar Quantity.")
        if not np.isfinite(a) or a <= 0:
            raise ValueError("a must be finite and > 0.")
        if not np.isfinite(u0) or not (0.0 < u0 < 1.0):
            raise ValueError("u0 must be in (0,1).")
        self.L = L.to(u.m)
        self.a = float(a)
        self.u0 = float(u0)
        self.n_grid = int(max(256, n_grid))

    def _shape(self, r: u.Quantity, r_ss: u.Quantity) -> np.ndarray:
        r = u.Quantity(r).to(u.m)
        r_ss = u.Quantity(r_ss).to(u.m)
        x = np.maximum((r - r_ss).to_value(u.m), 0.0) / self.L.to_value(u.m)
        x = np.power(x, self.a)
        s = self.u0 + (1.0 - self.u0) * (1.0 - np.exp(-x))
        return s

    def evaluate(self, *, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: u.Quantity, v_fallback: u.Quantity) -> TravelTimeResult:
        r_sc = _as1d(r_sc)
        V_bg = _as1d(V_bg)
        _validate_radii(r_sc, r_ss)

        V_eff, fb = _robust_speed(V_bg, v_min=v_min, v_fallback=v_fallback)

        r_max = np.nanmax(r_sc.to_value(u.m)) * u.m
        r_grid = np.linspace(r_ss.to_value(u.m), r_max.to_value(u.m), self.n_grid) * u.m

        s_grid = self._shape(r_grid, r_ss)
        if np.any(~np.isfinite(s_grid)) or np.any(s_grid <= 0):
            raise ValueError("Invalid acceleration shape: non-finite or non-positive s(r) on grid.")

        # cumulative integral J(r)=∫ dr/s(r)
        inv_s = 1.0 / s_grid
        dr = np.diff(r_grid.to_value(u.m))
        J = np.zeros_like(s_grid, dtype=float)
        J[1:] = np.cumsum(0.5 * (inv_s[1:] + inv_s[:-1]) * dr)

        # interpolate to r_sc
        r_grid_m = r_grid.to_value(u.m)
        r_sc_m = r_sc.to_value(u.m)
        s_sc = np.interp(r_sc_m, r_grid_m, s_grid)
        J_sc = np.interp(r_sc_m, r_grid_m, J)

        # tau = (s_sc/V_eff) * J_sc
        tau = (s_sc * J_sc) * u.m / V_eff.to(u.m / u.s)

        if np.nanmin(tau.to_value(u.s)) <= 0:
            raise ValueError("Internal error: non-positive tau produced in ScaledExpAcceleration.")

        meta = {
            "model": self.name,
            "assumption": "monotonic exp-shaped acceleration scaled to match V_bg at r_sc",
            "L_Rsun": float(self.L.to_value(u.R_sun)),
            "a": float(self.a),
            "u0": float(self.u0),
            "n_grid": int(self.n_grid),
        }
        return TravelTimeResult(tau=tau.to(u.s), fallback_mask=fb, meta=meta)

    def speed_profile(self, *, r_grid: u.Quantity, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: Optional[u.Quantity] = None) -> Optional[u.Quantity]:
        rr = _as1d(u.Quantity(r_grid).to(u.m))
        r_scq = u.Quantity(r_sc).to(u.m)
        V = u.Quantity(V_bg).to(u.m / u.s)
        s = self._shape(rr, r_ss)
        s_sc = _first_scalar(self._shape(r_scq, r_ss))
        s_sc = max(s_sc, 1e-12)
        U = V * (s / s_sc)
        if v_min is not None:
            U = np.maximum(U.to_value(u.m / u.s), u.Quantity(v_min).to_value(u.m / u.s)) * (u.m / u.s)
        return u.Quantity(U).to(u.km / u.s)



class HybridParker(TravelTimeModel):
    """Hybrid Parker approximations (large-distance + small-distance), scaled to match V_bg at r_sc.

    This implements a practical family of accelerating profiles inspired by
    common Parker-wind asymptotics. The free parameter ``rs`` controls the
    "acceleration scale" via the functional forms. For numerical robustness
    and for visually sensible profiles, the default joining mode is:

        switch="match": join the two asymptotics at the first radius r_join > rs
        where v_small(r_join) = v_large(r_join). This yields continuity in v(r)
        with a local jump in the derivative.

    If you explicitly request switch="rs", we switch at r=rs (less physical but
    matches the literal wording in some heuristic formulations).
    """

    name = "hybrid_parker"

    def __init__(self, *, rs: u.Quantity = 2.7 * u.R_sun, n_grid: int = 8192, switch: str = "match"):
        self.rs = u.Quantity(rs).to(u.m)
        self.n_grid = int(n_grid)
        self.switch = str(switch).strip().lower()
        if self.switch not in {"match", "rs"}:
            raise ValueError("HybridParker: switch must be 'match' or 'rs'.")

    @staticmethod
    def _f_large(r: u.Quantity, *, rs: u.Quantity, r_sc: u.Quantity) -> np.ndarray:
        rr = u.Quantity(r).to(u.m).to_value(u.m)
        rs_m = u.Quantity(rs).to(u.m).to_value(u.m)
        rsc_m = u.Quantity(r_sc).to(u.m).to_value(u.m)

        # Guard: large-distance form requires r > rs.
        x = np.maximum(rr / rs_m, 1.0 + 1e-12)
        xsc = max(rsc_m / rs_m, 1.0 + 1e-12)

        return np.sqrt(np.log(x)) / np.sqrt(np.log(xsc))

    @staticmethod
    def _f_small(r: u.Quantity, *, rs: u.Quantity, r_sc: u.Quantity) -> np.ndarray:
        rr = u.Quantity(r).to(u.m).to_value(u.m)
        rs_m = u.Quantity(rs).to(u.m).to_value(u.m)
        rsc_m = u.Quantity(r_sc).to(u.m).to_value(u.m)

        # Small-distance exponential form; scaled so f_small(r_sc)=1.
        return np.exp(2.0 * rs_m * (1.0 / rsc_m - 1.0 / rr))

    def _find_join_radius(self, *, r_ss: u.Quantity, r_sc: u.Quantity) -> Optional[u.Quantity]:
        """Find first r_join > rs with f_large(r)=f_small(r), if it exists."""
        rs = self.rs
        r_ss = u.Quantity(r_ss).to(u.m)
        r_sc = u.Quantity(r_sc).to(u.m)

        # We can only evaluate the large-distance form for r > rs.
        r_lo = max(r_ss, (1.0 + 1e-6) * rs)
        r_hi = r_sc
        if r_hi <= r_lo * (1.0 + 1e-9):
            return None

        # Bracket by scanning in log-space.
        rr = np.geomspace(r_lo.to_value(u.m), r_hi.to_value(u.m), num=1024) * u.m
        diff = self._f_large(rr, rs=rs, r_sc=r_sc) - self._f_small(rr, rs=rs, r_sc=r_sc)

        # Ignore NaNs (should not occur with guards, but be safe).
        ok = np.isfinite(diff)
        if ok.sum() < 4:
            return None
        rr = rr[ok]
        diff = diff[ok]

        s = np.sign(diff)
        # Look for the first sign change.
        idx = np.where(s[:-1] * s[1:] < 0.0)[0]
        if idx.size == 0:
            return None

        i = int(idx[0])
        r1 = rr[i].to_value(u.m)
        r2 = rr[i + 1].to_value(u.m)
        d1 = float(diff[i])
        d2 = float(diff[i + 1])

        # Linear interpolation is sufficient here (we only need a stable join radius).
        if d2 == d1:
            rj = 0.5 * (r1 + r2)
        else:
            rj = r1 - d1 * (r2 - r1) / (d2 - d1)

        return u.Quantity(rj, u.m)

    def speed_profile(
        self,
        *,
        r_grid: u.Quantity,
        r_sc: u.Quantity,
        V_bg: u.Quantity,
        r_ss: u.Quantity,
        v_min: Optional[u.Quantity] = None,
    ) -> Optional[u.Quantity]:
        rr = _as1d(u.Quantity(r_grid).to(u.m))
        r_scq = u.Quantity(r_sc).to(u.m)
        V = u.Quantity(V_bg).to(u.m / u.s)
        rs = self.rs

        fL = self._f_large(rr, rs=rs, r_sc=r_scq)
        fS = self._f_small(rr, rs=rs, r_sc=r_scq)

        if self.switch == "rs":
            r_join = rs
        else:
            r_join = self._find_join_radius(r_ss=u.Quantity(r_ss), r_sc=r_scq)
            if r_join is None:
                # If no join exists, fall back to a single branch:
                # - If r_ss > rs, large-distance is at least defined at r_ss.
                # - Otherwise, use the small-distance form.
                r_join = u.Quantity(r_ss).to(u.m)

        vL = V * fL
        vS = V * fS

        use_small = rr <= u.Quantity(r_join).to(u.m)
        U = np.where(use_small, vS.to_value(u.m / u.s), vL.to_value(u.m / u.s)) * (u.m / u.s)

        if v_min is not None:
            U = np.maximum(U.to_value(u.m / u.s), u.Quantity(v_min).to_value(u.m / u.s)) * (u.m / u.s)

        return u.Quantity(U).to(u.km / u.s)

    def evaluate(
        self,
        *,
        r_sc: u.Quantity,
        V_bg: u.Quantity,
        r_ss: u.Quantity,
        v_min: u.Quantity,
        v_fallback: u.Quantity,
    ) -> TravelTimeResult:
        r_sc = u.Quantity(r_sc).to(u.m)
        V_bg = u.Quantity(V_bg).to(u.m / u.s)
        r_ss = u.Quantity(r_ss).to(u.m)

        rr_sc = np.atleast_1d(r_sc.to_value(u.m))
        VV = np.atleast_1d(V_bg.to_value(u.m / u.s))

        tau = np.full(rr_sc.shape, np.nan, dtype=float) * u.s
        fb = np.zeros(rr_sc.shape, dtype=bool)

        # Diagnostics
        n_join_none = 0

        for i, (r_i, v_i) in enumerate(zip(rr_sc, VV)):
            if not np.isfinite(r_i) or not np.isfinite(v_i):
                fb[i] = True
                continue

            if r_i <= r_ss.to_value(u.m):
                fb[i] = True
                continue

            V_eff = max(v_i, u.Quantity(v_min).to_value(u.m / u.s))
            V_eff = max(V_eff, 1e-9)

            r_grid = np.geomspace(r_ss.to_value(u.m), r_i, num=self.n_grid) * u.m
            U = self.speed_profile(
                r_grid=r_grid,
                r_sc=u.Quantity(r_i, u.m),
                V_bg=u.Quantity(V_eff, u.m / u.s),
                r_ss=r_ss,
                v_min=v_min,
            )
            if U is None:
                fb[i] = True
                continue

            # Travel time integral
            r_m = r_grid.to_value(u.m)
            inv_u = 1.0 / U.to_value(u.m / u.s)
            tau_sec = float(np.trapz(inv_u, r_m))
            if not np.isfinite(tau_sec) or tau_sec <= 0:
                fb[i] = True
                continue
            tau[i] = tau_sec * u.s

            if self.switch == "match":
                # Count join failures for reporting (speed_profile falls back to r_ss in that case).
                rj = self._find_join_radius(r_ss=r_ss, r_sc=u.Quantity(r_i, u.m))
                if rj is None:
                    n_join_none += 1

        meta = {
            "model": self.name,
            "assumption": "hybrid Parker asymptotics scaled to match V_bg at r_sc",
            "rs_Rsun": float(self.rs.to_value(u.R_sun)),
            "switch": str(self.switch),
            "n_grid": int(self.n_grid),
            "diagnostics": {
                "join_not_found_count": int(n_join_none),
                "join_not_found_fraction": float(n_join_none / max(len(rr_sc), 1)),
            },
        }
        return TravelTimeResult(tau=tau.to(u.s), fallback_mask=fb, meta=meta)



class ParkerScaled(TravelTimeModel):
    """Isothermal Parker-wind *shape* scaled to match V_bg(t) at r_sc.

    This is the most defensible next-step beyond a purely ballistic mapping:
    the Parker solution provides a first-principles, monotonic, transonic
    acceleration profile. We use it only as a *shape* and scale it so that
    U(r_sc,t)=V_bg(t). The only remaining shape parameter is the critical
    radius r_c.
    """

    name = "parker_scaled"

    def __init__(self, *, r_c: u.Quantity = 6.0 * u.R_sun, n_grid: int = 8192):
        r_c = u.Quantity(r_c)
        if not r_c.isscalar or (r_c <= 0 * r_c.unit) or (not np.isfinite(r_c.to_value(r_c.unit))):
            raise ValueError("r_c must be a finite, positive scalar Quantity.")
        self.r_c = r_c.to(u.m)
        self.n_grid = int(max(512, n_grid))
        if not _HAS_LAMBERTW:
            raise RuntimeError("parker_scaled requires scipy (scipy.special.lambertw).")

    def _u_parker_shape(self, r: u.Quantity) -> np.ndarray:
        r = u.Quantity(r).to(u.m)
        rc = self.r_c
        rr = np.asarray(r.to_value(u.m), float)
        rcv = float(rc.to_value(u.m))
        J = 4.0 * (np.log(rr / rcv) + (rcv / rr) - 1.0)
        z = -np.exp(-(J + 1.0))
        w0 = lambertw(z, k=0).real
        w1 = lambertw(z, k=-1).real
        w = np.where(rr <= rcv, w0, w1)
        uhat = np.sqrt(np.maximum(-w, 0.0))
        uhat[~np.isfinite(uhat)] = np.nan
        return uhat

    def evaluate(self, *, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: u.Quantity, v_fallback: u.Quantity) -> TravelTimeResult:
        r_sc = _as1d(r_sc)
        V_bg = _as1d(V_bg)
        _validate_radii(r_sc, r_ss)

        V_eff, fb = _robust_speed(V_bg, v_min=v_min, v_fallback=v_fallback)

        r_max = np.nanmax(r_sc.to_value(u.m)) * u.m
        r_grid = np.linspace(r_ss.to_value(u.m), r_max.to_value(u.m), self.n_grid) * u.m

        u_grid = self._u_parker_shape(r_grid)
        if np.any(~np.isfinite(u_grid)) or np.nanmin(u_grid) <= 0:
            raise ValueError("Invalid Parker shape: non-finite or non-positive u(r) on grid.")

        inv_u = 1.0 / u_grid
        dr = np.diff(r_grid.to_value(u.m))
        I = np.zeros_like(u_grid, dtype=float)
        I[1:] = np.cumsum(0.5 * (inv_u[1:] + inv_u[:-1]) * dr)  # units: m

        r_grid_m = r_grid.to_value(u.m)
        r_sc_m = r_sc.to_value(u.m)
        u_sc = np.interp(r_sc_m, r_grid_m, u_grid)
        I_sc = np.interp(r_sc_m, r_grid_m, I)

        tau = (u_sc * I_sc) * u.m / V_eff.to(u.m / u.s)
        if np.nanmin(tau.to_value(u.s)) <= 0:
            raise ValueError("Internal error: non-positive tau produced in ParkerScaled.")

        meta = {
            "model": self.name,
            "assumption": "isothermal Parker-wind shape scaled to match V_bg at r_sc",
            "r_c_Rsun": float(self.r_c.to_value(u.R_sun)),
            "n_grid": int(self.n_grid),
        }
        return TravelTimeResult(tau=tau.to(u.s), fallback_mask=fb, meta=meta)

    def speed_profile(self, *, r_grid: u.Quantity, r_sc: u.Quantity, V_bg: u.Quantity, r_ss: u.Quantity, v_min: Optional[u.Quantity] = None) -> Optional[u.Quantity]:
        rr = _as1d(u.Quantity(r_grid).to(u.m))
        r_scq = u.Quantity(r_sc).to(u.m)
        V = u.Quantity(V_bg).to(u.m / u.s)
        u_grid = self._u_parker_shape(rr)
        u_sc = _first_scalar(self._u_parker_shape(r_scq))
        u_sc = max(u_sc, 1e-12)
        U = V * (u_grid / u_sc)
        if v_min is not None:
            U = np.maximum(U.to_value(u.m / u.s), u.Quantity(v_min).to_value(u.m / u.s)) * (u.m / u.s)
        return u.Quantity(U).to(u.km / u.s)


def build_model(name: str, *, model_kwargs: Optional[Dict[str, Any]] = None) -> TravelTimeModel:
    key = str(name).strip().lower()
    kw = dict(model_kwargs or {})

    if key in {"ballistic_bg", "ballistic", "background", "bg"}:
        return BallisticBackground()
    if key in {"constant", "const"}:
        if "V0" not in kw:
            raise ValueError("constant model requires model_kwargs={'V0': <Quantity>}.")
        return ConstantSpeed(kw.pop("V0"))
    if key in {"exp_accel", "accel", "acceleration", "empirical_exp"}:
        return ScaledExpAcceleration(**kw)

    if key in {"hybrid_parker", "hybrid"}:
        rs = kw.pop("rs", 2.7 * u.R_sun)
        n_grid = int(kw.pop("n_grid", 8192))
        switch = str(kw.pop("switch", "match"))
        return HybridParker(rs=rs, n_grid=n_grid, switch=switch)

    if key in {"parker_scaled", "parker"}:
        # Backward-compatible alias: use the hybrid Parker asymptotics (paper-style heuristic),
        # not the Lambert-W isothermal Parker solution.
        rs = kw.pop("rs", 2.7 * u.R_sun)
        n_grid = int(kw.pop("n_grid", 8192))
        switch = str(kw.pop("switch", "match"))
        return HybridParker(rs=rs, n_grid=n_grid, switch=switch)

    if key in {"parker_lambert", "parker_lambert_scaled", "parker_isothermal"}:
        # Optional: the exact isothermal Parker-wind (Lambert-W) shape, scaled to match V_bg at r_sc.
        if "r_c" in kw:
            rc = kw.pop("r_c")
        else:
            rc = 6.0 * u.R_sun
        n_grid = int(kw.pop("n_grid", 8192))
        return ParkerScaled(r_c=rc, n_grid=n_grid)

    raise ValueError(
        f"Unknown travel-time model {name!r}. Allowed: ballistic_bg, constant, exp_accel, parker_scaled, hybrid_parker, parker_lambert."
    )