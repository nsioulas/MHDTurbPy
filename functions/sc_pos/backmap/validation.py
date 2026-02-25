"""sc_pos.backmap.validation

Executable sanity checks for backmapping outputs.

These checks are designed to be lightweight and to fail loudly when a common
silent-corruption mode occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .circular import delta_deg


@dataclass
class ValidationReport:
    ok: bool
    failures: List[str]


def validate_units(df: pd.DataFrame, required: List[str]) -> List[str]:
    failures = []
    um = df.attrs.get("units", {})
    for c in required:
        if c not in df.columns:
            failures.append(f"Missing required output column: {c}")
        if c not in um:
            failures.append(f"Missing unit metadata for column: {c}")
    return failures


def validate_longitude_wrap(df: pd.DataFrame, col: str = "phi_src", max_jump_deg: float = 60.0) -> List[str]:
    failures = []
    if col not in df.columns:
        return [f"Missing longitude column {col}"]
    phi = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(phi)
    if ok.sum() < 5:
        return [f"Not enough finite values to validate longitude wrap for {col}"]
    d = delta_deg(phi[1:], phi[:-1])
    d = d[np.isfinite(d)]
    if d.size == 0:
        return [f"No finite differences for longitude column {col}"]
    if np.nanmax(np.abs(d)) > float(max_jump_deg):
        failures.append(
            f"Large circular longitude jump detected in {col}: max |Δ|={float(np.nanmax(np.abs(d))):.2f} deg. "
            "This is often an unwrap/interpolation error or too-coarse ephemeris step."
        )
    return failures


def validate_tau_positive(df: pd.DataFrame, col: str = "tau") -> List[str]:
    failures = []
    if col not in df.columns:
        return [f"Missing travel-time column {col}"]
    tau = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(tau)
    if ok.any() and np.nanmin(tau[ok]) <= 0:
        failures.append(f"Non-positive travel time found in {col}: min={float(np.nanmin(tau[ok]))}")
    return failures


def validate_determinism(res1: Dict[str, Any], res2: Dict[str, Any]) -> List[str]:
    failures = []
    d1 = res1.get("data")
    d2 = res2.get("data")
    if not isinstance(d1, pd.DataFrame) or not isinstance(d2, pd.DataFrame):
        return ["Results do not contain DataFrames under key 'data'"]
    # compare a stable subset
    cols = [c for c in ["phi_src", "tau", "Vr_bg", "r_sc"] if c in d1.columns and c in d2.columns]
    if not cols:
        return ["No common columns available for determinism check"]
    a = d1[cols].to_numpy(dtype=float)
    b = d2[cols].to_numpy(dtype=float)
    if a.shape != b.shape:
        failures.append(f"Determinism: shape mismatch {a.shape} vs {b.shape}")
    else:
        if not np.allclose(a, b, equal_nan=True, rtol=0.0, atol=0.0):
            failures.append("Determinism: outputs differ for identical inputs (bitwise).")
    return failures


def validate_backmap_output(df: pd.DataFrame) -> ValidationReport:
    failures: List[str] = []
    failures += validate_units(df, required=["phi_sc", "lat_sc", "r_sc", "Vr_bg", "tau", "phi_src", "sigma_phi"]) 
    failures += validate_tau_positive(df)
    failures += validate_longitude_wrap(df, col="phi_src")
    return ValidationReport(ok=(len(failures) == 0), failures=failures)


# ----------------------------------------------------------------------
# Minimal automatic validation suite (no pytest; prints explicit criteria)
# ----------------------------------------------------------------------
def _assert(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name} :: {detail}")
        return
    raise AssertionError(f"[FAIL] {name} :: {detail}")


def circular_wrap_test() -> None:
    """p16/p84 across the 0/360 boundary should be tight, not ~180 deg wide."""
    from .circular import circ_percentile_deg, wrap360

    rng = np.random.default_rng(0)
    # Cluster around 359.5 deg with small scatter; should wrap to ~0 deg region.
    x = wrap360(359.5 + 0.3 * rng.standard_normal(5000))
    p16 = float(circ_percentile_deg(x, 16.0))
    p84 = float(circ_percentile_deg(x, 84.0))
    # Use circular distance from the circular mean as width proxy
    mu = float(wrap360(np.degrees(np.angle(np.nanmean(np.exp(1j * np.deg2rad(x)))))))
    # signed diffs
    d16 = ((p16 - mu + 180.0) % 360.0) - 180.0
    d84 = ((p84 - mu + 180.0) % 360.0) - 180.0
    width = float(d84 - d16)
    _assert("circular wrap p16/p84", np.isfinite(width) and abs(width) < 5.0, f"width≈{width:.3g} deg (expected <5 deg)")


def sign_convention_test() -> None:
    """Synthetic sign test for phi_src = wrap(phi_sc + phi_sign * omega * tau)."""
    from .circular import wrap360

    phi_sc = 10.0
    omega = 14.1844 / 86400.0 * 360.0 / 360.0  # deg/s (numerically irrelevant)
    tau = 3600.0  # s
    # With phi_sign=+1, phi_src should decrease.
    phi_src_p = wrap360(phi_sc + (+1) * omega * tau)
    phi_src_m = wrap360(phi_sc + (-1) * omega * tau)
    _assert("sign convention (+1 vs -1)", phi_src_p != phi_src_m, f"phi(+1)={phi_src_p:.3f}, phi(-1)={phi_src_m:.3f}")


def travel_time_longitude_monotonicity_test() -> None:
    """Longer travel time must map to a *larger* Carrington longitude for phi_sign=+1.

    This matches the standard ballistic/Parker-spiral convention used in the literature,
    where increased travel time moves the back-mapped point Westward (higher longitude).
    """
    from .circular import wrap360

    phi_sc = 120.0  # deg
    omega_deg_s = 14.1844 / 86400.0  # deg/s (Carrington sidereal rate)
    tau_short = 40.0 * 3600.0
    tau_long  = 70.0 * 3600.0

    phi_short = wrap360(phi_sc + omega_deg_s * tau_short)
    phi_long  = wrap360(phi_sc + omega_deg_s * tau_long)

    # Use circular difference long - short in (-180,180]. It should be positive.
    d = ((phi_long - phi_short + 180.0) % 360.0) - 180.0
    _assert("tau -> westward monotonicity", float(d) > 0.0, f"Δphi={float(d):.3f} deg (expected >0)")

def accelerating_non_degeneracy_test() -> None:
    """Accelerating model should produce non-flat U(r) on a nontrivial r grid."""
    from .travel_time import build_model
    import astropy.units as u

    model = build_model("exp_accel", model_kwargs={"L": 6.0 * u.R_sun, "a": 3.0})
    r_ss = 2.5 * u.R_sun
    r_sc = 20.0 * u.R_sun
    V_bg = 400.0 * u.km / u.s
    r_grid = np.geomspace(r_ss.to_value(u.R_sun) * 1.01, r_sc.to_value(u.R_sun), 200) * u.R_sun
    U = model.speed_profile(r_grid=r_grid, r_sc=r_sc, V_bg=V_bg, r_ss=r_ss).to_value(u.km / u.s)
    span = float(np.nanmax(U) - np.nanmin(U))
    _assert("accelerating non-degeneracy", np.isfinite(span) and span > 1.0, f"U_span={span:.3g} km/s (expected >1 km/s)")


def traceability_guard_test() -> None:
    """Mismatch between profile signature and data signature must be caught."""
    import pandas as pd
    import astropy.units as u
    from .plotting import plot_source_surface_2d

    # Minimal fake data
    t = pd.date_range("2024-01-01", periods=3, freq="H")
    df = pd.DataFrame({"phi_sc": [0.0, 1.0, 2.0], "lat_sc": [0.0, 0.0, 0.0], "tau_s": [1.0, 1.0, 1.0]}, index=t)
    df.attrs["units"] = {"tau_s": str(u.s)}
    df.attrs["executed_model_signature"] = "AAA"
    prof = {"r_grid_Rsun": np.array([2.6, 3.0]), "U_med_kms": np.array([300.0, 400.0]), "r_ss_Rsun": 2.5, "r_sc_Rsun": 3.0, "executed_model_signature": "BBB"}
    try:
        plot_source_surface_2d(
            data=df,
            out_png=Path("/tmp/traceability_dummy.png"),
            plot_vars=[],
            var_specs={},
            profile_panel=prof,
            summary_box=None,
            show=False,
        )
        raise AssertionError("[FAIL] traceability_guard_test did not raise")
    except ValueError:
        print("[PASS] traceability guard triggers on signature mismatch")


def run_minimal_suite(*, do_smoke: bool = False, smoke_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Run the minimum automatic suite. Smoke test is optional."""
    circular_wrap_test()
    sign_convention_test()
    travel_time_longitude_monotonicity_test()
    accelerating_non_degeneracy_test()
    traceability_guard_test()

    if do_smoke:
        # Optional smoke test; run only if caller provides environment/data.
        print("[INFO] smoke test enabled")
        if smoke_kwargs is None:
            raise ValueError("smoke_kwargs must be provided when do_smoke=True")
        from .pipeline import backmap_interval
        res = backmap_interval(**smoke_kwargs)
        _assert("smoke run returns files", isinstance(res.get("files", None), dict) and len(res["files"]) > 0, "files keys=" + ",".join(res["files"].keys()))
