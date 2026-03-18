"""sc_pos.backmap.validation

Executable sanity checks for backmapping outputs.

These checks are designed to be lightweight and to fail loudly when a common
silent-corruption mode occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

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


def validate_tau_positive(df: pd.DataFrame) -> List[str]:
    failures = []
    if "tau_s" not in df.columns:
        return ["Missing tau_s column"]
    tau = pd.to_numeric(df["tau_s"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(tau) & (tau > 0)
    if int(np.sum(ok)) == 0:
        failures.append("All tau_s are invalid (NaN or <=0).")
    return failures


def validate_longitude_wrap(df: pd.DataFrame, col: str = "phi_src") -> List[str]:
    failures = []
    if col not in df.columns:
        return [f"Missing {col} column"]
    phi = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(phi)
    if int(np.sum(m)) == 0:
        return [f"All {col} are NaN"]
    if (np.nanmin(phi[m]) < -1e-6) or (np.nanmax(phi[m]) > 360.0 + 1e-6):
        failures.append(f"{col} not wrapped to [0,360): min={np.nanmin(phi[m]):.3g}, max={np.nanmax(phi[m]):.3g}")
    return failures


def validate_determinism(d1: pd.DataFrame, d2: pd.DataFrame) -> List[str]:
    failures: List[str] = []
    cols = [c for c in ["phi_src", "lat_src", "tau", "Vr_bg", "r_sc"] if c in d1.columns and c in d2.columns]
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


def _assert(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"[PASS] {name} :: {detail}")
        return
    raise AssertionError(f"[FAIL] {name} :: {detail}")


def circular_wrap_test() -> None:
    """p16/p84 across the 0/360 boundary should be tight, not ~180 deg wide."""
    from .circular import circ_percentile_deg, wrap_0_360

    rng = np.random.default_rng(0)
    x = wrap_0_360(359.5 + 0.3 * rng.standard_normal(5000))
    p16 = float(circ_percentile_deg(x, 16.0))
    p84 = float(circ_percentile_deg(x, 84.0))
    mu = float(wrap_0_360(np.degrees(np.angle(np.nanmean(np.exp(1j * np.deg2rad(x)))))))
    d16 = ((p16 - mu + 180.0) % 360.0) - 180.0
    d84 = ((p84 - mu + 180.0) % 360.0) - 180.0
    width = float(d84 - d16)
    _assert("circular wrap p16/p84", np.isfinite(width) and abs(width) < 5.0, f"width≈{width:.3g} deg (expected <5 deg)")


def sign_convention_test() -> None:
    """Synthetic sign test for phi_src = wrap(phi_sc + phi_sign * omega * tau)."""
    from .circular import wrap_0_360

    phi_sc = 10.0
    omega = 14.1844 / 86400.0
    tau = 3600.0
    phi_src_p = wrap_0_360(phi_sc + (+1) * omega * tau)
    phi_src_m = wrap_0_360(phi_sc + (-1) * omega * tau)
    _assert("sign convention (+1 vs -1)", phi_src_p != phi_src_m, f"phi(+1)={phi_src_p:.3f}, phi(-1)={phi_src_m:.3f}")


def travel_time_longitude_monotonicity_test() -> None:
    """Longer travel time must map to a larger Carrington longitude for phi_sign=+1."""
    from .circular import wrap_0_360

    phi_sc = 120.0
    omega_deg_s = 14.1844 / 86400.0
    tau_short = 40.0 * 3600.0
    tau_long = 70.0 * 3600.0

    phi_short = wrap_0_360(phi_sc + omega_deg_s * tau_short)
    phi_long = wrap_0_360(phi_sc + omega_deg_s * tau_long)

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
    """Quick guard that plot writing works end-to-end (no silent failures)."""
    import pandas as pd
    import astropy.units as u
    from .plotting import plot_source_surface_2d

    t = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame(
        {
            "phi_sc": [0.0, 1.0, 2.0],
            "lat_sc": [0.0, 0.0, 0.0],
            "phi_src": [10.0, 11.0, 12.0],
            "lat_src": [0.0, 0.0, 0.0],
            "tau_s": [3600.0, 3600.0, 3600.0],
            "tau": [1.0, 1.0, 1.0],
            "Vr_bg": [400.0, 400.0, 400.0],
            "r_sc": [20.0, 20.0, 20.0],
            "sigma_phi": [1.0, 1.0, 1.0],
            "marker_size": [20.0, 20.0, 20.0],
        },
        index=t,
    )
    df.attrs["units"] = {
        "phi_sc": u.deg,
        "lat_sc": u.deg,
        "phi_src": u.deg,
        "lat_src": u.deg,
        "tau_s": u.s,
        "tau": u.hour,
        "Vr_bg": u.km / u.s,
        "r_sc": u.R_sun,
        "sigma_phi": u.deg,
    }
    out = Path("traceability_guard_test.png")
    plot_source_surface_2d(
        data=df,
        out_png=out,
        plot_vars=["Vr_bg"],
        var_specs=None,
        percentiles=(2.0, 98.0),
        size_col="marker_size",
        show_uncertainty=False,
        summary_box="test",
        title="test",
        figsize=(6, 3),
        show=False,
    )
    _assert("traceability guard plot", out.exists() and out.stat().st_size > 0, f"{out}")
