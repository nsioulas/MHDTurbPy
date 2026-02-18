"""sc_pos.backmap.validation

Executable sanity checks for backmapping outputs.

These checks are designed to be lightweight and to fail loudly when a common
silent-corruption mode occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

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
