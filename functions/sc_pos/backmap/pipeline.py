from __future__ import annotations

"""sc_pos.backmap.pipeline

Minimal, physics-first source-surface backmapping for MHDTurbPy interval pickles.

Public API
----------
- ``backmap_interval``: the single canonical entry point.
- ``run_backmapping_interval``: backward-compatibility shim (thin wrapper).

What this module guarantees
---------------------------
- Deterministic interval loading by ``which_int`` using ``general_functions.load_files``.
- One canonical cadence index.
- Explicit units for all physical columns via ``df.attrs['units']``.
- Strictly positive travel times for r_sc > r_ss.
- Circular-safe longitude handling (no 0/360 interpolation spikes).
- Transparent metadata: no silent fallbacks.

What this module does NOT pretend to do
--------------------------------------
- It does not claim ballistic mapping is valid in shear/interaction regions.
- It does not model latitudinal evolution.
- It does not infer non-radial flow corrections unless you implement them explicitly.

The intent is a minimal, auditable baseline suitable for publication-grade figures.
"""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import astropy.units as u
    import astropy.constants as const
except Exception as e:  # pragma: no cover
    raise RuntimeError("MHDTurbPy backmapping requires astropy. Install with: pip install astropy") from e

# MHDTurbPy import style: notebooks commonly insert functions/ into sys.path.
_FUNCTIONS_DIR = Path(__file__).resolve().parents[2]
if str(_FUNCTIONS_DIR) not in sys.path:
    sys.path.insert(0, str(_FUNCTIONS_DIR))

import general_functions as func  # noqa: E402

from .circular import halfwidth_deg, circ_percentile_deg, wrap_0_360, circ_std_deg, delta_deg
from .ephemeris import get_ephemeris_hgc, interp_ephemeris_to_index
from .mapping import map_to_source_surface
from .azimuthal import compute_delta_phi_series
from .plotting import (
    VAR_SPECS,
    merge_var_specs,
    marker_sizes_from_metric,
    plot_source_surface_2d,
    plot_source_surface_3d,
    plot_velocity_profile,
    plot_carrington_diagnostics,
)
from .segmentation_figs import (
    plot_segmentation_score_timeseries,
    plot_segmentation_footpoints,
    plot_segmentation_schematic,
)
from .segmentation import segment_sources
from .travel_time import TravelTimeModel, TravelTimeResult, HybridParker, build_model
from .units import attach_units, default_input_units, q_from_df, to_value


# -----------------------------------------------------------------------------
# Strict column candidates (used only if explicit names are not supplied)
# -----------------------------------------------------------------------------
BR_CANDIDATES = ("Br", "B_R", "B_r", "BRTN_R")
VR_CANDIDATES = ("Vr", "V_R", "V_r", "Vsw", "|V|")
NP_CANDIDATES = ("Np", "np", "N_p", "n_p", "proton_density")


def _to_utc_index(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DatetimeIndex, got {type(df.index)}")
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    if out.empty:
        raise ValueError(f"{name} is empty after time parsing.")
    return out


def _extract_mag_par(fin: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministically extract MAG and PAR DataFrames from MHDTurbPy interval pickle."""
    if isinstance(fin, pd.DataFrame):
        df = _to_utc_index(fin, "fin")
        return df, df

    if not isinstance(fin, dict):
        raise TypeError(f"fin must be dict or DataFrame, got {type(fin)}")

    if "Mag" in fin and "Par" in fin and isinstance(fin["Mag"], (pd.DataFrame, dict)) and isinstance(fin["Par"], (pd.DataFrame, dict)):
        mag = fin["Mag"]
        par = fin["Par"]

        if isinstance(mag, dict):
            mag_frames = [(k, v) for k, v in mag.items() if isinstance(v, pd.DataFrame)]
            if len(mag_frames) != 1:
                raise ValueError(
                    "fin['Mag'] must contain exactly one DataFrame. "
                    f"Found {len(mag_frames)} frames: {[k for k,_ in mag_frames]}. "
                    "Fix the interval pickle to store a single MAG DataFrame under fin['Mag']."
                )
            mag = mag_frames[0][1]

        if isinstance(par, dict):
            par_frames = [(k, v) for k, v in par.items() if isinstance(v, pd.DataFrame)]
            if len(par_frames) != 1:
                raise ValueError(
                    "fin['Par'] must contain exactly one DataFrame. "
                    f"Found {len(par_frames)} frames: {[k for k,_ in par_frames]}. "
                    "Fix the interval pickle to store a single PAR DataFrame under fin['Par']."
                )
            par = par_frames[0][1]

        if isinstance(mag, pd.DataFrame) and isinstance(par, pd.DataFrame):
            return _to_utc_index(mag, "Mag"), _to_utc_index(par, "Par")

    # Strict fallback: scan all nested frames and require uniqueness.
    frames: List[pd.DataFrame] = []
    for v in fin.values():
        if isinstance(v, pd.DataFrame):
            frames.append(v)
        elif isinstance(v, dict):
            frames.extend([vv for vv in v.values() if isinstance(vv, pd.DataFrame)])

    if not frames:
        raise ValueError(f"fin dict contains no DataFrames. Keys={list(fin.keys())}")

    mag_hits, par_hits = [], []
    for df in frames:
        d = _to_utc_index(df, "candidate")
        if any(c in d.columns for c in BR_CANDIDATES):
            mag_hits.append(d)
        if any(c in d.columns for c in VR_CANDIDATES) and any(c in d.columns for c in NP_CANDIDATES):
            par_hits.append(d)

    if len(mag_hits) != 1 or len(par_hits) != 1:
        raise ValueError(
            "Could not uniquely identify MAG/PAR frames in fin. "
            f"MAG hits={len(mag_hits)}, PAR hits={len(par_hits)}. "
            "Fix the interval pickle to include fin['Mag'] and fin['Par'], or pass a single DataFrame with Br/Vr/Np."
        )

    return mag_hits[0], par_hits[0]


def _select_unique_col(df: pd.DataFrame, candidates: Sequence[str], label: str, explicit: Optional[str]) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Explicit {label} column {explicit!r} not in DataFrame.")
        return str(explicit)

    hits = [c for c in candidates if c in df.columns]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise KeyError(f"Missing {label}. Candidates={list(candidates)}. Available={list(df.columns)[:120]}...")
    raise KeyError(f"Ambiguous {label} column selection: matches={hits}. Pass an explicit column name.")


def load_interval_inputs(*, root_dir: Union[str, Path], sc: str, which_int: int) -> Dict[str, Any]:
    """Deterministic interval loading via MHDTurbPy's ``general_functions.load_files``."""
    root_dir = Path(root_dir)
    load_path = root_dir / "examples" / "downloaded_intervals" / str(sc)

    finnames = func.load_files(load_path, "final.pkl")
    gennames = func.load_files(load_path, "general.pkl")
    signames = func.load_files(load_path, "sig_c_sig_r.pkl")

    if which_int < 0 or which_int >= len(finnames):
        raise IndexError(f"which_int={which_int} out of range for {sc}. Available intervals: {len(finnames)}")

    fin = pd.read_pickle(finnames[which_int])
    gen = pd.read_pickle(gennames[which_int]) if len(gennames) else None
    sig = pd.read_pickle(signames[which_int]) if len(signames) else None
    if isinstance(sig, pd.DataFrame):
        sig = _to_utc_index(sig, "sig")

    return {
        "fin": fin,
        "gen": gen,
        "sig": sig if isinstance(sig, pd.DataFrame) else None,
        "paths": {
            "final": Path(finnames[which_int]),
            "general": Path(gennames[which_int]) if len(gennames) else None,
            "sig": Path(signames[which_int]) if len(signames) else None,
        },
    }




def load_interval_inputs_from_dir(*, interval_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load one interval directly from an interval folder.

    This is the robust path for range-mode selection: it does not rely on
    ``general_functions.load_files`` ordering, and instead reads the pickles
    that exist inside ``interval_dir``.

    Required
    --------
    - ``final.pkl`` must exist.

    Optional
    --------
    - ``general.pkl`` and ``sig_c_sig_r.pkl`` are loaded if present.
    """
    interval_dir = Path(interval_dir)

    fin_path = interval_dir / "final.pkl"
    if not fin_path.exists():
        raise FileNotFoundError(f"Missing final.pkl in interval_dir={interval_dir}")

    fin = pd.read_pickle(fin_path)

    gen_path = interval_dir / "general.pkl"
    gen = pd.read_pickle(gen_path) if gen_path.exists() else None

    sig_path = interval_dir / "sig_c_sig_r.pkl"
    sig = pd.read_pickle(sig_path) if sig_path.exists() else None
    if isinstance(sig, pd.DataFrame):
        sig = _to_utc_index(sig, "sig")

    return {
        "fin": fin,
        "gen": gen,
        "sig": sig if isinstance(sig, pd.DataFrame) else None,
        "paths": {"final": fin_path, "general": gen_path if gen_path.exists() else None, "sig": sig_path if sig_path.exists() else None},
    }
# -----------------------------------------------------------------------------
# Optional: remove data inside known gaps (MAG/PAR) before mapping/plotting
# -----------------------------------------------------------------------------

def load_padded_gaps(
    *,
    mag_gaps_path: Optional[Union[str, Path]] = None,
    par_gaps_path: Optional[Union[str, Path]] = None,
    gap_pad_frac: float = 0.5,
    index: Optional[Union[pd.DatetimeIndex, Sequence[Any]]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Load and standardize optional ``mag_gaps.pkl`` / ``par_gaps.pkl`` and build a padded union.

    Contract
    --------
    - Gap tables must contain ``Start`` and ``End`` columns.
    - All times are parsed as UTC (tz-aware) so comparisons against the pipeline's UTC index are valid.
    - Padding is symmetric: pad = gap_pad_frac * (End-Start) applied to BOTH sides.

    Returns
    -------
    gaps_padded : DataFrame or None
        Columns: Start, End (UTC tz-aware).
    keep : ndarray[bool] or None
        Returned only if ``index`` is provided. True means keep.
    """

    pad_frac = float(gap_pad_frac)
    if pad_frac < 0.0:
        raise ValueError("gap_pad_frac must be >= 0")

    def _read_df(p: Optional[Union[str, Path]]) -> Optional[pd.DataFrame]:
        if p is None:
            return None
        try:
            return pd.read_pickle(Path(p))
        except Exception:
            return None

    def _standardize(gaps: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if gaps is None or (not isinstance(gaps, pd.DataFrame)) or gaps.empty:
            return None
        cols = {str(c).lower(): c for c in gaps.columns}
        if "start" not in cols or "end" not in cols:
            raise KeyError(f"Gap dataframe must contain Start/End columns. Got: {list(gaps.columns)}")
        g = gaps[[cols["start"], cols["end"]]].copy()
        g.columns = ["Start", "End"]
        g["Start"] = pd.to_datetime(g["Start"], errors="coerce", utc=True)
        g["End"] = pd.to_datetime(g["End"], errors="coerce", utc=True)
        g = g.dropna()
        g = g[g["End"] > g["Start"]]
        if g.empty:
            return None
        return g.reset_index(drop=True)

    def _pad(g: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if g is None or g.empty:
            return None
        dt = (g["End"] - g["Start"])
        pad = dt * pad_frac
        out = g.copy()
        out["Start"] = out["Start"] - pad
        out["End"] = out["End"] + pad
        out = out[out["End"] > out["Start"]]
        if out.empty:
            return None
        return out.reset_index(drop=True)

    g_mag = _pad(_standardize(_read_df(mag_gaps_path)))
    g_par = _pad(_standardize(_read_df(par_gaps_path)))

    gaps_padded: Optional[pd.DataFrame] = None
    if g_mag is not None and g_par is not None:
        gaps_padded = pd.concat([g_mag, g_par], ignore_index=True)
    elif g_mag is not None:
        gaps_padded = g_mag
    elif g_par is not None:
        gaps_padded = g_par

    if gaps_padded is None or gaps_padded.empty:
        if index is None:
            return None, None
        idx = pd.DatetimeIndex(index)
        return None, np.ones(len(idx), dtype=bool)

    gaps_padded = gaps_padded.sort_values("Start").reset_index(drop=True)

    if index is None:
        return gaps_padded, None

    idx = pd.DatetimeIndex(index)
    in_gap = np.zeros(len(idx), dtype=bool)
    for row in gaps_padded.itertuples(index=False):
        in_gap |= (idx >= row.Start) & (idx <= row.End)

    keep = ~in_gap
    return gaps_padded, keep
def _resample(df: pd.DataFrame, cadence: str, *, agg: Mapping[str, str]) -> pd.DataFrame:
    if df.columns.has_duplicates:
        dups = sorted({c for c in df.columns[df.columns.duplicated()].tolist()})
        raise ValueError(f"Duplicate column labels in input: {dups}")
    d = df.select_dtypes(include=[np.number]).copy()
    if d.empty:
        raise ValueError("No numeric columns available to resample.")

    # Build a stable aggregation map
    agg_map: Dict[str, str] = {}
    for c in d.columns:
        agg_map[c] = agg.get(c, "mean")

    out = d.resample(cadence).agg(agg_map)
    out = _to_utc_index(out, "resampled")
    return out


def build_cadence_dataframe(
    *,
    fin: object,
    sig: Optional[pd.DataFrame],
    cadence: str,
    plot_vars: Sequence[str],
    br_col: Optional[str] = None,
    vr_col: Optional[str] = None,
    np_col: Optional[str] = None,
    join: str = "inner",
    input_units: Optional[Mapping[str, u.Unit]] = None,
) -> pd.DataFrame:
    """Build the canonical cadence DataFrame used for mapping and plotting.

    Resampling (explicit contract)
    ------------------------------
    The input pickles (final.pkl + optional sig_c_sig_r.pkl) are irregular and/or higher cadence.
    We resample onto a single pandas cadence grid using column-wise aggregations:

    - Magnetic field: Br -> mean within each bin.
    - Plasma: Vr and Np -> median within each bin (robust to outliers/spikes).
    - Other numeric columns:
        * if they originate in MAG/PAR: default mean unless overridden internally
        * if they originate in SIG (sigma_c, sigma_r, etc.): default mean.

    The resampled table is the *only* time base used downstream.
    Gap masking (mag_gaps.pkl/par_gaps.pkl) is applied later on this cadence grid
    so figures never interpolate across dropouts.

    """

    mag, par = _extract_mag_par(fin)

    br = _select_unique_col(mag, BR_CANDIDATES, "Br", br_col)
    vr = _select_unique_col(par, VR_CANDIDATES, "Vr", vr_col)
    np_ = _select_unique_col(par, NP_CANDIDATES, "Np", np_col)

    requested = [v for v in plot_vars if v not in ("polarity", "phi_src", "lat_src")]
    keep_mag = [br] + [c for c in requested if c in mag.columns and c != br]
    keep_par = [vr, np_] + [c for c in requested if c in par.columns and c not in (vr, np_)]

    # Resampling policy: background-flow quantities should be robust.
    agg_mag = {br: "mean"}
    agg_par = {vr: "median", np_: "median"}

    mag_c = _resample(mag[keep_mag], cadence, agg=agg_mag).rename(columns={br: "Br"})
    par_c = _resample(par[keep_par], cadence, agg=agg_par).rename(columns={vr: "Vr", np_: "Np"})

    if join not in ("inner", "outer"):
        raise ValueError("join must be 'inner' or 'outer'")

    data = mag_c.join(par_c, how=join)
    data = _to_utc_index(data, "cadence data")

    # Optionally join sigma diagnostics (dimensionless)
    if isinstance(sig, pd.DataFrame) and not sig.empty:
        sig_c = _resample(sig, cadence, agg={})
        data = data.join(sig_c, how=join)

    if len(data.columns) != len(set(data.columns)):
        dup = [c for c in data.columns if list(data.columns).count(c) > 1]
        raise ValueError(f"Duplicate columns after merge: {sorted(set(dup))}")

    # Attach the explicit unit contract
    iu = dict(default_input_units())
    if input_units is not None:
        iu.update({str(k): u.Unit(v) for k, v in dict(input_units).items()})

    attach_units(data, {"Br": iu["Br"], "Vr": iu["Vr"], "Np": iu["Np"]})

    # Dimensionless columns (if present)
    for c in ("sigma_c", "sigma_r"):
        if c in data.columns:
            attach_units(data, {c: u.one})

    return data


def _rolling_median(s: pd.Series, window: str) -> pd.Series:
    return s.rolling(window=window, min_periods=1, center=True).median()


def _rolling_mad(s: pd.Series, window: str) -> pd.Series:
    med = _rolling_median(s, window)
    mad = _rolling_median((s - med).abs(), window)
    return 1.4826 * mad


def _compute_background_speed(
    data: pd.DataFrame,
    *,
    vr_bg_window: str,
    vr_sigma_window: str,
    sigma_rel: float,
    sigma_abs: u.Quantity,
    v_min: u.Quantity,
    v_fallback: u.Quantity,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Compute Vr_bg and sigma_Vr on the canonical cadence index.

    Design goals (explicit)
    -----------------------
    - Vr_bg is a background speed by construction (rolling median).
    - Wave-scale contamination is suppressed by the windowing choice.
    - Invalid or non-physical Vr samples trigger a *transparent* fallback.
    - sigma_Vr combines MAD-scale variability with systematic components and is inflated
      on fallback samples (so downstream tau/phi uncertainties reflect degraded inputs).

    Returns
    -------
    Vr_bg : pd.Series
    sigma_Vr : pd.Series
    fallback_mask : np.ndarray[bool]
        True where raw Vr was invalid/non-physical and the fallback policy was used.
    """

    Vr_raw = pd.to_numeric(data["Vr"], errors="coerce")
    vmin_kms = float(u.Quantity(v_min).to_value(u.km / u.s))
    vfb_kms = float(u.Quantity(v_fallback).to_value(u.km / u.s))

    vr_arr = Vr_raw.to_numpy(dtype=float)
    fallback_mask = (~np.isfinite(vr_arr)) | (vr_arr <= 0.0)

    # Replace invalid samples before smoothing so Vr_bg remains defined.
    Vr_fill = Vr_raw.copy()
    Vr_fill[fallback_mask] = vfb_kms

    Vr_bg = _rolling_median(Vr_fill, vr_bg_window)

    # Enforce a physically meaningful floor.
    Vr_bg = Vr_bg.clip(lower=vmin_kms)

    # Variability estimate around the background.
    resid = Vr_fill - Vr_bg
    sig_mad = _rolling_mad(resid, vr_sigma_window)

    sys = float(max(0.0, sigma_rel)) * Vr_bg.abs()
    abs_term = float(u.Quantity(sigma_abs).to_value(u.km / u.s))

    sigma = np.sqrt(sig_mad.to_numpy(dtype=float) ** 2 + sys.to_numpy(dtype=float) ** 2 + abs_term ** 2)
    sigma = pd.Series(sigma, index=data.index)

    # Inflate uncertainty on fallback points (cannot trust instantaneous Vr).
    if np.any(fallback_mask):
        infl = 0.5 * Vr_bg.to_numpy(dtype=float) + abs_term
        sigma_vals = sigma.to_numpy(dtype=float)
        sigma_vals[fallback_mask] = np.maximum(sigma_vals[fallback_mask], infl[fallback_mask])
        sigma = pd.Series(sigma_vals, index=data.index)

    return Vr_bg, sigma, fallback_mask


def _derive_physical_diagnostics(data: pd.DataFrame, *, need: set[str]) -> None:
    """Add derived diagnostics requested by the plotting/output contract.

    Only computes what is explicitly needed. This keeps the pipeline auditable and avoids
    accidental hard dependencies on plasma columns that are irrelevant for a given run.
    """

    need = set(map(str, need))

    # Polarity (dimensionless): sign(Br). Treat exact zeros as missing.
    if "polarity" in need and "polarity" not in data.columns:
        br_val = pd.to_numeric(data["Br"], errors="coerce").to_numpy(dtype=float)
        pol = np.sign(br_val)
        pol[~np.isfinite(br_val)] = np.nan
        pol[pol == 0.0] = np.nan
        data["polarity"] = pol
        attach_units(data, {"polarity": u.one})

    # The remaining diagnostics require r_sc
    if "r_sc" not in data.columns:
        return

    r_sc = q_from_df(data, "r_sc").to(u.m)

    if ("Br_r2" in need) and ("Br_r2" not in data.columns):
        Br = q_from_df(data, "Br").to(u.T)
        Br_r2 = Br * r_sc ** 2  # [T m^2] == [Wb]
        data["Br_r2"] = to_value(Br_r2, u.Wb)
        attach_units(data, {"Br_r2": u.Wb})

    if ("P_ram" in need) and ("P_ram" not in data.columns):
        # Use the background speed when available (more representative of the large-scale flow
        # and consistent with the travel-time model inputs). Fall back to raw Vr otherwise.
        vcol = "Vr_bg" if "Vr_bg" in data.columns else "Vr"
        Vr = q_from_df(data, vcol).to(u.m / u.s)
        Np = q_from_df(data, "Np").to(u.m ** -3)
        P_ram = const.m_p * Np * Vr ** 2
        data["P_ram"] = to_value(P_ram, u.Pa)
        attach_units(data, {"P_ram": u.Pa})
# ---------------------------------------------------------------------
# Optional: same-source segmentation + hybrid-rs fitting
# ---------------------------------------------------------------------

def _estimate_dt_seconds(index: pd.Index) -> float:
    """Estimate the median sampling interval in seconds (NaN if unknown)."""
    try:
        if len(index) < 2:
            return float("nan")
        t = pd.to_datetime(index).asi8.astype("int64")
        dt = np.diff(t).astype("float64") * 1e-9
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        return float(np.nanmedian(dt)) if dt.size else float("nan")
    except Exception:
        return float("nan")


def _min_periods_for_window(index: pd.Index, window: str, *, frac: float = 0.5, floor: int = 3) -> int:
    """Choose a sensible ``min_periods`` for time-offset rolling windows."""
    try:
        wsec = float(pd.Timedelta(str(window)).total_seconds())
    except Exception:
        return int(max(1, floor))

    dt = _estimate_dt_seconds(index)
    if not np.isfinite(dt) or dt <= 0.0:
        return int(max(1, floor))

    n = int(max(1, round(wsec / dt)))
    return int(max(floor, np.ceil(frac * n)))


def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _normalize_feature_name(name: str) -> str:
    """Map common synonyms to canonical column names."""
    s = str(name).strip()
    aliases = {
        "Pram": "P_ram",
        "P_ram": "P_ram",
        "BrR2": "Br_r2",
        "Br_r2": "Br_r2",
        "|B|": "Bmag",
    }
    return aliases.get(s, s)


def _ensure_source_features(df: pd.DataFrame, features: Sequence[str]) -> Dict[str, str]:
    """Ensure requested source-segmentation features exist.

    Returns
    -------
    dict
        Mapping {requested_name -> executed_column_name} after canonicalization.

    Supported derived features (computed only if requested)
    ------------------------------------------------------
    - ``Br_r2``: ``Br * r_sc^2``  (open-flux proxy)  [Wb]
    - ``P_ram``: ``m_p * Np * Vr^2``  [Pa]
    - ``mass_flux``: ``m_p * Np * Vr``  [kg m^-2 s^-1]
      (uses ``Vr_bg`` if present, otherwise ``Vr``)
    - ``mag_mass_flux``: ``mass_flux / |Br|``  [kg m^-2 s^-1 T^-1]
      (guarded at small |Br|)
    - ``Bmag``: ``sqrt(Br^2 + Bt^2 + Bn^2)`` if components exist, else ``B`` if present.

    Notes
    -----
    We do *not* infer plasma beta; ``beta`` must be present upstream.
    """
    req_to_col: Dict[str, str] = {}
    feats = [_normalize_feature_name(f) for f in features]
    for raw, canon in zip(features, feats):
        req_to_col[str(raw)] = str(canon)

    need = set(feats)

    # Ensure derived physical diagnostics are computed even if they are not
    # in plot_vars.
    _derive_physical_diagnostics(df, need=need)

    # |B|
    if "Bmag" in need and "Bmag" not in df.columns:
        if all(c in df.columns for c in ("Br", "Bt", "Bn")):
            Br = pd.to_numeric(df["Br"], errors="coerce")
            Bt = pd.to_numeric(df["Bt"], errors="coerce")
            Bn = pd.to_numeric(df["Bn"], errors="coerce")
            df["Bmag"] = (Br * Br + Bt * Bt + Bn * Bn) ** 0.5
            attach_units(df, {"Bmag": q_from_df(df, "Br").unit})
        elif "B" in df.columns:
            df["Bmag"] = pd.to_numeric(df["B"], errors="coerce")
            attach_units(df, {"Bmag": q_from_df(df, "B").unit})

    # mass flux
    if "mass_flux" in need and "mass_flux" not in df.columns:
        if ("Np" in df.columns) and (("Vr_bg" in df.columns) or ("Vr" in df.columns)):
            vcol = "Vr_bg" if "Vr_bg" in df.columns else "Vr"
            Vr = q_from_df(df, vcol).to(u.m / u.s)
            Np = q_from_df(df, "Np").to(u.m ** -3)
            mf = const.m_p * Np * Vr
            df["mass_flux"] = to_value(mf, u.kg / (u.m ** 2 * u.s))
            attach_units(df, {"mass_flux": u.kg / (u.m ** 2 * u.s)})

    # magnetic mass flux proxy: (mass flux)/|Br|
    if "mag_mass_flux" in need and "mag_mass_flux" not in df.columns:
        if ("mass_flux" in df.columns) and ("Br" in df.columns):
            mf = q_from_df(df, "mass_flux").to(u.kg / (u.m ** 2 * u.s))
            Br = q_from_df(df, "Br").to(u.T)
            val = mf / np.maximum(np.abs(Br), 1e-30 * u.T)
            df["mag_mass_flux"] = to_value(val, u.kg / (u.m ** 2 * u.s * u.T))
            attach_units(df, {"mag_mass_flux": u.kg / (u.m ** 2 * u.s * u.T)})

    return req_to_col


def _compute_source_score(
    df: pd.DataFrame,
    *,
    vars: Sequence[str],
    window: str,
    mode: str = "variability",
    alpha: float = 0.5,
    weights: Optional[Dict[str, float]] = None,
    return_components: bool = False,
):
    """Deterministic, dimensionless source-change score.

    Parameters
    ----------
    mode:
        - "variability": robust within-window relative variability, MAD/|median|.
        - "diff": robust change-point sensitivity based on first differences.
        - "combo": variability + alpha * diff.

    Notes
    -----
    This is a segmentation aid, not a physical invariant. For coarse cadence,
    windows shorter than ~3 samples are ill-posed and are rejected.
    """
    req_to_col = _ensure_source_features(df, vars)

    if weights is None:
        weights = {str(v): 1.0 for v in vars}

    eps = 1e-12
    win = str(window)
    mp = _min_periods_for_window(df.index, win, frac=0.5, floor=3)

    # Guard: reject windows that cannot support the required min_periods.
    try:
        wsec = float(pd.Timedelta(win).total_seconds())
        dt = float(_estimate_dt_seconds(df.index))
        nwin = int(max(1, round(wsec / dt))) if (np.isfinite(dt) and dt > 0.0) else 0
    except Exception:
        nwin = 0
        dt = float("nan")
    if (nwin > 0) and (nwin < int(mp)):
        raise ValueError(
            f"source_segmentation_window={win!r} is too short for cadence~{dt:.3g} s "
            f"(needs >= {int(mp)} samples, got ~{nwin}). "
            "Increase the window (e.g. >= 3*cadence) or lower cadence."
        )

    mode_lc = str(mode).strip().lower()
    if mode_lc not in {"variability", "diff", "combo"}:
        raise ValueError(
            f"source_segmentation_mode must be one of 'variability','diff','combo', got {mode!r}"
        )

    score = pd.Series(0.0, index=df.index, dtype=float)
    comps: Dict[str, pd.Series] = {}
    used_any = False

    for v_raw in vars:
        v_raw = str(v_raw)
        v = req_to_col.get(v_raw, _normalize_feature_name(v_raw))
        if v not in df.columns:
            continue

        w = float(weights.get(v_raw, weights.get(v, 1.0)))
        x = pd.to_numeric(df[v], errors="coerce")

        # (A) Relative within-window variability
        med = x.rolling(win, center=True, min_periods=mp).median()
        mad = (x - med).abs().rolling(win, center=True, min_periods=mp).median()
        rel_var = mad / (med.abs() + eps)

        # (B) Change-point sensitivity from first differences
        dx = x.diff()
        med_dx = dx.rolling(win, center=True, min_periods=mp).median()
        mad_dx = (dx - med_dx).abs().rolling(win, center=True, min_periods=mp).median()
        rel_diff = (dx - med_dx).abs() / (mad_dx + eps)

        if mode_lc == "variability":
            rel = rel_var
        elif mode_lc == "diff":
            rel = rel_diff
        else:
            rel = rel_var + float(alpha) * rel_diff

        contrib = w * rel
        comps[v_raw] = contrib.astype(float)
        score = score + contrib.fillna(0.0)
        used_any = True

    if not used_any:
        out = pd.Series(np.nan, index=df.index, dtype=float)
        return (out, comps) if return_components else out

    return (score, comps) if return_components else score


def _segment_by_score(score: pd.Series, *, threshold: float, min_points: int) -> np.ndarray:
    """Assign integer segment labels using a threshold on score."""
    s = pd.to_numeric(score, errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(s)
    ok = good & (s <= float(threshold))

    seg = np.full(len(s), -1, dtype=int)
    k = 0
    i = 0
    while i < len(s):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < len(s) and ok[j]:
            j += 1
        if (j - i) >= int(min_points):
            seg[i:j] = k
            k += 1
        i = j
    return seg


def _fit_rs_min_circ_std(
    *,
    phi_sc_deg: np.ndarray,
    tau_s_by_rs: np.ndarray,
    omega_deg_per_s: float,
    phi_sign: int,
) -> int:
    """Return rs-index that minimizes circular std of mapped source longitudes."""
    phi_sc = np.asarray(phi_sc_deg, dtype=float)
    tau = np.asarray(tau_s_by_rs, dtype=float)  # (n_rs, n_t)
    phi_src = wrap_0_360(phi_sc[None, :] + float(phi_sign) * omega_deg_per_s * tau)
    std = circ_std_deg(phi_src, axis=1)
    return int(np.nanargmin(std))


def _to_jsonable(x: Any) -> Any:
    """Best-effort conversion to JSON-safe primitives for audit hashing."""
    if isinstance(x, u.Quantity):
        q = u.Quantity(x)
        if q.isscalar:
            return {"value": float(q.to_value(q.unit)), "unit": str(q.unit)}
        arr = np.asarray(q.to_value(q.unit), dtype=float)
        return {
            "unit": str(q.unit),
            "shape": list(arr.shape),
            "min": float(np.nanmin(arr)) if arr.size else float("nan"),
            "max": float(np.nanmax(arr)) if arr.size else float("nan"),
        }

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x[:16]] + (["..."] if len(x) > 16 else [])

    if isinstance(x, dict):
        out = {}
        for k in sorted(x.keys(), key=lambda z: str(z)):
            out[str(k)] = _to_jsonable(x[k])
        return out

    try:
        a = np.asarray(x)
        if a.ndim == 0:
            return float(a)
        if a.size <= 32:
            return a.astype(float).tolist()
        return {"shape": list(a.shape), "dtype": str(a.dtype), "min": float(np.nanmin(a.astype(float))), "max": float(np.nanmax(a.astype(float)))}
    except Exception:
        return repr(x)


def _model_signature(*, method_tag: str, model: Any, r_ss: u.Quantity, model_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Stable(ish) signature for traceability checks across compute/plot."""
    import hashlib

    payload = {
        "method_tag": str(method_tag),
        "model_class": type(model).__name__,
        "model_name": getattr(model, "name", None),
        "r_ss": _to_jsonable(r_ss),
        "model_kwargs": _to_jsonable(model_kwargs or {}),
        "model_state": _to_jsonable(getattr(model, "__dict__", {})),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _robust_speed_local(V: u.Quantity, *, v_min: u.Quantity, v_fallback: u.Quantity) -> tuple[u.Quantity, np.ndarray]:
    """Mirror travel_time._robust_speed without importing private helpers."""
    Vq = u.Quantity(V).to(u.Quantity(v_min).unit)
    vmin = u.Quantity(v_min).to(Vq.unit)
    vfb = u.Quantity(v_fallback).to(Vq.unit)

    arr = np.asarray(Vq.to_value(Vq.unit), dtype=float).reshape(-1)
    bad = (~np.isfinite(arr)) | (arr < float(vmin.to_value(Vq.unit)))

    Veff = Vq.reshape(-1).copy()
    if np.any(bad):
        Veff = Veff.to(vfb.unit)
        Veff[bad] = vfb
        Veff = Veff.to(vmin.unit)

    return Veff, bad


def _carrington_drift_stats(phi_deg: np.ndarray, idx: pd.DatetimeIndex) -> Dict[str, Any]:
    """Estimate d(phi)/dt for a Carrington longitude time series.

    We unwrap phi to avoid 0/360 discontinuities, then fit a line phi(t) = a + b t.
    Returned rate is in deg/day; wrap_time_days = 360/|rate| (inf if rate≈0).
    """
    out: Dict[str, Any] = {
        "n": 0,
        "rate_deg_per_day": float("nan"),
        "wrap_time_days": float("inf"),
        "delta_phi_deg": float("nan"),
        "n_rotations": float("nan"),
        "t0": None,
        "t1": None,
    }
    try:
        t = pd.to_datetime(pd.DatetimeIndex(idx), utc=True)
    except Exception:
        return out
    phi = np.asarray(phi_deg, dtype=float).reshape(-1)
    if len(phi) != len(t):
        return out
    m = np.isfinite(phi) & np.isfinite(t.view("int64"))
    if int(np.sum(m)) < 3:
        return out
    tt = t[m]
    ts = tt.view("int64").astype(float) / 1e9  # seconds
    phiu = np.rad2deg(np.unwrap(np.deg2rad(phi[m])))
    try:
        b, a = np.polyfit(ts, phiu, 1)
    except Exception:
        return out
    rate = float(b * 86400.0)  # deg/day
    delta = float(phiu[-1] - phiu[0])
    nrot = float(delta / 360.0)
    wrap_days = float("inf") if (not np.isfinite(rate) or abs(rate) < 1e-9) else float(360.0 / abs(rate))
    out.update(
        {
            "n": int(np.sum(m)),
            "rate_deg_per_day": rate,
            "wrap_time_days": wrap_days,
            "delta_phi_deg": delta,
            "n_rotations": nrot,
            "t0": str(tt.min()),
            "t1": str(tt.max()),
        }
    )
    return out


def _mapping_shift_sanity(
    *,
    phi_sc_deg: np.ndarray,
    phi_src_deg: np.ndarray,
    tau_s: np.ndarray,
    omega_deg_s: float,
    phi_sign: int,
    expected_shift_deg: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Sanity check: does (phi_src-phi_sc) match an expected shift (mod 360)?

    Default expected shift is phi_sign*Omega*tau. If expected_shift_deg is provided,
    it is used instead (e.g. Appendix-A delta_phi).
    """
    out: Dict[str, Any] = {
        "n": 0,
        "dphi_median_deg": float("nan"),
        "expected_median_deg": float("nan"),
        "err_median_deg": float("nan"),
        "err_p16_deg": float("nan"),
        "err_p84_deg": float("nan"),
        "sign_consistent": None,
    }
    phi_sc = np.asarray(phi_sc_deg, dtype=float).reshape(-1)
    phi_src = np.asarray(phi_src_deg, dtype=float).reshape(-1)
    tau = np.asarray(tau_s, dtype=float).reshape(-1)
    if not (len(phi_sc) == len(phi_src) == len(tau)):
        return out
    m = np.isfinite(phi_sc) & np.isfinite(phi_src) & np.isfinite(tau) & (tau > 0)
    if int(np.sum(m)) < 5:
        return out

    dphi = delta_deg(phi_src[m], phi_sc[m])  # (-180,180]
    if expected_shift_deg is None:
        expected = float(phi_sign) * float(omega_deg_s) * tau[m]  # degrees
    else:
        ex = np.asarray(expected_shift_deg, dtype=float).reshape(-1)
        if ex.shape[0] != tau.shape[0]:
            expected = float(phi_sign) * float(omega_deg_s) * tau[m]
        else:
            expected = ex[m]
    expected = ((expected + 180.0) % 360.0) - 180.0

    err = dphi - expected
    err = ((err + 180.0) % 360.0) - 180.0

    dmed = float(np.nanmedian(dphi))
    emed = float(np.nanmedian(expected))
    sign_ok = None
    try:
        if np.isfinite(dmed) and np.isfinite(emed) and (abs(emed) > 1.0):
            sign_ok = bool(np.sign(dmed) == np.sign(emed))
    except Exception:
        sign_ok = None

    out.update(
        {
            "n": int(np.sum(m)),
            "dphi_median_deg": dmed,
            "expected_median_deg": emed,
            "err_median_deg": float(np.nanmedian(err)),
            "err_p16_deg": float(np.nanpercentile(err, 16.0)),
            "err_p84_deg": float(np.nanpercentile(err, 84.0)),
            "sign_consistent": sign_ok,
        }
    )
    return out

def _format_audit_block(audit: Dict[str, Any]) -> str:
    """Human-readable audit block. Keep it ASCII + TeX-safe."""
    lines = []
    lines.append("[BACKMAP:AUDIT] requested_method={m} executed_model={em} class={cls}".format(
        m=audit.get("requested_method"),
        em=audit.get("executed_model"),
        cls=audit.get("executed_class"),
    ))
    lines.append("[BACKMAP:AUDIT] signature={sig}".format(sig=audit.get("model_signature")))
    lines.append("[BACKMAP:AUDIT] r_units={ru} phi_wrap={pw} circ_percentile={cid}".format(
        ru=audit.get("r_units"),
        pw=audit.get("phi_wrap"),
        cid=audit.get("circ_percentile_algo"),
    ))
    rs = audit["r_stats"]
    lines.append("[BACKMAP:AUDIT] r_ss={rss:.4f} R_sun  r_sc[min,med,max]={rmin:.4f},{rmed:.4f},{rmax:.4f}  min(r_sc-r_ss)={drmin:.4e} R_sun".format(
        rss=float(rs["r_ss_Rsun"]),
        rmin=float(rs["r_sc_min_Rsun"]),
        rmed=float(rs["r_sc_med_Rsun"]),
        rmax=float(rs["r_sc_max_Rsun"]),
        drmin=float(rs["dr_min_Rsun"]),
    ))
    pr = audit["profile"]
    lines.append("[BACKMAP:AUDIT] U_span={us:.6g} km/s  U_span_thr={thr:.6g} km/s  accel={acc}  degenerate={deg}  reason={rea}".format(
        us=float(pr["U_span_kms"]),
        thr=float(pr["U_span_thr_kms"]),
        acc=bool(pr["is_accelerating"]),
        deg=bool(pr["is_degenerate"]),
        rea=str(pr.get("degenerate_reason")),
    ))
    if ("U_ss_kms" in pr) and ("U_sc_kms" in pr):
        lines.append(
            "[BACKMAP:AUDIT] U_endpoints: U(r_ss)={uss:.3f} km/s  U(r_sc_med)={usc:.3f} km/s  U_ss/U_sc={rat:.3f}".format(
                uss=float(pr["U_ss_kms"]),
                usc=float(pr["U_sc_kms"]),
                rat=float(pr.get("U_ss_over_U_sc", float("nan"))),
            )
        )
    tvb = audit["tau_vs_ballistic"]
    lines.append("[BACKMAP:AUDIT] tau_vs_ballistic: median(eps)={med:.3e}  p16={p16:.3e}  p84={p84:.3e}  n={n}".format(
        med=float(tvb.get("eps_median", float("nan"))),
        p16=float(tvb.get("eps_p16", float("nan"))),
        p84=float(tvb.get("eps_p84", float("nan"))),
        n=int(tvb.get("n", 0)),
    ))
    if "parker_scaled" in str(audit.get("requested_method", "")):
        ps = audit.get("parker_scaled", {})
        lines.append("[BACKMAP:AUDIT] parker_scaled mismatch: fallback_frac={ff:.3f}  |Veff-Vbg|_median={mm:.3g} km/s".format(
            ff=float(ps.get("fallback_fraction", float("nan"))),
            mm=float(ps.get("mismatch_median_kms", float("nan"))),
        ))
    return "\n".join(lines)


def _summary_text(meta: Dict[str, Any]) -> str:
    """Return a compact diagnostics string that is TeX-safe.

    Users often have ``text.usetex=True`` globally. Literal ``%`` or unicode
    symbols can crash LaTeX rendering. We therefore place percent values in
    math-mode and use ``$R_\odot$`` for solar radii.
    """

    def _tex_escape(s: str) -> str:
        # Minimal escape set for common user-facing strings.
        return (
            s.replace("\\", r"\\")
             .replace("_", r"\_")
             .replace("%", r"\%")
             .replace("&", r"\&")
             .replace("#", r"\#")
        )

    lines = []
    if "model" in meta:
        lines.append(f"model: {_tex_escape(str(meta['model']))}")
    if "r_ss_Rsun" in meta:
        lines.append(f"r_ss: {float(meta['r_ss_Rsun']):.2f} $R_\\odot$")
    if "omega_deg_per_day" in meta:
        lines.append(f"$\\Omega$: {float(meta['omega_deg_per_day']):.4g} deg/day")
    if "ephem_step" in meta:
        lines.append(f"ephem: {_tex_escape(str(meta['ephem_step']))}")
    if "frame3d" in meta:
        lines.append(f"frame: {_tex_escape(str(meta['frame3d']))}")

    # IMPORTANT: do NOT use ``\\%`` here.
    # Under TeX rendering, ``\\`` is a newline command; followed by ``%`` it
    # becomes a comment delimiter and can break parsing.
    # Use the literal TeX percent escape ``\%`` (single backslash) and avoid
    # wrapping these values in math-mode.
    if "fallback_fraction" in meta:
        lines.append(f"fallback: {100*float(meta['fallback_fraction']):.1f}\\%")
    if "masked_fraction" in meta:
        lines.append(f"masked: {100*float(meta['masked_fraction']):.1f}\\%")
    if "tau_median_hr" in meta:
        lines.append(f"$\\tau_{{\\rm med}}$: {float(meta['tau_median_hr']):.2f} h")
    if "sigma_phi_p84_deg" in meta:
        lines.append(f"$\\sigma_\\phi(84)$: {float(meta['sigma_phi_p84_deg']):.2f} deg")
    if "r_sc_median_Rsun" in meta:
        rs = float(meta["r_sc_median_Rsun"])
        lines.append(f"$r_{{\\rm sc,med}}$: {rs:.2f} $R_\\odot$")

    # Profile sanity flags (non-ballistic models must not silently degenerate)
    if bool(meta.get("profile_fallback_used", False)):
        lines.append("profile: FALLBACK")
    if meta.get("profile_degenerate", False):
        lines.append("profile: DEGENERATE")

    # Final TeX-safety guard: convert accidental \\% to \% (newline + comment is fatal).
    for ii, ln in enumerate(lines):
        if "\\\\%" in ln:
            lines[ii] = ln.replace("\\\\%", "\\%")

    return "\n".join(lines)


def _compute_3d_cartesian(
    data: pd.DataFrame,
    *,
    r_ss: u.Quantity,
    frame3d: str,
    observer: str = "earth",
) -> None:
    """Compute AU Cartesian coordinates for spacecraft and source-surface points."""

    try:
        from astropy.time import Time
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import get_body_heliographic_stonyhurst
        from sunpy.coordinates.frames import (
            HeliographicCarrington,
            HeliocentricEarthEcliptic,
            HeliocentricInertial,
        )
        import astropy.units as uu
    except Exception as e:
        raise RuntimeError("3D outputs require sunpy+astropy.") from e

    frame3d = str(frame3d).upper().strip()
    if frame3d not in {"HEE", "HCI"}:
        raise ValueError("frame3d must be 'HEE' or 'HCI'")

    t_index = pd.DatetimeIndex(data.index)
    obstime = Time(t_index.to_pydatetime())

    if str(observer).lower().strip() != "earth":
        raise ValueError("Only observer='earth' is supported in this minimal implementation.")

    earth_obs = get_body_heliographic_stonyhurst("earth", obstime)
    hgc = HeliographicCarrington(obstime=obstime, observer=earth_obs)

    lon = (pd.to_numeric(data["phi_src"], errors="coerce").to_numpy(dtype=float) * uu.deg)
    lat = (pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float) * uu.deg)

    rss_au = r_ss.to_value(u.AU)

    fp_ss = SkyCoord(lon=lon, lat=lat, radius=(rss_au * uu.AU), frame=hgc)

    if frame3d == "HEE":
        tf = HeliocentricEarthEcliptic(obstime=obstime)
    else:
        tf = HeliocentricInertial(obstime=obstime)

    ss = fp_ss.transform_to(tf)
    data["ss_x_au"] = ss.cartesian.x.to_value(uu.AU)
    data["ss_y_au"] = ss.cartesian.y.to_value(uu.AU)
    data["ss_z_au"] = ss.cartesian.z.to_value(uu.AU)

    # Spacecraft Cartesian coordinates in target frame (needed for link lines)
    if {"hee_x_au", "hee_y_au", "hee_z_au"}.issubset(data.columns):
        xsc = pd.to_numeric(data["hee_x_au"], errors="coerce").to_numpy(dtype=float)
        ysc = pd.to_numeric(data["hee_y_au"], errors="coerce").to_numpy(dtype=float)
        zsc = pd.to_numeric(data["hee_z_au"], errors="coerce").to_numpy(dtype=float)

        if frame3d == "HEE":
            data["sc_x_au"], data["sc_y_au"], data["sc_z_au"] = xsc, ysc, zsc
        else:
            rep = SkyCoord(
                x=xsc * uu.AU,
                y=ysc * uu.AU,
                z=zsc * uu.AU,
                frame=HeliocentricEarthEcliptic(obstime=obstime),
                representation_type="cartesian",
            )
            rep2 = rep.transform_to(HeliocentricInertial(obstime=obstime))
            data["sc_x_au"] = rep2.cartesian.x.to_value(uu.AU)
            data["sc_y_au"] = rep2.cartesian.y.to_value(uu.AU)
            data["sc_z_au"] = rep2.cartesian.z.to_value(uu.AU)
    else:
        raise ValueError("3D plotting requires HEE spacecraft Cartesian columns hee_x_au/hee_y_au/hee_z_au (set include_hee=True in ephemeris).")

    # Optional: Cartesian endpoints for longitude-interval uncertainty (p16/p84) if present.
    # This keeps the uncertainty visualization frame-consistent.
    if {"phi_src_p16", "phi_src_p84", "lat_src"}.issubset(data.columns):
        lon_lo = (pd.to_numeric(data["phi_src_p16"], errors="coerce").to_numpy(dtype=float) * uu.deg)
        lon_hi = (pd.to_numeric(data["phi_src_p84"], errors="coerce").to_numpy(dtype=float) * uu.deg)
        lat_u = (pd.to_numeric(data["lat_src"], errors="coerce").to_numpy(dtype=float) * uu.deg)

        fp_lo = SkyCoord(lon=lon_lo, lat=lat_u, radius=(rss_au * uu.AU), frame=hgc)
        fp_hi = SkyCoord(lon=lon_hi, lat=lat_u, radius=(rss_au * uu.AU), frame=hgc)

        lo_tf = fp_lo.transform_to(tf)
        hi_tf = fp_hi.transform_to(tf)

        data["ss_p16_x_au"] = lo_tf.cartesian.x.to_value(uu.AU)
        data["ss_p16_y_au"] = lo_tf.cartesian.y.to_value(uu.AU)
        data["ss_p16_z_au"] = lo_tf.cartesian.z.to_value(uu.AU)

        data["ss_p84_x_au"] = hi_tf.cartesian.x.to_value(uu.AU)
        data["ss_p84_y_au"] = hi_tf.cartesian.y.to_value(uu.AU)
        data["ss_p84_z_au"] = hi_tf.cartesian.z.to_value(uu.AU)


def _compute_sc_cartesian_from_hee(
    track: pd.DataFrame,
    *,
    frame3d: str,
) -> pd.DataFrame:
    """Compute spacecraft Cartesian coordinates in the requested plotting frame.

    Parameters
    ----------
    track
        DataFrame indexed by time with HEE Cartesian columns: hee_x_au, hee_y_au, hee_z_au.
    frame3d
        "HEE" or "HCI".

    Returns
    -------
    DataFrame with added columns sc_x_au, sc_y_au, sc_z_au.
    """

    frame3d = str(frame3d).upper().strip()
    if frame3d not in {"HEE", "HCI"}:
        raise ValueError("frame3d must be 'HEE' or 'HCI'")

    if not {"hee_x_au", "hee_y_au", "hee_z_au"}.issubset(track.columns):
        raise ValueError("track must contain hee_x_au, hee_y_au, hee_z_au")

    out = track.copy()
    x = pd.to_numeric(out["hee_x_au"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(out["hee_y_au"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(out["hee_z_au"], errors="coerce").to_numpy(dtype=float)

    if frame3d == "HEE":
        out["sc_x_au"], out["sc_y_au"], out["sc_z_au"] = x, y, z
        return out

    try:
        from astropy.time import Time
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates.frames import HeliocentricEarthEcliptic, HeliocentricInertial
        import astropy.units as uu
    except Exception as e:
        raise RuntimeError("3D outputs require sunpy+astropy.") from e

    t_index = pd.DatetimeIndex(out.index)
    obstime = Time(t_index.to_pydatetime())

    rep = SkyCoord(
        x=x * uu.AU,
        y=y * uu.AU,
        z=z * uu.AU,
        frame=HeliocentricEarthEcliptic(obstime=obstime),
        representation_type="cartesian",
    )
    rep2 = rep.transform_to(HeliocentricInertial(obstime=obstime))
    out["sc_x_au"] = rep2.cartesian.x.to_value(uu.AU)
    out["sc_y_au"] = rep2.cartesian.y.to_value(uu.AU)
    out["sc_z_au"] = rep2.cartesian.z.to_value(uu.AU)
    return out








# -----------------------------------------------------------------------------
# Human-readable report (writes to outdir)
# -----------------------------------------------------------------------------

def _write_report_md(outdir: Path, *, meta: Dict[str, Any], files: Dict[str, Any]) -> Path:
    """Write a compact, user-facing report.

    This is intentionally *not* a raw dump of meta.json. The goal is to provide
    a quick audit trail and an actionable checklist.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "REPORT.md"

    diag = (meta.get("diagnostics", {}) or {})
    ep = meta.get("ephemeris", {}) or {}

    lines: List[str] = []
    lines.append(f"# Backmapping report: {meta.get('sc','?')} interval {meta.get('which_int','?')}")
    lines.append("")
    lines.append("## What was computed")
    lines.append(
        "This run maps each in-situ timestamp to a Carrington longitude/latitude on the **source surface** "
        "at radius $r_{SS}$. The mapping is ballistic in the sense of radial propagation above $r_{SS}$."
    )
    lines.append("")
    lines.append("Core relations (Carrington):")
    lines.append("- Travel time $\tau$: $\tau = \int_{r_{SS}}^{r_{sc}} \mathrm{d}r / U(r)$ (or $\tau = (r_{sc}-r_{SS})/U$ for the ballistic baseline).")
    lines.append("- Longitude shift: $\Delta\phi = \Omega\,\tau$.")
    lines.append("- Source-surface longitude: $\phi_{SS}=\mathrm{wrap}_{[0,360)}(\phi_{sc} + s\,\Delta\phi)$ where $s=\texttt{phi\_sign}$. Latitude is held fixed in this baseline.")
    lines.append("")

    lines.append("## Model choices")
    method = str(meta.get("method") or "?").strip().lower()
    lines.append(f"- method: `{meta.get('method')}`")
    if method in {"constant"}:
        lines.append("- speed profile across the interval: **single** (constant $U$ for all timestamps)")
    elif method in {"ballistic_bg"}:
        lines.append("- speed profile across the interval: **time-dependent scaling** ($U(r,t)=V_{bg}(t)$, constant in $r$)")
    elif method in {"hybrid_parker", "parker_scaled", "exp_accel"}:
        lines.append("- speed profile across the interval: **time-dependent scaling** (shape fixed; scaled per timestamp to satisfy $U(r_{sc}(t),t)=V_{bg}(t)$)")
    else:
        lines.append("- speed profile across the interval: **see method documentation** (this method is not in the production set)")
    lines.append(f"- $r_{{SS}}$: {meta.get('r_ss_Rsun')} $R_\odot$")
    lines.append(f"- $\Omega$: {meta.get('omega_deg_per_day')} deg/day")
    lines.append(f"- cadence: `{meta.get('cadence')}`")
    lines.append(f"- ephemeris: step=`{ep.get('step','?')}` observer=`{ep.get('observer','?')}`")
    lines.append("")
    # Plot variable contract (requested vs used)
    plotm = meta.get("plot", {}) or {}
    pv_req = plotm.get("plot_vars_requested", None)
    pv_used = plotm.get("plot_vars_used", None)
    pv_drop = plotm.get("plot_vars_dropped", None)
    if pv_req is not None:
        lines.append("## Plot variables")
        lines.append("Requested variables are filtered to what exists in the loaded pickles; missing variables are dropped (with a warning).")
        lines.append(f"- 2D requested: {pv_req}")
        lines.append(f"- 2D used: {pv_used}")
        if pv_drop:
            lines.append(f"- 2D dropped: {pv_drop}")
        v3_req = plotm.get("plot_3d_vars_requested", None)
        v3_used = plotm.get("plot_3d_vars_used", None)
        v3_drop = plotm.get("plot_3d_vars_dropped", None)
        if v3_req is not None:
            lines.append(f"- 3D requested: {v3_req}")
            lines.append(f"- 3D used: {v3_used}")
            if v3_drop:
                lines.append(f"- 3D dropped: {v3_drop}")
        lines.append("")

    # Gaps removal summary
    if bool(diag.get("gaps_removed", False)):
        lines.append("## Data gaps")
        lines.append("Samples inside padded MAG/PAR gaps were removed before mapping/plotting.")
        lines.append(f"- gap_pad_frac: {diag.get('gap_pad_frac', None)}")
        lines.append(f"- n_samples_before: {diag.get('n_samples_before_gaps', None)}")
        lines.append(f"- n_samples_after: {diag.get('n_samples_after_gaps', None)}")
        gf = diag.get('gaps_file', None)
        if gf:
            lines.append(f"- gaps table: `{gf}`")
        lines.append("")

    lines.append("## Uncertainty (what the error bars mean)")
    rs_family = bool(((meta.get("model_meta", {}) or {}).get("diagnostics", {}) or {}).get("rs_family_enabled", False))
    if rs_family:
        lines.append(
            "This run used a **hybrid-Parker profile family scan** over $r_s$ to produce circular-safe percentiles "
            "of $\phi_{SS}$. The plotted bars correspond to the 16th-84th percentile envelope across that family."
        )
    else:
        um = meta.get("uncertainty_model", {}) or {}
        lines.append(
            "This run propagated uncertainty from the background speed estimate $V_{bg}(t)$. "
            "We compute $\sigma_\tau$ from $\tau(V_{bg}\pm\sigma_{V_r})$, then propagate to longitude via "
            "$\sigma_\phi=\Omega\,\sigma_\tau$ and form a symmetric circular interval around $\phi_{SS}$."
        )
        lines.append(f"- $V_{{bg}}$ window: `{um.get('vr_bg_window')}`")
        lines.append(f"- $\sigma_{{V_r}}$ window: `{um.get('vr_sigma_window')}`")
        lines.append(f"- systematic: sigma_rel={um.get('sigma_rel')} and sigma_abs={um.get('sigma_abs_kms')} km/s")
    lines.append("")

    # --- segmentation section (optional) ---
    segdiag = diag.get("source_segmentation", None)
    if isinstance(segdiag, dict) and bool(segdiag.get("enabled", False)):
        lines.append("## Source segmentation (state-change detector)")
        lines.append("Segmentation labels *stable* time ranges in a multivariate physical feature space. It does **not** change $\tau$ unless you enable `source_fit`.")

        mode0 = str(segdiag.get('mode', '?'))
        win0 = segdiag.get('window', '?')
        thr0 = segdiag.get('threshold', float('nan'))
        mp0 = segdiag.get('min_points', '?')
        a0 = segdiag.get('alpha', float('nan'))
        if isinstance(a0, (int, float)) and np.isfinite(float(a0)):
            lines.append(f"- window: `{win0}`  mode: `{mode0}`  alpha: {float(a0):.3g}  threshold: {float(thr0):.4g}  min_points: {mp0}")
        else:
            lines.append(f"- window: `{win0}`  mode: `{mode0}`  threshold: {float(thr0):.4g}  min_points: {mp0}")

        mode = str(segdiag.get('mode', '')).strip().lower()
        if mode in {'mv_cpd', 'gmm_cpd'}:
            lines.append("Method (CPD):")
            lines.append("- Build a feature vector per time from the requested variables: robust baseline (rolling median) + constancy proxy (rolling MAD with a scale that avoids median~0 blow-ups).")
            lines.append("- Robust-standardize the feature matrix using median/MAD across the interval.")
            lines.append("- Define `source_score(t)` as the two-sided window mean shift: $\|\mu_R-\mu_L\|_2/\sqrt{p}$ in the standardized space.")
            lines.append("- Stable points satisfy: finite features and `source_score(t) \le` threshold (threshold is auto-set from median+MAD unless you supplied one).")
            gmm = segdiag.get('gmm', None)
            if isinstance(gmm, dict) and bool(gmm.get('enabled', False)):
                lines.append("- ML regularization (GMM): fit a GaussianMixture on the standardized feature vectors (BIC selects $k$) to stabilize regime labels; segments split when labels change inside stable spans.")

        used = segdiag.get("used_features", []) or []
        if used:
            lines.append("- features used: " + ", ".join([str(x) for x in used]))
        lines.append(f"- n_segments: {segdiag.get('n_segments', 0)}")

        segs = segdiag.get('segments', None)
        if isinstance(segs, list) and len(segs) > 0:
            lines.append("")
            lines.append("Segment summaries (medians within each stable segment):")
            lines.append("")
            cols = [
                'segment', 't_start', 't_end', 'duration_hr', 'n_points',
                'Vr_bg', 'Np', 'P_ram', 'Br_r2', 'mass_flux', 'sigma_c', 'beta', 'instability_median'
            ]
            present_cols = []
            for c in cols:
                if any((isinstance(r, dict) and (c in r)) for r in segs):
                    present_cols.append(c)
            if present_cols:
                lines.append("| " + " | ".join(present_cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(present_cols)) + " |")
                for r in segs[:12]:
                    row = []
                    for c in present_cols:
                        v = r.get(c)
                        if isinstance(v, float):
                            row.append(f"{v:.4g}")
                        else:
                            row.append(str(v))
                    lines.append("| " + " | ".join(row) + " |")
        lines.append("Diagnostic figures (if present):")
        for kk in ("segmentation_score", "segmentation_footpoints", "segmentation_schematic"):
            vv = files.get(kk)
            if vv:
                lines.append(f"- **{kk}**: `{vv}`")
        lines.append("")

    lines.append("## Quick sanity checks")
    lines.append(f"- masked fraction (NaNs etc.): {diag.get('masked_fraction', float('nan')):.3f}")
    lines.append(f"- Vr fallback fraction: {diag.get('fallback_fraction', float('nan')):.3f}")
    lines.append(f"- median $\tau$ [h]: {diag.get('tau_median_hr', float('nan')):.3g}")
    lines.append(f"- p84($\sigma_\phi$) [deg]: {diag.get('sigma_phi_p84_deg', float('nan')):.3g}")
    lines.append("")

    lines.append("## Outputs")
    for k in (
        "timeseries",
        "maps_2d",
        "maps_3d",
        "velocity_profile",
        "segmentation_score",
        "segmentation_footpoints",
        "segmentation_schematic",
        "gaps_padded",
        "meta",
    ):
        v = files.get(k)
        if v:
            lines.append(f"- **{k}**: `{v}`")
    lines.append("")

    lines.append("## Notes")
    lines.append("- This tool stops at the source surface. Any magnetic mapping below $r_{SS}$ (PFSS/MHD) is deliberately out of scope here.")
    lines.append("- If you enable `source_fit` for `hybrid_parker`, the code selects $r_s$ per stable segment by minimizing the **circular dispersion** of $\phi_{SS}$ computed from $(\phi_{sc},\tau(r_s))$.")
    lines.append("  - The objective is evaluated on a decimated subset of points (parameter `rs_fit_decimate`) for speed, but the fitted $r_s$ is then applied to **all points** in that segment when recomputing $\tau$.")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p
def backmap_interval(
    *,
    root_dir: Union[str, Path],
    sc: str,
    which_int: int,
    interval_dir: Optional[Union[str, Path]] = None,
    data_override: Optional[pd.DataFrame] = None,
    interval_label: Optional[str] = None,
    method: str = "ballistic_bg",
    allow_experimental_methods: bool = False,
    cadence: str = "60min",
    r_ss: u.Quantity = 2.5 * u.R_sun,
    omega: u.Quantity = 14.1844 * u.deg / u.day,
    phi_sign: int = +1,
    # Optional: Appendix-A azimuthal correction (v_phi)
    azimuthal_correction: bool = False,
    r_A: u.Quantity = 20 * u.R_sun,
    az_n_grid: int = 2048,
    az_ma2_tol: float = 1e-3,
    ephem_step: str = "1h",
    ephem_observer: str = "earth",
    plot_vars: Sequence[str] = ("polarity", "Vr_bg", "P_ram", "sigma_c"),
    var_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    plot_percentiles: Tuple[float, float] = (2.0, 98.0),
    plot_3d: bool = False,
    plot_3d_var: Optional[str] = None,
    plot_3d_vars: Optional[Sequence[str]] = None,
    plot_3d_ncols: int = 2,
    plot_3d_camera: str = "iso",
    # Decimation for interactive 3D HTML only (every Nth sample).
    # This does NOT change the underlying cadence-grid data saved to disk.
    plot_3d_decimate: int = 1,
    # Carrington diagnostics: quantify drift rates and check phi_src shift consistency.
    plot_carrington_diag: bool = True,
    frame3d: str = "HEE",
    show_uncertainty: bool = True,
    show: bool = False,
    vr_bg_window: str = "6h",
    vr_sigma_window: str = "6h",
    sigma_rel: float = 0.15,
    sigma_abs: u.Quantity = 30 * u.km / u.s,
    v_min: u.Quantity = 50 * u.km / u.s,
    v_fallback: u.Quantity = 400 * u.km / u.s,
    input_units: Optional[Mapping[str, u.Unit]] = None,

    # ------------------------------------------------------------------
    # Optional: remove padded MAG/PAR gaps from the cadence-grid series
    # (affects plots + mapping; prevents interpolation across dropouts)
    # ------------------------------------------------------------------
    remove_gaps: bool = True,
    gap_pad_frac: float = 0.5,
    mag_gaps_path: Optional[Union[str, Path]] = None,
    par_gaps_path: Optional[Union[str, Path]] = None,

    # Optional: provide an index for the spacecraft orbit trace.
    # If None, the orbit is drawn on the pre-gap cadence index.
    orbit_index: Optional[Union[pd.DatetimeIndex, Sequence[Any]]] = None,

    model_kwargs: Optional[Dict[str, Any]] = None,
    br_col: Optional[str] = None,
    vr_col: Optional[str] = None,
    np_col: Optional[str] = None,
    join: str = "inner",
    size_by: str = "P_ram",
    figsize_2d: Optional[Tuple[float, float]] = None,
    figsize_3d: Optional[Tuple[int, int]] = None,

    # ------------------------------------------------------------------
    # Optional: "same-source" segmentation + rs fitting for hybrid Parker
    # ------------------------------------------------------------------
    # source segmentation (optional): multi-diagnostic same-source candidate intervals
    # - source_segmentation only labels stable intervals (no change to tau)
    # - source_fit performs a hybrid-Parker rs calibration within those intervals
    source_segmentation: bool = False,
    source_segmentation_vars: Optional[Sequence[str]] = None,
    source_segmentation_window: Optional[str] = None,
    source_segmentation_weights: Optional[Dict[str, float]] = None,
    source_segmentation_mode: str = "variability",
    source_segmentation_alpha: float = 0.5,
    source_segmentation_ridge_alpha: float = 0.2,
    source_segmentation_threshold: Optional[float] = None,
    source_segmentation_transition_pad: int = 1,
    source_segmentation_min_points: Optional[int] = None,

    # same-source segmentation + rs fitting for hybrid Parker (backward-compatible API)
    source_fit: bool = False,
    source_fit_vars: Sequence[str] = ("Vr_bg", "P_ram", "mass_flux", "mag_mass_flux", "Br_r2", "sigma_c", "sigma_r", "Bmag", "Np", "beta"),
    source_fit_window: str = "2h",
    source_fit_weights: Optional[Dict[str, float]] = None,
    source_fit_threshold: float = 1.0,
    source_fit_min_points: int = 3,
    rs_fit_range_Rsun: Tuple[float, float] = (1.0, 10.0),
    rs_fit_n: int = 25,
    rs_fit_decimate: int = 3,
    rs_fit_allow_boundary: bool = False,

    verbose: bool = True,
    outdir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Backmap one MHDTurbPy interval to the source surface.

    Returns
    -------
    dict with keys: data, meta, files
    """

    root_dir = Path(root_dir)

    # ------------------------------------------------------------
    # Interval I/O (or externally supplied cadence-grid data)
    # ------------------------------------------------------------
    if data_override is None:
        if interval_dir is not None:
            inp = load_interval_inputs_from_dir(interval_dir=interval_dir)
        else:
            inp = load_interval_inputs(root_dir=root_dir, sc=sc, which_int=which_int)

        fin = inp["fin"]
        sig = inp.get("sig", None)

        # Interval folder used for default gap tables (mag_gaps.pkl/par_gaps.pkl)
        interval_folder = Path(inp["paths"]["final"]).parent
    else:
        inp = {"paths": {"final": None, "general": None, "sig": None}}
        fin = None
        sig = None

        # Best-effort folder for outputs/caches
        if outdir is not None:
            interval_folder = Path(outdir)
        elif interval_dir is not None:
            interval_folder = Path(interval_dir)
        else:
            interval_folder = root_dir

    method_tag = str(method).strip().lower()

    # ------------------------------------------------------------------
    # Method contract
    # ------------------------------------------------------------------
    # Default (minimal, publication-grade) methods:
    #   - ballistic_bg  : constant-in-r propagation with U=V_bg(t)
    #   - hybrid_parker : one-parameter accelerating profile (rs) scaled to V_bg(t)
    # Experimental methods are available only behind an explicit opt-in flag.

    aliases = {
        "ballistic": "ballistic_bg",
        "hybrid": "hybrid_parker",
        "const": "constant",
    }
    method_tag = aliases.get(method_tag, method_tag)

    _SUPPORTED_CORE = {"ballistic_bg", "hybrid_parker"}
    _SUPPORTED_EXPERIMENTAL = {"constant", "exp_accel", "parker_scaled"}
    _SUPPORTED_METHODS = set(_SUPPORTED_CORE) | (set(_SUPPORTED_EXPERIMENTAL) if bool(allow_experimental_methods) else set())

    if (method_tag in _SUPPORTED_EXPERIMENTAL) and (not bool(allow_experimental_methods)):
        raise ValueError(
            (
                "method={!r} is not part of the default, minimal backmapping API. "
                "Use method in {} for production runs, or set allow_experimental_methods=True to enable {}."
            ).format(method_tag, sorted(_SUPPORTED_CORE), sorted(_SUPPORTED_EXPERIMENTAL))
        )

    if method_tag not in _SUPPORTED_METHODS:
        raise ValueError(
            "Unsupported method={!r}. Supported methods are: {}".format(method_tag, sorted(_SUPPORTED_METHODS))
        )
    out_base = Path(outdir) if outdir is not None else (interval_folder / "back_mapping" / method_tag)
    out_base.mkdir(parents=True, exist_ok=True)

    # Build canonical cadence DataFrame
    if data_override is None:
        data = build_cadence_dataframe(
            fin=fin,
            sig=sig,
            cadence=cadence,
            plot_vars=plot_vars,
            br_col=br_col,
            vr_col=vr_col,
            np_col=np_col,
            join=join,
            input_units=input_units,
        )
    else:
        if not isinstance(data_override, pd.DataFrame):
            raise TypeError("data_override must be a pandas.DataFrame")
        data = data_override.copy()
        data = _to_utc_index(data, "data_override")

    # Orbit index used for drawing a continuous spacecraft trajectory in 3D.
    # IMPORTANT: this is intentionally set *before* any gap masking, so the orbit
    # line remains present even when data samples are removed.
    if orbit_index is None:
        orbit_index_use = pd.DatetimeIndex(data.index)
    else:
        try:
            orbit_index_use = pd.to_datetime(pd.DatetimeIndex(orbit_index), utc=True)
            orbit_index_use = pd.DatetimeIndex(orbit_index_use).sort_values().unique()
        except Exception as e:
            raise TypeError("orbit_index must be coercible to a pandas.DatetimeIndex") from e
        # Ensure the orbit index covers all cadence-grid timestamps used for mapping.
        orbit_index_use = orbit_index_use.union(pd.DatetimeIndex(data.index)).sort_values().unique()

    try:
        plot_3d_decimate = max(1, int(plot_3d_decimate))
    except Exception:
        plot_3d_decimate = 1


    # ------------------------------------------------------------------
    # Optional: remove samples that fall inside known MAG/PAR gaps
    # (press-release figures should not interpolate across data dropouts)
    # ------------------------------------------------------------------
    gaps_padded = None
    gap_keep = None
    gaps_csv = None
    if bool(remove_gaps):
        mgp = Path(mag_gaps_path) if mag_gaps_path is not None else (interval_folder / "mag_gaps.pkl")
        pgp = Path(par_gaps_path) if par_gaps_path is not None else (interval_folder / "par_gaps.pkl")
        mgp_use = mgp if mgp.exists() else None
        pgp_use = pgp if pgp.exists() else None
        if mgp_use is not None or pgp_use is not None:
            gaps_padded, gap_keep = load_padded_gaps(
                mag_gaps_path=mgp_use,
                par_gaps_path=pgp_use,
                gap_pad_frac=float(gap_pad_frac),
                index=data.index,
            )
            if gap_keep is not None:
                n_before = int(len(data))
                data = data.loc[gap_keep].copy()
                n_after = int(len(data))
                if n_after < 2:
                    raise ValueError(
                        f"After gap removal (pad_frac={gap_pad_frac}), too few samples remain: {n_after}/{n_before}. "
                        "Reduce gap_pad_frac or disable remove_gaps."
                    )
                if isinstance(gaps_padded, pd.DataFrame) and (not gaps_padded.empty):
                    gaps_csv = out_base / "gaps_padded.csv"
                    try:
                        gaps_padded.to_csv(gaps_csv, index=False)
                    except Exception:
                        gaps_csv = None


    # Ephemeris (Carrington; observer explicit)
    # We evaluate/interpolate ephemeris on `orbit_index_use` (pre-gap cadence index by default)
    # so the *spacecraft orbit line* can remain visible even if `data` has been gap-masked.
    cache_file = out_base / "ephemeris_cache.pkl"
    eph = get_ephemeris_hgc(target=sc, times=orbit_index_use, step=ephem_step, observer=ephem_observer, cache_file=cache_file, include_hee=True)
    eph_i_orbit = interp_ephemeris_to_index(eph.df, orbit_index_use, circular_cols=("phi_sc_deg",))
    eph_i = eph_i_orbit.reindex(data.index)

    data["phi_sc"] = eph_i["phi_sc_deg"].to_numpy(dtype=float)
    data["lat_sc"] = eph_i["lat_sc_deg"].to_numpy(dtype=float)
    data["r_sc"] = eph_i["r_sc_au"].to_numpy(dtype=float)
    attach_units(data, {"phi_sc": u.deg, "lat_sc": u.deg, "r_sc": u.AU})

    for c in ("hee_x_au", "hee_y_au", "hee_z_au"):
        if c in eph_i.columns:
            data[c] = eph_i[c].to_numpy(dtype=float)

    # Optional: spacecraft orbit track (for continuous trajectory rendering in 3D)
    sc_track_df = None
    if bool(plot_3d):
        try:
            if {"hee_x_au", "hee_y_au", "hee_z_au"}.issubset(eph_i_orbit.columns):
                trk = eph_i_orbit[["hee_x_au", "hee_y_au", "hee_z_au"]].copy()
                trk = _compute_sc_cartesian_from_hee(trk, frame3d=frame3d)
                sc_track_df = trk[["sc_x_au", "sc_y_au", "sc_z_au"]].copy()
        except Exception:
            sc_track_df = None

    # Background speed + uncertainty model
    Vr_bg, sigma_Vr, fb_vr = _compute_background_speed(
        data,
        vr_bg_window=vr_bg_window,
        vr_sigma_window=vr_sigma_window,
        sigma_rel=sigma_rel,
        sigma_abs=sigma_abs,
        v_min=v_min,
        v_fallback=v_fallback,
    )

    data["Vr_bg"] = Vr_bg.to_numpy(dtype=float)
    attach_units(data, {"Vr_bg": q_from_df(data, "Vr").unit})

    data["sigma_Vr"] = sigma_Vr.to_numpy(dtype=float)
    attach_units(data, {"sigma_Vr": q_from_df(data, "Vr").unit})

    # Travel time model
    model: TravelTimeModel
    if str(method_tag) in {"constant", "const"}:
        V0 = u.Quantity(model_kwargs.get("V0")) if model_kwargs else np.nan * (u.km / u.s)
        if not np.isfinite(V0.to_value(u.km / u.s)):
            V0 = np.nanmedian(data["Vr_bg"].to_numpy(dtype=float)) * (u.km / u.s)
        model = build_model("constant", model_kwargs={"V0": V0})
    else:
        model = build_model(method_tag, model_kwargs=model_kwargs)

    model_sig = _model_signature(method_tag=method_tag, model=model, r_ss=u.Quantity(r_ss), model_kwargs=model_kwargs)
    data.attrs["executed_model_signature"] = str(model_sig)
    data.attrs["executed_method"] = str(method_tag)

    r_sc_q = q_from_df(data, "r_sc")
    V_bg_q = q_from_df(data, "Vr_bg")


    # ------------------------------------------------------------------
    # Run-time audit printout (no silent behavior)
    # ------------------------------------------------------------------
    if verbose:
        fb_frac = float(np.mean(fb_vr)) if fb_vr is not None else 0.0
        print("\n[BACKMAP] interval={sc} which_int={which_int} cadence={cadence}".format(
            sc=str(sc), which_int=int(which_int), cadence=str(cadence)
        ))
        print("[BACKMAP] method={m}  (class={cls})".format(m=str(method_tag), cls=type(model).__name__))
        print("[BACKMAP] ephem: step={s} observer={o}  (r_sc must be heliocentric)".format(
            s=str(ephem_step), o=str(ephem_observer)
        ))
        print("[BACKMAP] mapping: r_ss={rss:.3f} R_sun  omega={om:.4f} deg/day  phi_sign={ps:+d}".format(
            rss=float(u.Quantity(r_ss).to_value(u.R_sun)),
            om=float(u.Quantity(omega).to_value(u.deg/u.day)),
            ps=int(phi_sign),
        ))

        # Background speed estimator (what is fitted / how)
        print("[BACKMAP] V_bg(t): rolling-median(Vr_fill) with window={w}".format(w=str(vr_bg_window)))
        print("[BACKMAP] Vr_fill: invalid/non-physical Vr replaced with v_fallback={vfb:.1f} km/s".format(
            vfb=float(u.Quantity(v_fallback).to_value(u.km/u.s))
        ))
        print("[BACKMAP] floor: V_bg >= v_min={vmin:.1f} km/s".format(
            vmin=float(u.Quantity(v_min).to_value(u.km/u.s))
        ))
        print("[BACKMAP] sigma_Vr(t): MAD(residuals) window={ws} + sigma_rel*|V_bg| + sigma_abs (quadrature)".format(
            ws=str(vr_sigma_window)
        ))
        print("[BACKMAP] sigma_rel={sr:.3f}  sigma_abs={sa:.1f} km/s  fallback_fraction={ff:.3f}".format(
            sr=float(sigma_rel),
            sa=float(u.Quantity(sigma_abs).to_value(u.km/u.s)),
            ff=fb_frac,
        ))

        # Travel-time profile used (physics statement)
        mk = dict(model_kwargs) if model_kwargs else {}
        if str(method_tag) in {"parker_scaled", "parker_lambert"}:
            print("[BACKMAP] U(r): isothermal Parker (Lambert-W) shape scaled so U(r_sc,t)=V_bg(t).")
        elif str(method_tag) in {"hybrid_parker", "hybrid"}:
            rs_min = mk.get("rs_min", None)
            rs_max = mk.get("rs_max", None)
            n_rs = mk.get("n_rs", None)
            switch = mk.get("switch", "rs")
            mode = mk.get("rs_sample_mode", "tau_uniform")
            print("[BACKMAP] U(r): hybrid Parker approximations; scaled so U(r_sc,t)=V_bg(t).")
            print("[BACKMAP] hybrid family: switch={sw}  rs_sample_mode={md}  rs_min={rmin}  rs_max={rmax}  n_rs={nrs}".format(
                sw=str(switch), md=str(mode), rmin=str(rs_min), rmax=str(rs_max), nrs=str(n_rs)
            ))
        elif str(method_tag) in {"exp_accel"}:
            print("[BACKMAP] U(r): exponential/parametric acceleration shape scaled so U(r_sc,t)=V_bg(t).")
        elif str(method_tag) in {"ballistic_bg"}:
            print("[BACKMAP] U(r): constant in r (ballistic) with U=V_bg(t).")
        elif str(method_tag) in {"constant", "const"}:
            print("[BACKMAP] U(r): constant in r with U=V0 (constant-speed model).")
        else:
            print("[BACKMAP] U(r): (unrecognized method tag) {m}".format(m=str(method_tag)))
    # Base evaluation (deterministic)
    tt = model.evaluate(r_sc=r_sc_q, V_bg=V_bg_q, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback)
    tau = tt.tau

    # Executed-path truth: machine-checkable dispatch record.
    tt.meta.setdefault("executed", {})
    tt.meta["executed"].update(
        {
            "requested_method": str(method_tag),
            "executed_model": str(getattr(model, "name", None)),
            "executed_class": str(type(model).__name__),
            "evaluate_fn": str(getattr(getattr(model, "evaluate", None), "__qualname__", None)),
            "profile_fn": str(getattr(getattr(model, "speed_profile", None), "__qualname__", None)),
            "model_signature": str(model_sig),
        }
    )

    # --------------------------------------------------------------
    # Optional: "same-source" segmentation + rs fitting (hybrid Parker)
    #
    # Physics intent:
    # If a contiguous subset of the time series plausibly originates from a
    # single coronal source region, its source-surface longitude should be
    # relatively tight after ballistic mapping. We exploit this by choosing
    # the hybrid-Parker profile parameter rs that *minimizes the circular
    # dispersion* of phi_src within each stable segment.
    #
    # IMPORTANT:
    # - This is an *internal consistency* fit, not a unique inversion.
    # - It is only enabled when you explicitly set source_fit=True and the
    #   method is hybrid_parker/hybrid.
    # --------------------------------------------------------------
    rs_fit_used = None
    source_score = None
    source_segment = None
    source_score_components = None
    seg_window_used = None
    seg_mode_used = None
    seg_alpha_used = None
    seg_threshold_used = None
    seg_model_meta = None

    segmentation_enabled = bool(locals().get('source_segmentation', False)) or bool(source_fit)
    if segmentation_enabled:
        # Parameter aliasing for backward compatibility
        seg_vars_user = locals().get('source_segmentation_vars')
        seg_vars = None if seg_vars_user is None else tuple(seg_vars_user or source_fit_vars)
        seg_window = str(locals().get('source_segmentation_window') or source_fit_window)
        seg_weights = locals().get('source_segmentation_weights') if locals().get('source_segmentation_weights') is not None else source_fit_weights
        seg_mode = str(locals().get('source_segmentation_mode') or 'variability')
        seg_alpha = float(locals().get('source_segmentation_alpha') if locals().get('source_segmentation_alpha') is not None else 0.5)
        seg_ridge_alpha = float(locals().get('source_segmentation_ridge_alpha') if locals().get('source_segmentation_ridge_alpha') is not None else 0.2)
        seg_transition_pad = int(locals().get('source_segmentation_transition_pad') if locals().get('source_segmentation_transition_pad') is not None else 1)
        seg_threshold_user = locals().get('source_segmentation_threshold')
        seg_threshold = float(seg_threshold_user) if seg_threshold_user is not None else float(source_fit_threshold)
        seg_min_points = int(locals().get('source_segmentation_min_points') if locals().get('source_segmentation_min_points') is not None else source_fit_min_points)

        seg_window_used = seg_window
        seg_mode_used = seg_mode
        seg_alpha_used = seg_alpha
        seg_threshold_used = seg_threshold

        if verbose:
            print('[BACKMAP] source_segmentation: ENABLED')

        # Ensure derived features exist.
        # If the user did not specify vars, allow the segmentation model to auto-select
        # a balanced physical diagnostic set from what exists in `data` (including derived).
        if seg_vars is None:
            # Populate a conservative superset of derived diagnostics that are often useful.
            _ensure_source_features(data, source_fit_vars)
            req_to_col = {str(v): _normalize_feature_name(str(v)) for v in source_fit_vars}
            seg_cols_present = None
        else:
            req_to_col = _ensure_source_features(data, seg_vars)
            seg_cols = [req_to_col.get(str(v), _normalize_feature_name(str(v))) for v in seg_vars]
            seg_cols_present = []
            for c in seg_cols:
                if (c in data.columns) and (c not in seg_cols_present):
                    seg_cols_present.append(c)

        # Map weights (if provided) from requested names to canonical columns.
        seg_w_cols = {}
        if (seg_cols_present is not None) and isinstance(seg_weights, dict):
            for v_req, v_col in zip([str(v) for v in seg_vars], seg_cols):
                if v_col not in seg_cols_present:
                    continue
                if v_req in seg_weights:
                    seg_w_cols[v_col] = float(seg_weights[v_req])
                elif v_col in seg_weights:
                    seg_w_cols[v_col] = float(seg_weights[v_col])

        # New, physics-motivated segmentation modes (vectorized + optional ML)
        seg_mode_lc = str(seg_mode).strip().lower()
        if seg_mode_lc in {'mv_cpd', 'gmm_cpd'}:
            # Use automatic threshold unless the user explicitly provided one.
            thr = float(seg_threshold) if (seg_threshold_user is not None) else None
            res = segment_sources(
                data,
                vars=seg_cols_present,
                window=seg_window,
                weights=seg_w_cols if seg_w_cols else None,
                mode=seg_mode_lc,
                threshold=thr,
                min_points=int(seg_min_points),
                transition_pad=int(seg_transition_pad),
                ridge_alpha=float(seg_ridge_alpha),
            )
            source_score = res.score
            source_segment = res.segment
            source_score_components = res.score_components
            seg_model_meta = res.meta
            # Update used values from the actual run
            seg_threshold_used = float(res.meta.get('threshold', float('nan')))
            seg_alpha_used = float('nan')
            seg_mode_used = str(res.meta.get('mode', seg_mode_lc))
        else:
            # Legacy score (kept for backward compatibility)
            source_score, source_score_components = _compute_source_score(
                data,
                vars=seg_vars,
                window=seg_window,
                mode=seg_mode,
                alpha=seg_alpha,
                weights=seg_weights,
                return_components=True,
            )
            source_segment = _segment_by_score(source_score, threshold=seg_threshold, min_points=seg_min_points)
            seg_threshold_used = float(seg_threshold)
            seg_alpha_used = float(seg_alpha)
            seg_mode_used = str(seg_mode)
            seg_model_meta = None

        data['source_score'] = pd.to_numeric(source_score, errors='coerce').to_numpy(dtype=float)
        data['source_segment'] = np.asarray(source_segment, dtype=int)

        # Diagnostics
        present = []
        missing = []
        if seg_vars is not None:
            present = [str(v) for v in seg_vars if req_to_col.get(str(v), _normalize_feature_name(str(v))) in data.columns]
            missing = [str(v) for v in seg_vars if req_to_col.get(str(v), _normalize_feature_name(str(v))) not in data.columns]

        seg_lengths = []
        for sid in sorted(set(int(x) for x in np.unique(source_segment) if x >= 0)):
            seg_lengths.append(int(np.sum(source_segment == sid)))

        # Segment summaries (compact, physically interpretable)
        seg_summaries = []
        key_cols = [
            'Vr_bg', 'Vr', 'Np', 'P_ram', 'mass_flux', 'mag_mass_flux', 'Br_r2', 'Bmag', 'sigma_c', 'sigma_r', 'beta'
        ]
        key_cols = [c for c in key_cols if c in data.columns]
        for sid in sorted(set(int(x) for x in np.unique(source_segment) if x >= 0)):
            m = (source_segment == sid)
            if not np.any(m):
                continue
            t_idx = data.index[m]
            t0 = pd.to_datetime(t_idx[0])
            t1 = pd.to_datetime(t_idx[-1])
            dur_hr = float((t1 - t0).total_seconds() / 3600.0)
            row = {
                'segment': int(sid),
                't_start': str(t0),
                't_end': str(t1),
                'duration_hr': dur_hr,
                'n_points': int(np.sum(m)),
            }
            for c in key_cols:
                try:
                    row[c] = float(np.nanmedian(pd.to_numeric(data.loc[m, c], errors='coerce').to_numpy(dtype=float)))
                except Exception:
                    row[c] = float('nan')
            # constancy proxy: median of per-variable instability (if available)
            try:
                inst = []
                for kk, ss in (source_score_components or {}).items():
                    if isinstance(ss, pd.Series) and (len(ss) == len(data)):
                        inst.append(np.nanmedian(pd.to_numeric(ss.loc[m], errors='coerce').to_numpy(dtype=float)))
                row['instability_median'] = float(np.nanmedian(np.asarray(inst, dtype=float))) if len(inst) else float('nan')
            except Exception:
                row['instability_median'] = float('nan')
            seg_summaries.append(row)

        feat_mean = {}
        for k, s in (source_score_components or {}).items():
            try:
                feat_mean[str(k)] = float(np.nanmean(pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)))
            except Exception:
                feat_mean[str(k)] = float('nan')

        tt.meta.setdefault("diagnostics", {})
        tt.meta["diagnostics"].setdefault("source_segmentation", {})
        requested_features = ["AUTO"] if (seg_vars is None) else [str(v) for v in seg_vars]
        # Prefer the authoritative list of actually-used diagnostics from the segmentation model.
        used_features = None
        if isinstance(seg_model_meta, dict):
            uf = seg_model_meta.get("used_features", None)
            if isinstance(uf, (list, tuple)):
                used_features = [str(x) for x in uf]
        if used_features is None:
            used_features = present
        missing_features = missing if (seg_vars is not None) else []
        ridge_alpha_used = float(seg_ridge_alpha)
        ridge_lambda_used = float("nan")
        if isinstance(seg_model_meta, dict):
            try:
                ridge_alpha_used = float(seg_model_meta.get("ridge_alpha", ridge_alpha_used))
            except Exception:
                pass
            try:
                ridge_lambda_used = float(seg_model_meta.get("ridge_lambda", ridge_lambda_used))
            except Exception:
                pass
        tt.meta["diagnostics"]["source_segmentation"].update(
            {
                'enabled': True,
                'window': seg_window,
                'mode': str(seg_mode_used),
                'alpha': float(seg_alpha_used),
                'ridge_alpha': float(ridge_alpha_used),
                'ridge_lambda': float(ridge_lambda_used),
                'threshold': float(seg_threshold_used),
                'min_points': int(seg_min_points),
                'requested_features': requested_features,
                'used_features': used_features,
                'missing_features': missing_features,
                'n_segments': int(len([x for x in np.unique(source_segment) if int(x) >= 0])),
                'segment_lengths': seg_lengths,
                'segments': seg_summaries,
                'gmm': (seg_model_meta or {}).get('gmm', None) if isinstance(seg_model_meta, dict) else None,
                'score_stats': {
                    'p16': float(np.nanpercentile(data['source_score'], 16)) if np.isfinite(data['source_score']).any() else float('nan'),
                    'p50': float(np.nanpercentile(data['source_score'], 50)) if np.isfinite(data['source_score']).any() else float('nan'),
                    'p84': float(np.nanpercentile(data['source_score'], 84)) if np.isfinite(data['source_score']).any() else float('nan'),
                    'max': float(np.nanmax(data['source_score'])) if np.isfinite(data['source_score']).any() else float('nan'),
                    'frac_above_threshold': float(np.nanmean(data['source_score'] > float(seg_threshold_used)))
                    if np.isfinite(data['source_score']).any()
                    else float('nan'),
                },
                'feature_mean_contrib': feat_mean,
            }
        )

        if verbose:
            msg_used = used_features if isinstance(used_features, list) else present
            msg = '[BACKMAP] source_segmentation: used=' + ','.join([str(x) for x in msg_used])
            if missing_features:
                msg += ' | missing=' + ','.join([str(x) for x in missing_features])
            msg += f' | mode={seg_mode}'
            msg += f' | n_segments={tt.meta["diagnostics"]["source_segmentation"]["n_segments"]}'
            print(msg)

    # --------------------------------------------------------------
    # Optional: rs fitting for hybrid Parker (internal-consistency calibration)
    # --------------------------------------------------------------
    if bool(source_fit):
        if bool(azimuthal_correction):
            raise ValueError('azimuthal_correction=True is not supported with source_fit (objective would require Appendix-A delta_phi). Disable source_fit or disable azimuthal_correction.')
        if source_segment is None or source_score is None:
            raise RuntimeError('source_fit=True requires source_segmentation to be enabled (or source_fit itself).')

        if str(method_tag).lower().strip() not in {'hybrid_parker', 'hybrid'}:
            raise ValueError('source_fit=True currently supports only method="hybrid_parker" (internal-consistency rs fit).')

        if verbose:
            print('[BACKMAP] source_fit: ENABLED (hybrid Parker rs fit by minimizing circular dispersion of phi_src).')

        # Candidate rs grid
        rs_lo, rs_hi = float(rs_fit_range_Rsun[0]), float(rs_fit_range_Rsun[1])
        rs_grid = (np.linspace(rs_lo, rs_hi, int(rs_fit_n)) * u.R_sun).to(u.R_sun)

        # Fit rs per segment
        rs_fit_used = np.full(len(data), np.nan, dtype=float)
        omega_deg_per_s = float(u.Quantity(omega).to_value(u.deg / u.s))
        phi_sc_deg = data['phi_sc'].to_numpy(dtype=float)

        for seg_id in sorted(set(int(x) for x in np.unique(source_segment) if x >= 0)):
            idxs = np.where(source_segment == seg_id)[0]
            if idxs.size < int(source_fit_min_points):
                continue

            idx_fit = idxs[:: max(1, int(rs_fit_decimate))]

            ok = np.isfinite(r_sc_q.to_value(u.AU))[idx_fit] & np.isfinite(V_bg_q.to_value(u.km / u.s))[idx_fit]
            idx_fit = idx_fit[ok]
            if idx_fit.size < max(3, int(0.5 * max(3, source_fit_min_points))):
                continue

            tau_by_rs = np.full((len(rs_grid), len(idx_fit)), np.nan, dtype=float)
            for j, rs_j in enumerate(rs_grid):
                mrs = HybridParker(rs=rs_j, n_grid=getattr(model, 'n_grid', 8192), switch=getattr(model, 'switch', 'rs'))
                ttj = mrs.evaluate(
                    r_sc=r_sc_q[idx_fit],
                    V_bg=V_bg_q[idx_fit],
                    r_ss=u.Quantity(r_ss),
                    v_min=v_min,
                    v_fallback=v_fallback,
                )
                tau_by_rs[j, :] = ttj.tau.to_value(u.s)

            j_best = _fit_rs_min_circ_std(
                phi_sc_deg=phi_sc_deg[idx_fit],
                tau_s_by_rs=tau_by_rs,
                omega_deg_per_s=omega_deg_per_s,
                phi_sign=int(phi_sign),
            )

            rs_best = float(rs_grid[j_best].to_value(u.R_sun))
            hit_boundary = (int(j_best) == 0) or (int(j_best) == (len(rs_grid) - 1))
            if hit_boundary and (not bool(rs_fit_allow_boundary)):
                # Boundary solutions indicate that the objective is flat/ill-posed over the
                # admissible grid. Do not apply such a "fit" unless the user explicitly
                # allows it.
                # Store the objective curve for traceability.
                try:
                    phi_src_all = wrap_0_360(
                        phi_sc_deg[idx_fit][None, :]
                        + int(phi_sign) * omega_deg_per_s * tau_by_rs
                    )
                    std_all = np.asarray(circ_std_deg(phi_src_all, axis=1), dtype=float)
                    curve = {
                        'rs_Rsun': [float(v) for v in rs_grid.to_value(u.R_sun).tolist()],
                        'circ_std_deg': [float(v) for v in std_all.tolist()],
                    }
                except Exception:
                    curve = None

                tt.meta.setdefault("diagnostics", {})
                tt.meta["diagnostics"].setdefault("source_fit", {})
                tt.meta["diagnostics"]["source_fit"].setdefault("boundary_segments", [])
                tt.meta["diagnostics"]["source_fit"].setdefault("boundary_rs_Rsun", {})
                tt.meta["diagnostics"]["source_fit"].setdefault("boundary_objective", {})
                tt.meta["diagnostics"]["source_fit"]["boundary_segments"].append(int(seg_id))
                tt.meta["diagnostics"]["source_fit"]["boundary_rs_Rsun"][int(seg_id)] = float(rs_best)
                if curve is not None:
                    tt.meta["diagnostics"]["source_fit"]["boundary_objective"][int(seg_id)] = curve
                if verbose:
                    print(
                        f"[BACKMAP] source_fit: segment={seg_id} hit rs boundary (rs_best={rs_best:.3f} R_sun) "
                        "-> NOT applied (set rs_fit_allow_boundary=True to force)."
                    )
                continue

            rs_fit_used[idxs] = rs_best

            if verbose:
                phi_src_best = wrap_0_360(phi_sc_deg[idx_fit] + int(phi_sign) * omega_deg_per_s * tau_by_rs[j_best, :])
                std_best = float(circ_std_deg(phi_src_best, axis=None))
                print(f'[BACKMAP] source_fit: segment={seg_id}  n={idx_fit.size}  rs_best={rs_best:.3f} R_sun  circ_std(phi_src)~{std_best:.2f} deg')

        data['rs_fit_Rsun'] = rs_fit_used

        # Re-evaluate tau using segment-median fitted rs where available.
        tau_new = np.full(len(data), np.nan, dtype=float) * u.s
        fb_new = np.zeros(len(data), dtype=bool)

        for seg_id in sorted(set(int(x) for x in np.unique(source_segment) if x >= 0)):
            idxs = np.where(source_segment == seg_id)[0]
            rs_vals = rs_fit_used[idxs]
            if not np.isfinite(np.nanmedian(rs_vals)):
                continue
            rs_seg = float(np.nanmedian(rs_vals)) * u.R_sun
            mrs = HybridParker(rs=rs_seg, n_grid=getattr(model, 'n_grid', 8192), switch=getattr(model, 'switch', 'rs'))
            tts = mrs.evaluate(
                r_sc=r_sc_q[idxs],
                V_bg=V_bg_q[idxs],
                r_ss=u.Quantity(r_ss),
                v_min=v_min,
                v_fallback=v_fallback,
            )
            tau_new[idxs] = tts.tau
            fb_new[idxs] = tts.fallback_mask

        use = np.isfinite(tau_new.to_value(u.s))
        tau = np.where(use, tau_new.to_value(u.s), tau.to_value(u.s)) * u.s
        tt = TravelTimeResult(tau=tau.to(u.s), fallback_mask=(tt.fallback_mask | fb_new), meta=tt.meta)

        tt.meta.setdefault("diagnostics", {})
        tt.meta["diagnostics"].setdefault("source_fit", {})
        tt.meta["diagnostics"]["source_fit"].update(
            {
                'enabled': True,
                'rs_grid_Rsun': [float(rs_grid[0].to_value(u.R_sun)), float(rs_grid[-1].to_value(u.R_sun)), int(len(rs_grid))],
                'rs_fit_decimate': int(rs_fit_decimate),
                'rs_fit_allow_boundary': bool(rs_fit_allow_boundary),
                'n_fit_points': int(np.isfinite(rs_fit_used).sum()),
            }
        )

        if verbose:
            nfit = int(np.isfinite(rs_fit_used).sum())
            print(f'[BACKMAP] source_fit: fitted rs on {nfit}/{len(data)} points (others kept base tau).')
    # --------------------------------------------------------------
    # Uncertainty propagation
    #
    # Default: treat Vr_bg uncertainty as Gaussian with sigma_Vr.
    # For the hybrid Parker profile family (paper-style heuristic),
    # allow a shape-uncertainty scan over rs to produce tau/phi percentiles.
    # --------------------------------------------------------------
    tau_samples_s = None  # shape (n_rs, n_time) in seconds
    phi_samples_deg = None  # shape (n_rs, n_time) in degrees
    rs_samples_used = None  # Quantity array of rs samples (optional)
    rs_switch_used = None
    rs_n_grid_used = None

    # ------------------------------------------------------------------
    # Profile-family uncertainty (paper-style): ONLY for the hybrid family.
    #
    # IMPORTANT: Do NOT enable this for parker_scaled (Lambert-W Parker shape).
    # Doing so silently swaps the physics model (hybrid vs Parker) and is the
    # exact kind of over-parameterized, non-defensible behavior we must avoid.
    # ------------------------------------------------------------------
    method_lc = str(method_tag).strip().lower()
    use_rs_family = bool(
        method_lc in {"hybrid_parker", "hybrid"}
        and model_kwargs
        and any(k in model_kwargs for k in ("rs_samples", "rs_min", "rs_max", "n_rs"))
    )

    if use_rs_family:
        if bool(azimuthal_correction):
            raise ValueError('azimuthal_correction=True is not supported with the rs-family uncertainty scan. Disable azimuthal_correction or omit rs_samples/rs_min/rs_max/n_rs in model_kwargs.')

        # Parse rs samples (in R_sun unless already a Quantity)
        # Defaults chosen for *defensibility + simplicity*:
        # - switch at r=rs (stable, predictable; avoids pathological "match" degeneracy)
        # - moderate n_grid (fast, reproducible)
        switch = str(model_kwargs.get("switch", "rs"))
        n_grid = int(model_kwargs.get("n_grid", 4096))

        rs_mode = str(model_kwargs.get("rs_sample_mode", "tau_uniform")).strip().lower()
        rs_samples = model_kwargs.get("rs_samples", None)

        if rs_samples is None:
            rs_min = u.Quantity(model_kwargs.get("rs_min", 0.1 * u.R_sun)).to(u.R_sun)
            rs_max = u.Quantity(model_kwargs.get("rs_max", 3.0 * u.R_sun)).to(u.R_sun)
            n_rs = int(model_kwargs.get("n_rs", 20))

            if rs_mode == "tau_uniform":
                # Choose rs so that a representative travel time spans uniformly between
                # the ballistic and most-accelerating cases (heuristic but stable).
                rs_dense = np.linspace(rs_min.to_value(u.R_sun), rs_max.to_value(u.R_sun), num=max(200, 10 * n_rs)) * u.R_sun

                r_sc_rep = float(np.nanmedian(r_sc_q.to_value(u.R_sun))) * u.R_sun
                V_rep = float(np.nanmedian(V_bg_q.to_value(u.km / u.s))) * (u.km / u.s)

                tau_dense = np.full(len(rs_dense), np.nan, dtype=float)
                for j, rs_j in enumerate(rs_dense):
                    m_j = build_model("hybrid_parker", model_kwargs={"rs": rs_j, "n_grid": n_grid, "switch": switch})
                    tt_j = m_j.evaluate(r_sc=r_sc_rep, V_bg=V_rep, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback)
                    tau_dense[j] = float(np.atleast_1d(tt_j.tau.to_value(u.s))[0])

                # Target travel times: uniform between ballistic and max-accel (largest tau)
                tau_ball = float(((r_sc_rep - u.Quantity(r_ss)).to(u.m) / V_rep.to(u.m / u.s)).to_value(u.s))
                ok = np.isfinite(tau_dense)
                if ok.sum() < 5:
                    rs_samples = np.linspace(rs_min.to_value(u.R_sun), rs_max.to_value(u.R_sun), n_rs) * u.R_sun
                    rs_mode = "rs_uniform_fallback"
                else:
                    tau_ok = tau_dense[ok]
                    rs_ok = rs_dense.to_value(u.R_sun)[ok]

                    # Enforce monotonic interpolation (sort by tau)
                    srt = np.argsort(tau_ok)
                    tau_ok = tau_ok[srt]
                    rs_ok = rs_ok[srt]

                    tau_hi = float(np.nanmax(tau_ok))
                    tau_targets = np.linspace(min(tau_ball, tau_hi), tau_hi, n_rs)
                    rs_samples = np.interp(tau_targets, tau_ok, rs_ok) * u.R_sun
            else:
                rs_samples = np.linspace(rs_min.to_value(u.R_sun), rs_max.to_value(u.R_sun), n_rs) * u.R_sun
        else:
            if isinstance(rs_samples, u.Quantity):
                rs_samples = rs_samples.to(u.R_sun)
            else:
                rs_samples = np.asarray(rs_samples, dtype=float) * u.R_sun

        # Evaluate tau (and mapped phi) for each rs profile
        rs_samples_used = rs_samples
        rs_switch_used = str(switch)
        rs_n_grid_used = int(n_grid)

        tau_stack = []
        phi_stack = []
        phi_sc = data["phi_sc"].to_numpy(dtype=float)
        lat_sc = data["lat_sc"].to_numpy(dtype=float)

        for rs in rs_samples:
            m_rs = build_model("hybrid_parker", model_kwargs={"rs": rs, "n_grid": n_grid, "switch": switch})
            tt_rs = m_rs.evaluate(r_sc=r_sc_q, V_bg=V_bg_q, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback)
            tau_rs = tt_rs.tau

            tau_stack.append(tau_rs.to_value(u.s))

            # Map phi for each sample (lat is unchanged in our baseline)
            map_rs = map_to_source_surface(
                phi_sc_deg=phi_sc,
                lat_sc_deg=lat_sc,
                tau=tau_rs,
                omega=omega,
                phi_sign=phi_sign,
            )
            phi_stack.append(map_rs.phi_src_deg)

        tau_samples_s = np.stack(tau_stack, axis=0)  # seconds
        phi_samples_deg = np.stack(phi_stack, axis=0)

        # Central + percentiles
        tau = (np.nanmedian(tau_samples_s, axis=0) * u.s).to(u.s)
        tau_p16 = (np.nanpercentile(tau_samples_s, 16.0, axis=0) * u.s).to(u.s)
        tau_p84 = (np.nanpercentile(tau_samples_s, 84.0, axis=0) * u.s).to(u.s)

        # Phi percentiles on the circle
        phi_src = circ_percentile_deg(phi_samples_deg, 50.0, axis=0)
        phi_src_p16 = circ_percentile_deg(phi_samples_deg, 16.0, axis=0)
        phi_src_p84 = circ_percentile_deg(phi_samples_deg, 84.0, axis=0)

        sigma_tau = 0.5 * (tau_p84 - tau_p16)

        # Surface this explicitly for diagnostics/fig annotation downstream.
        tt.meta.setdefault("diagnostics", {})
        tt.meta["diagnostics"].update(
            {
                "rs_family_enabled": True,
                "rs_min_Rsun": float(np.nanmin(u.Quantity(rs_samples).to_value(u.R_sun))),
                "rs_max_Rsun": float(np.nanmax(u.Quantity(rs_samples).to_value(u.R_sun))),
                "rs_n": int(len(rs_samples)),
                "rs_switch": str(switch),
                "rs_sample_mode": str(rs_mode),
            }
        )

    else:
        # Uncertainty propagation: treat V_bg uncertainty as Gaussian with sigma_Vr
        V_sig_q = q_from_df(data, "sigma_Vr")

        V_p16 = (V_bg_q - V_sig_q).to(u.km / u.s)
        V_p84 = (V_bg_q + V_sig_q).to(u.km / u.s)

        # Floor to keep times positive and avoid hidden singularities
        V_p16 = np.maximum(V_p16.to_value(u.km / u.s), u.Quantity(v_min).to_value(u.km / u.s)) * (u.km / u.s)
        V_p84 = np.maximum(V_p84.to_value(u.km / u.s), u.Quantity(v_min).to_value(u.km / u.s)) * (u.km / u.s)

        tau_p16 = model.evaluate(r_sc=r_sc_q, V_bg=V_p84, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback).tau
        tau_p84 = model.evaluate(r_sc=r_sc_q, V_bg=V_p16, r_ss=u.Quantity(r_ss), v_min=v_min, v_fallback=v_fallback).tau

        sigma_tau = 0.5 * (tau_p84 - tau_p16)

    # ------------------------------------------------------------------
    # Longitude shift Δφ used for mapping.
    #
    # Baseline (ballistic in rotating frame): Δφ = Ωτ.
    # Optional (Appendix A): include v_ϕ(r) via Weber--Davis,
    #   Δφ = ∫ (Ω - v_ϕ/r) / v_r \, dr.
    # ------------------------------------------------------------------
    omega_deg_s = u.Quantity(omega).to(u.deg / u.s)
    omega_deg_s_val = float(omega_deg_s.to_value(u.deg / u.s))

    tau_s_arr = tau.to_value(u.s)
    delta_phi_omega_tau = (omega_deg_s * u.Quantity(tau_s_arr, u.s)).to(u.deg)

    delta_phi = delta_phi_omega_tau
    delta_phi_meta = {"mode": "omega_tau"}

    if bool(azimuthal_correction):
        az_res = compute_delta_phi_series(
            model=model,
            r_sc=r_sc_q,
            V_bg=V_bg_q,
            r_ss=u.Quantity(r_ss),
            omega=omega,
            r_A=u.Quantity(r_A),
            n_grid=int(az_n_grid),
            v_min=v_min,
            v_fallback=v_fallback,
            ma2_tol=float(az_ma2_tol),
        )
        delta_phi = u.Quantity(az_res.delta_phi).to(u.deg)
        delta_phi_meta = dict(az_res.meta)
        delta_phi_meta["mode"] = "appendixA"

    # Convert travel-time uncertainty into a longitude uncertainty scale.
    # Baseline: sigma_phi ≈ Ω*sigma_tau. With Appendix-A enabled,
    # sigma_phi is derived from Δφ(V_bg±σ) below.
    sigma_phi_from_tau = (abs(int(phi_sign)) * omega_deg_s * u.Quantity(sigma_tau).to(u.s)).to(u.deg)
    sigma_phi_from_tau_deg = np.asarray(sigma_phi_from_tau.to_value(u.deg), dtype=float)

    # Valid mask (recomputed after uncertainty logic)
    valid = (
        np.isfinite(tau.to_value(u.s))
        & np.isfinite(data["phi_sc"].to_numpy(dtype=float))
        & np.isfinite(data["lat_sc"].to_numpy(dtype=float))
    )

    fallback_fraction = float(np.nanmean(fb_vr[valid])) if valid.any() else 1.0
    masked_fraction = 1.0 - float(np.nanmean(valid)) if len(valid) else 1.0

    # ------------------------------------------------------------------
    # Physics diagnostic: tau(t) compared to ballistic-bg reference under the
    # same V_bg floor/fallback policy.
    # ------------------------------------------------------------------
    tau_vs_ballistic: Dict[str, Any] = {
        "eps_median": float("nan"),
        "eps_p16": float("nan"),
        "eps_p84": float("nan"),
        "n": int(np.sum(valid)),
        "error": None,
    }
    try:
        tau_ball = build_model("ballistic_bg").evaluate(
            r_sc=r_sc_q,
            V_bg=V_bg_q,
            r_ss=u.Quantity(r_ss),
            v_min=v_min,
            v_fallback=v_fallback,
        ).tau
        tb = tau_ball.to_value(u.s)
        tc = tau.to_value(u.s)
        eps = (tc - tb) / tb
        eps = eps[valid]
        if eps.size:
            tau_vs_ballistic["eps_median"] = float(np.nanmedian(eps))
            tau_vs_ballistic["eps_p16"] = float(np.nanpercentile(eps, 16.0))
            tau_vs_ballistic["eps_p84"] = float(np.nanpercentile(eps, 84.0))
            tau_vs_ballistic["n"] = int(eps.size)
    except Exception as _e_tau_ref:
        tau_vs_ballistic["error"] = str(_e_tau_ref)

    # Mapping
    phi_sc_arr = data["phi_sc"].to_numpy(dtype=float)
    lat_sc_arr = data["lat_sc"].to_numpy(dtype=float)

    # Central (median) mapping
    map0 = map_to_source_surface(
        phi_sc_deg=phi_sc_arr,
        lat_sc_deg=lat_sc_arr,
        tau=tau,
        omega=omega,
        delta_phi_deg=delta_phi.to_value(u.deg),
        phi_sign=phi_sign,
    )

    data["lat_src"] = map0.lat_src_deg

    if phi_samples_deg is not None:
        # Percentiles computed directly from the rs-family mapping (circular-safe).
        data["phi_src"] = phi_src
        data["phi_src_p16"] = phi_src_p16
        data["phi_src_p84"] = phi_src_p84
        # In rs-family mode, define sigma_phi from the circular half-width of the percentile envelope.
        sigma_phi_deg = halfwidth_deg(np.asarray(phi_src_p16, float), np.asarray(phi_src_p84, float))
    else:
        # For V_bg uncertainty only, avoid mapping tau_p16/tau_p84 directly through wrap(0,360),
        # which can produce misleading "flipped" intervals near 0/360.
        # Instead, propagate the time-scale uncertainty into an angular scale sigma_phi = Omega*sigma_tau,
        # then construct a symmetric circular interval around the central phi_src.
        phi_center = np.asarray(map0.phi_src_deg, dtype=float)

        if bool(azimuthal_correction):
            # Derive an angular uncertainty scale from Appendix-A Δφ(V_bg±σ).
            # Note: tau_p16 uses V_p84 (fast) and tau_p84 uses V_p16 (slow); mirror that ordering here.
            az_lo = compute_delta_phi_series(
                model=model,
                r_sc=r_sc_q,
                V_bg=V_p84,
                r_ss=u.Quantity(r_ss),
                omega=omega,
                r_A=u.Quantity(r_A),
                n_grid=int(az_n_grid),
                v_min=v_min,
                v_fallback=v_fallback,
                ma2_tol=float(az_ma2_tol),
            ).delta_phi.to(u.deg)
            az_hi = compute_delta_phi_series(
                model=model,
                r_sc=r_sc_q,
                V_bg=V_p16,
                r_ss=u.Quantity(r_ss),
                omega=omega,
                r_A=u.Quantity(r_A),
                n_grid=int(az_n_grid),
                v_min=v_min,
                v_fallback=v_fallback,
                ma2_tol=float(az_ma2_tol),
            ).delta_phi.to(u.deg)
            sigma_phi_deg = 0.5 * (az_hi.to_value(u.deg) - az_lo.to_value(u.deg))
        else:
            sigma_phi_deg = sigma_phi_from_tau_deg
        data["phi_src"] = phi_center
        data["phi_src_p16"] = wrap_0_360(phi_center - sigma_phi_deg)
        data["phi_src_p84"] = wrap_0_360(phi_center + sigma_phi_deg)

    attach_units(data, {"phi_src": u.deg, "lat_src": u.deg, "phi_src_p16": u.deg, "phi_src_p84": u.deg})

    # Store the longitude shift used for mapping (degrees).
    data["delta_phi_omega_tau"] = np.asarray(delta_phi_omega_tau.to_value(u.deg), dtype=float)
    data["delta_phi"] = np.asarray(u.Quantity(delta_phi).to_value(u.deg), dtype=float)
    data["delta_phi_signed"] = int(phi_sign) * np.asarray(u.Quantity(delta_phi).to_value(u.deg), dtype=float)
    attach_units(data, {"delta_phi": u.deg, "delta_phi_signed": u.deg, "delta_phi_omega_tau": u.deg})

    # Tau in hours for output clarity.
    # NOTE: define tau_s *before* any downstream use (e.g. phi_src_at_t).
    data["tau"] = tau.to_value(u.hour)
    data["tau_s"] = tau.to_value(u.s)
    data["tau_p16"] = tau_p16.to_value(u.hour)
    data["tau_p16_s"] = tau_p16.to_value(u.s)
    data["tau_p84"] = tau_p84.to_value(u.hour)
    data["tau_p84_s"] = tau_p84.to_value(u.s)
    data["sigma_tau"] = sigma_tau.to_value(u.hour)
    attach_units(data, {"tau": u.hour, "tau_p16": u.hour, "tau_p84": u.hour, "sigma_tau": u.hour})

    # Carrington longitude of the same source-surface footpoint rotated forward to the measurement time.
    # This uses the *same* omega as the mapping and the travel time tau(t) used to compute phi_src.
    tau_s_arr = np.asarray(data["tau_s"], dtype=float)
    data["phi_src_at_t"] = wrap_0_360(np.asarray(data["phi_src"], float) - omega_deg_s_val * tau_s_arr)
    if "phi_src_p16" in data.columns and "phi_src_p84" in data.columns:
        data["phi_src_at_t_p16"] = wrap_0_360(np.asarray(data["phi_src_p16"], float) - omega_deg_s_val * tau_s_arr)
        data["phi_src_at_t_p84"] = wrap_0_360(np.asarray(data["phi_src_p84"], float) - omega_deg_s_val * tau_s_arr)
    attach_units(data, {"phi_src_at_t": u.deg, "phi_src_at_t_p16": u.deg, "phi_src_at_t_p84": u.deg})

    # Longitude uncertainty scale (deg)
    data["sigma_phi"] = np.asarray(sigma_phi_deg, dtype=float)
    attach_units(data, {"sigma_phi": u.deg})

    # ------------------------------------------------------------------
    # Carrington sanity diagnostics.
    # These are explicit numerical checks that help catch frame/sign mistakes.
    # (For typical heliocentric orbits, Carrington longitudes drift and can wrap
    # by ~360 deg over weeks; the exact drift rate depends on the orbit.)
    # ------------------------------------------------------------------
    carr_diag: Dict[str, Any] = {"ok": True, "error": None}
    try:
        carr_diag["phi_sc_drift"] = _carrington_drift_stats(
            eph_i_orbit["phi_sc_deg"].to_numpy(dtype=float),
            pd.DatetimeIndex(eph_i_orbit.index),
        )
        if "phi_src" in data.columns:
            carr_diag["phi_src_drift"] = _carrington_drift_stats(
                data["phi_src"].to_numpy(dtype=float),
                pd.DatetimeIndex(data.index),
            )

        omega_deg_s = float(u.Quantity(omega).to_value(u.deg / u.s))
        carr_diag["mapping_shift"] = _mapping_shift_sanity(
            phi_sc_deg=data["phi_sc"].to_numpy(dtype=float),
            phi_src_deg=data["phi_src"].to_numpy(dtype=float),
            tau_s=data["tau_s"].to_numpy(dtype=float),
            omega_deg_s=omega_deg_s,
            phi_sign=int(phi_sign),
            expected_shift_deg=(data["delta_phi_signed"].to_numpy(dtype=float) if "delta_phi_signed" in data.columns else None),
        )
    except Exception as _e_carr:
        carr_diag["ok"] = False
        carr_diag["error"] = str(_e_carr)

    # Derived physical diagnostics (needs r_sc)
        # Derived diagnostics computed only when requested by the plotting/size contract
    need_diag: set[str] = set(map(str, plot_vars)) | {str(size_by)}
    if plot_3d:
        vars3d = list(plot_3d_vars) if plot_3d_vars is not None else ([str(plot_3d_var)] if plot_3d_var is not None else list(plot_vars))
        need_diag |= set(map(str, vars3d))
    _derive_physical_diagnostics(data, need=need_diag)

    # Marker size for scatter maps
    if size_by in data.columns:
        data["marker_size"] = marker_sizes_from_metric(pd.to_numeric(data[size_by], errors="coerce"))
    else:
        data["marker_size"] = 30.0

    # Plotting
    specs = merge_var_specs(VAR_SPECS, var_specs)

    out_pkl = out_base / "backmap_timeseries.pkl"
    out_png = out_base / "source_surface_maps_2d.png"
    out_html = out_base / "source_surface_3d.html"
    out_meta = out_base / "meta.json"

    # Titles:
    #   - 2D should be clean (date/time only; no parameter soup).
    #   - 3D must explicitly include a radial distance in $R_\odot$.
    try:
        t0 = pd.to_datetime(data.index.min()).to_pydatetime()
        t1 = pd.to_datetime(data.index.max()).to_pydatetime()
        tspan = f"{t0:%Y-%m-%d %H:%M}-{t1:%H:%M} UTC"
    except Exception:
        tspan = ""

    run_label = str(interval_label).strip() if interval_label else f"int {which_int}"
    title = f"{sc} | {run_label}" + (f" | {tspan}" if tspan else "")

    r_sc_med_Rsun = (
        float(np.nanmedian(r_sc_q[valid].to_value(u.R_sun)))
        if valid.any()
        else float(np.nanmedian(r_sc_q.to_value(u.R_sun)))
    )
    title3d = (
        f"{title} | "
        f"$r_{{\\rm sc,med}}$={r_sc_med_Rsun:.1f} $R_\\odot$"
    )
    # ------------------------------------------------------------------
    # Preflight: robust plotting contract (drop missing *optional* variables)
    # ------------------------------------------------------------------
    def _available_preview(cols: Sequence[str]) -> str:
        avail = ", ".join(list(map(str, data.columns[:40]))) + (" ..." if len(data.columns) > 40 else "")
        req = ", ".join(list(map(str, cols)))
        return f"requested=[{req}] | available=[{avail}]"

    # 2D variables: drop those not present
    plot_vars_requested = [str(v) for v in list(plot_vars)]
    plot_vars_used = [v for v in plot_vars_requested if v in data.columns]
    plot_vars_dropped = [v for v in plot_vars_requested if v not in data.columns]
    if plot_vars_dropped and verbose:
        print(f"[BACKMAP][WARN] 2D plotting: dropping missing columns: {plot_vars_dropped}")
    if len(plot_vars_used) == 0:
        raise ValueError(f"2D plotting: no requested variables exist. {_available_preview(plot_vars_requested)}")

    # 3D variables
    vars3d_requested = None
    vars3d_used = None
    vars3d_dropped = None
    if plot_3d:
        if plot_3d_vars is not None:
            vars3d_requested = [str(v) for v in list(plot_3d_vars)]
        elif plot_3d_var is not None:
            vars3d_requested = [str(plot_3d_var)]
        else:
            vars3d_requested = list(plot_vars_requested)

        vars3d_used = [v for v in vars3d_requested if v in data.columns]
        vars3d_dropped = [v for v in vars3d_requested if v not in data.columns]
        if vars3d_dropped and verbose:
            print(f"[BACKMAP][WARN] 3D plotting: dropping missing columns: {vars3d_dropped}")
        if len(vars3d_used) == 0:
            raise ValueError(f"3D plotting: no requested variables exist. {_available_preview(vars3d_requested)}")
        if show_uncertainty:
            miss_ci = [c for c in ["phi_src_p16", "phi_src_p84"] if c not in data.columns]
            if miss_ci:
                raise ValueError(f"3D uncertainty requested but missing columns: {miss_ci}")
# ------------------------------------------------------------------
    # Velocity profile used for travel-time integration (mandatory 2D panel)
    # ------------------------------------------------------------------
    prof_png = None
    profile_panel = None
    diag_extra: Dict[str, Any] = {
        "profile_model": str(method_tag),
        "profile_ok": False,
        "profile_error": None,
        "profile_degenerate": None,
        "profile_fallback_used": False,
        "U_min_kms": None,
        "U_max_kms": None,
        "U_span_kms": None,
        "r_sc_profile_Rsun": None,
        "source_fit_enabled": bool(source_fit) and (str(method_tag).lower().strip() in {"hybrid_parker","hybrid"}),
        "source_fit_threshold": float(source_fit_threshold),
        "source_fit_window": str(source_fit_window),
        "source_fit_vars": [str(v) for v in source_fit_vars],
        "rs_fit_range_Rsun": (float(rs_fit_range_Rsun[0]), float(rs_fit_range_Rsun[1])),
        "rs_fit_n": int(rs_fit_n),
        "rs_fit_decimate": int(rs_fit_decimate),
        "rs_fit_fraction": float(np.isfinite(rs_fit_used).sum() / max(len(data), 1)) if rs_fit_used is not None else 0.0,
        "rs_fit_unique_segments": int(len(set(int(x) for x in np.unique(data.get("source_segment", np.array([-1]))) if int(x) >= 0))) if "source_segment" in data.columns else 0,
        "rs_fit_Rsun_median": float(np.nanmedian(rs_fit_used)) if rs_fit_used is not None and np.isfinite(np.nanmedian(rs_fit_used)) else None,

    }
    try:
        out_prof = out_base / "velocity_profile.png"

        if valid.any():
            r_sc_med = float(np.nanmedian(r_sc_q[valid].to_value(u.R_sun)))
            V_med = float(np.nanmedian(V_bg_q[valid].to_value(u.km / u.s)))
            V_sig_med = float(np.nanmedian(V_sig_q[valid].to_value(u.km / u.s)))
        else:
            r_sc_med = float(np.nanmedian(r_sc_q.to_value(u.R_sun)))
            V_med = float(np.nanmedian(V_bg_q.to_value(u.km / u.s)))
            V_sig_med = float(np.nanmedian(V_sig_q.to_value(u.km / u.s)))

        r_ss_R = float(u.Quantity(r_ss).to_value(u.R_sun))
        r_hi = max(r_ss_R * 1.02, r_sc_med)
        r_grid = np.geomspace(max(1.01, r_ss_R), r_hi, 420)

        v_min_profile = None
        if model_kwargs and ("v_min_profile" in model_kwargs):
            v_min_profile = u.Quantity(model_kwargs["v_min_profile"])

        # If we ran an rs-family scan (hybrid Parker uncertainty), show the envelope across profiles.
        if (rs_samples_used is not None) and (str(method_tag).lower().strip() in {"hybrid_parker", "hybrid"}):
            U_stack = []
            for rs_i in u.Quantity(rs_samples_used).to(u.R_sun):
                m_i = build_model(
                    "hybrid_parker",
                    model_kwargs={
                        "rs": rs_i,
                        "n_grid": int(rs_n_grid_used or 8192),
                        "switch": str(rs_switch_used or "rs"),
                    },
                )
                Ui = m_i.speed_profile(
                    r_grid=r_grid * u.R_sun,
                    r_sc=r_sc_med * u.R_sun,
                    V_bg=V_med * (u.km / u.s),
                    r_ss=u.Quantity(r_ss),
                    v_min=v_min_profile,
                )
                if Ui is not None:
                    U_stack.append(Ui.to_value(u.km / u.s))

            if len(U_stack) < 3:
                U_med = None
                U_lo = None
                U_hi = None
            else:
                U_arr = np.asarray(U_stack, dtype=float)
                U_med = u.Quantity(np.nanmedian(U_arr, axis=0), u.km / u.s)
                U_lo = u.Quantity(np.nanpercentile(U_arr, 16.0, axis=0), u.km / u.s)
                U_hi = u.Quantity(np.nanpercentile(U_arr, 84.0, axis=0), u.km / u.s)
        else:
            U_med = model.speed_profile(
                r_grid=r_grid * u.R_sun,
                r_sc=r_sc_med * u.R_sun,
                V_bg=V_med * (u.km / u.s),
                r_ss=u.Quantity(r_ss),
                v_min=v_min_profile,
            )
            U_lo = None
            U_hi = None
            if U_med is not None:
                # Uncertainty envelope: vary the boundary speed by sigma_V.
                V_lo = max(float(u.Quantity(v_min).to_value(u.km / u.s)), V_med - V_sig_med)
                V_hi = V_med + V_sig_med

                U_lo = model.speed_profile(
                    r_grid=r_grid * u.R_sun,
                    r_sc=r_sc_med * u.R_sun,
                    V_bg=V_lo * (u.km / u.s),
                    r_ss=u.Quantity(r_ss),
                    v_min=v_min_profile,
                )
                U_hi = model.speed_profile(
                    r_grid=r_grid * u.R_sun,
                    r_sc=r_sc_med * u.R_sun,
                    V_bg=V_hi * (u.km / u.s),
                    r_ss=u.Quantity(r_ss),
                    v_min=v_min_profile,
                )

        if U_med is not None:
            profile_panel = {
                "r_grid_Rsun": r_grid,
                "U_med_kms": U_med.to_value(u.km / u.s),
                "U_lo_kms": (U_lo.to_value(u.km / u.s) if U_lo is not None else None),
                "U_hi_kms": (U_hi.to_value(u.km / u.s) if U_hi is not None else None),
                "r_ss_Rsun": float(r_ss_R),
                "r_sc_Rsun": float(r_sc_med),
                "U_min_kms": diag_extra.get("U_min_kms", None),
                "U_max_kms": diag_extra.get("U_max_kms", None),
                "U_span_kms": diag_extra.get("U_span_kms", None),
                "U_span_thr_kms": diag_extra.get("U_span_thr_kms", None),
                "profile_degenerate": diag_extra.get("profile_degenerate", None),
                "degenerate_reason": diag_extra.get("degenerate_reason", None),
                "executed_model_signature": str(model_sig),
            }
            # Optional: time-varying U(r,t) samples for visualization (colored by time in the U(r) panel).
            # This makes explicit that, for ballistic_bg and the scaled-profile models, the assumed profile
            # is evaluated per timestamp using V_bg(t) and r_sc(t) (shape fixed, scaling time-dependent).
            try:
                n_prof = 12
                if model_kwargs and ("profile_n_time_samples" in model_kwargs):
                    n_prof = int(model_kwargs["profile_n_time_samples"])
                n_prof = int(max(0, min(30, n_prof)))
                if n_prof >= 2:
                    r_sc_arr = r_sc_q.to_value(u.R_sun)
                    V_arr = V_bg_q.to_value(u.km / u.s)
                    okm = np.isfinite(r_sc_arr) & np.isfinite(V_arr)
                    idx_ok = np.where(okm)[0]
                    if idx_ok.size >= 2:
                        sel = idx_ok[np.linspace(0, idx_ok.size - 1, min(n_prof, idx_ok.size)).astype(int)]
                        t0 = pd.to_datetime(data.index[sel[0]])
                        U_samp: List[np.ndarray] = []
                        t_hr: List[float] = []
                        t_iso: List[str] = []
                        r_samp: List[float] = []
                        V_samp: List[float] = []
                        for ii in sel:
                            Ui = model.speed_profile(
                                r_grid=r_grid * u.R_sun,
                                r_sc=float(r_sc_arr[ii]) * u.R_sun,
                                V_bg=float(V_arr[ii]) * (u.km / u.s),
                                r_ss=u.Quantity(r_ss),
                                v_min=v_min_profile,
                            )
                            if Ui is None:
                                continue
                            U_samp.append(Ui.to_value(u.km / u.s))
                            try:
                                ti = pd.to_datetime(data.index[ii])
                                t_iso.append(str(ti))
                                t_hr.append(float((ti - t0).total_seconds() / 3600.0))
                            except Exception:
                                t_iso.append(str(ii))
                                t_hr.append(float(ii))
                            r_samp.append(float(r_sc_arr[ii]))
                            V_samp.append(float(V_arr[ii]))
                        if len(U_samp) >= 2:
                            profile_panel["U_samples_kms"] = np.asarray(U_samp, float)
                            profile_panel["t_samples_hr"] = np.asarray(t_hr, float)
                            profile_panel["t_samples_iso"] = list(t_iso)
                            profile_panel["r_sc_samples_Rsun"] = list(r_samp)
                            profile_panel["V_bg_samples_kms"] = list(V_samp)
            except Exception:
                pass

            # Diagnostics: detect degenerate/flat profiles (should not occur for accelerating models).
            try:
                uarr = np.asarray(profile_panel.get("U_med_kms", None), dtype=float)
                # helpful scalar diagnostics for plotting/readout
                try:
                    if uarr.size >= 2 and np.isfinite(uarr[[0, -1]]).all() and uarr[-1] != 0.0:
                        profile_panel["U_ss_kms"] = float(uarr[0])
                        profile_panel["U_sc_kms"] = float(uarr[-1])
                        profile_panel["U_ss_over_U_sc"] = float(uarr[0] / uarr[-1])
                        try:
                            diag_extra["U_ss_kms"] = float(uarr[0])
                            diag_extra["U_sc_kms"] = float(uarr[-1])
                            diag_extra["U_ss_over_U_sc"] = float(uarr[0] / uarr[-1])
                        except Exception:
                            pass
                except Exception:
                    pass
                if uarr.size > 0 and np.isfinite(uarr).any():
                    umin = float(np.nanmin(uarr))
                    umax = float(np.nanmax(uarr))
                    span = float(umax - umin)
                else:
                    umin = umax = span = float('nan')
                diag_extra["profile_ok"] = True
                diag_extra["U_min_kms"] = umin
                diag_extra["U_max_kms"] = umax
                diag_extra["U_span_kms"] = span
                diag_extra["r_sc_profile_Rsun"] = float(r_sc_med)
                accel = str(method_tag).lower().strip() in {"parker_scaled", "exp_accel", "hybrid_parker", "hybrid"}
                if accel and np.isfinite(span) and np.isfinite(V_med):
                    thr = max(1e-6, 1e-4 * float(V_med))
                    diag_extra["U_span_thr_kms"] = float(thr)
                    diag_extra["profile_degenerate"] = bool(span < thr)

                    # Distinguish true profile failure from trivial r_sc~r_ss span.
                    dr = float(r_sc_med - r_ss_R)
                    if diag_extra["profile_degenerate"]:
                        if dr <= 0.05:
                            diag_extra["degenerate_reason"] = "r_sc~r_ss (tiny integration interval)"
                        else:
                            diag_extra["degenerate_reason"] = "flat_U(r) over nontrivial r-range"
                            raise RuntimeError(
                                "Degenerate accelerating profile: U_span={:.3g} km/s < thr={:.3g} km/s with r_sc-r_ss={:.3g} R_sun. "
                                "This indicates a broken dispatch/profile/integration path.".format(span, thr, dr)
                            )
                    else:
                        diag_extra["degenerate_reason"] = None
                else:
                    diag_extra["U_span_thr_kms"] = None
                    diag_extra["profile_degenerate"] = False
                    diag_extra["degenerate_reason"] = None
            except Exception as _e_prof:
                diag_extra["profile_ok"] = False
                diag_extra["profile_error"] = f"profile_diagnostics_failed: {_e_prof}"


            # Keep a standalone diagnostic PNG as well.
            prof_title = r"$U(r)$ profile used for travel-time integration"
            prof_png, _ = plot_velocity_profile(
                out_png=out_prof,
                r_grid=r_grid,
                U_med=U_med.to_value(u.km / u.s),
                U_lo=(U_lo.to_value(u.km / u.s) if U_lo is not None else None),
                U_hi=(U_hi.to_value(u.km / u.s) if U_hi is not None else None),
                r_ss=r_ss_R,
                r_sc=r_sc_med,
                title=prof_title,
                show=show,
            )
    except Exception as _e_prof:
        prof_png = None
        profile_panel = None
        diag_extra["profile_ok"] = False
        diag_extra["profile_error"] = str(_e_prof)

    # Hard guard: the 2D figure must always contain a velocity-profile panel.
    if profile_panel is None:
        raise RuntimeError(
            "Mandatory velocity-profile panel construction failed (no silent fallback). "
            "profile_error={!r}".format(diag_extra.get("profile_error", None))
        )


# Summary box (TeX-safe; includes profile diagnostics)
    
    # ------------------------------------------------------------------
    # Mandatory audit block (executed-path truthful; stored in meta)
    # ------------------------------------------------------------------
    from .circular import CIRC_PERCENTILE_ALGO_ID  # local import for ID only
    r_sc_Rsun = r_sc_q.to_value(u.R_sun)
    r_ss_Rsun = float(u.Quantity(r_ss).to_value(u.R_sun))
    r_stats = {
        "r_ss_Rsun": float(r_ss_Rsun),
        "r_sc_min_Rsun": float(np.nanmin(r_sc_Rsun)) if np.isfinite(r_sc_Rsun).any() else float("nan"),
        "r_sc_med_Rsun": float(np.nanmedian(r_sc_Rsun)) if np.isfinite(r_sc_Rsun).any() else float("nan"),
        "r_sc_max_Rsun": float(np.nanmax(r_sc_Rsun)) if np.isfinite(r_sc_Rsun).any() else float("nan"),
        "dr_min_Rsun": float(np.nanmin(r_sc_Rsun - r_ss_Rsun)) if np.isfinite(r_sc_Rsun).any() else float("nan"),
    }

    # Parker-scaled mismatch stats (only nonzero where floor/fallback applied)
    parker_scaled_diag = None
    if method_tag == "parker_scaled":
        Veff, bad = _robust_speed_local(V_bg_q, v_min=v_min, v_fallback=v_fallback)
        mm = np.abs(Veff.to_value(u.km / u.s) - V_bg_q.to_value(u.km / u.s))
        mm = mm[np.isfinite(mm)]
        parker_scaled_diag = {
            "fallback_fraction": float(np.mean(bad)) if bad.size else 0.0,
            "mismatch_median_kms": float(np.nanmedian(mm)) if mm.size else float("nan"),
        }

    audit: Dict[str, Any] = {
        "requested_method": str(method_tag),
        "executed_model": str(getattr(model, "name", None)),
        "executed_class": str(type(model).__name__),
        "evaluate_fn": str(getattr(getattr(model, "evaluate", None), "__qualname__", None)),
        "profile_fn": str(getattr(getattr(model, "speed_profile", None), "__qualname__", None)),
        "model_signature": str(model_sig),
        "r_units": "R_sun (internal)",
        "phi_wrap": "[0,360) deg",
        "circ_percentile_algo": str(CIRC_PERCENTILE_ALGO_ID),
        "r_stats": r_stats,
        "profile": {
            "is_accelerating": bool(str(method_tag) in {"exp_accel", "parker_scaled", "hybrid_parker"}),
            "U_span_kms": float(diag_extra.get("U_span_kms", float("nan"))),
            "U_ss_kms": float(diag_extra.get("U_ss_kms", float("nan"))),
            "U_sc_kms": float(diag_extra.get("U_sc_kms", float("nan"))),
            "U_ss_over_U_sc": float(diag_extra.get("U_ss_over_U_sc", float("nan"))),
            "U_span_thr_kms": float(diag_extra.get("U_span_thr_kms", float("nan"))) if diag_extra.get("U_span_thr_kms", None) is not None else float("nan"),
            "is_degenerate": bool(diag_extra.get("profile_degenerate", False)),
            "degenerate_reason": diag_extra.get("degenerate_reason", None),
        },
        "tau_vs_ballistic": dict(tau_vs_ballistic),
        "parker_scaled": parker_scaled_diag or {},
    }

    audit_block = _format_audit_block(audit)
    if verbose:
        print("\n" + audit_block + "\n")

    summary_box = _summary_text(
        {
            "model": str(method_tag),
            "r_ss_Rsun": float(u.Quantity(r_ss).to_value(u.R_sun)),
            "omega_deg_per_day": float(u.Quantity(omega).to_value(u.deg / u.day)),
            "ephem_step": str(ephem_step),
            "frame3d": str(frame3d),
            "fallback_fraction": float(fallback_fraction),
            "masked_fraction": float(masked_fraction),
            "r_sc_median_Rsun": float(r_sc_med_Rsun),
            "tau_median_hr": float(np.nanmedian(data.loc[valid, "tau"])) if valid.any() else np.nan,
            "sigma_phi_p84_deg": float(np.nanpercentile(data.loc[valid, "sigma_phi"], 84.0)) if valid.any() else np.nan,
            "profile_degenerate": diag_extra.get("profile_degenerate", None),
            "profile_fallback_used": bool(diag_extra.get("profile_fallback_used", False)),
            "segmentation_plot_ok": diag_extra.get("segmentation_plot_ok", None),
            "segmentation_plot_error": diag_extra.get("segmentation_plot_error", None),
        }
    )


    out_png_ret, fig2d = plot_source_surface_2d(
        data=data,
        out_png=out_png,
        plot_vars=list(plot_vars_used),
        var_specs=specs,
        percentiles=plot_percentiles,
        size_col="marker_size",
        show_uncertainty=bool(show_uncertainty),
        summary_box=summary_box,
        title=title,
        figsize=figsize_2d,
        show=show,
        profile_panel=profile_panel,
    )

    # 3D output (optional)
    fig3d = None
    if plot_3d:
        _compute_3d_cartesian(data, r_ss=u.Quantity(r_ss), frame3d=frame3d, observer=ephem_observer)
        # Geometry must exist after Cartesian conversion
        miss_ss = [c for c in ("ss_x_au", "ss_y_au", "ss_z_au") if c not in data.columns]
        if miss_ss:
            avail = ", ".join(list(map(str, data.columns[:40]))) + (" ..." if len(data.columns) > 40 else "")
            raise ValueError(f"3D geometry (source surface): missing columns {miss_ss}. Available columns: {avail}")

        miss_sc = [c for c in ("sc_x_au", "sc_y_au", "sc_z_au") if c not in data.columns]
        if miss_sc:
            avail = ", ".join(list(map(str, data.columns[:40]))) + (" ..." if len(data.columns) > 40 else "")
            raise ValueError(f"3D geometry (spacecraft): missing columns {miss_sc}. Available columns: {avail}")
        r_sun_au = const.R_sun.to_value(u.AU)
        r_ss_au = u.Quantity(r_ss).to_value(u.AU)
        w3d, h3d = None, None
        if figsize_3d is not None:
            try:
                w3d, h3d = int(figsize_3d[0]), int(figsize_3d[1])
            except Exception:
                w3d, h3d = None, None

        out_html_ret, fig3d = plot_source_surface_3d(
            data=data,
            out_html=out_html,
            var_specs=specs,
            plot_vars=list(vars3d_used) if vars3d_used is not None else list(plot_vars_used),
            ncols_vars=int(plot_3d_ncols),
            r_ss_au=float(r_ss_au),
            r_sun_au=float(r_sun_au),
            r_sc_med_rsun=float(r_sc_med_Rsun),
            frame3d=frame3d,
            decimate=int(plot_3d_decimate),
            sc_track=sc_track_df,
            sc_track_decimate=int(plot_3d_decimate),
            percentiles=plot_percentiles,
            show_uncertainty_arcs=bool(show_uncertainty),
            # requested: links + RTN axes + camera sync across all sub-panels
            show_links=True,
            link_count=12,
            show_rtn_axes=True,
            rtn_axis_frac=0.22,
            sync_cameras=True,
            camera=str(plot_3d_camera),
            title=title3d,
            width=w3d,
            height=h3d,
            draw_panel_boxes=False,
            show_ecliptic_circles=True,
            ecliptic_circle_count=5,
            ecliptic_circle_label_units="both",
            show_ecliptic_axes=True,
            show=show,
        )


    # ------------------------------------------------------------------
    # Carrington diagnostics figure (phi drift + mapping shift sanity).
    # In a Carrington (Sun-fixed rotating) longitude, a spacecraft that is not
    # rigidly co-rotating with the Sun typically drifts by O(10 deg/day) and can
    # sweep ~360 deg on ~weeks timescales (order a Carrington rotation),
    # depending on its orbital angular rate.
    # ------------------------------------------------------------------
    carr_png = None
    if bool(plot_carrington_diag):
        try:
            carr_png = out_base / "carrington_diagnostics.png"
            plot_carrington_diagnostics(
                data=data,
                ephem_orbit=eph_i_orbit,
                omega=omega,
                phi_sign=int(phi_sign),
                out_png=carr_png,
                title=f"{title} | Carrington drift diagnostics",
                show=show,
            )
        except Exception as _e_cfig:
            carr_png = None
            meta.setdefault("diagnostics", {})
            meta["diagnostics"]["carrington_plot_error"] = str(_e_cfig)

    # ------------------------------------------------------------------
    # Segmentation diagnostic figures (only if segmentation was enabled)
    # ------------------------------------------------------------------
    seg_score_png = None
    seg_score_pdf = None
    seg_foot_png = None
    seg_foot_pdf = None
    seg_schematic_pdf = None
    seg_schematic_png = None

    if segmentation_enabled:
        try:
            diag_extra["segmentation_plot_ok"] = True
            seg_score_png = out_base / "segmentation_score.png"
            seg_score_pdf = out_base / "segmentation_score.pdf"
            plot_segmentation_score_timeseries(
                data=data,
                plot_vars=(seg_model_meta.get('used_features') if isinstance(seg_model_meta, dict) and seg_model_meta.get('used_features') else (present if present else list(data.columns))),
                threshold=float(seg_threshold_used) if seg_threshold_used is not None else float(source_fit_threshold),
                window=str(seg_window_used) if seg_window_used is not None else str(source_fit_window),
                mode=str(seg_mode_used) if seg_mode_used is not None else 'variability',
                metric=str(seg_model_meta.get('metric')) if isinstance(seg_model_meta, dict) and seg_model_meta.get('metric') is not None else 'legacy',
                ridge_alpha=float(seg_model_meta.get('ridge_alpha')) if isinstance(seg_model_meta, dict) and np.isfinite(float(seg_model_meta.get('ridge_alpha', float('nan')))) else float('nan'),
                out_png=seg_score_png,
                out_pdf=seg_score_pdf,
                show=False,
            )

            seg_foot_png = out_base / "segmentation_footpoints.png"
            seg_foot_pdf = out_base / "segmentation_footpoints.pdf"
            plot_segmentation_footpoints(
                data=data,
                out_png=seg_foot_png,
                out_pdf=seg_foot_pdf,
                show=False,
            )

            seg_schematic_pdf = out_base / "segmentation_schematic.pdf"
            seg_schematic_png = out_base / "segmentation_schematic.png"
            plot_segmentation_schematic(out_pdf=seg_schematic_pdf, out_png=seg_schematic_png)

        except Exception as _e_segplot:
            diag_extra["segmentation_plot_ok"] = False
            diag_extra["segmentation_plot_error"] = str(_e_segplot)
            seg_score_png = None
            seg_score_pdf = None
            seg_foot_png = None
            seg_foot_pdf = None
            seg_schematic_pdf = None
            seg_schematic_png = None

    # Build meta
    meta: Dict[str, Any] = {
        "sc": str(sc),
        "which_int": int(which_int),
        "cadence": str(cadence),
        "method": str(method_tag),
        "r_ss_Rsun": float(u.Quantity(r_ss).to_value(u.R_sun)),
        "omega_deg_per_day": float(u.Quantity(omega).to_value(u.deg / u.day)),
        "phi_sign": int(phi_sign),
        "ephemeris": dict(eph.meta),
        "plot": {
            "plot_vars_requested": list(plot_vars_requested),
            "plot_vars_used": list(plot_vars_used),
            "plot_vars_dropped": list(plot_vars_dropped),
            "plot_vars": list(plot_vars_used),
            "plot_percentiles": list(plot_percentiles),
            "plot_3d": bool(plot_3d),
            "plot_3d_var": (str(plot_3d_var) if plot_3d_var is not None else None) if plot_3d else None,
            "plot_3d_vars_requested": list(vars3d_requested) if plot_3d and vars3d_requested is not None else None,
            "plot_3d_vars_used": list(vars3d_used) if plot_3d and vars3d_used is not None else None,
            "plot_3d_vars_dropped": list(vars3d_dropped) if plot_3d and vars3d_dropped is not None else None,
            "plot_3d_vars": list(vars3d_used) if plot_3d and vars3d_used is not None else None,
            "plot_3d_ncols": int(plot_3d_ncols) if plot_3d else None,
            "frame3d": str(frame3d) if plot_3d else None,
            "show_uncertainty": bool(show_uncertainty),
        },
        "uncertainty_model": {
            "vr_bg_window": str(vr_bg_window),
            "vr_sigma_window": str(vr_sigma_window),
            "sigma_rel": float(sigma_rel),
            "sigma_abs_kms": float(u.Quantity(sigma_abs).to_value(u.km / u.s)),
            "v_min_kms": float(u.Quantity(v_min).to_value(u.km / u.s)),
            "v_fallback_kms": float(u.Quantity(v_fallback).to_value(u.km / u.s)),
        },
        "diagnostics": {
            "fallback_fraction": fallback_fraction,
            "masked_fraction": masked_fraction,
            "carrington": carr_diag,
            "gaps_removed": bool(remove_gaps) and (gap_keep is not None) and (gaps_padded is not None),
            "gap_pad_frac": float(gap_pad_frac),
            "n_samples_before_gaps": int(n_before) if (gap_keep is not None) else None,
            "n_samples_after_gaps": int(len(data)) if (gap_keep is not None) else None,
            "gaps_file": str(gaps_csv) if gaps_csv is not None else None,
            "r_sc_median_Rsun": float(np.nanmedian(r_sc_q[valid].to_value(u.R_sun))) if valid.any() else float(np.nanmedian(r_sc_q.to_value(u.R_sun))),
            "r_sc_median_AU": float(np.nanmedian(r_sc_q[valid].to_value(u.AU))) if valid.any() else float(np.nanmedian(r_sc_q.to_value(u.AU))),
            "tau_median_hr": float(np.nanmedian(data.loc[valid, "tau"])) if valid.any() else np.nan,
            "tau_p16_hr": float(np.nanpercentile(data.loc[valid, "tau"], 16.0)) if valid.any() else np.nan,
            "tau_p84_hr": float(np.nanpercentile(data.loc[valid, "tau"], 84.0)) if valid.any() else np.nan,
            "sigma_phi_median_deg": float(np.nanmedian(data.loc[valid, "sigma_phi"])) if valid.any() else np.nan,
            "sigma_phi_p84_deg": float(np.nanpercentile(data.loc[valid, "sigma_phi"], 84.0)) if valid.any() else np.nan,
            "profile_ok": bool(diag_extra.get("profile_ok", False)),
            "profile_error": diag_extra.get("profile_error", None),
            "profile_degenerate": diag_extra.get("profile_degenerate", None),
            "profile_fallback_used": bool(diag_extra.get("profile_fallback_used", False)),
            "segmentation_plot_ok": diag_extra.get("segmentation_plot_ok", None),
            "segmentation_plot_error": diag_extra.get("segmentation_plot_error", None),
            "U_min_kms": diag_extra.get("U_min_kms", None),
            "U_max_kms": diag_extra.get("U_max_kms", None),
            "U_span_kms": diag_extra.get("U_span_kms", None),
            "r_sc_profile_Rsun": diag_extra.get("r_sc_profile_Rsun", None),
            "audit": dict(audit),
            "audit_block": str(audit_block),
            # optional: segmentation + rs fitting diagnostics
            "source_segmentation": (tt.meta.get("diagnostics", {}) or {}).get("source_segmentation", None),
            "source_fit": (tt.meta.get("diagnostics", {}) or {}).get("source_fit", None),
        },
        "model_meta": dict(tt.meta),
        "mapping_meta": dict(map0.meta),
        "azimuthal": {
            "enabled": bool(azimuthal_correction),
            "r_A_Rsun": (float(u.Quantity(r_A).to_value(u.R_sun)) if bool(azimuthal_correction) else None),
            "n_grid": (int(az_n_grid) if bool(azimuthal_correction) else None),
            "ma2_tol": (float(az_ma2_tol) if bool(azimuthal_correction) else None),
            "delta_phi_meta": (dict(delta_phi_meta) if bool(azimuthal_correction) else {"mode": "omega_tau"}),
        },
        "units": dict(data.attrs.get("units", {})),
        "inputs": {
            "final_pkl": str(inp["paths"]["final"]),
            "sig_pkl": str(inp["paths"]["sig"]) if inp["paths"]["sig"] else None,
        },
    }

    # Storage policy
    pd.to_pickle(data, out_pkl)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True, default=str)

    files = {
        "timeseries": str(out_pkl),
        "maps_2d": str(out_png),
        "maps_3d": str(out_html) if plot_3d else None,
        "carrington_diagnostics": str(carr_png) if carr_png is not None else None,
        "velocity_profile": str(prof_png) if prof_png is not None else None,
        "segmentation_score": str(seg_score_png) if seg_score_png is not None else None,
        "segmentation_score_pdf": str(seg_score_pdf) if seg_score_pdf is not None else None,
        "segmentation_footpoints": str(seg_foot_png) if seg_foot_png is not None else None,
        "segmentation_footpoints_pdf": str(seg_foot_pdf) if seg_foot_pdf is not None else None,
        "segmentation_schematic": str(seg_schematic_pdf) if seg_schematic_pdf is not None else None,
        "segmentation_schematic_png": str(seg_schematic_png) if seg_schematic_png is not None else None,
        "meta": str(out_meta),
        "outdir": str(out_base),
        "ephemeris_cache": str(cache_file),
    }
    # User-facing report
    try:
        report_path = _write_report_md(out_base, meta=meta, files=files)
        files['report'] = str(report_path)
    except Exception as _e_rep:
        files['report'] = None
        meta.setdefault('diagnostics', {})
        meta['diagnostics']['report_error'] = str(_e_rep)

    return {"data": data, "meta": meta, "files": files, "fig2d": fig2d, "fig3d": fig3d}


# -----------------------------------------------------------------------------
# Backward-compatibility shim
# -----------------------------------------------------------------------------

def run_backmapping_interval(
    *,
    root_dir: Union[str, Path],
    sc: str,
    which_int: int,
    interval_dir: Optional[Union[str, Path]] = None,
    method: str = "ballistic_bg",
    cadence: str = "60min",
    smooth: Optional[str] = None,  # deprecated
    plot_vars: Sequence[str] = ("polarity", "Vr_bg", "P_ram"),
    plot_3d: bool = False,
    # NOTE: plot_3d_var is historically a *single* variable name.
    # Some notebooks passed a tuple/list here; we accept that for backward
    # compatibility and route it to plot_3d_vars.
    plot_3d_var: Optional[Union[str, Sequence[str]]] = None,
    plot_3d_vars: Optional[Sequence[str]] = None,
    plot_3d_ncols: int = 2,
    r_ss_rsun: float = 2.5,
    omega_deg_per_day: float = 14.1844,
    phi_sign: int = +1,
    vsw_fallback_kms: float = 400.0,
    ephem_step: str = "1h",
    ephem_observer: str = "earth",
    join: str = "inner",
    show: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Legacy entry point retained as a thin wrapper.

    Notes
    -----
    - ``smooth`` is deprecated; use ``vr_bg_window`` in ``backmap_interval``.
    - ``vsw_fallback_kms`` maps to ``v_fallback``.
    """

    if smooth is not None:
        kwargs.setdefault("vr_bg_window", str(smooth))

    # Backward-compatibility: if the user provided a list/tuple via plot_3d_var,
    # treat it as plot_3d_vars.
    if plot_3d_vars is None and isinstance(plot_3d_var, (list, tuple)):
        plot_3d_vars = [str(v) for v in plot_3d_var]
        plot_3d_var = None

    return backmap_interval(
        root_dir=root_dir,
        sc=sc,
        which_int=which_int,
        interval_dir=interval_dir,
        method=method,
        cadence=cadence,
        r_ss=float(r_ss_rsun) * u.R_sun,
        omega=float(omega_deg_per_day) * u.deg / u.day,
        phi_sign=phi_sign,
        ephem_step=ephem_step,
        ephem_observer=ephem_observer,
        plot_vars=plot_vars,
        plot_3d=plot_3d,
        plot_3d_var=str(plot_3d_var) if isinstance(plot_3d_var, str) else None,
        plot_3d_vars=plot_3d_vars,
        join=join,
        v_fallback=float(vsw_fallback_kms) * u.km / u.s,
        show=show,
        verbose=verbose,
        **kwargs,
    )

# -----------------------------------------------------------------------------
# Batch runner: select intervals by start/end datetime and run all
# -----------------------------------------------------------------------------

def _parse_interval_dir_times(name: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Parse (t0,t1) from an interval folder name.

    Expected pattern:
        YYYY-MM-DD_HH-MM-SS_YYYY-MM-DD_HH-MM-SS_sc_0

    Returns (t0,t1) as UTC tz-aware pandas Timestamps, or None.
    """
    s = str(name)
    m = re.search(
        r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})",
        s,
    )
    if m is None:
        return None
    fmt = "%Y-%m-%d_%H-%M-%S"
    try:
        t0 = pd.to_datetime(m.group(1), format=fmt, utc=True)
        t1 = pd.to_datetime(m.group(2), format=fmt, utc=True)
    except Exception:
        return None
    if pd.isna(t0) or pd.isna(t1):
        return None
    return t0, t1



def build_merged_cadence_data_from_interval_dirs(
    selected: Sequence[Mapping[str, Any]],
    *,
    cadence: str,
    plot_vars: Sequence[str],
    br_col: Optional[str] = None,
    vr_col: Optional[str] = None,
    np_col: Optional[str] = None,
    join: str = "outer",
    input_units: Optional[Mapping[str, u.Unit]] = None,
    remove_gaps: bool = True,
    gap_pad_frac: float = 0.5,
    verbose: bool = False,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DatetimeIndex]]:
    """Load multiple downloaded intervals, resample each to a single cadence grid, optionally
    remove padded MAG/PAR gaps, and concatenate into one time-ordered DataFrame.

    Notes
    -----
    - Resampling occurs *per interval* via :func:`build_cadence_dataframe`.
      No further averaging occurs in the plotting functions: each cadence bin becomes one point.
    - Gap masking is applied *after* resampling, on the cadence grid, using padded
      mag_gaps.pkl / par_gaps.pkl if present in each interval folder.
    - Concatenation is a simple vertical stack. If duplicate timestamps exist (overlaps),
      the first occurrence is kept and the rest are dropped (with a warning).
    """

    if not selected:
        raise ValueError("build_merged_cadence_data_from_interval_dirs: empty selection")

    dfs: list[pd.DataFrame] = []
    gaps_all: list[pd.DataFrame] = []
    units_all: dict[str, u.Unit] = {}
    orbit_indices: list[pd.DatetimeIndex] = []

    for item in selected:
        interval_dir = Path(item["interval_dir"])
        inp = load_interval_inputs_from_dir(interval_dir=interval_dir)
        fin = inp["fin"]
        sig = inp.get("sig", None)

        data_i = build_cadence_dataframe(
            fin=fin,
            sig=sig,
            cadence=cadence,
            plot_vars=plot_vars,
            br_col=br_col,
            vr_col=vr_col,
            np_col=np_col,
            join=join,
            input_units=input_units,
        )

        # Preserve the *pre-gap* cadence index so the spacecraft orbit can be drawn
        # continuously even if data samples are removed.
        try:
            orbit_indices.append(pd.DatetimeIndex(data_i.index))
        except Exception:
            pass

        # Optional per-interval gap masking (recommended for figures)
        if remove_gaps:
            mgp = interval_dir / "mag_gaps.pkl"
            pgp = interval_dir / "par_gaps.pkl"
            mgp_use = mgp if mgp.exists() else None
            pgp_use = pgp if pgp.exists() else None
            gaps_padded, keep = load_padded_gaps(
                mag_gaps_path=mgp_use,
                par_gaps_path=pgp_use,
                gap_pad_frac=float(gap_pad_frac),
                index=data_i.index,
            )
            if gaps_padded is not None and not gaps_padded.empty:
                g = gaps_padded.copy()
                g["interval"] = str(item.get("name", interval_dir.name))
                gaps_all.append(g)
            if keep is not None:
                data_i = data_i.loc[keep].copy()

        # Helpful provenance
        data_i["interval_name"] = str(item.get("name", interval_dir.name))
        data_i["interval_which_int"] = int(item.get("which_int", -1))

        # Merge units metadata (best-effort)
        um = dict(data_i.attrs.get("units", {}))
        for k, v in um.items():
            if k not in units_all:
                units_all[k] = u.Unit(v)

        dfs.append(data_i)

        if verbose:
            try:
                n0 = int(item.get("n_points_raw", -1))
            except Exception:
                n0 = -1
            print(f"[BACKMAP] merge: loaded {interval_dir.name} -> n={len(data_i)}")

    merged = pd.concat(dfs, axis=0).sort_index()
    merged = _to_utc_index(merged, "merged cadence data")

    if merged.index.has_duplicates:
        ndup = int(merged.index.duplicated().sum())
        warnings.warn(
            f"Merged cadence data has {ndup} duplicate timestamps (overlapping intervals). "
            "Keeping the first occurrence.",
            RuntimeWarning,
        )
        merged = merged.loc[~merged.index.duplicated(keep="first")].copy()

    # Attach merged unit map
    if units_all:
        attach_units(merged, units_all)

    # Union of cadence timestamps before gap masking (for continuous orbit rendering).
    orbit_index = None
    if orbit_indices:
        try:
            orbit_index = pd.DatetimeIndex(np.concatenate([idx.values for idx in orbit_indices]))
            orbit_index = pd.to_datetime(orbit_index, utc=True)
            orbit_index = pd.DatetimeIndex(orbit_index).sort_values().unique()
        except Exception:
            orbit_index = None

    gaps_union = None
    if gaps_all:
        gaps_union = pd.concat(gaps_all, ignore_index=True).sort_values("Start").reset_index(drop=True)

    return merged, gaps_union, orbit_index

def run_backmapping_range(
    *,
    root_dir: Union[str, Path],
    sc: str,
    start: str,
    end: str,
    selection: str = "start_in_range",
    method: str = "ballistic_bg",
    cadence: str = "60min",
    outdir: Optional[Union[str, Path]] = None,
    merge_intervals: bool = True,
    save_per_interval: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run backmapping for all downloaded intervals selected by a UTC date window.

    This selector is intentionally folder-driven (``*/final.pkl``) so it will not
    silently miss intervals if any upstream helper returns an incomplete list.

    Parameters
    ----------
    start, end
        Datetime strings parseable by pandas. Interpreted as UTC.

    selection
        - "start_in_range" (default): include if t0 in [start,end]
        - "end_in_range"            : include if t1 in [start,end]
        - "overlap"                 : include if [t0,t1] overlaps [start,end]
        - "contained"               : include if [t0,t1] fully contained in [start,end]
    """

    root_dir = Path(root_dir)
    interval_root = root_dir / "examples" / "downloaded_intervals" / str(sc)

    t_start = pd.to_datetime(start, utc=True)
    t_end = pd.to_datetime(end, utc=True)
    if pd.isna(t_start) or pd.isna(t_end):
        raise ValueError("start/end could not be parsed as datetimes")
    if t_end < t_start:
        raise ValueError("end must be >= start")

    sel = str(selection).strip().lower()
    if sel not in {"start_in_range", "end_in_range", "overlap", "contained"}:
        raise ValueError("selection must be one of: start_in_range, end_in_range, overlap, contained")

    # Candidate intervals: any folder with a final.pkl
    final_paths = sorted(interval_root.glob("*/final.pkl"))
    n_final = len(final_paths)
    if n_final == 0:
        raise FileNotFoundError(f"No final.pkl found under {interval_root}")

    # Optional: map each final.pkl to the canonical which_int used elsewhere (if available)
    which_map: Dict[Path, int] = {}
    try:
        finnames = func.load_files(interval_root, "final.pkl")
        for i, p in enumerate(finnames):
            which_map[Path(p).resolve()] = int(i)
    except Exception:
        which_map = {}

    parsed: List[Dict[str, Any]] = []
    n_unparsed = 0
    for i, fin_path in enumerate(final_paths):
        idir = fin_path.parent
        parsed_times = _parse_interval_dir_times(idir.name)
        if parsed_times is None:
            n_unparsed += 1
            continue
        t0, t1 = parsed_times
        parsed.append(
            {
                "interval_dir": str(idir),
                "name": str(idir.name),
                "final": str(fin_path),
                "t0": t0,
                "t1": t1,
                "which_int": int(which_map.get(fin_path.resolve(), int(i))),
            }
        )

    if len(parsed) == 0:
        raise ValueError(f"Could not parse any interval folder names under {interval_root}. Unparsed={n_unparsed}")

    # Apply selection
    selected: List[Dict[str, Any]] = []
    for it in parsed:
        t0 = it["t0"]
        t1 = it["t1"]

        if sel == "start_in_range":
            ok = (t0 >= t_start) and (t0 <= t_end)
        elif sel == "end_in_range":
            ok = (t1 >= t_start) and (t1 <= t_end)
        elif sel == "overlap":
            ok = (t1 >= t_start) and (t0 <= t_end)
        else:  # contained
            ok = (t0 >= t_start) and (t1 <= t_end)

        if ok:
            selected.append(it)

    selected.sort(key=lambda d: d["t0"])

    if verbose:
        tmin = min([d["t0"] for d in parsed])
        tmax = max([d["t1"] for d in parsed])
        print(
            f"[BACKMAP] range candidates: final.pkl={n_final} parsed={len(parsed)} unparsed={n_unparsed} | "
            f"folder span {tmin} -> {tmax}"
        )
        print(f"[BACKMAP] range selection: {len(selected)} intervals | selection='{sel}' | {t_start} -> {t_end}")
        if len(selected) > 0:
            n_show = min(len(selected), 20)
            for it in selected[:n_show]:
                print(f"    - which_int={int(it.get('which_int', -1))} | {it.get('name', '')} | {it.get('t0', '')} -> {it.get('t1', '')}")
            if len(selected) > n_show:
                print(f"    ... ({len(selected)-n_show} more)")



    # Output base folder
    if outdir is None:
        t0_tag = pd.to_datetime(start, utc=True).strftime("%Y-%m-%d_%H-%M-%S")
        t1_tag = pd.to_datetime(end, utc=True).strftime("%Y-%m-%d_%H-%M-%S")
        base_out = (
            root_dir
            / "examples"
            / "figures"
            / f"backmap_{sc}_range_{str(method).strip().lower()}"
            / f"range_{t0_tag}_to_{t1_tag}"
        )
    else:
        base_out = Path(outdir)
    base_out.mkdir(parents=True, exist_ok=True)

    # Common build parameters used when merging cadence data
    plot_vars_use = kwargs.get("plot_vars", ("polarity", "Vr_bg", "P_ram", "sigma_c", "Br_r2", "mass_flux"))
    br_col_use = kwargs.get("br_col", None)
    vr_col_use = kwargs.get("vr_col", None)
    np_col_use = kwargs.get("np_col", None)
    join_use = kwargs.get("join", "outer")
    input_units_use = kwargs.get("input_units", None)

    remove_gaps_use = bool(kwargs.get("remove_gaps", True))
    gap_pad_frac_use = float(kwargs.get("gap_pad_frac", 0.5))
    # Range behavior controls (handled by explicit function parameters)
    # Write selection table (always)
    try:
        sel_df = pd.DataFrame(
            [
                dict(
                    which_int=int(it.get("which_int", -1)),
                    name=str(it.get("name", "")),
                    start=str(it.get("t0", "")),
                    end=str(it.get("t1", "")),
                    interval_dir=str(it.get("interval_dir", "")),
                )
                for it in selected
            ]
        )
        sel_df.to_csv(base_out / "selected_intervals.csv", index=False)
    except Exception:
        pass

    merged_result: Optional[dict] = None
    results: list[dict] = []

    # ------------------------------------------------------------
    # Mode A: MERGE -> ONE combined set of figures
    # ------------------------------------------------------------
    if bool(merge_intervals):
        merged_data, gaps_union, orbit_index = build_merged_cadence_data_from_interval_dirs(
            selected,
            cadence=cadence,
            plot_vars=plot_vars_use,
            br_col=br_col_use,
            vr_col=vr_col_use,
            np_col=np_col_use,
            join=join_use,
            input_units=input_units_use,
            remove_gaps=remove_gaps_use,
            gap_pad_frac=gap_pad_frac_use,
            verbose=verbose,
        )

        gaps_csv = None
        if gaps_union is not None and not gaps_union.empty:
            gaps_csv = base_out / "gaps_padded_union.csv"
            try:
                gaps_union.to_csv(gaps_csv, index=False)
            except Exception:
                gaps_csv = None

        # Prevent double gap-masking inside backmap_interval (we already applied it above).
        kwargs_merged = dict(kwargs)
        for k in ("remove_gaps", "gap_pad_frac", "mag_gaps_path", "par_gaps_path", "merge_intervals", "save_per_interval"):
            kwargs_merged.pop(k, None)

        t0 = pd.to_datetime(start, utc=True)
        t1 = pd.to_datetime(end, utc=True)
        merged_label = f"range {t0.strftime('%Y-%m-%d')} to {t1.strftime('%Y-%m-%d')} | n_int={len(selected)}"

        merged_result = run_backmapping_interval(
            root_dir=root_dir,
            sc=sc,
            which_int=-1,
            interval_dir=None,
            method=method,
            cadence=cadence,
            outdir=str(base_out),
            data_override=merged_data,
            orbit_index=orbit_index,
            interval_label=merged_label,
            remove_gaps=False,
            verbose=verbose,
            **kwargs_merged,
        )
        results.append(merged_result)

    # ------------------------------------------------------------
    # Mode B: PER-INTERVAL backmapping in subfolders (optional)
    # ------------------------------------------------------------
    if bool(save_per_interval):
        per_base = base_out / "per_interval"
        per_base.mkdir(parents=True, exist_ok=True)

        for item in selected:
            wi_int = int(item["which_int"])
            sub_out = per_base / item["name"]
            if verbose:
                print(f"[BACKMAP] per-interval: interval_dir={item['name']} which_int={wi_int}")

            res = run_backmapping_interval(
                root_dir=root_dir,
                sc=sc,
                which_int=wi_int,
                interval_dir=item["interval_dir"],
                method=method,
                cadence=cadence,
                outdir=str(sub_out),
                verbose=verbose,
                **kwargs,
            )
            results.append(res)

    # Manifest (selection + outputs)
    manifest = dict(
        sc=sc,
        method=str(method),
        cadence=str(cadence),
        start=str(t_start),
        end=str(t_end),
        selection=str(selection),
        merge_intervals=bool(merge_intervals),
        save_per_interval=bool(save_per_interval),
        selected=selected,
        outputs=dict(
            base_out=str(base_out),
            merged_files=(merged_result.get("files", {}) if isinstance(merged_result, dict) else {}),
        ),
    )

    manifest_path = base_out / "manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=_json_safe)
    except Exception:
        manifest_path = None

    return {
        "selected": selected,
        "results": results,
        "merged": merged_result,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "outdir": str(base_out),
    }