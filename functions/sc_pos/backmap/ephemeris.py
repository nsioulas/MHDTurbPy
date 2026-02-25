"""sc_pos.backmap.ephemeris

Ephemeris retrieval and interpolation for heliographic Carrington coordinates.

This module is deliberately strict because ephemeris assumptions can silently
corrupt source-surface longitudes.

Key contracts
-------------
- Output is in **heliographic Carrington** lon/lat.
- Longitudes are wrapped to [0, 360).
- Time interpolation of longitudes is done on an *unwrapped* angle to avoid
  0/360 discontinuity artifacts.

Dependencies
------------
Requires sunpy + astropy (and their Horizons access). If they are missing,
we raise with an actionable error.

Determinism
-----------
We avoid online "search"/"lookup" logic. If a spacecraft name is not in the
explicit SPKID map and is not already a numeric SPKID, we raise and require the
user to pass an explicit SPKID string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd


KNOWN_SPKID: Dict[str, str] = {
    # Extend explicitly; do not guess.
    "ACE": "-92",
    "WIND": "-8",
    "IMAP": "-43",
    "SWFO-L1": "-231",
    "DSCOVR": "-78",
    "ADITYA-L1": "-156",
    "SOHO": "-21",
    "PSP": "-96",
    "SOLO": "-144",
}


def _require_sunpy() -> None:
    try:
        import sunpy  # noqa: F401
        import astropy  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: sunpy/astropy.\n"
            "Install with: pip install sunpy astropy"
        ) from e


def canonicalize_target(target: str) -> str:
    s = str(target).strip()
    if re.fullmatch(r"-?\d+", s):
        return s
    cleaned = " ".join(s.upper().replace("_", " ").split())
    compact = cleaned.replace(" ", "").replace("-", "")
    alias = {
        "ACE": "ACE",
        "WIND": "WIND",
        "IMAP": "IMAP",
        "DSCOVR": "DSCOVR",
        "DISCOVR": "DSCOVR",
        "ADITYA": "ADITYA-L1",
        "ADITYAL1": "ADITYA-L1",
        "PSP": "PSP",
        "PARKERSOLARPROBE": "PSP",
        "SOLO": "SOLO",
        "SOLARORBITER": "SOLO",
        "SOHO": "SOHO",
        "SWFOL1": "SWFO-L1",
    }
    return alias.get(compact, cleaned)


def resolve_spkid(target: str) -> str:
    t = canonicalize_target(target)
    if re.fullmatch(r"-?\d+", t):
        return t
    key = t.upper()
    if key in KNOWN_SPKID:
        return KNOWN_SPKID[key]
    raise ValueError(
        f"Unknown spacecraft target {target!r}. "
        "Pass an explicit numeric SPKID string (e.g., '-92') or extend KNOWN_SPKID explicitly."
    )


def _to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def validate_time_window(start: pd.Timestamp, stop: pd.Timestamp) -> Tuple[str, str]:
    t0 = _to_utc(start)
    t1 = _to_utc(stop)
    if t1 <= t0:
        raise ValueError(f"stop must be later than start (start={t0}, stop={t1}).")
    return (
        t0.tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
        t1.tz_localize(None).strftime("%Y-%m-%dT%H:%M:%S"),
    )


def normalize_horizons_step(step: str) -> str:
    c = str(step).strip()
    c = re.sub(r"\s+", "", c)
    c = c.replace("minutes", "min").replace("minute", "min")
    c = c.replace("hours", "h").replace("hour", "h")
    c = c.replace("days", "d").replace("day", "d")
    c = c.replace("min", "m")
    return c


@dataclass(frozen=True)
class EphemResult:
    target: str
    spkid: str
    observer: str
    step: str
    df: pd.DataFrame
    cache_file: Optional[Path]
    meta: Dict[str, Any]


def _cache_payload(df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"meta": dict(meta), "df": df.copy()}


def _load_cache(cache_path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = pd.read_pickle(cache_path)
    except Exception:
        return None
    if isinstance(obj, dict) and "df" in obj and isinstance(obj["df"], pd.DataFrame):
        return obj
    if isinstance(obj, pd.DataFrame):
        return {"meta": {"legacy": True}, "df": obj}
    return None


def _cache_covers(df: pd.DataFrame, times: pd.DatetimeIndex) -> bool:
    if df.empty:
        return False
    idx = pd.to_datetime(df.index, utc=True)
    t = pd.to_datetime(times, utc=True)
    return (idx.min() <= t.min()) and (idx.max() >= t.max())


def get_ephemeris_hgc(
    *,
    target: str,
    times: pd.DatetimeIndex,
    step: str = "1h",
    observer: str = "earth",
    cache_file: Optional[Union[str, Path]] = None,
    include_hee: bool = True,
) -> EphemResult:
    """Retrieve spacecraft ephemeris and return Carrington lon/lat/r.

    Returns a DataFrame indexed by UTC times (from Horizons sampling), with columns:

    - ``phi_sc_deg`` : Carrington longitude in degrees, wrapped to [0, 360)
    - ``lat_sc_deg`` : heliographic latitude in degrees
    - ``r_sc_au``     : heliocentric distance in AU

    Optionally also provides HEE Cartesian coordinates (AU): ``hee_x_au`` etc.

    Notes
    -----
    Carrington longitude is a Sun-fixed rotating longitude (not an
    observer-centric longitude like Stonyhurst). However, some frame
    transformations in ``sunpy`` require an explicit observer when converting
    *through* observer-dependent frames.

    In this minimal implementation we only support ``observer='earth'`` and we
    prefer a direct transformation from the Horizons coordinate to
    ``HeliographicCarrington``. A conservative fallback through
    ``HeliographicStonyhurst`` is kept for compatibility with older ``sunpy``
    versions.
    """

    _require_sunpy()
    import astropy.units as u
    from sunpy.coordinates import get_horizons_coord, get_body_heliographic_stonyhurst
    from sunpy.coordinates.frames import (
        HeliographicCarrington,
        HeliographicStonyhurst,
        HeliocentricEarthEcliptic,
    )

    times = pd.to_datetime(times, utc=True)
    if len(times) == 0:
        raise ValueError("times is empty.")
    times = pd.DatetimeIndex(times).sort_values().unique()

    step_n = normalize_horizons_step(step)
    canonical = canonicalize_target(target)
    spkid = resolve_spkid(canonical)

    obs = str(observer).lower().strip()
    if obs != "earth":
        raise ValueError("Only observer='earth' is supported in this minimal implementation.")

    cache_path = Path(cache_file) if cache_file else None
    if cache_path and cache_path.exists():
        cached = _load_cache(cache_path)
        if cached is not None:
            df = cached["df"].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df = df[~df.index.duplicated(keep="first")].sort_index()
            base_ok = {"phi_sc_deg", "lat_sc_deg", "r_sc_au"}.issubset(df.columns) and _cache_covers(df, times)
            hee_ok = (not include_hee) or {"hee_x_au", "hee_y_au", "hee_z_au"}.issubset(df.columns)
            if base_ok and hee_ok:
                meta = dict(cached.get("meta", {}))
                meta.update({"cache_hit": True})
                return EphemResult(target=str(canonical), spkid=str(spkid), observer=str(obs), step=str(step_n), df=df, cache_file=cache_path, meta=meta)

    start, stop = validate_time_window(times.min(), times.max())
    time_query = {"start": start, "stop": stop, "step": step_n}

    coord0 = get_horizons_coord(spkid, time_query)

    # Prefer a direct transformation into Carrington.
    # This avoids accidental entanglement with observer-centric frames.
    try:
        hgc = coord0.transform_to(HeliographicCarrington(obstime=coord0.obstime))
    except Exception:
        # Fallback: normalize through HGS and then into Carrington.
        # Keep the observer explicit for the HGS<->HGC conversion.
        hgs = coord0.transform_to(HeliographicStonyhurst(obstime=coord0.obstime))
        earth_obs = get_body_heliographic_stonyhurst("earth", hgs.obstime)
        hgc = hgs.transform_to(HeliographicCarrington(obstime=hgs.obstime, observer=earth_obs))

    # Distance accessor compatible across frames
    def _dist_au(c):
        if hasattr(c, "radius"):
            return c.radius.to_value(u.AU)
        if hasattr(c, "distance"):
            return c.distance.to_value(u.AU)
        return c.spherical.distance.to_value(u.AU)

    df = pd.DataFrame(
        {
            "phi_sc_deg": np.mod(hgc.lon.to_value(u.deg), 360.0),
            "lat_sc_deg": hgc.lat.to_value(u.deg),
            "r_sc_au": _dist_au(hgc),
        },
        index=pd.to_datetime(hgc.obstime.datetime64, utc=True),
    ).sort_index()

    if include_hee:
        hee = coord0.transform_to(HeliocentricEarthEcliptic(obstime=coord0.obstime))
        df["hee_x_au"] = hee.cartesian.x.to_value(u.AU)
        df["hee_y_au"] = hee.cartesian.y.to_value(u.AU)
        df["hee_z_au"] = hee.cartesian.z.to_value(u.AU)

    meta = {
        "cache_hit": False,
        "target": str(canonical),
        "spkid": str(spkid),
        "observer": str(obs),
        "step": str(step_n),
        "include_hee": bool(include_hee),
        "t_min": str(times.min()),
        "t_max": str(times.max()),
    }

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(_cache_payload(df, meta), cache_path)

    return EphemResult(target=str(canonical), spkid=str(spkid), observer=str(obs), step=str(step_n), df=df, cache_file=cache_path, meta=meta)


def interp_ephemeris_to_index(
    ephem: pd.DataFrame,
    idx: pd.DatetimeIndex,
    *,
    circular_cols: tuple[str, ...] = ("phi_sc_deg",),
) -> pd.DataFrame:
    """Time-interpolate ephemeris to a target index (UTC), unwrapping circular columns."""

    if not isinstance(ephem, pd.DataFrame) or ephem.empty:
        raise ValueError("ephem must be a non-empty DataFrame.")
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        raise ValueError("idx must be a non-empty DatetimeIndex.")

    e = ephem.copy()
    e.index = pd.to_datetime(e.index, utc=True)
    e = e[~e.index.duplicated(keep="first")].sort_index()

    t = pd.to_datetime(idx, utc=True).sort_values().unique()
    t = pd.DatetimeIndex(t)

    e2 = e.copy()

    for c in circular_cols:
        if c not in e2.columns:
            continue
        ang = pd.to_numeric(e2[c], errors="coerce").to_numpy(dtype=float)
        if (~np.isfinite(ang)).any():
            raise ValueError(f"Circular column {c!r} contains non-finite values; cannot unwrap/interpolate.")

        d_raw = np.diff(ang)
        d_circ = (d_raw + 180.0) % 360.0 - 180.0
        if d_circ.size:
            max_step = float(np.nanmax(np.abs(d_circ)))
            if max_step > 120.0:
                raise ValueError(
                    f"Ephemeris longitude steps for {c!r} are too large (max={max_step:.2f} deg). "
                    "Use a finer ephem_step (e.g., '1h' or '30m')."
                )

        e2[c] = np.rad2deg(np.unwrap(np.deg2rad(ang)))

    out = e2.reindex(e2.index.union(t)).sort_index().interpolate(method="time").reindex(t)
    out.index = t

    for c in circular_cols:
        if c in out.columns:
            out[c] = np.mod(pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float), 360.0)

    return out
