# -*- coding: utf-8 -*-
"""sc_pos.backmap.pfss

Standalone PFSS utilities (no external repos).

What this module does
---------------------
- Download (or load) a GONG MRZQS synoptic magnetogram for a given date.
- Run a PFSS extrapolation (pfsspy) from that synoptic boundary.
- Extract Br on either:
    - the photosphere (boundary map)
    - the source surface (PFSS output map)
- Cache extracted 2D Br maps to disk (NPZ) for fast time-series use.
- Provide geometry utilities for:
    - sampling Br at backmapped lon/lat
    - estimating distance to the source-surface neutral line (HCS proxy)
- Provide optional Plotly 3D sphere visualization.

Key design choice
-----------------
This module is deliberately **backmap-local**: it does not depend on any "toy" repo.
It relies only on the standard PFSS stack:
    - pfsspy
    - sunpy
    - astropy

Download strategy
-----------------
We use SunPy's `Fido` + `GONGClient` to locate and fetch GONG synoptic products.
This is more robust than hardcoding URLs and matches SunPy's supported interface.

Notes / limitations
-------------------
- PFSS is a potential-field model; the HCS proxy is the Br=0 neutral line at the
  source surface.
- Synoptic inputs are not instantaneous global maps; interpret the PFSS neutral
  line as a context proxy, not a ground-truth current sheet.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import hashlib


def robust_symmetric_clim(
    arrays: Sequence[np.ndarray],
    *,
    percentiles: Tuple[float, float] = (2.0, 98.0),
    fallback: float = 1.0,
) -> Tuple[float, float]:
    """Return a symmetric (cmin,cmax) for diverging Br visualizations.

    The goal is *cross-plot consistency* (static PNG/HTML + MP4 + Plotly animations):
    use a single robust scale so that polarity does not appear to flip between frames
    due to autoscaling.

    Parameters
    ----------
    arrays:
        Iterable of 2D (or 1D) arrays containing Br-like values.

    percentiles:
        Robust percentiles used to ignore outliers. The returned limits are symmetric
        about zero with magnitude max(|p_lo|, |p_hi|).

    fallback:
        Used if all arrays are empty or non-finite.

    Returns
    -------
    (cmin, cmax)
        Symmetric limits about zero.
    """
    lo_p, hi_p = float(percentiles[0]), float(percentiles[1])
    vv_list = []
    for a in arrays:
        try:
            x = np.asarray(a, dtype=float)
            x = x[np.isfinite(x)]
            if x.size:
                vv_list.append(x)
        except Exception:
            continue
    if not vv_list:
        mm = float(abs(fallback)) if np.isfinite(fallback) and float(fallback) > 0 else 1.0
        return (-mm, +mm)

    vv = np.concatenate(vv_list)
    lo, hi = np.nanpercentile(vv, [lo_p, hi_p])
    mm = float(max(abs(float(lo)), abs(float(hi))))
    if not np.isfinite(mm) or mm <= 0.0:
        mm = float(abs(fallback)) if np.isfinite(fallback) and float(fallback) > 0 else 1.0
    return (-mm, +mm)



# ============================================================
# Config objects
# ============================================================


@dataclass(frozen=True)
class PFSSConfig:
    """Configuration for PFSS background generation.

    Parameters
    ----------
    out_dir:
        Directory for downloads + cached derived maps.

    date_str:
        Date string in "YYYY-MM-DD" used to select the synoptic magnetogram.

    prefer_hhmm:
        Preferred UT time code ("HHMM") when multiple synoptic maps exist.
        We choose the available map whose *start time* is closest to this.

    overwrite_download:
        If True, force re-download / re-fetch.

    N:
        Target *latitudinal* resolution for the PFSS input map.
        Longitudinal resolution defaults to 2*N unless `nlon` is provided.

    nlon:
        Optional longitudinal resolution. If None, uses 2*N.

    rss_rsun:
        Source surface radius in units of R_sun.

    nr:
        PFSS radial grid points (pfsspy uses `nrho` or equivalent). If None, uses N.

    enforce_flux_balance:
        If True, subtract the mean of Br from the input map (removes monopole).
        This is common in PFSS workflows; keep it explicit.

    local_path:
        If provided, skip download and use this local file (fits or fits.gz).

    magnetogram_id:
        Optional identifier to disambiguate synoptic inputs that share the same date. If not provided,
        the code derives a deterministic identifier from the resolved local magnetogram file (size+mtime).

    search_days:
        If a PFSS boundary file cannot be retrieved for `date_str`, search
        within +/- search_days for the nearest available day. Set to 0 to
        disable this fallback window.
    """

    out_dir: Path = Path("./pfss_out")
    date_str: str = "2024-10-05"

    prefer_hhmm: str = "1204"
    overwrite_download: bool = False

    N: int = 180
    nlon: Optional[int] = None

    rss_rsun: float = 2.5
    nr: Optional[int] = None

    enforce_flux_balance: bool = True
    local_path: Optional[Path] = None
    magnetogram_id: Optional[str] = None

    search_days: int = 7



def _magnetogram_id_from_file(p: Optional[Path]) -> str:
    """Deterministic magnetogram identifier from a local file.

    We avoid hashing the full file contents (can be large) and instead use a stable
    fingerprint from resolved path + size + mtime_ns. This is sufficient to prevent
    accidental cache collisions across different local synoptic inputs.

    Returns an empty string if the path is missing or stat() fails.
    """
    if p is None:
        return ""
    try:
        pp = Path(p).expanduser().resolve()
        st = pp.stat()
        s = f"{pp}|{st.st_size}|{getattr(st, 'st_mtime_ns', int(st.st_mtime*1e9))}"
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return ""


def _as_path_list(files: Any) -> Sequence[Path]:
    """Coerce a SunPy/Parfive fetch return into a list of local file Paths.

    SunPy's `Fido.fetch(...)` often returns a `parfive.Results` object. Depending on versions,
    iterating that object can yield strings, Paths, or small Result objects.

    This helper tries multiple strategies:
      1) direct string/Path/list/tuple
      2) common attributes on Results objects (.files, .downloaded_files, .paths, .file_paths)
      3) iteration over the object
    and then coercion of each element via common fields (.path/.filepath/.local_path/.file_path/.filename)
    before falling back to str(x).

    It also handles a "file://" prefix if present.
    """
    if files is None:
        return []

    def _coerce_one(x: Any) -> Optional[Path]:
        if x is None:
            return None
        if isinstance(x, Path):
            return x.expanduser().resolve()
        if isinstance(x, str):
            s = x
        else:
            s = ""
            for attr in ("path", "filepath", "local_path", "file_path", "filename", "name"):
                try:
                    v = getattr(x, attr, None)
                except Exception:
                    v = None
                if v:
                    s = str(v)
                    break
            if not s:
                s = str(x)

        s = s.strip()
        if not s:
            return None
        if s.startswith("file://"):
            s = s[7:]
        # Extract a path-like token from wrapper strings if present.
        try:
            import re as _re
            m_win = _re.search(r"([A-Za-z]:\\[^\n\r'\"()]+?\.fits(?:\.gz)?)", s)
            m_pos = _re.search(r"(/[^\n\r'\"()]+?\.fits(?:\.gz)?)", s)
            m_any = _re.search(r"([^\s'\"()]+?\.fits(?:\.gz)?)", s)
            cand = None
            for _m in (m_win, m_pos, m_any):
                if _m is not None:
                    cand = _m.group(1)
                    break
            if cand:
                s = cand.strip()
                if s.startswith("file://"):
                    s = s[7:]
        except Exception:
            pass

        p = Path(s).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        return p

    if isinstance(files, (str, Path)):
        p = _coerce_one(files)
        return [p] if p is not None else []
    if isinstance(files, (list, tuple)):
        out = []
        for it in files:
            p = _coerce_one(it)
            if p is not None:
                out.append(p)
        return out

    for attr in ("files", "downloaded_files", "paths", "file_paths"):
        try:
            seq = getattr(files, attr, None)
        except Exception:
            seq = None
        if seq:
            out = []
            try:
                for it in list(seq):
                    p = _coerce_one(it)
                    if p is not None:
                        out.append(p)
            except Exception:
                out = []
            if out:
                return out

    try:
        out = []
        for it in list(files):
            p = _coerce_one(it)
            if p is not None:
                out.append(p)
        return out
    except Exception:
        return []


@dataclass(frozen=True)
class SpherePlotConfig:
    r_sphere: float = 1.0
    close_lon: bool = True

    colorscale: str = "RdBu"
    opacity: float = 1.0
    showscale: bool = True
    cbar_title: str = "Br"

    clim: Optional[Tuple[float, float]] = None

    show_axes: bool = False
    title: str = "PFSS Br on sphere"

    width: int = 980
    height: int = 820
    margin: Tuple[int, int, int, int] = (0, 0, 45, 0)  # (l, r, t, b)


@dataclass(frozen=True)
class Points3DConfig:
    enabled: bool = True
    # If project_to_sphere=True, points are drawn on the PFSS sphere surface (r = SpherePlotConfig.r_sphere),
    # regardless of the `r` value. This avoids "floating" points and keeps the 3D context unambiguous.
    project_to_sphere: bool = True
    r: float = 1.0
    size: int = 3
    symbol: str = "square"
    name: str = "points"
    colorscale: str = "Viridis"
    showscale: bool = False
    edge_rgba: str = "rgba(0,0,0,0.75)"
    edge_width: float = 1.0


# ============================================================
# Lazy imports (keep module import cheap)
# ============================================================


@lru_cache(maxsize=1)
def _import_pfss_stack() -> Dict[str, Any]:
    """Import pfsspy/sunpy/astropy pieces lazily."""
    import astropy.units as u  # noqa: WPS433
    import pfsspy  # noqa: WPS433
    import sunpy.map  # noqa: WPS433
    from sunpy.net import Fido, attrs as a  # noqa: WPS433

    return {"u": u, "pfsspy": pfsspy, "sunpy_map": sunpy.map, "Fido": Fido, "a": a}


def pfss_available() -> bool:
    try:
        _import_pfss_stack()
        return True
    except Exception:
        return False


# ============================================================
# Download + load
# ============================================================


def _parse_hhmm(hhmm: str) -> int:
    s = str(hhmm).strip()
    if len(s) != 4 or (not s.isdigit()):
        raise ValueError("prefer_hhmm must be a 4-digit string 'HHMM'")
    hh = int(s[:2])
    mm = int(s[2:])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError("prefer_hhmm must encode a valid time")
    return hh * 60 + mm


def download_gong_synoptic(cfg: PFSSConfig) -> Path:
    """Download a GONG synoptic magnetogram for cfg.date_str.

    Uses SunPy Fido + GONGClient. Returns a local file path.

    If cfg.local_path is provided, this returns it directly.
    """
    if cfg.local_path is not None:
        p = Path(cfg.local_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"PFSSConfig.local_path does not exist: {p}")
        return p

    tb = _import_pfss_stack()
    Fido, a = tb["Fido"], tb["a"]

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _local_candidates_for_window(date_str: str) -> list:
        """Find local GONG-like synoptic files in out_dir (recursive) within +/- search_days.

        This is a network-free fallback used when SunPy/Fido cannot reach GONG
        (e.g. DNS/proxy restrictions) or when users run in offline environments.
        """
        import re as _re
        import pandas as _pd

        target = _pd.Timestamp(date_str)
        max_d = int(max(0, int(cfg.search_days)))
        try:
            target_min = _parse_hhmm(cfg.prefer_hhmm)
        except Exception:
            target_min = None

        # Allow nested cache layouts (some users keep downloaded FITS in subfolders).
        cand = []
        for pat in ("*.fits", "*.fits.gz"):
            try:
                cand.extend(list(out_dir.rglob(pat)))
            except Exception:
                cand.extend(list(out_dir.glob(pat)))

        out = []
        for p in cand:
            name = p.name

            # Date parsing: look for YYYY-MM-DD or YYYYMMDD tokens.
            d = None
            m = _re.search(r"(20\d{2}-\d{2}-\d{2})", name)
            if m is not None:
                try:
                    d = _pd.Timestamp(m.group(1))
                except Exception:
                    d = None
            if d is None:
                m2 = _re.search(r"(20\d{2}[01]\d[0-3]\d)", name)
                if m2 is not None:
                    try:
                        d = _pd.Timestamp(m2.group(1), format="%Y%m%d")
                    except Exception:
                        d = None
            if d is None:
                # Last resort: direct token check against requested date.
                tok = date_str.replace("-", "")
                if (date_str in name) or (tok in name):
                    d = target

            if d is None:
                continue

            dd = int(abs((d - target).days))
            if dd > max_d:
                continue

            # Time parsing: ...tHHMM... (optional)
            td = 10**9
            mt = _re.search(r"t(\d{4})", name)
            if (mt is not None) and (target_min is not None):
                try:
                    td = abs(_parse_hhmm(mt.group(1)) - target_min)
                except Exception:
                    td = 10**9

            out.append((dd, td, str(p)))

        out.sort(key=lambda x: (x[0], x[1], x[2]))
        return [Path(x[2]).expanduser().resolve() for x in out]

    def _try_one_day(date_str: str) -> Optional[Path]:
        """Return a local file for one day, or None if unavailable."""

        # Local-only reuse path (recursive + +/- search_days). This avoids any network calls.
        if not bool(cfg.overwrite_download):
            try:
                cands = _local_candidates_for_window(date_str)
                if cands:
                    use = cands[0]
                    if use.exists():
                        return use
            except Exception:
                pass

        t0 = f"{date_str} 00:00"
        t1 = f"{date_str} 23:59"
        try:
            res = Fido.search(a.Time(t0, t1), a.Instrument("GONG"))
        except Exception:
            # Offline / DNS / proxy issues.
            return None
        if len(res) == 0:
            return None

        tbl = res[0]
        target_min = _parse_hhmm(cfg.prefer_hhmm)

        try:
            st = tbl["Start Time"]
            if hasattr(st, "to_datetime"):
                dt = st.to_datetime()
            else:
                dt = np.array([getattr(x, "to_datetime", lambda: None)() for x in st], dtype=object)
                if any(x is None for x in dt):
                    raise TypeError
            mins = np.array([int(x.hour) * 60 + int(x.minute) for x in dt], dtype=int)
            idx = int(np.argmin(np.abs(mins - target_min)))
            sub = tbl[idx: idx + 1]
            fetched = Fido.fetch(sub, path=str(out_dir / "{file}"), overwrite=bool(cfg.overwrite_download))
            files = [p for p in _as_path_list(fetched) if p is not None and Path(p).exists()]
            if files:
                return files[0]
            raise RuntimeError("empty fetch")
        except Exception:
            fetched = Fido.fetch(res, path=str(out_dir / "{file}"), overwrite=bool(cfg.overwrite_download))
            files = [p for p in _as_path_list(fetched) if p is not None and Path(p).exists()]
            if not files:
                return None

            import re

            def _score(p: Path) -> int:
                m = re.search(r"t(\d{4})", p.name)
                if m is None:
                    return 10**9
                try:
                    mm = _parse_hhmm(m.group(1))
                    return abs(mm - target_min)
                except Exception:
                    return 10**9

            files.sort(key=_score)
            return files[0]

    p0 = _try_one_day(cfg.date_str)
    if p0 is not None:
        return p0

    max_d = int(max(0, int(cfg.search_days)))
    if max_d <= 0:
        raise RuntimeError(f"GONG download failed for {cfg.date_str}")

    import pandas as pd

    t_ref = pd.Timestamp(cfg.date_str)
    for k in range(1, max_d + 1):
        for sgn in (-1, +1):
            dstr = (t_ref + pd.Timedelta(days=int(sgn * k))).strftime("%Y-%m-%d")
            p = _try_one_day(dstr)
            if p is not None:
                try:
                    logp = out_dir / "pfss_fallback_log.txt"
                    with logp.open("a", encoding="utf-8") as f:
                        f.write(f"requested={cfg.date_str} -> used={dstr} file={p.name}\n")
                except Exception:
                    pass
                return p

    # One last attempt: if networking is down but there exist local files in the window,
    # return the best local candidate.
    try:
        cands = _local_candidates_for_window(cfg.date_str)
        if cands:
            use = cands[0]
            if use.exists():
                return use
    except Exception:
        pass

    raise RuntimeError(
        f"GONG synoptic magnetogram could not be resolved for date={cfg.date_str}. "
        "This typically occurs when the machine has no working network/DNS (SunPy Fido cannot reach GONG) "
        "and no local magnetogram exists in pfss_config['out_dir']. "
        "Fix options: (i) enable internet/DNS/proxy, or (ii) set pfss_config['local_path'] to a local GONG FITS/FITS.GZ."
    )


def load_gong_map(cfg: PFSSConfig, local_path: Path) -> Any:
    """Load a synoptic magnetogram as a SunPy Map."""
    tb = _import_pfss_stack()
    sunpy_map = tb["sunpy_map"]

    m = sunpy_map.Map(str(local_path))
    return m


def resample_map(cfg: PFSSConfig, gong_map: Any) -> Any:
    """Resample the SunPy map to a desired (nlon, nlat) resolution."""
    tb = _import_pfss_stack()
    u = tb["u"]

    nlat = int(cfg.N)
    nlon = int(cfg.nlon) if cfg.nlon is not None else int(2 * nlat)

    new_dims = u.Quantity([nlon, nlat], u.pixel)
    try:
        m2 = gong_map.resample(new_dims)
    except Exception:
        m2 = gong_map.resample(tuple(new_dims))

    if bool(cfg.enforce_flux_balance):
        data = np.asarray(m2.data, dtype=float)

        # Enforce zero net flux with an area-weighted mean (weights ∝ cos(lat)).
        # Using an unweighted mean on a lon/lat grid leaves a residual monopole.
        lat_ax = None
        try:
            axes = _sunpy_lonlat_axes_deg(m2)
            if axes is not None:
                lat_ax = axes[1]
        except Exception:
            lat_ax = None

        if lat_ax is None:
            lat_ax = np.linspace(-90.0, 90.0, int(data.shape[0]), endpoint=True, dtype=float)

        w = np.cos(np.deg2rad(np.asarray(lat_ax, dtype=float))).reshape(-1, 1)
        ok = np.isfinite(data)
        den = np.nansum(w * ok)
        if den > 0.0:
            mean = np.nansum(data * w * ok) / den
        else:
            mean = np.nanmean(data)

        data = data - float(mean)
        m2 = type(m2)(data, m2.meta)

    return m2


# ============================================================
# PFSS solve
# ============================================================


class PFSSRun:
    """A thin wrapper to keep our extraction API stable."""

    def __init__(self, *, gong_map: Any, pfss_in: Any, pfss_out: Any):
        self.gong_map = gong_map
        self.pfss_in = pfss_in
        self.output = pfss_out
        self.br_photosphere = np.asarray(getattr(gong_map, "data", gong_map), dtype=float)


def run_pfss(cfg: PFSSConfig, gong_map: Any) -> PFSSRun:
    """Run PFSS with pfsspy from a SunPy synoptic map."""
    tb = _import_pfss_stack()
    pfsspy = tb["pfsspy"]

    nr = int(cfg.nr) if cfg.nr is not None else int(cfg.N)
    pfss_in = pfsspy.Input(gong_map, nr, float(cfg.rss_rsun))
    pfss_out = pfsspy.pfss(pfss_in)
    return PFSSRun(gong_map=gong_map, pfss_in=pfss_in, pfss_out=pfss_out)


def pfss_from_gong(cfg: PFSSConfig) -> Dict[str, Any]:
    """One-shot PFSS pipeline from GONG synoptic to PFSS output."""
    local_path = download_gong_synoptic(cfg)
    gong_map = load_gong_map(cfg, local_path)
    gong_mapN = resample_map(cfg, gong_map)
    run = run_pfss(cfg, gong_mapN)

    return {
        "pfss": run,
        "brN": np.asarray(run.br_photosphere, dtype=float),
        "hdrN": dict(getattr(run.gong_map, "meta", {}) or {}),
        "local_path": str(local_path),
        "date_str": str(cfg.date_str),
        "nlat": int(cfg.N),
        "nlon": int(cfg.nlon) if cfg.nlon is not None else int(2 * int(cfg.N)),
    }


# ============================================================
# Extract Br map to plot
# ============================================================


def _as_2d_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.ndim == 2:
        return x
    try:
        d = getattr(x, "data", None)
        if d is not None:
            a = np.asarray(d)
            if a.ndim == 2:
                return a
    except Exception:
        pass
    try:
        a = np.asarray(x)
        if a.ndim == 2:
            return a
    except Exception:
        return None
    return None


def extract_br_map(pfss: Any, brN: np.ndarray, which: str = "photosphere") -> np.ndarray:
    """Extract a 2D Br map.

    - photosphere: returns the boundary map (input)
    - source_surface: returns PFSS output source-surface Br if available

    Falls back to brN if needed.
    """
    w = str(which).strip().lower()

    if w == "photosphere":
        br = getattr(pfss, "br_photosphere", None)
        a = _as_2d_array(br)
        return np.asarray(a if a is not None else brN, dtype=float)

    if w in ("source_surface", "ss", "rss"):
        out = getattr(pfss, "output", None)
        if out is not None:
            for name in ("source_surface_br", "br_source_surface", "brss"):
                a = _as_2d_array(getattr(out, name, None))
                if a is not None:
                    return np.asarray(a, dtype=float)

            bc = getattr(out, "bc", None)
            if bc is not None:
                try:
                    b0 = np.asarray(bc[0])
                    if b0.ndim == 3:
                        ss = b0[:, :, -1]
                        return np.asarray(ss.T, dtype=float)
                except Exception:
                    pass

        brp = getattr(pfss, "br_photosphere", None)
        a = _as_2d_array(brp)
        return np.asarray(a if a is not None else brN, dtype=float)

    raise ValueError("which must be 'photosphere' or 'source_surface'.")


# ============================================================
# Cacheable PFSS-derived Br maps + neutral-line geometry
# ============================================================


def _pfss_cache_key(cfg: PFSSConfig, *, nlat: int, nlon: int) -> str:
    """Stable cache key for PFSS-derived Br maps."""
    import hashlib
    import json

    lp = ""
    try:
        if cfg.local_path is not None:
            lp = str(Path(cfg.local_path).expanduser().resolve())
    except Exception:
        lp = str(cfg.local_path) if cfg.local_path is not None else ""

    meta = dict(
        date_str=str(cfg.date_str),
        prefer_hhmm=str(cfg.prefer_hhmm),
        nlat=int(nlat),
        nlon=int(nlon),
        rss_rsun=float(cfg.rss_rsun),
        nr=(int(cfg.nr) if cfg.nr is not None else None),
        enforce_flux_balance=bool(cfg.enforce_flux_balance),
        local_path=str(lp),
        magnetogram_id=str(cfg.magnetogram_id or _magnetogram_id_from_file(Path(lp) if lp else None)),
    )
    s = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(s).hexdigest()[:12]


def _sunpy_lonlat_axes_deg(m: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return 1D (lon, lat) axes in degrees from a SunPy Map via WCS."""
    try:
        tb = _import_pfss_stack()
        u = tb["u"]
    except Exception:
        return None

    if m is None or (not hasattr(m, "pixel_to_world")) or (not hasattr(m, "data")):
        return None

    try:
        a = np.asarray(m.data, dtype=float)
        if a.ndim != 2:
            return None
        nlat, nlon = a.shape

        x = (np.arange(nlon, dtype=float) + 0.5) * u.pixel
        ymid = (np.full(nlon, 0.5 * (nlat - 1), dtype=float) + 0.5) * u.pixel
        c1 = m.pixel_to_world(x, ymid)
        lon = np.mod(np.asarray(c1.spherical.lon.to_value(u.deg), dtype=float), 360.0)

        xmid = (np.full(nlat, 0.5 * (nlon - 1), dtype=float) + 0.5) * u.pixel
        y = (np.arange(nlat, dtype=float) + 0.5) * u.pixel
        c2 = m.pixel_to_world(xmid, y)
        lat = np.asarray(c2.spherical.lat.to_value(u.deg), dtype=float)

        if lon.size != nlon or lat.size != nlat:
            return None
        if not (np.isfinite(lon).any() and np.isfinite(lat).any()):
            return None

        return lon, lat
    except Exception:
        return None


def _regrid_to_uniform_lonlat(
    br2d: np.ndarray,
    *,
    lon_axis_deg: Optional[np.ndarray],
    lat_axis_deg: Optional[np.ndarray],
    nlat: int,
    nlon: int,
) -> np.ndarray:
    """Regrid a Br map onto a uniform lon/lat grid in degrees."""
    br = np.asarray(br2d, dtype=float)
    if br.ndim != 2:
        raise ValueError("br2d must be 2D")
    if br.shape != (int(nlat), int(nlon)):
        raise ValueError(f"Unexpected Br shape {br.shape}; expected {(int(nlat), int(nlon))}.")

    if lon_axis_deg is None or lat_axis_deg is None:
        return br

    lon = np.asarray(lon_axis_deg, dtype=float).reshape(-1)
    lat = np.asarray(lat_axis_deg, dtype=float).reshape(-1)
    if lon.size != int(nlon) or lat.size != int(nlat):
        return br

    lon_u = np.linspace(0.0, 360.0, int(nlon), endpoint=False, dtype=float)
    lat_u = np.linspace(-90.0, 90.0, int(nlat), endpoint=True, dtype=float)

    br_work = br
    if np.nanmean(np.diff(lat)) < 0:
        lat = lat[::-1]
        br_work = br_work[::-1, :]

    lon_w = np.mod(lon, 360.0)
    order = np.argsort(lon_w)
    lon_w = lon_w[order]
    br_work = br_work[:, order]

    lon_ext = np.r_[lon_w, lon_w[0] + 360.0]
    br_ext = np.c_[br_work, br_work[:, :1]]

    tmp = np.full((int(nlat), int(nlon)), np.nan, dtype=float)
    for i in range(int(nlat)):
        row = br_ext[i, :]
        ok = np.isfinite(row) & np.isfinite(lon_ext)
        if int(np.sum(ok)) < 2:
            continue
        tmp[i, :] = np.interp(lon_u, lon_ext[ok], row[ok])

    out = np.full_like(tmp, np.nan, dtype=float)
    for j in range(int(nlon)):
        col = tmp[:, j]
        ok = np.isfinite(col) & np.isfinite(lat)
        if int(np.sum(ok)) < 2:
            continue
        out[:, j] = np.interp(lat_u, lat[ok], col[ok])

    if not np.isfinite(out).any():
        return br
    return out


def _map_obj_for_br(pfss_run: Any, *, which: str) -> Any:
    """Return the SunPy Map object corresponding to the requested Br surface."""
    w = str(which).strip().lower()
    if w == "photosphere":
        return getattr(pfss_run, "gong_map", None)
    if w in ("source_surface", "ss", "rss"):
        out = getattr(pfss_run, "output", None)
        if out is None:
            return None
        for nm in ("source_surface_br", "br_source_surface", "brss"):
            m = getattr(out, nm, None)
            if m is not None:
                return m
        return None
    return None


def pfss_maps_cached(
    cfg: PFSSConfig,
    *,
    which: Tuple[str, ...] = ("source_surface",),
    cache: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Return PFSS-derived Br maps with a deterministic on-disk cache (NPZ).

    Determinism contract
    --------------------
    The cache key is computed from:
      - the PFSS configuration (grid size, rss, nr, flux-balance option, etc.)
      - the *resolved* synoptic magnetogram identity (magnetogram_id)

    This prevents silent cache collisions when different synoptic inputs exist for the
    same date or when SunPy falls back to nearby availability windows.

    Legacy-cache reuse
    ------------------
    Older cache files can be reused only if their embedded metadata matches the
    requested configuration. If the request is ambiguous (e.g. multiple cache files
    match but you did not specify local_path or magnetogram_id), this function raises
    rather than guessing.
    """

    which_l = tuple([str(w).strip().lower() for w in which])
    for w in which_l:
        if w not in {"photosphere", "source_surface"}:
            raise ValueError("which must contain only 'photosphere' and/or 'source_surface'")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nlat = int(cfg.N)
    nlon = int(cfg.nlon) if cfg.nlon is not None else int(2 * nlat)
    date_tag = str(cfg.date_str).replace("-", "")

    def _load_npz(pth: Path) -> Optional[Dict[str, Any]]:
        if not pth.exists():
            return None
        try:
            d = np.load(pth, allow_pickle=False)
        except Exception:
            return None

        out: Dict[str, Any] = {"cache_file": str(pth), "date_str": str(cfg.date_str)}

        # Optional cached neutral line (source surface)
        if ("neutral_lon_source_surface" in d) and ("neutral_lat_source_surface" in d):
            out["neutral_lon_source_surface"] = np.asarray(d["neutral_lon_source_surface"], dtype=float)
            out["neutral_lat_source_surface"] = np.asarray(d["neutral_lat_source_surface"], dtype=float)

        for w0 in which_l:
            k = f"br_{w0}"
            if k not in d:
                return None
            out[w0] = np.asarray(d[k], dtype=float)

        # Attach meta if present (useful for debugging)
        if "meta" in d:
            try:
                out["meta"] = str(np.asarray(d["meta"], dtype=str))
            except Exception:
                pass

        return out

    def _read_meta(pth: Path) -> Optional[Dict[str, Any]]:
        try:
            d = np.load(pth, allow_pickle=False)
        except Exception:
            return None
        if "meta" not in d:
            return None
        try:
            import json as _json
            meta_s = str(np.asarray(d["meta"], dtype=str))
            if not meta_s:
                return None
            return _json.loads(meta_s)
        except Exception:
            return None

    def _resolve_local_gong_file_for_date() -> Optional[Path]:
        """Resolve an already-downloaded GONG synoptic file for cfg.date_str (no network)."""
        if cfg.local_path is not None:
            try:
                pp = Path(cfg.local_path).expanduser().resolve()
            except Exception:
                pp = Path(cfg.local_path)
            return pp if pp.exists() else None

        try:
            cand = []
            for ext in ("*.fits", "*.fits.gz"):
                cand.extend(list(out_dir.glob(ext)))
            if not cand:
                return None

            date_str = str(cfg.date_str)
            date_token = date_str.replace("-", "")
            cand_date = [p for p in cand if (date_str in p.name) or (date_token in p.name)]
            if not cand_date:
                return None

            # Prefer the file with Start Time closest to prefer_hhmm, if encoded.
            try:
                import re as _re
                target_min = _parse_hhmm(cfg.prefer_hhmm)

                def _score(pp: Path) -> int:
                    m = _re.search(r"t(\d{4})", pp.name)
                    if m is None:
                        return 10**9
                    try:
                        mm = _parse_hhmm(m.group(1))
                        return abs(mm - target_min)
                    except Exception:
                        return 10**9

                cand_date = sorted(cand_date, key=_score)
            except Exception:
                cand_date = sorted(cand_date)

            use = Path(cand_date[0]).expanduser().resolve()
            return use if use.exists() else None
        except Exception:
            return None

    # -----------------------
    # 1) Fast path: load a uniquely matching cache without downloading/solving.
    # -----------------------
    if cache and (not overwrite):
        # If the caller explicitly pins the magnetogram via local_path or magnetogram_id,
        # we can compute the exact cache filename.
        eff_local = None
        if cfg.local_path is not None:
            eff_local = _resolve_local_gong_file_for_date()

        eff_mag_id = str(cfg.magnetogram_id or _magnetogram_id_from_file(eff_local))
        if (cfg.local_path is not None) or (cfg.magnetogram_id is not None) or (eff_local is not None and eff_mag_id):
            # Build the exact key from the resolved identity.
            import hashlib, json as _json
            meta_key = dict(
                date_str=str(cfg.date_str),
                prefer_hhmm=str(cfg.prefer_hhmm),
                nlat=int(nlat),
                nlon=int(nlon),
                rss_rsun=float(cfg.rss_rsun),
                nr=(int(cfg.nr) if cfg.nr is not None else None),
                enforce_flux_balance=bool(cfg.enforce_flux_balance),
                local_path=str(eff_local) if eff_local is not None else str(Path(cfg.local_path).expanduser().resolve()) if cfg.local_path is not None else "",
                magnetogram_id=str(eff_mag_id),
            )
            key = hashlib.sha1(_json.dumps(meta_key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
            cache_file = out_dir / f"pfss_maps_{date_tag}_{key}.npz"
            out0 = _load_npz(cache_file)
            if out0 is not None:
                return out0

        # Otherwise, scan cache files for a match on the *configuration*.
        base_want = dict(
            date_str=str(cfg.date_str),
            prefer_hhmm=str(cfg.prefer_hhmm),
            nlat=int(nlat),
            nlon=int(nlon),
            rss_rsun=float(cfg.rss_rsun),
            nr=(int(cfg.nr) if cfg.nr is not None else None),
            enforce_flux_balance=bool(cfg.enforce_flux_balance),
        )

        matches = []
        for pth in sorted(out_dir.glob(f"pfss_maps_{date_tag}_*.npz")):
            meta = _read_meta(pth)
            if not meta:
                continue
            ok = True
            for k, v in base_want.items():
                if meta.get(k, None) != v:
                    ok = False
                    break
            if ok:
                matches.append(pth)

        if len(matches) == 1:
            out0 = _load_npz(matches[0])
            if out0 is not None:
                return out0
        if len(matches) > 1:
            raise ValueError(
                "Ambiguous PFSS cache selection: multiple cache files match the requested PFSS configuration "
                "but no unique synoptic magnetogram was specified. "
                "Specify pfss_config['local_path'] or PFSSConfig.magnetogram_id to disambiguate. "
                f"Matches={[m.name for m in matches]}"
            )

    # -----------------------
    # 2) Need to compute: resolve magnetogram -> run PFSS -> cache.
    # -----------------------
    eff_local = _resolve_local_gong_file_for_date()
    if eff_local is None:
        eff_local = download_gong_synoptic(cfg)

    eff_mag_id = str(cfg.magnetogram_id or _magnetogram_id_from_file(eff_local))

    import hashlib, json as _json
    meta_key = dict(
        date_str=str(cfg.date_str),
        prefer_hhmm=str(cfg.prefer_hhmm),
        nlat=int(nlat),
        nlon=int(nlon),
        rss_rsun=float(cfg.rss_rsun),
        nr=(int(cfg.nr) if cfg.nr is not None else None),
        enforce_flux_balance=bool(cfg.enforce_flux_balance),
        local_path=str(Path(eff_local).expanduser().resolve()) if eff_local is not None else "",
        magnetogram_id=str(eff_mag_id),
    )
    key = hashlib.sha1(_json.dumps(meta_key, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]
    cache_file = out_dir / f"pfss_maps_{date_tag}_{key}.npz"

    if cache and cache_file.exists() and (not overwrite):
        out0 = _load_npz(cache_file)
        if out0 is not None:
            return out0

    # Run PFSS (local_path is passed explicitly to avoid any re-selection ambiguity).
    cfg_eff = PFSSConfig(
        out_dir=Path(cfg.out_dir),
        date_str=str(cfg.date_str),
        prefer_hhmm=str(cfg.prefer_hhmm),
        overwrite_download=bool(cfg.overwrite_download),
        N=int(cfg.N),
        nlon=(int(cfg.nlon) if cfg.nlon is not None else None),
        rss_rsun=float(cfg.rss_rsun),
        nr=(int(cfg.nr) if cfg.nr is not None else None),
        enforce_flux_balance=bool(cfg.enforce_flux_balance),
        local_path=Path(eff_local),
        magnetogram_id=str(eff_mag_id),
        search_days=int(cfg.search_days),
    )

    res = pfss_from_gong(cfg_eff)
    pfss = res["pfss"]
    brN = res["brN"]

    out: Dict[str, Any] = {
        "cache_file": str(cache_file),
        "date_str": str(cfg.date_str),
        "local_path": str(res.get("local_path", "")),
        "nlat": int(nlat),
        "nlon": int(nlon),
    }

    arrays: Dict[str, np.ndarray] = {}
    regrid_flags: Dict[str, bool] = {}

    for w0 in which_l:
        br_raw = extract_br_map(pfss, brN, which=w0)
        mobj = _map_obj_for_br(pfss, which=w0)
        axes = _sunpy_lonlat_axes_deg(mobj)
        if axes is None:
            arrays[f"br_{w0}"] = np.asarray(br_raw, dtype=float)
            regrid_flags[w0] = False
        else:
            lon_ax, lat_ax = axes
            arrays[f"br_{w0}"] = _regrid_to_uniform_lonlat(
                np.asarray(br_raw, dtype=float),
                lon_axis_deg=lon_ax,
                lat_axis_deg=lat_ax,
                nlat=int(nlat),
                nlon=int(nlon),
            )
            regrid_flags[w0] = True

        out[w0] = arrays[f"br_{w0}"]

    # Optional derived product: source-surface neutral line (Br=0) as an HCS proxy.
    if "source_surface" in which_l:
        try:
            br_ss = arrays.get("br_source_surface", out.get("source_surface", None))
            if br_ss is not None:
                lon_nl, lat_nl = neutral_line_vertices_lonlat(np.asarray(br_ss, dtype=float), level=0.0, stride=1)
                arrays["neutral_lon_source_surface"] = np.asarray(lon_nl, dtype=float)
                arrays["neutral_lat_source_surface"] = np.asarray(lat_nl, dtype=float)
                out["neutral_lon_source_surface"] = arrays["neutral_lon_source_surface"]
                out["neutral_lat_source_surface"] = arrays["neutral_lat_source_surface"]
        except Exception:
            pass

    if cache:
        import os as _os
        import threading as _threading

        meta = dict(meta_key)
        meta.update(dict(
            cache_version="v0.3",
            regridded=dict(regrid_flags),
        ))

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_name(
            f"{cache_file.stem}.tmp_pid{_os.getpid()}_tid{_threading.get_ident()}{cache_file.suffix}"
        )
        try:
            with open(tmp, "wb") as _f:
                np.savez_compressed(_f, **arrays, meta=np.asarray(_json.dumps(meta), dtype=str))
            tmp.replace(cache_file)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

        out["cache_file"] = str(cache_file)

    return out


def sample_br_nearest(br2d: np.ndarray, *, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Nearest-neighbor Br sampling on a (lat,lon) grid with lon periodicity."""
    br = np.asarray(br2d, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    if br.ndim != 2:
        raise ValueError("br2d must be 2D (lat, lon)")
    nlat, nlon = br.shape

    lonw = np.mod(lon, 360.0)
    ilon = np.floor(lonw / 360.0 * nlon).astype(int)
    ilon = np.mod(ilon, nlon)

    latc = np.clip(lat, -90.0, 90.0)
    ilat = np.round((latc + 90.0) / 180.0 * (nlat - 1)).astype(int)
    ilat = np.clip(ilat, 0, nlat - 1)

    out = br[ilat, ilon]
    out[~np.isfinite(lon) | ~np.isfinite(lat)] = np.nan
    return out


def _lon_lat_axes_from_br(br2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(br2d)
    if a.ndim != 2:
        raise ValueError("br2d must be 2D (lat, lon)")
    nlat, nlon = int(a.shape[0]), int(a.shape[1])
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=float)
    lat = np.linspace(-90.0, 90.0, nlat, endpoint=True, dtype=float)
    return lon, lat


def decimate_nan_polyline(lon_deg: np.ndarray, lat_deg: np.ndarray, *, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate a NaN-separated lon/lat polyline while preserving segment breaks.

    Parameters
    ----------
    lon_deg, lat_deg
        Arrays of identical shape. Non-finite entries are treated as segment separators.
    stride
        Keep every `stride`-th point within each contiguous finite segment.

    Returns
    -------
    (lon_out, lat_out) as float arrays, with NaN separators preserved.
    """
    lon = np.asarray(lon_deg, dtype=float).reshape(-1)
    lat = np.asarray(lat_deg, dtype=float).reshape(-1)
    if lon.shape != lat.shape:
        raise ValueError('lon_deg and lat_deg must have the same shape')

    s = int(max(1, int(stride)))
    if s <= 1:
        return lon.copy(), lat.copy()

    fin = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(fin):
        return lon.copy(), lat.copy()

    out_lon = []
    out_lat = []
    n = lon.size
    i = 0
    while i < n:
        if not fin[i]:
            i += 1
            continue
        j = i
        while j < n and fin[j]:
            j += 1
        seg_lon = lon[i:j:s]
        seg_lat = lat[i:j:s]
        if seg_lon.size > 0:
            out_lon.append(seg_lon)
            out_lat.append(seg_lat)
            out_lon.append(np.array([np.nan], dtype=float))
            out_lat.append(np.array([np.nan], dtype=float))
        i = j

    if not out_lon:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(out_lon), np.concatenate(out_lat)


def neutral_line_vertices_lonlat(
    br2d: np.ndarray,
    *,
    level: float = 0.0,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Br=level contour vertices as a NaN-separated lon/lat polyline.

    Returns
    -------
    (lon_deg, lat_deg) arrays. Disjoint contour paths are separated by NaNs, which
    downstream plotting code can use to break line segments cleanly.
    """
    import matplotlib.pyplot as plt

    br = np.asarray(br2d, dtype=float)
    lon, lat = _lon_lat_axes_from_br(br)

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(111)
    cs = ax.contour(lon, lat, br, levels=[float(level)])

    lon_out = []
    lat_out = []
    s = int(max(1, int(stride)))

    for coll in cs.collections:
        for path in coll.get_paths():
            v = np.asarray(path.vertices, dtype=float)
            if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] != 2:
                continue
            vv = v[::s, :] if s > 1 else v
            lon_seg = np.mod(vv[:, 0], 360.0)
            lat_seg = np.clip(vv[:, 1], -90.0, 90.0)
            lon_out.append(lon_seg.astype(float))
            lat_out.append(lat_seg.astype(float))
            # NaN separator between disjoint paths
            lon_out.append(np.array([np.nan], dtype=float))
            lat_out.append(np.array([np.nan], dtype=float))

    plt.close(fig)

    if not lon_out:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(lon_out), np.concatenate(lat_out)


def _great_circle_dist_deg(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Great-circle distance on a unit sphere in degrees."""
    lam1 = np.deg2rad(lon1)
    phi1 = np.deg2rad(lat1)
    lam2 = np.deg2rad(lon2)
    phi2 = np.deg2rad(lat2)

    s1, c1 = np.sin(phi1), np.cos(phi1)
    s2, c2 = np.sin(phi2), np.cos(phi2)
    dlam = lam1 - lam2
    cosd = s1 * s2 + c1 * c2 * np.cos(dlam)
    cosd = np.clip(cosd, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosd))


def angular_distance_to_neutral_line_deg(
    *,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    nl_lon_deg: np.ndarray,
    nl_lat_deg: np.ndarray,
    chunk: int = 1024,
) -> np.ndarray:
    """Distance (deg) to nearest neutral-line vertex."""
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    nl_lon = np.asarray(nl_lon_deg, dtype=float)
    nl_lat = np.asarray(nl_lat_deg, dtype=float)

    out = np.full(lon.shape, np.nan, dtype=float)
    if lon.size == 0 or nl_lon.size == 0:
        return out

    # Drop NaN separators / non-finite vertices (neutral lines are often stored as NaN-separated polylines).
    mnl = np.isfinite(nl_lon) & np.isfinite(nl_lat)
    if not np.any(mnl):
        return out
    nl_lon = nl_lon[mnl]
    nl_lat = nl_lat[mnl]

    good = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(good):
        return out

    idx = np.where(good)[0]
    for i0 in range(0, idx.size, int(max(1, chunk))):
        ii = idx[i0:i0 + int(max(1, chunk))]
        d = _great_circle_dist_deg(lon[ii, None], lat[ii, None], nl_lon[None, :], nl_lat[None, :])
        out[ii] = np.nanmin(d, axis=1)

    return out


# ============================================================
# Plotly colorscale helpers
# ============================================================


def _normalize_pfss_plotly_colorscale(colorscale: Any) -> Any:
    """Normalize diverging PFSS Br colorscales so that +Br maps to red and -Br to blue.

    Plotly's built-in string "RdBu" follows the ColorBrewer convention (red->blue),
    which makes +Br (high) appear blue when using symmetric limits. For heliophysics
    overlays we want the opposite: +Br red, -Br blue.

    If a non-string colorscale is provided (e.g. a list of [t,color] pairs), it is
    returned unchanged.
    """
    if not isinstance(colorscale, str):
        return colorscale

    name = str(colorscale).strip()
    low = name.lower()
    if low in {"rdbu", "rdbu_r"}:
        try:
            import plotly.colors as _pc  # noqa: WPS433
            cols = list(getattr(_pc.diverging, 'RdBu'))
            cols = cols[::-1]  # blue->red so that high (+) is red
            n = max(2, len(cols))
            return [[i / (n - 1), cols[i]] for i in range(n)]
        except Exception:
            # Fallback: many Plotly installs accept this alias; if not, caller will fall back to default.
            return 'RdBu_r'

    return name

# ============================================================
# Plotly 3D sphere utilities (optional)
# ============================================================


def _lonlat_to_xyz(r: float, lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    cl = np.cos(lat)
    x = r * cl * np.cos(lon)
    y = r * cl * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def write_br_sphere_html(
    *,
    br2d: np.ndarray,
    out_html: Path,
    plot_cfg: Optional[SpherePlotConfig] = None,
    points_lonlat: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    points_value: Optional[np.ndarray] = None,
    points_cfg: Optional[Points3DConfig] = None,
) -> Path:
    """Write a self-contained Plotly HTML sphere for a Br map (and optional points)."""
    try:
        import plotly.graph_objects as go
        import math
    except Exception as e:
        raise RuntimeError("Plotly is required for HTML sphere outputs.") from e

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    br = np.asarray(br2d, dtype=float)
    if br.ndim != 2:
        raise ValueError("br2d must be 2D")

    pcfg = plot_cfg or SpherePlotConfig()
    lons, lats = _lon_lat_axes_from_br(br)
    if bool(pcfg.close_lon):
        lons2 = np.concatenate([lons, [360.0]])
        br2 = np.concatenate([br, br[:, :1]], axis=1)
    else:
        lons2 = lons
        br2 = br


    # Default Br limits: robust symmetric about zero (helps polarity interpretation).
    if pcfg.clim is None:
        vv = br2[np.isfinite(br2)]
        if vv.size > 0:
            lo, hi = np.nanpercentile(vv, [2.0, 98.0])
            mm = float(max(abs(float(lo)), abs(float(hi))))
            pcfg_clim = (-mm, +mm)
        else:
            pcfg_clim = (-1.0, +1.0)
    else:
        pcfg_clim = (float(pcfg.clim[0]), float(pcfg.clim[1]))

    lon_grid, lat_grid = np.meshgrid(lons2, lats)
    x, y, z = _lonlat_to_xyz(float(pcfg.r_sphere), lon_grid, lat_grid)

    surf = go.Surface(
        x=x, y=y, z=z,
        surfacecolor=br2,
        colorscale=_normalize_pfss_plotly_colorscale(pcfg.colorscale),
        opacity=float(pcfg.opacity),
        showscale=bool(pcfg.showscale),
        colorbar=dict(title=str(pcfg.cbar_title)),
    )

    fig = go.Figure(data=[surf])
    fig.update_traces(cmin=float(pcfg_clim[0]), cmax=float(pcfg_clim[1]))

    if points_lonlat is not None and points_cfg is not None and bool(points_cfg.enabled):
        lon_p, lat_p = points_lonlat
        r_pts = float(pcfg.r_sphere) if bool(getattr(points_cfg, 'project_to_sphere', True)) else float(points_cfg.r)
        xp, yp, zp = _lonlat_to_xyz(r_pts, np.asarray(lon_p, float), np.asarray(lat_p, float))
        col = None
        if points_value is not None:
            col = np.asarray(points_value, float)
        fig.add_trace(
            go.Scatter3d(
                x=xp, y=yp, z=zp,
                mode="markers",
                name=str(points_cfg.name),
                marker=dict(
                    size=int(points_cfg.size),
                    symbol=str(points_cfg.symbol),
                    color=col,
                    colorscale=str(points_cfg.colorscale),
                    showscale=bool(points_cfg.showscale),
                    opacity=0.95,
                    line=dict(color=str(getattr(points_cfg, 'edge_rgba', 'rgba(0,0,0,0.75)')), width=float(getattr(points_cfg, 'edge_width', 1.0))),
                ),
            )
        )

    fig.update_layout(
        title=str(pcfg.title),
        width=int(pcfg.width),
        height=int(pcfg.height),
        margin=dict(l=int(pcfg.margin[0]), r=int(pcfg.margin[1]), t=int(pcfg.margin[2]), b=int(pcfg.margin[3])),
        scene=dict(
            xaxis=dict(visible=bool(pcfg.show_axes)),
            yaxis=dict(visible=bool(pcfg.show_axes)),
            zaxis=dict(visible=bool(pcfg.show_axes)),
            aspectmode="data",
        ),
    )

    out_html.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")
    return out_html



# ============================================================
# Dynamic PFSS + backmapped-footpoint 3D HTML (Plotly animation)
# ============================================================

# ============================================================
# Dynamic PFSS + backmapped-footpoint 3D HTML (Plotly animation)
# ============================================================

def write_pfss_backmap_dynamic_sphere_html(
    *,
    out_html: Path,
    br_by_day: Dict[str, np.ndarray],
    which_br: str,
    lon_fp_deg: np.ndarray,
    lat_fp_deg: np.ndarray,
    pfss_date: Sequence[str],
    # --- temporal rendering ---
    stride: int = 4,
    max_frames: int = 1500,
    # How the footpoint track is drawn as frames advance:
    #   - "cumulative": show all points from start -> current (decimated by stride_eff)
    #   - "tail":       show only the last `tail` samples (default in older versions)
    fp_draw_mode: str = "cumulative",
    tail: int = 240,
    # --- geometry ---
    # PFSS texture sphere (typically photosphere, r=1)
    r_sphere: float = 1.0,
    # Footpoint sphere (typically source surface). If None, uses point_r if provided else r_sphere.
    r_fp: Optional[float] = None,
    # Legacy alias for footpoint radius (kept for backward compatibility).
    point_r: Optional[float] = None,
    # Optional: draw a faint source-surface shell to give the footpoint context.
    show_source_surface_shell: bool = True,
    r_shell: Optional[float] = None,
    shell_opacity: float = 0.10,
    shell_color: str = "rgba(0,0,0,0.10)",
    # --- footpoint "projected square" patch (publication look) ---
    # Plotly Scatter3d markers are billboarded (screen-space). For a marker that actually lives on the sphere,
    # we render a tiny Mesh3d patch on the source surface and optionally color it by point_value.
    fp_patch_size_deg: float = 1.6,  # full width (deg) of the patch on the sphere
    fp_patch_opacity: float = 0.0,
    fp_patch_outline_rgba: str = "rgba(0,0,0,0.85)",
    fp_patch_outline_width: int = 0,
    # --- legacy marker args (kept for compatibility with older pipeline calls) ---
    # These are mapped onto the projected patch settings above.
    point_size: Optional[int] = None,
    point_symbol: Optional[str] = None,
    point_edge_rgba: Optional[str] = None,
    point_edge_width: Optional[float] = None,
    show_colorbar: Optional[bool] = None,
    # --- footpoint path styling ---
    fp_path_rgba: str = "rgba(15,15,15,0.78)",
    fp_path_width: int = 5,
    fp_recent_rgba: str = "rgba(0,0,0,0.55)",
    fp_recent_width: int = 7,
    fp_recent_tail: int = 80,
    # --- optional: color the footpoint track by a time series ---
    point_value: Optional[np.ndarray] = None,
    point_value_label: str = "",
    point_colorscale: str = "Viridis",
    show_point_colorbar: bool = False,
    # --- PFSS texture styling ---
    pfss_colorscale: str = "RdBu",
    pfss_clim: Optional[Tuple[float, float]] = None,
    pfss_show_colorbar: bool = True,
    pfss_colorbar_title: Optional[str] = None,
    # --- Neutral line (HCS proxy) ---
    neutral_by_day: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    neutral_line_rgba: str = "rgba(0,0,0,0.75)",
    neutral_line_width: int = 5,
    # --- spacecraft trajectory (Carrington lon/lat) ---
    sc_lon_deg: Optional[np.ndarray] = None,
    sc_lat_deg: Optional[np.ndarray] = None,
    sc_shell_r: Optional[float] = None,
    sc_orbit_stride: int = 2,
    sc_tail: int = 720,
    sc_line_rgba: str = "rgba(0,0,0,0.40)",
    sc_line_width: int = 5,
    sc_trail_rgba: str = "rgba(0,0,0,0.58)",
    sc_trail_width: int = 7,
    sc_point_size: int = 7,
    sc_point_rgba: str = "rgba(0,0,0,0.92)",
    show_sc_connector: bool = True,
    connector_rgba: str = "rgba(0,0,0,0.30)",
    connector_width: int = 4,
    # --- grids + camera ---
    show_sphere_grids: bool = True,
    grid_lon_step_deg: int = 45,
    grid_lat_step_deg: int = 30,
    grid_rgba: str = "rgba(0,0,0,0.18)",
    grid_width: int = 2,
    camera_follow_sc: bool = True,
    camera_distance: float = 1.65,   # in units of scene "lim"
    camera_z_boost: float = 0.28,    # adds a fixed +z component before normalization
    # --- playback ---
    play_fps: int = 3,
    title: str = "PFSS (photosphere) + source-surface backmapping + spacecraft trajectory",
) -> Path:
    """Write a publication-style Plotly HTML animation.

    Design goals (what this tries to fix)
    -------------------------------------
    1) **Match static-figure aesthetics**: smooth spheres, stable colorbars, clear geometry.
    2) **Show all required context simultaneously**:
       - PFSS Br texture on the photosphere (inner sphere).
       - HCS proxy (Br=0 neutral line) as a curve on the source surface.
       - Spacecraft trajectory on a slightly larger shell (so it is visible).
       - Backmapped footpoint track on the source surface, built frame-by-frame.
    3) Avoid common long-range pitfalls:
       - no per-frame autoscaling (stable diverging Br scale),
       - no disappearing SC orbit due to NaNs (finite filtering),
       - no "billboard square" artifacts (use a tiny Mesh3d patch on the sphere).

    Notes
    -----
    - PFSS is not cadence-resolved; `pfss_date` defines the piecewise-constant background.
    - For very long intervals, use `stride`/`max_frames` to keep the HTML size reasonable.
    """
    try:
        import plotly.graph_objects as go
        import math
    except Exception as e:
        raise RuntimeError("Plotly is required for dynamic 3D HTML outputs.") from e

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    which = str(which_br).strip().lower()
    if which not in {"photosphere", "source_surface"}:
        which = "source_surface"

    # -----------------------
    # Backward-compatible argument remapping
    # -----------------------
    # Older pipeline versions passed Scatter3d marker settings; in this publication-style
    # renderer we map them onto the projected patch controls.
    if point_size is not None:
        try:
            fp_patch_size_deg = float(max(0.6, min(5.0, float(point_size) * 0.28)))
        except Exception:
            pass
    if point_edge_rgba is not None:
        fp_patch_outline_rgba = str(point_edge_rgba)
    if point_edge_width is not None:
        try:
            fp_patch_outline_width = int(max(3, round(float(point_edge_width) * 4.0)))
        except Exception:
            pass
    if show_colorbar is not None:
        show_point_colorbar = bool(show_colorbar)


    # -----------------------
    # Coerce and validate arrays
    # -----------------------
    lon_fp = np.asarray(lon_fp_deg, dtype=float).reshape(-1)
    lat_fp = np.asarray(lat_fp_deg, dtype=float).reshape(-1)
    pfss_date_arr = np.asarray([str(x) for x in pfss_date], dtype=object).reshape(-1)
    n = int(lon_fp.size)
    if n == 0:
        raise ValueError("Empty lon_fp_deg/lat_fp_deg.")
    if lat_fp.size != n or pfss_date_arr.size != n:
        raise ValueError("lon_fp_deg, lat_fp_deg, pfss_date must have identical length.")

    pv = None
    if point_value is not None:
        pv = np.asarray(point_value, dtype=float).reshape(-1)
        if pv.size != n:
            raise ValueError("point_value must have the same length as lon_fp_deg.")

    # Spacecraft lon/lat (Carrington) are optional; they often contain NaNs depending on upstream ephemeris.
    have_sc = (sc_lon_deg is not None) and (sc_lat_deg is not None)
    if have_sc:
        lon_sc = np.asarray(sc_lon_deg, dtype=float).reshape(-1)
        lat_sc = np.asarray(sc_lat_deg, dtype=float).reshape(-1)
        if lon_sc.size != n or lat_sc.size != n:
            raise ValueError("sc_lon_deg/sc_lat_deg must have the same length as lon_fp_deg.")
        sc_ok = np.isfinite(lon_sc) & np.isfinite(lat_sc)
        have_sc = bool(np.any(sc_ok))
    else:
        lon_sc = lat_sc = None
        sc_ok = None

    # -----------------------
    # Validate PFSS maps
    # -----------------------
    days = sorted([str(k) for k in br_by_day.keys() if str(k).strip()])
    if not days:
        raise ValueError("br_by_day is empty.")
    shape0 = None
    for d in days:
        a = np.asarray(br_by_day[d], dtype=float)
        if a.ndim != 2:
            raise ValueError(f"br_by_day[{d}] must be 2D.")
        if shape0 is None:
            shape0 = a.shape
        elif a.shape != shape0:
            raise ValueError(f"Inconsistent PFSS shapes: {d} has {a.shape}, expected {shape0}.")

    # Global clim for stable polarity and stable colorbar.
    if pfss_clim is None:
        cmin, cmax = robust_symmetric_clim(
            [np.asarray(br_by_day[d], dtype=float) for d in days],
            percentiles=(2.0, 98.0),
            fallback=1.0,
        )
    else:
        cmin, cmax = (float(pfss_clim[0]), float(pfss_clim[1]))

    # -----------------------
    # Frame decimation
    # -----------------------
    stride0 = int(max(1, int(stride)))
    idx_all = np.arange(n, dtype=int)[::stride0]
    if idx_all.size > int(max_frames):
        stride_eff = int(np.ceil(n / float(max_frames)))
        stride_eff = int(max(stride0, stride_eff))
        idx_all = np.arange(n, dtype=int)[::stride_eff]
    else:
        stride_eff = stride0
    if idx_all.size < 2:
        raise RuntimeError("Not enough frames after stride/max_frames constraints.")

    fp_draw_mode = str(fp_draw_mode).lower().strip()
    if fp_draw_mode not in {"cumulative", "tail"}:
        fp_draw_mode = "cumulative"

    # -----------------------
    # Geometry helpers
    # -----------------------
    fp_r = float(r_fp) if (r_fp is not None) else (float(point_r) if (point_r is not None) else float(r_sphere))
    shell_r = float(r_shell) if (r_shell is not None) else float(fp_r)

    if have_sc:
        sc_r = float(sc_shell_r) if (sc_shell_r is not None) else float(max(fp_r * 1.28, float(r_sphere) * 1.08))
    else:
        sc_r = None

    def _xyz_on_r(r: float, lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _lonlat_to_xyz(float(r), np.mod(lon_deg, 360.0), np.clip(lat_deg, -90.0, 90.0))

    def _patch_vertices_on_sphere(r: float, lon_deg: float, lat_deg: float, size_deg: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 4 vertices of a tiny square patch centered at (lon,lat) on a sphere of radius r."""
        lon = float(lon_deg)
        lat = float(lat_deg)
        if not (np.isfinite(lon) and np.isfinite(lat)):
            return (np.full(4, np.nan), np.full(4, np.nan), np.full(4, np.nan))
        lonr = math.radians(lon % 360.0)
        latr = math.radians(max(-90.0, min(90.0, lat)))
        # unit basis
        er = np.array([math.cos(latr) * math.cos(lonr), math.cos(latr) * math.sin(lonr), math.sin(latr)], dtype=float)
        elon = np.array([-math.sin(lonr), math.cos(lonr), 0.0], dtype=float)
        elat = np.array([-math.sin(latr) * math.cos(lonr), -math.sin(latr) * math.sin(lonr), math.cos(latr)], dtype=float)
        # half-width (radians) -> tangent-plane amplitude
        hh = max(1e-6, math.radians(float(size_deg)) * 0.5)
        a = math.tan(hh)
        corners = [(-1.0, -1.0), (+1.0, -1.0), (+1.0, +1.0), (-1.0, +1.0)]
        xyz = []
        for sx, sy in corners:
            v = er + sx * a * elon + sy * a * elat
            v = v / float(np.linalg.norm(v))
            xyz.append(v * float(r))
        xyz = np.asarray(xyz, dtype=float)
        return (xyz[:, 0], xyz[:, 1], xyz[:, 2])

    def _grid_traces(r: float) -> Sequence[Any]:
        if not bool(show_sphere_grids):
            return []
        lon_step = int(max(1, int(grid_lon_step_deg)))
        lat_step = int(max(1, int(grid_lat_step_deg)))

        traces = []
        # constant longitude lines
        for lon0 in range(0, 360, lon_step):
            lats = np.linspace(-90.0, 90.0, 181, dtype=float)
            lons = np.full_like(lats, float(lon0))
            xg, yg, zg = _xyz_on_r(float(r), lons, lats)
            traces.append(
                go.Scatter3d(
                    x=xg, y=yg, z=zg,
                    mode="lines",
                    line=dict(color=str(grid_rgba), width=int(grid_width)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        # constant latitude lines
        for lat0 in range(-90 + lat_step, 90, lat_step):
            lons = np.linspace(0.0, 360.0, 361, dtype=float)
            lats = np.full_like(lons, float(lat0))
            xg, yg, zg = _xyz_on_r(float(r), lons, lats)
            traces.append(
                go.Scatter3d(
                    x=xg, y=yg, z=zg,
                    mode="lines",
                    line=dict(color=str(grid_rgba), width=int(grid_width)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        return traces

    # -----------------------
    # PFSS surface geometry from map grid
    # -----------------------
    nlat, nlon = int(shape0[0]), int(shape0[1])
    lon_ax = np.linspace(0.0, 360.0, nlon, endpoint=False, dtype=float)
    lat_ax = np.linspace(-90.0, 90.0, nlat, endpoint=True, dtype=float)
    lon2 = np.concatenate([lon_ax, [360.0]])
    lon_g, lat_g = np.meshgrid(lon2, lat_ax)
    lonr = np.deg2rad(lon_g)
    latr = np.deg2rad(lat_g)
    ux = np.cos(latr) * np.cos(lonr)
    uy = np.cos(latr) * np.sin(lonr)
    uz = np.sin(latr)
    X = float(r_sphere) * ux
    Y = float(r_sphere) * uy
    Z = float(r_sphere) * uz

    # Initial PFSS day
    day0 = str(pfss_date_arr[idx_all[0]]).strip()
    if (not day0) or (day0 not in br_by_day):
        day0 = days[0]
    br0 = np.asarray(br_by_day[day0], dtype=float)
    br0 = np.concatenate([br0, br0[:, :1]], axis=1)

    # Neutral line initial (draw on the source-surface shell)
    nl0_xyz = None
    if neutral_by_day is not None and day0 in neutral_by_day:
        nl_lon, nl_lat = neutral_by_day[day0]
        nl_lon = np.asarray(nl_lon, dtype=float).reshape(-1)
        nl_lat = np.asarray(nl_lat, dtype=float).reshape(-1)
        ok = np.isfinite(nl_lon) & np.isfinite(nl_lat)
        if np.any(ok):
            xnl, ynl, znl = _xyz_on_r(shell_r, nl_lon[ok], nl_lat[ok])
            nl0_xyz = (xnl, ynl, znl)

    # Footpoint value scaling
    vmin_p, vmax_p = None, None
    if pv is not None and np.isfinite(pv).any():
        lo, hi = np.nanpercentile(pv[np.isfinite(pv)], [2.0, 98.0])
        vmin_p, vmax_p = float(lo), float(hi)

    # Spacecraft precomputed xyz (finite filtering)
    if have_sc:
        x_sc_all, y_sc_all, z_sc_all = _xyz_on_r(sc_r, lon_sc, lat_sc)
    else:
        x_sc_all = y_sc_all = z_sc_all = None

    # -----------------------
    # Base traces (publication-style)
    # -----------------------
    traces = []

    # PFSS surface
    idx_pfss = len(traces)
    traces.append(
        go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=br0,
            cmin=float(cmin), cmax=float(cmax),
            colorscale=_normalize_pfss_plotly_colorscale(pfss_colorscale),
            opacity=0.995,
            showscale=bool(pfss_show_colorbar),
            colorbar=(
                dict(
                    title=str(pfss_colorbar_title or f"PFSS Br ({which})"),
                    len=0.82,
                    thickness=20,
                    x=1.02,
                    y=0.50,
                    yanchor="middle",
                    xanchor="left",
                    outlinewidth=0,
                ) if bool(pfss_show_colorbar) else None
            ),
            lighting=dict(ambient=0.55, diffuse=0.92, specular=0.10, roughness=0.95, fresnel=0.02),
            lightposition=dict(x=2.2, y=1.8, z=1.4),
            hoverinfo="skip",
            name="PFSS",
        )
    )

    # Neutral line (HCS proxy) on source surface
    idx_nl = len(traces)
    if nl0_xyz is not None:
        xnl, ynl, znl = nl0_xyz
        traces.append(go.Scatter3d(x=xnl, y=ynl, z=znl, mode="lines",
                                   line=dict(color=str(neutral_line_rgba), width=int(neutral_line_width)),
                                   hoverinfo="skip", showlegend=False))
    else:
        traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="lines",
                                   line=dict(color=str(neutral_line_rgba), width=int(neutral_line_width)),
                                   hoverinfo="skip", showlegend=False))

    # Faint source-surface shell
    idx_shell = len(traces)
    if bool(show_source_surface_shell) and (float(shell_r) != float(r_sphere)):
        shell_sc = np.zeros_like(br0, dtype=float)
        traces.append(
            go.Surface(
                x=float(shell_r) * ux, y=float(shell_r) * uy, z=float(shell_r) * uz,
                surfacecolor=shell_sc,
                cmin=0.0, cmax=1.0,
                colorscale=[[0.0, str(shell_color)], [1.0, str(shell_color)]],
                opacity=float(shell_opacity),
                showscale=False,
                lighting=dict(ambient=0.85, diffuse=0.10, specular=0.0, roughness=1.0, fresnel=0.0),
                lightposition=dict(x=1.0, y=1.0, z=1.0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers",
                                   marker=dict(size=1, opacity=0.0), hoverinfo="skip", showlegend=False))

    # Spacecraft full orbit (static, finite-filtered)
    idx_orbit = len(traces)
    if have_sc:
        ok = sc_ok.copy()
        # Downsample the orbit for visual clarity
        s_orb = int(max(1, int(sc_orbit_stride)))
        ok_idx = np.where(ok)[0]
        if ok_idx.size:
            ok_idx = ok_idx[::s_orb]
            traces.append(go.Scatter3d(
                x=x_sc_all[ok_idx], y=y_sc_all[ok_idx], z=z_sc_all[ok_idx],
                mode="lines",
                line=dict(color=str(sc_line_rgba), width=int(sc_line_width)),
                hoverinfo="skip",
                showlegend=False,
            ))
        else:
            traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="lines", hoverinfo="skip", showlegend=False))
    else:
        traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="lines", hoverinfo="skip", showlegend=False))

    # Spacecraft trail (dynamic)
    idx_sc_trail = len(traces)
    traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="lines",
                               line=dict(color=str(sc_trail_rgba), width=int(sc_trail_width)),
                               hoverinfo="skip", showlegend=False))

    # Spacecraft marker (dynamic)
    idx_sc_point = len(traces)
    traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers",
                               marker=dict(size=int(sc_point_size), symbol="circle", color=str(sc_point_rgba)),
                               hoverinfo="skip", showlegend=False))

    # Connector (dynamic)
    idx_conn = len(traces)
    traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="lines",
                               line=dict(color=str(connector_rgba), width=int(connector_width)),
                               hoverinfo="skip", showlegend=False))

    # Footpoint path markers (dynamic; colored by pv if provided)
    idx_fp_mark = len(traces)
    traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="markers",
                               marker=dict(size=4, symbol="circle",
                                           color=("rgba(20,20,20,0.60)" if pv is None else np.array([0.0])),
                                           colorscale=str(point_colorscale),
                                           cmin=vmin_p, cmax=vmax_p,
                                           showscale=bool(show_point_colorbar) if (pv is not None) else False,
                                           colorbar=(dict(title=str(point_value_label), len=0.55, thickness=18, x=0.94, y=0.15, yanchor="middle", outlinewidth=0)
                                                     if (pv is not None and bool(show_point_colorbar)) else None),
                                           ),
                               hoverinfo="skip", showlegend=False))

    # Footpoint recent tail (dynamic, a thicker line segment)
    idx_fp_tail = len(traces)
    traces.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan], mode="lines",
                               line=dict(color=str(fp_recent_rgba), width=int(fp_recent_width)),
                               hoverinfo="skip", showlegend=False))

    # Footpoint projected square patch: Mesh3d (dynamic)
    idx_fp_patch = len(traces)
    # placeholder vertices
    traces.append(go.Mesh3d(
        x=[np.nan]*4, y=[np.nan]*4, z=[np.nan]*4,
        i=[0, 0], j=[1, 2], k=[2, 3],
        opacity=float(fp_patch_opacity),
        intensity=[0.0, 0.0, 0.0, 0.0],
        colorscale=str(point_colorscale),
        cmin=vmin_p,
        cmax=vmax_p,
        showscale=False,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
    ))

    # Patch outline (dynamic)
    idx_fp_outline = len(traces)
    traces.append(go.Scatter3d(
        x=[np.nan], y=[np.nan], z=[np.nan],
        mode="lines",
        line=dict(color=str(fp_patch_outline_rgba), width=int(fp_patch_outline_width)),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Static grids (photosphere + source surface)
    traces.extend(_grid_traces(float(r_sphere)))
    traces.extend(_grid_traces(float(shell_r)))

    fig = go.Figure(data=traces)

    # Scene extent
    lim = float(max(float(r_sphere), float(shell_r), float(fp_r), float(sc_r) if sc_r is not None else 0.0)) * 1.12
    fig.update_layout(
        template="plotly_white",
        title=dict(text=str(title), x=0.02, xanchor="left"),
        width=1160,
        height=920,
        scene=dict(
            xaxis=dict(visible=False, range=[-lim, lim]),
            yaxis=dict(visible=False, range=[-lim, lim]),
            zaxis=dict(visible=False, range=[-lim, lim]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.35, y=1.15, z=0.95), up=dict(x=0, y=0, z=1)),
            bgcolor="white",
        ),
        margin=dict(l=0, r=140, t=62, b=40),
        showlegend=False,
    )

    # -----------------------
    # Frames
    # -----------------------
    frames = []
    last_day = day0

    def _camera_for_sc(x: float, y: float, z: float) -> Dict[str, Any]:
        v = np.array([x, y, z], dtype=float)
        if not np.isfinite(v).all() or float(np.linalg.norm(v)) <= 0.0:
            return dict(eye=dict(x=1.35, y=1.15, z=0.95), up=dict(x=0, y=0, z=1))
        v = v / float(np.linalg.norm(v))
        v = v + np.array([0.0, 0.0, float(camera_z_boost)], dtype=float)
        v = v / float(np.linalg.norm(v))
        d = float(max(1.6, float(camera_distance))) * float(lim)
        eye = v * d
        return dict(eye=dict(x=float(eye[0]), y=float(eye[1]), z=float(eye[2])), up=dict(x=0, y=0, z=1))

    for jj, ii in enumerate(idx_all):
        ii = int(ii)
        day = str(pfss_date_arr[ii]).strip()
        if (not day) or (day not in br_by_day):
            day = last_day

        # Footpoint indices shown this frame
        if fp_draw_mode == "cumulative":
            tidx = np.arange(0, ii + 1, dtype=int)
        else:
            t0 = max(0, ii - int(max(0, int(tail))))
            tidx = np.arange(t0, ii + 1, dtype=int)

        if stride_eff > 1:
            tidx = tidx[::stride_eff]
            if tidx.size == 0 or tidx[-1] != ii:
                tidx = np.r_[tidx, ii]

        # Footpoint xyz
        xt, yt, zt = _xyz_on_r(fp_r, lon_fp[tidx], lat_fp[tidx])

        # recent segment (for visual direction)
        rtail = int(max(0, int(fp_recent_tail)))
        if rtail > 0:
            r0 = max(0, ii - rtail)
            ridx = np.arange(r0, ii + 1, dtype=int)
            if stride_eff > 1:
                ridx = ridx[::stride_eff]
                if ridx.size == 0 or ridx[-1] != ii:
                    ridx = np.r_[ridx, ii]
            xr, yr, zr = _xyz_on_r(fp_r, lon_fp[ridx], lat_fp[ridx])
        else:
            xr = yr = zr = np.array([np.nan])

        # Footpoint patch
        xq, yq, zq = _patch_vertices_on_sphere(fp_r, float(lon_fp[ii]), float(lat_fp[ii]), float(fp_patch_size_deg))
        # Outline (closed loop)
        xol = np.r_[xq, xq[:1]]
        yol = np.r_[yq, yq[:1]]
        zol = np.r_[zq, zq[:1]]

        frame_data = []
        frame_traces = []
        frame_layout = {}

        # PFSS background updates (only when day changes)
        if day != last_day:
            br = np.asarray(br_by_day[day], dtype=float)
            br = np.concatenate([br, br[:, :1]], axis=1)
            frame_data.append(go.Surface(surfacecolor=br))
            frame_traces.append(idx_pfss)

            # HCS proxy
            if neutral_by_day is not None and day in neutral_by_day:
                nl_lon, nl_lat = neutral_by_day[day]
                nl_lon = np.asarray(nl_lon, dtype=float).reshape(-1)
                nl_lat = np.asarray(nl_lat, dtype=float).reshape(-1)
                ok = np.isfinite(nl_lon) & np.isfinite(nl_lat)
                if np.any(ok):
                    xnl, ynl, znl = _xyz_on_r(shell_r, nl_lon[ok], nl_lat[ok])
                    frame_data.append(go.Scatter3d(x=xnl, y=ynl, z=znl))
                else:
                    frame_data.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan]))
            else:
                frame_data.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan]))
            frame_traces.append(idx_nl)

            last_day = day

        # Footpoint path markers (build-up)
        if pv is not None:
            frame_data.append(go.Scatter3d(
                x=xt, y=yt, z=zt,
                marker=dict(
                    size=4,
                    symbol="circle",
                    color=np.asarray(pv[tidx], dtype=float),
                    colorscale=str(point_colorscale),
                    cmin=vmin_p,
                    cmax=vmax_p,
                    showscale=bool(show_point_colorbar),
                    line=dict(width=0),
                ),
            ))
        else:
            frame_data.append(go.Scatter3d(
                x=xt, y=yt, z=zt,
                marker=dict(size=4, symbol="circle", color=str(fp_path_rgba)),
            ))
        frame_traces.append(idx_fp_mark)

        # recent tail (thicker line)
        frame_data.append(go.Scatter3d(x=xr, y=yr, z=zr))
        frame_traces.append(idx_fp_tail)

        # patch mesh (colored by current pv if available)
        if pv is not None and np.isfinite(pv[ii]):
            inten = [float(pv[ii])] * 4
        else:
            inten = [0.0] * 4
        frame_data.append(go.Mesh3d(x=xq, y=yq, z=zq, intensity=inten))
        frame_traces.append(idx_fp_patch)

        # patch outline
        frame_data.append(go.Scatter3d(x=xol, y=yol, z=zol))
        frame_traces.append(idx_fp_outline)

        # Spacecraft trail + marker + connector + camera
        if have_sc:
            # trail indices
            s0 = max(0, ii - int(max(0, int(sc_tail))))
            sidx = np.arange(s0, ii + 1, dtype=int)
            if stride_eff > 1:
                sidx = sidx[::stride_eff]
                if sidx.size == 0 or sidx[-1] != ii:
                    sidx = np.r_[sidx, ii]
            # keep only finite sc points
            ok = np.isfinite(x_sc_all[sidx]) & np.isfinite(y_sc_all[sidx]) & np.isfinite(z_sc_all[sidx])
            if np.any(ok):
                xsct, ysct, zsct = x_sc_all[sidx][ok], y_sc_all[sidx][ok], z_sc_all[sidx][ok]
                frame_data.append(go.Scatter3d(x=xsct, y=ysct, z=zsct, line=dict(color=str(sc_trail_rgba), width=int(sc_trail_width))))
            else:
                frame_data.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan]))
            frame_traces.append(idx_sc_trail)

            # current sc point (if finite)
            if np.isfinite(x_sc_all[ii]) and np.isfinite(y_sc_all[ii]) and np.isfinite(z_sc_all[ii]):
                xscp = np.array([x_sc_all[ii]])
                yscp = np.array([y_sc_all[ii]])
                zscp = np.array([z_sc_all[ii]])
            else:
                xscp = yscp = zscp = np.array([np.nan])
            frame_data.append(go.Scatter3d(x=xscp, y=yscp, z=zscp))
            frame_traces.append(idx_sc_point)

            if bool(show_sc_connector) and np.isfinite(xscp[0]) and np.isfinite(xq[0]):
                # link to patch center (approx: mean of vertices)
                xc = float(np.nanmean(xq))
                yc = float(np.nanmean(yq))
                zc = float(np.nanmean(zq))
                frame_data.append(go.Scatter3d(x=[float(xscp[0]), xc], y=[float(yscp[0]), yc], z=[float(zscp[0]), zc],
                                               line=dict(color=str(connector_rgba), width=int(connector_width))))
            else:
                frame_data.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan]))
            frame_traces.append(idx_conn)

            if bool(camera_follow_sc) and np.isfinite(xscp[0]):
                cam = _camera_for_sc(float(xscp[0]), float(yscp[0]), float(zscp[0]))
                frame_layout = dict(scene=dict(camera=cam))
        else:
            # keep connector hidden
            frame_data.append(go.Scatter3d(x=[np.nan], y=[np.nan], z=[np.nan]))
            frame_traces.append(idx_conn)

        frames.append(
            go.Frame(
                data=frame_data,
                traces=frame_traces,
                name=str(jj),
                layout=frame_layout,
            )
        )

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=int(1000 / max(1, int(play_fps))), redraw=True), transition=dict(duration=0), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="frame: "),
                pad=dict(t=25, b=8),
                len=0.96,
                x=0.02,
                y=0.0,
                steps=[
                    dict(method="animate",
                         args=[[str(j)], dict(mode="immediate", frame=dict(duration=0), transition=dict(duration=0), redraw=True)],
                         label=str(j))
                    for j in range(len(frames))
                ],
            )
        ],
    )

    out_html.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")
    return out_html
