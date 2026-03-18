from __future__ import annotations

"""sc_pos.backmap.movie

Efficient PFSS + backmapping movie generation.

Design goals
------------
- Zero recompute of PFSS per frame: compute/load PFSS Br maps once per unique date.
- Two rendering modes:
    1) mode="reuse" (default): single-process, reuses Matplotlib artists + streams frames directly to ffmpeg.
       This is typically the fastest end-to-end method when ffmpeg is available.
    2) mode="parallel": render PNG frames in parallel (joblib), then assemble with ffmpeg.
       Useful when frame render dominates and you have many CPU cores + fast disk.

This module assumes you already ran backmap_interval(...) with pfss_config enabled,
so the saved DataFrame contains:
    - phi_src, lat_src (degrees)
    - tau_s (seconds)
    - t_src (optional; otherwise computed from index - tau_s)
    - pfss_date (optional; otherwise computed from t_src day)

If those columns are missing, we compute them deterministically.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MovieConfig:
    out_mp4: Path
    which_br: str = "source_surface"  # 'source_surface' or 'photosphere'
    fps: int = 3
    stride: int = 1
    tail: int = 240  # number of samples in the trailing path
    panel_vars: Tuple[str, ...] = ("sigma_c", "sigma_r")
    title: str = "Backmapping + PFSS"
    dpi: int = 150
    mode: str = "reuse"  # 'reuse' or 'parallel'
    n_jobs: int = -1
    overwrite: bool = True

    # PFSS background selection for the movie.
    # - "from_data": use data['pfss_date'] if present (default; consistent with pipeline)
    # - "t_src_day": piecewise-constant PFSS background by source-surface launch day
    # - "t_obs_day": piecewise-constant PFSS background by observation day
    # - "interval_mid_day": a single representative PFSS background for the interval
    # - "fixed": force a single PFSS background given by `pfss_date_str`
    pfss_date_mode: str = "from_data"
    pfss_date_str: str = ""


def _ensure_columns(
    df: pd.DataFrame,
    *,
    pfss_date_mode: str = "from_data",
    pfss_date_str: str = "",
) -> pd.DataFrame:
    d = df.copy()

    if "tau_s" not in d.columns:
        raise KeyError("Movie requires tau_s (seconds) in the input DataFrame.")
    if not {"phi_src", "lat_src"}.issubset(d.columns):
        raise KeyError("Movie requires phi_src and lat_src in the input DataFrame.")

    mode = str(pfss_date_mode).strip().lower()
    if mode in {"from_data", "data", "use_data"}:
        mode = "from_data"

    t_obs = pd.to_datetime(pd.DatetimeIndex(d.index), utc=True)
    tau_s = pd.to_numeric(d["tau_s"], errors="coerce").to_numpy(dtype=float)

    # Always define t_src deterministically (tz-naive UTC) for downstream use.
    if "t_src" in d.columns:
        try:
            t_src_col = pd.to_datetime(d["t_src"], errors="coerce")
            if getattr(t_src_col.dt, "tz", None) is not None:
                t_src_col = t_src_col.dt.tz_convert(None)
            d["t_src"] = t_src_col
        except Exception:
            d["t_src"] = (t_obs - pd.to_timedelta(tau_s, unit="s")).tz_convert(None)
    else:
        d["t_src"] = (t_obs - pd.to_timedelta(tau_s, unit="s")).tz_convert(None)

    # Decide whether to (re)compute pfss_date.
    have_pfss_date = "pfss_date" in d.columns

    if (mode == "from_data") and have_pfss_date:
        # Keep the existing column as-is.
        return d

    if mode == "fixed":
        ds = str(pfss_date_str).strip()
        if not ds:
            raise ValueError("pfss_date_mode='fixed' requires pfss_date_str='YYYY-MM-DD'.")
        d["pfss_date"] = np.array([ds] * len(d), dtype=object)
        return d

    if mode == "interval_mid_day":
        dt = pd.to_datetime(d["t_src"], errors="coerce")
        dt_valid = dt[~pd.isna(dt)]
        if len(dt_valid) == 0:
            raise ValueError("pfss_date_mode='interval_mid_day' but all t_src are NaT.")
        mid = pd.DatetimeIndex(dt_valid).tz_convert(None)[int(len(dt_valid) // 2)]
        ds = pd.Timestamp(mid).strftime("%Y-%m-%d")
        d["pfss_date"] = np.array([ds] * len(d), dtype=object)
        return d

    if mode == "t_obs_day":
        dt = pd.to_datetime(t_obs, utc=True, errors="coerce")
        s = pd.DatetimeIndex(dt).tz_convert(None).strftime("%Y-%m-%d").astype(object)
        s = np.where(s == "NaT", "", s)
        d["pfss_date"] = np.array(s, dtype=object)
        return d

    # Default: t_src_day
    dt = pd.to_datetime(d["t_src"], errors="coerce")
    s = dt.dt.strftime("%Y-%m-%d").astype(object).to_numpy()
    s = np.where(pd.isna(dt).to_numpy(), "", s)
    d["pfss_date"] = np.array(s, dtype=object)
    return d



def _load_pfss_map(
    *,
    date_str: str,
    which_br: str,
    pfss_out_dir: Union[str, Path],
    N: int,
    rss_rsun: float,
    nr: Optional[int],
    prefer_hhmm: str,
    overwrite_download: bool,
) -> np.ndarray:
    from .pfss import PFSSConfig, pfss_maps_cached

    cfg = PFSSConfig(
        out_dir=Path(pfss_out_dir),
        date_str=str(date_str),
        prefer_hhmm=str(prefer_hhmm),
        overwrite_download=bool(overwrite_download),
        N=int(N),
        rss_rsun=float(rss_rsun),
        nr=(int(nr) if nr is not None else None),
    )
    maps = pfss_maps_cached(cfg, which=(str(which_br).strip().lower(),), cache=True, overwrite=bool(overwrite_download))
    br2d = np.asarray(maps.get(str(which_br).strip().lower()), dtype=float)
    if br2d.ndim != 2:
        raise ValueError("PFSS Br map must be 2D (lat,lon)")
    return br2d


def make_pfss_backmap_movie(
    *,
    data: Union[pd.DataFrame, str, Path],
    cfg: MovieConfig,
    pfss_out_dir: Union[str, Path],
    N: int = 180,
    rss_rsun: float = 2.5,
    nr: Optional[int] = None,
    prefer_hhmm: str = "1204",
    overwrite_download: bool = False,
    clim: Optional[Tuple[float, float]] = None,
) -> str:
    """Create an MP4 movie combining PFSS background + moving backmapped point.

    Parameters
    ----------
    data:
        Either a DataFrame or a path to the backmap_interval timeseries pickle.

    cfg:
        MovieConfig controlling fps/stride/tail/etc.

    pfss_out_dir:
        Directory used for PFSS map caches (NPZ) and any intermediate frames.

    Notes
    -----
    - mode='reuse' requires ffmpeg available to Matplotlib.
    - mode='parallel' requires ffmpeg on PATH and joblib installed.
    """

    if isinstance(data, (str, Path)):
        df = pd.read_pickle(Path(data))
    else:
        df = data
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("data must be a non-empty DataFrame (or a path to one).")

    d = _ensure_columns(df, pfss_date_mode=str(cfg.pfss_date_mode), pfss_date_str=str(cfg.pfss_date_str))
    out_mp4 = Path(cfg.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    which_br = str(cfg.which_br).strip().lower()
    if which_br not in {"photosphere", "source_surface"}:
        raise ValueError("cfg.which_br must be 'photosphere' or 'source_surface'")

    idx_all = np.arange(len(d), dtype=int)[:: max(1, int(cfg.stride))]
    if idx_all.size < 2:
        raise ValueError("Not enough frames after stride.")

    # Preload PFSS maps per unique date
    dates = [str(x) for x in d["pfss_date"].astype(str).to_numpy()]
    def _ok_date_key(s: str) -> bool:
        ss = str(s).strip()
        if not ss:
            return False
        if ss.lower() in {'nan','nat','none','<na>'}:
            return False
        import re as _re
        return bool(_re.match(r'^\d{4}-\d{2}-\d{2}$', ss))
    uniq = sorted({dates[i] for i in idx_all if _ok_date_key(dates[i])})
    if len(uniq) == 0:
        raise RuntimeError("No valid PFSS dates found in data['pfss_date'] for the requested frames.")
    br_by_date: Dict[str, np.ndarray] = {}
    for ds in uniq:
        br_by_date[ds] = _load_pfss_map(
            date_str=ds,
            which_br=which_br,
            pfss_out_dir=pfss_out_dir,
            N=int(N),
            rss_rsun=float(rss_rsun),
            nr=nr,
            prefer_hhmm=prefer_hhmm,
            overwrite_download=overwrite_download,
        )
    
    # If no explicit color limits were provided, use a single robust symmetric clim
    # across all loaded PFSS maps. This makes the sign/polarity visually consistent
    # and prevents day-to-day autoscaling from looking like a sign change.
    if clim is None:
        try:
            from .pfss import robust_symmetric_clim
            clim = robust_symmetric_clim(list(br_by_date.values()), percentiles=(2.0, 98.0), fallback=1.0)
        except Exception:
            clim = (-1.0, +1.0)

    mode = str(cfg.mode).strip().lower()
    if mode == "reuse":
        return _movie_reuse(
            d=d,
            idxs=idx_all,
            dates=dates,
            br_by_date=br_by_date,
            cfg=cfg,
            clim=clim,
            which_br=which_br,
        )
    if mode == "parallel":
        return _movie_parallel(
            d=d,
            idxs=idx_all,
            dates=dates,
            br_by_date=br_by_date,
            cfg=cfg,
            clim=clim,
            which_br=which_br,
        )
    raise ValueError("cfg.mode must be 'reuse' or 'parallel'")


def _movie_reuse(
    *,
    d: pd.DataFrame,
    idxs: np.ndarray,
    dates: Sequence[str],
    br_by_date: Dict[str, np.ndarray],
    cfg: MovieConfig,
    clim: Optional[Tuple[float, float]],
    which_br: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FFMpegWriter

    from .pfss import neutral_line_vertices_lonlat

    # Check ffmpeg availability
    try:
        available = bool(matplotlib.animation.writers.is_available("ffmpeg"))
    except Exception:
        available = False
    if not available:
        raise RuntimeError("Matplotlib ffmpeg writer is not available. Install ffmpeg and ensure it is on PATH.")

    out_mp4 = Path(cfg.out_mp4)
    if out_mp4.exists() and (not cfg.overwrite):
        return str(out_mp4)

    # SAFE_MPL_RCPARAMS_MOVIE: disable TeX + mathtext wrapping + constrained_layout for rendering
    _rc_backup = {
        'text.usetex': matplotlib.rcParams.get('text.usetex', False),
        'axes.formatter.use_mathtext': matplotlib.rcParams.get('axes.formatter.use_mathtext', False),
        'figure.constrained_layout.use': matplotlib.rcParams.get('figure.constrained_layout.use', False),
    }
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['axes.formatter.use_mathtext'] = False
    matplotlib.rcParams['figure.constrained_layout.use'] = False

    t_obs = pd.to_datetime(pd.DatetimeIndex(d.index), utc=True).tz_convert(None)
    t_src = pd.to_datetime(d["t_src"], errors="coerce")
    if getattr(t_src.dt, "tz", None) is not None:
        t_src = t_src.dt.tz_convert(None)

    lon = pd.to_numeric(d["phi_src"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(d["lat_src"], errors="coerce").to_numpy(dtype=float)

    # Figure layout: (top) map; (bottom) panels.
    # Only split when the user requested BOTH the HCS-distance variable and sigma variables.
    want_dist = ('pfss_hcs_dist_deg' in tuple(getattr(cfg, 'panel_vars', ())))
    want_sigma = any((v in tuple(getattr(cfg, 'panel_vars', ()))) for v in ('sigma_c', 'sigma_r'))
    has_dist = bool(want_dist and ('pfss_hcs_dist_deg' in d.columns))
    has_sigma = bool(want_sigma and any((v in d.columns) for v in ('sigma_c', 'sigma_r')))
    split_panels = bool(has_dist and has_sigma)

    if split_panels:
        fig = plt.figure(figsize=(12.2, 8.6), constrained_layout=False)
        gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.25, 0.55, 0.75])
        axm = fig.add_subplot(gs[0, 0])
        axd = fig.add_subplot(gs[1, 0])
        axs = fig.add_subplot(gs[2, 0], sharex=axd)
        axt = None
        for _tl in axd.get_xticklabels():
            _tl.set_visible(False)
    else:
        fig = plt.figure(figsize=(12.2, 7.1), constrained_layout=False)
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.25, 1.0])
        axm = fig.add_subplot(gs[0, 0])
        axt = fig.add_subplot(gs[1, 0])
        axd = None
        axs = None

    # Initial map
    def _first_valid_date() -> str:
        for _ii in idxs:
            _ds = str(dates[int(_ii)]).strip()
            if _ds in br_by_date:
                return _ds
        raise RuntimeError("No valid PFSS maps for the selected frames (pfss_date is empty/NaT).")

    d0 = _first_valid_date()
    br0 = br_by_date[d0]
    im = axm.imshow(br0, origin="lower", aspect="auto", extent=[0, 360, -90, 90], cmap="RdBu_r")
    if clim is not None:
        im.set_clim(float(clim[0]), float(clim[1]))
    cb = fig.colorbar(im, ax=axm, pad=0.01)
    cb.set_label(f"PFSS Br ({which_br})")
    try:
        cb.ax.set_facecolor("0.92")
        cb.outline.set_edgecolor("0.30")
    except Exception:
        pass

    # Neutral line (source surface only)
    nl_artist = None
    if which_br == "source_surface":
        try:
            nl_lon, nl_lat = neutral_line_vertices_lonlat(br0, level=0.0, stride=2)
            if nl_lon.size:
                nl_artist = axm.scatter(nl_lon, nl_lat, s=1.0, c="k", alpha=0.35, linewidths=0)
        except Exception:
            nl_artist = None

    # Trajectory tail + current point
    tail_line, = axm.plot([], [], lw=2.0, alpha=0.60)
    cur_pt = axm.scatter([], [], s=70, c="k")

    axm.set(xlabel="Carrington lon [deg]", ylabel="Carrington lat [deg]")
    axm.set_xlim(0, 360)
    axm.set_ylim(-90, 90)

    # Time series panels
    vlines = []
    ts_lines = []

    if split_panels and (axd is not None) and (axs is not None):
        # (1) PFSS-HCS distance panel
        axd.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        axd.grid(True, alpha=0.25)
        ydist = pd.to_numeric(d.get('pfss_hcs_dist_deg', np.nan), errors='coerce').to_numpy(dtype=float)
        lnd, = axd.plot(t_obs, ydist, label='PFSS-HCS dist [deg]')
        ts_lines.append(lnd)
        axd.set_ylabel('dist [deg]')
        axd.legend(loc='upper right', frameon=False)
        vlines.append(axd.axvline(t_obs[int(idxs[0])], lw=2.0, alpha=0.55))

        # (2) sigma panel (sigma_c, sigma_r + any other requested variables)
        axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        axs.set_xlabel('t_obs (UTC)')
        axs.grid(True, alpha=0.25)
        for v in cfg.panel_vars:
            if v == 'pfss_hcs_dist_deg':
                continue
            if v in d.columns:
                y = pd.to_numeric(d[v], errors='coerce').to_numpy(dtype=float)
                ln, = axs.plot(t_obs, y, label=str(v))
                ts_lines.append(ln)
        axs.legend(loc='upper right', frameon=False)
        vlines.append(axs.axvline(t_obs[int(idxs[0])], lw=2.0, alpha=0.55))
    else:
        # Backward-compatible single panel
        axt.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        axt.set_xlabel('t_obs (UTC)')
        axt.grid(True, alpha=0.25)
        for v in cfg.panel_vars:
            if v in d.columns:
                y = pd.to_numeric(d[v], errors='coerce').to_numpy(dtype=float)
                ln, = axt.plot(t_obs, y, label=str(v))
                ts_lines.append(ln)
        axt.legend(loc='upper right', frameon=False)
        vlines.append(axt.axvline(t_obs[int(idxs[0])], lw=2.0, alpha=0.55))

    # Writer
    writer = FFMpegWriter(fps=int(cfg.fps), codec="libx264")
    try:
        with writer.saving(fig, str(out_mp4), dpi=int(cfg.dpi)):
            prev_date = d0
            for k, ii in enumerate(idxs):
                i = int(ii)
                ds_raw = str(dates[i]).strip()
                # If pfss_date is missing/invalid for this frame (e.g., tau_s is NaN -> t_src is NaT),
                # fall back to the previous valid PFSS map to keep the movie continuous.
                ds = ds_raw if ds_raw in br_by_date else prev_date
                if ds != prev_date:
                    br = br_by_date[ds]
                    im.set_data(br)
                    if clim is not None:
                        im.set_clim(float(clim[0]), float(clim[1]))

                    if nl_artist is not None:
                        nl_artist.remove()
                        nl_artist = None
                    if which_br == "source_surface":
                        try:
                            nl_lon, nl_lat = neutral_line_vertices_lonlat(br, level=0.0, stride=2)
                            if nl_lon.size:
                                nl_artist = axm.scatter(nl_lon, nl_lat, s=1.0, c="k", alpha=0.35, linewidths=0)
                        except Exception:
                            nl_artist = None

                    prev_date = ds

                # Tail indices
                tlen = int(max(2, int(cfg.tail)))
                j0 = max(0, i - tlen + 1)
                j = np.arange(j0, i + 1, dtype=int)
                jj = j[np.isfinite(lon[j]) & np.isfinite(lat[j])]
                tail_line.set_data(lon[jj], lat[jj])

                # Current point
                if np.isfinite(lon[i]) and np.isfinite(lat[i]):
                    cur_pt.set_offsets(np.array([[lon[i], lat[i]]], dtype=float))
                else:
                    cur_pt.set_offsets(np.array([[np.nan, np.nan]], dtype=float))

                # Time cursor
                for _vl in vlines:
                    _vl.set_xdata([t_obs[i], t_obs[i]])

                # Title
                tt_obs = pd.Timestamp(t_obs[i])
                tt_src = pd.Timestamp(t_src.iloc[i]) if i < len(t_src) else pd.NaT
                if ds_raw and (ds_raw != ds):
                    axm.set_title(f"{cfg.title} | t_obs={tt_obs} | t_src={tt_src} | PFSS date={ds_raw} (using {ds})")
                else:
                    axm.set_title(f"{cfg.title} | t_obs={tt_obs} | t_src={tt_src} | PFSS date={ds}")

                writer.grab_frame()
    finally:
        # restore rcParams
        try:
            for _k, _v in _rc_backup.items():
                matplotlib.rcParams[_k] = _v
        except Exception:
            pass
        try:
            plt.close(fig)
        except Exception:
            pass

    return str(out_mp4)




def _movie_parallel(
    *,
    d: pd.DataFrame,
    idxs: np.ndarray,
    dates: Sequence[str],
    br_by_date: Dict[str, np.ndarray],
    cfg: MovieConfig,
    clim: Optional[Tuple[float, float]],
    which_br: str,
) -> str:
    import shutil
    import subprocess

    import matplotlib

    try:
        from joblib import Parallel, delayed
    except Exception as e:
        raise RuntimeError("mode='parallel' requires joblib. Install joblib in this environment.") from e

    out_mp4 = Path(cfg.out_mp4)
    if out_mp4.exists() and (not cfg.overwrite):
        return str(out_mp4)

    # SAFE_MPL_RCPARAMS_MOVIE: disable TeX + mathtext wrapping + constrained_layout for rendering
    _rc_backup = {
        'text.usetex': matplotlib.rcParams.get('text.usetex', False),
        'axes.formatter.use_mathtext': matplotlib.rcParams.get('axes.formatter.use_mathtext', False),
        'figure.constrained_layout.use': matplotlib.rcParams.get('figure.constrained_layout.use', False),
    }
    matplotlib.rcParams['text.usetex'] = False
    matplotlib.rcParams['axes.formatter.use_mathtext'] = False
    matplotlib.rcParams['figure.constrained_layout.use'] = False

    frames_dir = out_mp4.parent / (out_mp4.stem + "_frames")
    if frames_dir.exists() and cfg.overwrite:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    t_obs = pd.to_datetime(pd.DatetimeIndex(d.index), utc=True).tz_convert(None)
    t_src = pd.to_datetime(d["t_src"], errors="coerce")
    if getattr(t_src.dt, "tz", None) is not None:
        t_src = t_src.dt.tz_convert(None)

    lon = pd.to_numeric(d["phi_src"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(d["lat_src"], errors="coerce").to_numpy(dtype=float)

    # Resolve PFSS map date per frame. If some frames have missing/invalid pfss_date (e.g. tau_s NaN -> t_src NaT),
    # fall back to the previous valid PFSS map so the movie can still be rendered.
    ds_raw_list = [str(dates[int(_i)]).strip() for _i in idxs]
    first_valid = next((_ds for _ds in ds_raw_list if _ds in br_by_date), None)
    if first_valid is None:
        raise RuntimeError("No valid PFSS maps for the selected frames (pfss_date is empty/NaT).")
    ds_used_list = []
    prev = first_valid
    for _ds_raw in ds_raw_list:
        if _ds_raw in br_by_date:
            prev = _ds_raw
        ds_used_list.append(prev)

    def _render_one(k: int, i: int) -> str:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        from .pfss import neutral_line_vertices_lonlat

        ds_raw = ds_raw_list[int(k)]
        ds = ds_used_list[int(k)]
        br = br_by_date[ds]

        want_dist = ('pfss_hcs_dist_deg' in tuple(getattr(cfg, 'panel_vars', ())))
        want_sigma = any((v in tuple(getattr(cfg, 'panel_vars', ()))) for v in ('sigma_c', 'sigma_r'))
        has_dist = bool(want_dist and ('pfss_hcs_dist_deg' in d.columns))
        has_sigma = bool(want_sigma and any((v in d.columns) for v in ('sigma_c', 'sigma_r')))
        split_panels = bool(has_dist and has_sigma)

        if split_panels:
            fig = plt.figure(figsize=(12.2, 8.6), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.25, 0.55, 0.75])
            axm = fig.add_subplot(gs[0, 0])
            axd = fig.add_subplot(gs[1, 0])
            axs = fig.add_subplot(gs[2, 0], sharex=axd)
            axt = None
            for _tl in axd.get_xticklabels():
                _tl.set_visible(False)
        else:
            fig = plt.figure(figsize=(12.2, 7.1), constrained_layout=False)
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.25, 1.0])
            axm = fig.add_subplot(gs[0, 0])
            axt = fig.add_subplot(gs[1, 0])
            axd = None
            axs = None

        im = axm.imshow(br, origin="lower", aspect="auto", extent=[0, 360, -90, 90], cmap="RdBu_r")
        if clim is not None:
            im.set_clim(float(clim[0]), float(clim[1]))
        cb = fig.colorbar(im, ax=axm, pad=0.01)
        cb.set_label(f"PFSS Br ({which_br})")
        try:
            cb.ax.set_facecolor("0.92")
            cb.outline.set_edgecolor("0.30")
        except Exception:
            pass

        if which_br == "source_surface":
            try:
                nl_lon, nl_lat = neutral_line_vertices_lonlat(br, level=0.0, stride=2)
                if nl_lon.size:
                    axm.scatter(nl_lon, nl_lat, s=1.0, c="k", alpha=0.35, linewidths=0)
            except Exception:
                pass

        tlen = int(max(2, int(cfg.tail)))
        j0 = max(0, int(i) - tlen + 1)
        j = np.arange(j0, int(i) + 1, dtype=int)
        jj = j[np.isfinite(lon[j]) & np.isfinite(lat[j])]
        axm.plot(lon[jj], lat[jj], lw=2.0, alpha=0.60)
        if np.isfinite(lon[int(i)]) and np.isfinite(lat[int(i)]):
            axm.scatter([lon[int(i)]], [lat[int(i)]], s=70, c="k")

        axm.set(xlabel="Carrington lon [deg]", ylabel="Carrington lat [deg]")
        axm.set_xlim(0, 360)
        axm.set_ylim(-90, 90)

        if split_panels and (axd is not None) and (axs is not None):
            axd.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            axd.grid(True, alpha=0.25)
            ydist = pd.to_numeric(d.get('pfss_hcs_dist_deg', np.nan), errors='coerce').to_numpy(dtype=float)
            axd.plot(t_obs, ydist, label='PFSS-HCS dist [deg]')
            axd.set_ylabel('dist [deg]')
            axd.legend(loc='upper right', frameon=False)
            axd.axvline(t_obs[int(i)], lw=2.0, alpha=0.55)

            axs.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            axs.set_xlabel('t_obs (UTC)')
            axs.grid(True, alpha=0.25)
            for v in cfg.panel_vars:
                if v == 'pfss_hcs_dist_deg':
                    continue
                if v in d.columns:
                    y = pd.to_numeric(d[v], errors='coerce').to_numpy(dtype=float)
                    axs.plot(t_obs, y, label=str(v))
            axs.legend(loc='upper right', frameon=False)
            axs.axvline(t_obs[int(i)], lw=2.0, alpha=0.55)
        else:
            axt.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            axt.set_xlabel('t_obs (UTC)')
            axt.grid(True, alpha=0.25)
            for v in cfg.panel_vars:
                if v in d.columns:
                    y = pd.to_numeric(d[v], errors='coerce').to_numpy(dtype=float)
                    axt.plot(t_obs, y, label=str(v))
            axt.legend(loc='upper right', frameon=False)
            axt.axvline(t_obs[int(i)], lw=2.0, alpha=0.55)

        tt_obs = pd.Timestamp(t_obs[int(i)])
        tt_src = pd.Timestamp(t_src.iloc[int(i)]) if int(i) < len(t_src) else pd.NaT
        if ds_raw and (ds_raw != ds):
            axm.set_title(f"{cfg.title} | t_obs={tt_obs} | t_src={tt_src} | PFSS date={ds_raw} (using {ds})")
        elif not ds_raw:
            axm.set_title(f"{cfg.title} | t_obs={tt_obs} | t_src={tt_src} | PFSS date missing (using {ds})")
        else:
            axm.set_title(f"{cfg.title} | t_obs={tt_obs} | t_src={tt_src} | PFSS date={ds}")

        out_png = frames_dir / f"frame_{k:06d}.png"
        fig.savefig(out_png, dpi=int(cfg.dpi))
        plt.close(fig)
        return str(out_png)

    try:
        Parallel(n_jobs=int(cfg.n_jobs), prefer="processes")(delayed(_render_one)(k, int(i)) for k, i in enumerate(idxs))

        # Assemble with ffmpeg
        ffmpeg = "ffmpeg"
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(int(cfg.fps)),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            raise RuntimeError("ffmpeg failed. Ensure ffmpeg is installed and on PATH.") from e
    finally:
        # restore rcParams
        try:
            for _k, _v in _rc_backup.items():
                matplotlib.rcParams[_k] = _v
        except Exception:
            pass

    return str(out_mp4)


# =============================================================================
# 3D movie: PFSS texture on a sphere + moving backmapped footpoint (+ optional panels)
# =============================================================================

@dataclass(frozen=True)
class Movie3DConfig:
    """Configuration for 3D MP4 rendering.

    IMPORTANT: A true "carbon copy" of the Plotly HTML requires engine='plotly'.
    The Matplotlib engine is kept only as a fallback.
    """

    out_mp4: Path

    # Rendering backend:
    #   - 'plotly' : build the exact Plotly figure used for HTML and export frames (carbon copy)
    #   - 'mpl'    : legacy Matplotlib 3D renderer (approximate)
    engine: str = "plotly"

    # PFSS texture sphere:
    #   - 'photosphere'   : render PFSS Br on r=1 sphere (required by your spec)
    #   - 'source_surface': render PFSS Br on r=rss_rsun sphere
    which_br: str = "photosphere"

    fps: int = 3

    # If stride <= 0, it is chosen automatically to satisfy `max_frames` and `min_step_s`.
    stride: int = 0
    tail: int = 240

    # Plotly carbon-copy layout controls
    plot_vars: Tuple[str, ...] = ("polarity", "Vr_bg", "P_ram", "sigma_c")
    ncols_vars: int = 2
    frame3d: str = "HGC"
    decimate: int = 1
    panel_px: int = 650
    export_scale: float = 1.6
    export_width: Optional[int] = None
    export_height: Optional[int] = None

    # Tail rendering mode:
    #   - 'tail'      : only the last `tail` samples are shown (recommended; readable)
    #   - 'cumulative': all samples up to current time are shown
    draw_mode: str = "tail"

    title: str = "PFSS (photosphere) + source-surface backmapping + spacecraft trajectory"
    dpi: int = 150  # (mpl only)
    overwrite: bool = True

    # PFSS background selection for the movie (same semantics as MovieConfig.pfss_date_mode).
    pfss_date_mode: str = "from_data"
    pfss_date_str: str = ""

    # Performance guards
    max_frames: int = 1500
    min_step_s: float = 3600.0  # 1 hour

    # Camera behavior (Plotly and mpl)
    follow_sc: bool = True
    follow_sc_smooth: float = 0.18  # 0..1 (higher -> faster response)
    follow_sc_azim_offset_deg: float = 35.0
    follow_sc_elev_offset_deg: float = 20.0

    # Plotly camera geometry (eye distance tuning)
    camera_distance: float = 1.75
    camera_z_boost: float = 0.22

    # Geometry context
    show_source_surface_shell: bool = True
    shell_opacity: float = 0.10
    shell_color: str = "0.65"
    show_sphere_grid: bool = False  # keep off (ugly)

    sc_tail: int = 720
    show_sc: bool = True
    show_sc_connector: bool = True

    lim_factor: float = 1.10  # (mpl only)

    # PFSS rendering
    cmap: str = "RdBu_r"

    # If None: a single global symmetric clim is computed over all loaded texture maps (robust percentiles)
    clim: Optional[Tuple[float, float]] = None

    show_colorbar: bool = True
    colorbar_marker: bool = True
    colorbar_label: str = "PFSS $B_r$ (photosphere)"

    # What value should the PFSS colorbar marker represent?
    #   - 'source_surface_fp' : Br at the footpoint sampled on the source-surface map (recommended for HCS context)
    #   - 'texture_fp'        : Br sampled on the rendered texture map at (lon_fp,lat_fp)
    pfss_marker_from: str = "source_surface_fp"

    # Column name in the DataFrame that can provide PFSS Br at the footpoint.
    # If absent, the movie will sample from PFSS maps.
    pfss_value_var: str = "pfss_br"

    # HCS proxy (neutral line at source surface)
    neutral_line: bool = True
    neutral_stride: int = 2
    show_hcs_metric: bool = True
    hcs_dist_var: str = "pfss_hcs_dist_deg"
    hcs_cross_thresh_deg: float = 3.0
    crossing_flash_frames: int = 8

    # Footpoint highlight aesthetics (Plotly engine uses marker highlight; mpl uses curved patch)
    highlight_size: int = 14
    highlight_edge_rgba: str = "rgba(0,0,0,0.95)"
    highlight_edge_width: int = 5
    highlight_flash_edge_rgba: str = "rgba(255,0,0,0.95)"

    # Optional: per-panel colorbar markers for plotted variables (Plotly engine only)
    show_panel_colorbar_markers: bool = True

    # Optional: color the moving footpoint by a time series (e.g., sigma_c)
    point_value_var: str = ""       # column name
    point_cmap: str = "viridis"
    point_clim: Optional[Tuple[float, float]] = None
    show_point_colorbar: bool = False
    point_colorbar_label: str = ""




def _unit_xyz_from_lonlat(lon_deg: np.ndarray, lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    cl = np.cos(lat)
    x = cl * np.cos(lon)
    y = cl * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def _wrap180(deg: float) -> float:
    x = float(deg)
    x = (x + 180.0) % 360.0 - 180.0
    return x


def _smooth_angle(prev_deg: float, target_deg: float, alpha: float) -> float:
    a = float(max(0.0, min(1.0, alpha)))
    d = _wrap180(float(target_deg) - float(prev_deg))
    return float(prev_deg) + a * d


def _sphere_patch_quad(
    lon_deg: float,
    lat_deg: float,
    *,
    size_deg: float,
    r: float,
) -> np.ndarray:
    """Return a 4x3 array of vertices for a small curved 'square' patch on a sphere.

    The patch is defined in the local tangent plane and then projected back to the sphere
    by renormalization, which yields a curved quadrilateral that visually reads as
    'living on' the sphere (unlike billboard scatter markers).
    """
    lon = np.deg2rad(float(lon_deg))
    lat = np.deg2rad(float(lat_deg))
    u = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], dtype=float)

    # Local tangent basis: east (increasing lon), north (increasing lat)
    e_east = np.array([-np.sin(lon), np.cos(lon), 0.0], dtype=float)
    e_north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)], dtype=float)

    # Normalize (numerical safety near poles)
    ee = np.linalg.norm(e_east)
    en = np.linalg.norm(e_north)
    if ee > 0:
        e_east = e_east / ee
    if en > 0:
        e_north = e_north / en

    half = np.deg2rad(float(size_deg) * 0.5)
    # Tangent-plane offsets
    corners = [
        u + (+half) * e_east + (+half) * e_north,
        u + (-half) * e_east + (+half) * e_north,
        u + (-half) * e_east + (-half) * e_north,
        u + (+half) * e_east + (-half) * e_north,
    ]
    V = np.stack(corners, axis=0)
    # Project back onto the sphere
    nn = np.linalg.norm(V, axis=1)
    nn = np.where(nn > 0, nn, 1.0)
    V = (V.T / nn).T
    return V * float(r)


def _sample_br_nearest(br2d: np.ndarray, lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sampling of a (lat,lon) PFSS map on a regular grid.

    Assumes:
      - lon grid covers [0,360) uniformly with endpoint excluded
      - lat grid covers [-90,90] uniformly with endpoint included
    """
    a = np.asarray(br2d, dtype=float)
    if a.ndim != 2:
        raise ValueError("br2d must be 2D (lat,lon)")
    nlat, nlon = a.shape
    lon = np.mod(np.asarray(lon_deg, dtype=float), 360.0)
    lat = np.clip(np.asarray(lat_deg, dtype=float), -90.0, 90.0)

    j = np.floor(lon / 360.0 * float(nlon)).astype(int)
    j = np.clip(j, 0, nlon - 1)

    # Map lat=-90 -> i=0, lat=+90 -> i=nlat-1
    i = np.round((lat + 90.0) / 180.0 * float(nlat - 1)).astype(int)
    i = np.clip(i, 0, nlat - 1)

    out = a[i, j]
    out = np.asarray(out, dtype=float).reshape(lon.shape)
    return out


def _min_gc_dist_deg_to_polyline(
    lon_deg: float,
    lat_deg: float,
    nl_lon_deg: np.ndarray,
    nl_lat_deg: np.ndarray,
) -> float:
    """Minimum great-circle distance (deg) between a point and a polyline of lon/lat vertices.

    Uses unit-vector dot products; robust to NaN separators in the polyline (ignored).
    """
    lon0 = float(lon_deg)
    lat0 = float(lat_deg)
    if (not np.isfinite(lon0)) or (not np.isfinite(lat0)):
        return float("nan")

    nl_lon = np.asarray(nl_lon_deg, dtype=float).reshape(-1)
    nl_lat = np.asarray(nl_lat_deg, dtype=float).reshape(-1)
    ok = np.isfinite(nl_lon) & np.isfinite(nl_lat)
    if not np.any(ok):
        return float("nan")
    nl_lon = nl_lon[ok]
    nl_lat = nl_lat[ok]

    x0, y0, z0 = _unit_xyz_from_lonlat(np.array([lon0]), np.array([lat0]))
    u0 = np.array([x0[0], y0[0], z0[0]], dtype=float)

    x1, y1, z1 = _unit_xyz_from_lonlat(nl_lon, nl_lat)
    U = np.vstack([x1, y1, z1]).T  # (M,3)

    # angle = arccos(dot) ; min angle <-> max dot
    d = np.clip(U @ u0, -1.0, 1.0)
    mu = float(np.nanmax(d))  # max dot
    ang = float(np.rad2deg(np.arccos(mu)))
    return ang


def make_pfss_backmap_movie_3d(
    *,
    data: Union[str, Path],
    cfg: Movie3DConfig,
    pfss_out_dir: Union[str, Path],
    N: int = 180,
    rss_rsun: float = 2.5,
    nr: Optional[int] = None,
    prefer_hhmm: str = "1204",
    overwrite_download: bool = False,
    cache_maps: bool = True,
    clim: Optional[Tuple[float, float]] = None,
) -> Path:
    """Create a Matplotlib 3D MP4 designed to be a *carbon copy* of the dynamic Plotly HTML.

    What this movie guarantees (logic + visuals)
    -------------------------------------------
    - PFSS Br texture rendered on an inner sphere (photosphere by default, r=1).
    - Backmapped footpoint shown as a *curved square patch* on the source surface (r=rss_rsun).
    - HCS proxy shown as a black neutral line on the source surface (Br_ss=0 contour).
    - Spacecraft Carrington trajectory drawn on a slightly larger shell, with optional connector to the footpoint.
    - Camera can follow the spacecraft (smoothly) to make crossings interpretable.
    - A numeric HCS-distance metric is displayed (deg), and crossings are highlighted consistently.
    - Colorbars are stable across the movie (no per-frame autoscaling) and can show a moving marker for the
      instantaneous value (PFSS Br at the footpoint; plus an optional point-value variable).
    """
    
    engine = str(getattr(cfg, "engine", "mpl")).strip().lower()
    if engine in {"plotly", "html"}:
        return _movie3d_plotly_export(
            data=data,
            cfg=cfg,
            pfss_out_dir=pfss_out_dir,
            N=N,
            rss_rsun=rss_rsun,
            nr=nr,
            prefer_hhmm=prefer_hhmm,
            overwrite_download=overwrite_download,
            cache_maps=cache_maps,
            clim=clim,
        )
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import Normalize
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    out_mp4 = Path(cfg.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if out_mp4.exists() and (not bool(cfg.overwrite)):
        return out_mp4

    df = pd.read_pickle(Path(data)) if str(data).lower().endswith((".pkl", ".pickle")) else pd.read_parquet(Path(data))
    df = _ensure_columns(df, pfss_date_mode=str(cfg.pfss_date_mode), pfss_date_str=str(cfg.pfss_date_str))

    n = int(len(df))
    if n <= 0:
        raise RuntimeError("Empty DataFrame.")

    # ------------------------------------------------------------------
    # Frame selection: choose a stride that respects physical time scales
    # and caps the number of frames.
    # ------------------------------------------------------------------
    t_obs = pd.to_datetime(pd.DatetimeIndex(df.index), utc=True)
    t_ns = t_obs.view("i8")
    dt_ns = np.diff(t_ns)
    dt_ns = dt_ns[np.isfinite(dt_ns)]
    native_dt_s = float(np.nanmedian(dt_ns)) * 1e-9 if dt_ns.size else 0.0
    native_dt_s = float(native_dt_s) if np.isfinite(native_dt_s) else 0.0
    native_dt_s = max(native_dt_s, 0.0)

    user_stride = int(cfg.stride) if cfg.stride is not None else 0
    stride = int(max(1, user_stride)) if user_stride > 0 else 1

    max_frames = int(max(2, int(cfg.max_frames)))
    if user_stride <= 0:
        stride_frames = int(np.ceil(n / float(max_frames)))
        stride = max(stride, max(1, stride_frames))

    min_step_s = float(cfg.min_step_s) if cfg.min_step_s is not None else 0.0
    if (user_stride <= 0) and (n > max_frames) and (native_dt_s > 0.0) and (min_step_s > 0.0):
        stride_min = int(np.ceil(min_step_s / native_dt_s))
        stride = max(stride, max(1, stride_min))

    idx_all = np.arange(n, dtype=int)[::stride]
    if idx_all.size < 2:
        idx_all = np.array([0, n - 1], dtype=int)

    if "pfss_date" not in df.columns:
        raise KeyError("Input DataFrame lacks pfss_date. Re-run backmapping with PFSS enabled, or add pfss_date.")

    pfss_out_dir = Path(pfss_out_dir)
    pfss_out_dir.mkdir(parents=True, exist_ok=True)

    from .pfss import PFSSConfig, pfss_maps_cached, neutral_line_vertices_lonlat, decimate_nan_polyline

    which_tex = str(cfg.which_br).strip().lower()
    if which_tex not in {"photosphere", "source_surface"}:
        raise ValueError("Movie3DConfig.which_br must be 'photosphere' or 'source_surface'")

    pfss_date = df["pfss_date"].astype("string").fillna("").to_numpy()

    def _ok_date_key(s: str) -> bool:
        ss = str(s).strip()
        if (not ss) or (ss.lower() in {"nan", "nat", "none", "<na>"}):
            return False
        import re as _re
        return bool(_re.match(r"^\d{4}-\d{2}-\d{2}$", ss))

    unique_days = sorted({str(pfss_date[i]) for i in idx_all if _ok_date_key(str(pfss_date[i]))})
    if not unique_days:
        raise RuntimeError("pfss_date is empty for all frames; cannot build PFSS background.")

    # Surface decimation factors (dominant speed lever)
    s_lat = int(max(1, int(cfg.surface_stride_lat)))
    s_lon = int(max(1, int(cfg.surface_stride_lon)))

    # Load maps once per day (photosphere + source_surface always), cache for speed.
    raw_photo_by_day: Dict[str, np.ndarray] = {}
    raw_ss_by_day: Dict[str, np.ndarray] = {}
    tex_by_day: Dict[str, np.ndarray] = {}
    nl_by_day: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

    all_vals_for_clim: List[np.ndarray] = []
    all_vals_for_point: List[np.ndarray] = []

    cmap_pfss = mpl.cm.get_cmap(str(cfg.cmap))

    # Optional point-value for patch coloring
    point_vals = None
    if str(cfg.point_value_var).strip() and (str(cfg.point_value_var) in df.columns):
        point_vals = pd.to_numeric(df[str(cfg.point_value_var)], errors="coerce").to_numpy(dtype=float)
        if cfg.point_clim is None:
            vv = point_vals[np.isfinite(point_vals)]
            if vv.size:
                all_vals_for_point.append(vv)

    for dstr in unique_days:
        p_cfg = PFSSConfig(
            out_dir=pfss_out_dir,
            date_str=str(dstr),
            prefer_hhmm=str(prefer_hhmm),
            overwrite_download=bool(overwrite_download),
            N=int(N),
            nlon=None,
            rss_rsun=float(rss_rsun),
            nr=(int(nr) if nr is not None else None),
            enforce_flux_balance=True,
            local_path=None,
            search_days=0,
        )
        maps = pfss_maps_cached(
            p_cfg,
            which=("photosphere", "source_surface"),
            cache=bool(cache_maps),
            overwrite=bool(overwrite_download),
        )
        br_photo = np.asarray(maps.get("photosphere"), dtype=float)
        br_ss = np.asarray(maps.get("source_surface"), dtype=float)

        raw_photo_by_day[dstr] = br_photo
        raw_ss_by_day[dstr] = br_ss

        br_tex_full = br_photo if (which_tex == "photosphere") else br_ss
        if cfg.clim is None and (clim is None):
            vv = br_tex_full[np.isfinite(br_tex_full)]
            if vv.size:
                all_vals_for_clim.append(vv)

        # decimated texture
        tex_by_day[dstr] = br_tex_full[::s_lat, ::s_lon]

        # neutral line always derived from source-surface
        if bool(cfg.neutral_line):
            try:
                if ("neutral_lon_source_surface" in maps) and ("neutral_lat_source_surface" in maps):
                    nl_lon = np.asarray(maps["neutral_lon_source_surface"], dtype=float)
                    nl_lat = np.asarray(maps["neutral_lat_source_surface"], dtype=float)
                    if int(cfg.neutral_stride) > 1:
                        nl_lon, nl_lat = decimate_nan_polyline(nl_lon, nl_lat, stride=int(cfg.neutral_stride))
                else:
                    nl_lon, nl_lat = neutral_line_vertices_lonlat(br_ss, level=0.0, stride=int(max(1, int(cfg.neutral_stride))))
                nl_by_day[dstr] = (nl_lon, nl_lat)
            except Exception:
                nl_by_day[dstr] = None
        else:
            nl_by_day[dstr] = None

    # Global PFSS clim (stable diverging scale)
    clim_use = clim if (clim is not None) else (cfg.clim if cfg.clim is not None else None)
    if clim_use is None:
        if all_vals_for_clim:
            vv = np.concatenate(all_vals_for_clim)
            lo, hi = np.nanpercentile(vv, [2.0, 98.0])
            mm = float(max(abs(float(lo)), abs(float(hi))))
            vmin_glob, vmax_glob = (-mm, mm) if mm > 0 else (-1.0, 1.0)
        else:
            vmin_glob, vmax_glob = (-1.0, 1.0)
    else:
        vmin_glob, vmax_glob = (float(clim_use[0]), float(clim_use[1]))

    norm_pfss = Normalize(vmin=float(vmin_glob), vmax=float(vmax_glob))
    sm_pfss = plt.cm.ScalarMappable(cmap=cmap_pfss, norm=norm_pfss)
    sm_pfss.set_array([])

    # Point-value normalization (optional)
    cmap_point = mpl.cm.get_cmap(str(cfg.point_cmap))
    point_norm = None
    sm_point = None
    if point_vals is not None:
        if cfg.point_clim is not None:
            pvmin, pvmax = (float(cfg.point_clim[0]), float(cfg.point_clim[1]))
        else:
            vv = point_vals[np.isfinite(point_vals)]
            if vv.size:
                pvmin, pvmax = (float(np.nanpercentile(vv, 2.0)), float(np.nanpercentile(vv, 98.0)))
                if pvmin == pvmax:
                    pvmin, pvmax = (pvmin - 1.0, pvmax + 1.0)
            else:
                pvmin, pvmax = (0.0, 1.0)
        point_norm = Normalize(vmin=float(pvmin), vmax=float(pvmax))
        sm_point = plt.cm.ScalarMappable(cmap=cmap_point, norm=point_norm)
        sm_point.set_array([])

    # Footpoint lon/lat (source surface)
    lon_fp = pd.to_numeric(df["phi_src"], errors="coerce").to_numpy(dtype=float)
    lat_fp = pd.to_numeric(df["lat_src"], errors="coerce").to_numpy(dtype=float)
    lon_fp = np.mod(lon_fp, 360.0)
    lat_fp = np.clip(lat_fp, -90.0, 90.0)

    x_fp_u, y_fp_u, z_fp_u = _unit_xyz_from_lonlat(lon_fp, lat_fp)
    x_fp = x_fp_u * float(rss_rsun)
    y_fp = y_fp_u * float(rss_rsun)
    z_fp = z_fp_u * float(rss_rsun)

    # Spacecraft Carrington trajectory context: project to a shell slightly outside source surface
    have_sc = ("phi_sc" in df.columns) and ("lat_sc" in df.columns)
    if have_sc:
        lon_sc = pd.to_numeric(df["phi_sc"], errors="coerce").to_numpy(dtype=float)
        lat_sc = pd.to_numeric(df["lat_sc"], errors="coerce").to_numpy(dtype=float)
        lon_sc = np.mod(lon_sc, 360.0)
        lat_sc = np.clip(lat_sc, -90.0, 90.0)
        x_sc_u, y_sc_u, z_sc_u = _unit_xyz_from_lonlat(lon_sc, lat_sc)
        sc_shell_r = float(cfg.sc_shell_r) if (cfg.sc_shell_r is not None) else (float(rss_rsun) * 1.25)
        x_sc = x_sc_u * sc_shell_r
        y_sc = y_sc_u * sc_shell_r
        z_sc = z_sc_u * sc_shell_r
        sc_ok = np.isfinite(x_sc) & np.isfinite(y_sc) & np.isfinite(z_sc)
        have_sc = bool(np.any(sc_ok))
    else:
        lon_sc = lat_sc = None
        x_sc = y_sc = z_sc = None
        sc_ok = None

    # HCS distance: prefer column if present; else compute from neutral line vertices.
    hcs_dist = None
    if str(cfg.hcs_dist_var).strip() and (str(cfg.hcs_dist_var) in df.columns):
        hcs_dist = pd.to_numeric(df[str(cfg.hcs_dist_var)], errors="coerce").to_numpy(dtype=float)

    # Br at footpoint: prefer df column; else sample from maps (source-surface for HCS logic).
    br_fp_col = None
    if str(cfg.pfss_value_var).strip() and (str(cfg.pfss_value_var) in df.columns):
        br_fp_col = pd.to_numeric(df[str(cfg.pfss_value_var)], errors="coerce").to_numpy(dtype=float)

    # Precompute sampled values for selected frames (cheap, and avoids re-sampling at every frame)
    br_ss_fp = np.full(n, np.nan, dtype=float)
    br_tex_fp = np.full(n, np.nan, dtype=float)
    hcs_dist_calc = np.full(n, np.nan, dtype=float)

    # Per-day vectorized sampling for selected frames only
    for dstr in unique_days:
        m = np.array([str(pfss_date[i]).strip() == dstr for i in idx_all], dtype=bool)
        if not np.any(m):
            continue
        ii = idx_all[m]
        lon_m = lon_fp[ii]
        lat_m = lat_fp[ii]
        br_ss_fp[ii] = _sample_br_nearest(raw_ss_by_day[dstr], lon_m, lat_m)

        # texture sample at same lon/lat (purely visual)
        br_tex_full = raw_photo_by_day[dstr] if (which_tex == "photosphere") else raw_ss_by_day[dstr]
        br_tex_fp[ii] = _sample_br_nearest(br_tex_full, lon_m, lat_m)

        if hcs_dist is None:
            nl = nl_by_day.get(dstr, None)
            if nl is not None:
                nl_lon, nl_lat = nl
                # compute pointwise for selected frames; neutral line length is moderate => OK
                out = []
                for LON, LAT in zip(lon_m, lat_m):
                    out.append(_min_gc_dist_deg_to_polyline(float(LON), float(LAT), nl_lon, nl_lat))
                hcs_dist_calc[ii] = np.asarray(out, dtype=float)

    # Choose HCS distance source
    if hcs_dist is None:
        hcs_dist_use = hcs_dist_calc
    else:
        # Fill missing values (if any) with computed values
        hcs_dist_use = np.asarray(hcs_dist, dtype=float)
        miss = ~np.isfinite(hcs_dist_use)
        if np.any(miss):
            hcs_dist_use[miss] = hcs_dist_calc[miss]

    # Choose Br to drive crossing logic
    br_ss_use = br_fp_col if (br_fp_col is not None) else br_ss_fp
    # Marker value for PFSS colorbar
    marker_from = str(cfg.pfss_marker_from).strip().lower()
    if marker_from not in {"source_surface_fp", "texture_fp"}:
        marker_from = "source_surface_fp"
    br_marker_use = br_ss_use if (marker_from == "source_surface_fp") else br_tex_fp

    # ------------------------------------------------------------------
    # Layout: top = 3D; optional bottom row panels
    # ------------------------------------------------------------------
    if bool(cfg.show_panels) and len(cfg.panel_vars) >= 1:
        from matplotlib.gridspec import GridSpec
        n_pan = int(len(cfg.panel_vars))
        fig = plt.figure(figsize=tuple(cfg.figsize), constrained_layout=False)
        gs = GridSpec(2, n_pan, figure=fig, height_ratios=[3.1, 1.2])
        ax3d = fig.add_subplot(gs[0, :], projection="3d")
        ax_pan = [fig.add_subplot(gs[1, j]) for j in range(n_pan)]
        fig.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.07, hspace=0.24, wspace=0.18)
    else:
        fig = plt.figure(figsize=tuple(cfg.figsize), constrained_layout=False)
        ax3d = fig.add_subplot(111, projection="3d")
        ax_pan = []
        fig.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.07)

    ax3d.set_axis_off()
    try:
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass

    # Build lon/lat grid consistent with native PFSS map; apply decimation.
    d0 = unique_days[0]
    br0_raw = raw_photo_by_day[d0] if (which_tex == "photosphere") else raw_ss_by_day[d0]
    nlat_full, nlon_full = br0_raw.shape
    lon_full = np.linspace(0.0, 360.0, int(nlon_full), endpoint=False)
    lat_full = np.linspace(-90.0, 90.0, int(nlat_full), endpoint=True)
    lon_ax = lon_full[::s_lon] if s_lon > 1 else lon_full
    lat_ax = lat_full[::s_lat] if s_lat > 1 else lat_full
    lon_grid, lat_grid = np.meshgrid(lon_ax, lat_ax)

    x_u, y_u, z_u = _unit_xyz_from_lonlat(lon_grid, lat_grid)
    r_tex = 1.0 if (which_tex == "photosphere") else float(rss_rsun)
    xs = x_u * float(r_tex)
    ys = y_u * float(r_tex)
    zs = z_u * float(r_tex)

    # Smooth source-surface shell for context (no grid lines)
    xs_ss = x_u * float(rss_rsun)
    ys_ss = y_u * float(rss_rsun)
    zs_ss = z_u * float(rss_rsun)

    # Artists
    surf = None
    neutral_line_artist = None
    fp_patch = None
    fp_tail_artist = None

    sc_orbit_artist = None
    sc_tail_artist = None
    sc_point_artist = None
    sc_link_artist = None

    # Colorbars + marker lines
    cb_pfss = None
    pfss_marker_line = None

    cb_point = None
    point_marker_line = None

    # Panel cursor lines + panel marker lines
    vlines = []
    panel_cbar_markers = []

    # Info overlay (2D text anchored to axes)
    info_text = ax3d.text2D(0.02, 0.02, "", transform=ax3d.transAxes)

    def _build_surface(day: str) -> None:
        nonlocal surf, neutral_line_artist

        if surf is not None:
            try:
                surf.remove()
            except Exception:
                pass
            surf = None
        if neutral_line_artist is not None:
            try:
                neutral_line_artist.remove()
            except Exception:
                pass
            neutral_line_artist = None

        br = tex_by_day[day]
        fc = cmap_pfss(norm_pfss(br))
        surf = ax3d.plot_surface(
            xs, ys, zs,
            rstride=1, cstride=1,
            facecolors=fc,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

        if bool(cfg.neutral_line) and (nl_by_day.get(day) is not None):
            nl_lon, nl_lat = nl_by_day[day]
            # neutral line always on source surface
            xnl, ynl, znl = _unit_xyz_from_lonlat(nl_lon, nl_lat)
            neutral_line_artist, = ax3d.plot(
                xnl * float(rss_rsun),
                ynl * float(rss_rsun),
                znl * float(rss_rsun),
                color="k",
                lw=2.8,
                alpha=0.85,
            )

    # Init surface
    day0 = str(pfss_date[idx_all[0]]).strip()
    if (not day0) or (day0 not in tex_by_day):
        day0 = unique_days[0]
    _build_surface(day0)

    # Context shell (source surface)
    if bool(cfg.show_source_surface_shell) and float(rss_rsun) != float(r_tex):
        try:
            ax3d.plot_surface(
                xs_ss, ys_ss, zs_ss,
                rstride=1, cstride=1,
                color=str(cfg.shell_color),
                alpha=float(cfg.shell_opacity),
                linewidth=0.0,
                antialiased=False,
                shade=False,
            )
        except Exception:
            pass

    # Spacecraft context
    if have_sc and bool(cfg.show_sc):
        try:
            sc_orbit_artist, = ax3d.plot(
                x_sc[idx_all], y_sc[idx_all], z_sc[idx_all],
                color="k", lw=2.6, alpha=0.30,
            )
            sc_tail_artist, = ax3d.plot(
                [float(x_sc[idx_all[0]])], [float(y_sc[idx_all[0]])], [float(z_sc[idx_all[0]])],
                color="k", lw=3.0, alpha=0.55,
            )
            sc_point_artist = ax3d.scatter(
                [float(x_sc[idx_all[0]])], [float(y_sc[idx_all[0]])], [float(z_sc[idx_all[0]])],
                s=38, marker="o", depthshade=False, c="k",
            )
            if bool(cfg.show_sc_connector):
                sc_link_artist, = ax3d.plot(
                    [float(x_sc[idx_all[0]]), float(x_fp[idx_all[0]])],
                    [float(y_sc[idx_all[0]]), float(y_fp[idx_all[0]])],
                    [float(z_sc[idx_all[0]]), float(z_fp[idx_all[0]])],
                    color="k", lw=2.0, alpha=0.25,
                )
        except Exception:
            sc_orbit_artist = None
            sc_tail_artist = None
            sc_point_artist = None
            sc_link_artist = None

    # Footpoint patch + tail (on source surface)
    i0 = int(idx_all[0])
    quad0 = _sphere_patch_quad(float(lon_fp[i0]), float(lat_fp[i0]), size_deg=float(cfg.fp_patch_size_deg), r=float(rss_rsun))
    face0 = str(cfg.fp_face_rgba)
    if (point_vals is not None) and (point_norm is not None):
        v0 = float(point_vals[i0]) if np.isfinite(point_vals[i0]) else np.nan
        if np.isfinite(v0):
            face0 = cmap_point(point_norm(v0))
    fp_patch = Poly3DCollection([quad0], facecolors=[face0], edgecolors=[str(cfg.fp_edge_rgba)], linewidths=[float(cfg.fp_edge_lw)], alpha=float(cfg.fp_patch_opacity))
    ax3d.add_collection3d(fp_patch)

    x0, y0, z0 = float(x_fp[i0]), float(y_fp[i0]), float(z_fp[i0])
    fp_tail_artist, = ax3d.plot([x0], [y0], [z0], color="k", lw=2.2, alpha=0.65)

    # Axes limits (tight, stable)
    rmax = float(max(float(rss_rsun), float(r_tex), float((float(cfg.sc_shell_r) if cfg.sc_shell_r is not None else (float(rss_rsun) * 1.25)))))
    lim = rmax * float(cfg.lim_factor)
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)

    # Colorbars (PFSS + optional point variable)
    if bool(cfg.show_colorbar):
        cax_pfss = fig.add_axes([0.92, 0.32, 0.016, 0.44])
        cb_pfss = fig.colorbar(sm_pfss, cax=cax_pfss)
        cb_pfss.set_label(str(cfg.colorbar_label))
        try:
            cb_pfss.ax.set_facecolor("0.92")
            cb_pfss.outline.set_edgecolor("0.30")
        except Exception:
            pass
        if bool(cfg.colorbar_marker):
            pfss_marker_line = cb_pfss.ax.axhline(0.0, color="k", lw=2.0, alpha=0.85)

    if bool(cfg.show_point_colorbar) and (sm_point is not None):
        cax_pv = fig.add_axes([0.945, 0.32, 0.016, 0.44])
        cb_point = fig.colorbar(sm_point, cax=cax_pv)
        lab = str(cfg.point_colorbar_label).strip() if str(cfg.point_colorbar_label).strip() else str(cfg.point_value_var)
        cb_point.set_label(lab)
        try:
            cb_point.ax.set_facecolor("0.92")
            cb_point.outline.set_edgecolor("0.30")
        except Exception:
            pass
        point_marker_line = cb_point.ax.axhline(0.0, color="k", lw=2.0, alpha=0.85)

    # Panels: time series + per-panel colorbar with marker
    if ax_pan:
        for ax, vname in zip(ax_pan, cfg.panel_vars):
            vlines.append(None)
            panel_cbar_markers.append(None)

            if vname not in df.columns:
                ax.text(0.5, 0.5, f"missing: {vname}", ha="center", va="center", transform=ax.transAxes)
                continue

            vv = pd.to_numeric(df[vname], errors="coerce").to_numpy(dtype=float)
            ax.plot(df.index, vv, lw=1.2)
            ax.set_title(str(vname))
            ax.grid(True, alpha=0.25)
            vl = ax.axvline(df.index[i0], lw=1.4, alpha=0.70)
            vlines[-1] = vl

            if bool(cfg.panel_colorbars):
                # Small inset colorbar at the right of each panel
                try:
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    cax = inset_axes(ax, width="6%", height="80%", loc="center right",
                                     bbox_to_anchor=(0.08, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0.0)
                except Exception:
                    # Fallback: manual placement using axes bounding box
                    bb = ax.get_position()
                    cax = fig.add_axes([bb.x1 + 0.006, bb.y0 + 0.05 * bb.height, 0.010, 0.90 * bb.height])

                vv_f = vv[np.isfinite(vv)]
                if vv_f.size:
                    vmin = float(np.nanpercentile(vv_f, 2.0))
                    vmax = float(np.nanpercentile(vv_f, 98.0))
                    if vmin == vmax:
                        vmin, vmax = (vmin - 1.0, vmax + 1.0)
                else:
                    vmin, vmax = (0.0, 1.0)

                norm = Normalize(vmin=vmin, vmax=vmax)
                sm = plt.cm.ScalarMappable(cmap=cmap_point, norm=norm)
                sm.set_array([])
                cbp = fig.colorbar(sm, cax=cax)
                cbp.ax.tick_params(labelsize=7)
                cbp.outline.set_edgecolor("0.30")
                cbp.ax.set_facecolor("0.92")
                mk = cbp.ax.axhline(float(vv[i0]) if np.isfinite(vv[i0]) else vmin, color="k", lw=2.0, alpha=0.85)
                panel_cbar_markers[-1] = (mk, vv, vmin, vmax)

    # Camera init
    ax3d.view_init(elev=float(cfg.elev), azim=float(cfg.azim))
    prev_az = float(cfg.azim)
    prev_el = float(cfg.elev)

    # Crossing detection state
    flash_left = 0
    prev_sign = 0

    # Encoding
    metadata = dict(title=str(cfg.title))
    extra_args = ["-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", "23"]
    writer = FFMpegWriter(fps=int(cfg.fps), codec="libx264", extra_args=extra_args, metadata=metadata)

    with writer.saving(fig, str(out_mp4), dpi=int(cfg.dpi)):
        last_day = day0
        report_every = max(1, int(round(len(idx_all) / 20.0)))

        for k, ii in enumerate(idx_all):
            ii = int(ii)
            day = str(pfss_date[ii]).strip()
            if (not day) or (day not in tex_by_day):
                day = last_day
            if (day != last_day) and (day in tex_by_day):
                _build_surface(day)
                last_day = day

            # Update footpoint tail
            t0 = max(0, ii - int(max(0, int(cfg.tail))))
            jj = np.arange(t0, ii + 1, dtype=int)
            fp_tail_artist.set_data(x_fp[jj], y_fp[jj])
            fp_tail_artist.set_3d_properties(z_fp[jj])

            # Update footpoint patch (geometry + color)
            quad = _sphere_patch_quad(float(lon_fp[ii]), float(lat_fp[ii]), size_deg=float(cfg.fp_patch_size_deg), r=float(rss_rsun))
            fp_patch.set_verts([quad])

            face = str(cfg.fp_face_rgba)
            if (point_vals is not None) and (point_norm is not None) and np.isfinite(point_vals[ii]):
                face = cmap_point(point_norm(float(point_vals[ii])))
            fp_patch.set_facecolor(face)

            # HCS distance and crossing
            d_hcs = float(hcs_dist_use[ii]) if (hcs_dist_use is not None and np.isfinite(hcs_dist_use[ii])) else float("nan")
            br_here = float(br_ss_use[ii]) if np.isfinite(br_ss_use[ii]) else float("nan")

            sgn = int(np.sign(br_here)) if np.isfinite(br_here) else 0
            crossed = False
            if (k > 0) and (sgn != 0) and (prev_sign != 0) and (sgn != prev_sign):
                if (not bool(cfg.show_hcs_metric)) or (np.isfinite(d_hcs) and (d_hcs <= float(cfg.hcs_cross_thresh_deg))):
                    crossed = True
            prev_sign = sgn

            if crossed:
                flash_left = int(max(1, int(cfg.crossing_flash_frames)))
            if flash_left > 0:
                fp_patch.set_edgecolor(str(cfg.fp_edge_flash_rgba))
                fp_patch.set_linewidth(float(cfg.fp_edge_flash_lw))
                flash_left -= 1
            else:
                fp_patch.set_edgecolor(str(cfg.fp_edge_rgba))
                fp_patch.set_linewidth(float(cfg.fp_edge_lw))

            # Spacecraft updates + camera follow
            if have_sc and bool(cfg.show_sc) and (x_sc is not None) and (sc_tail_artist is not None) and (sc_point_artist is not None):
                s0 = max(0, ii - int(max(0, int(cfg.sc_tail))))
                sidx = np.arange(s0, ii + 1, dtype=int)
                sc_tail_artist.set_data(x_sc[sidx], y_sc[sidx])
                sc_tail_artist.set_3d_properties(z_sc[sidx])
                try:
                    sc_point_artist._offsets3d = ([float(x_sc[ii])], [float(y_sc[ii])], [float(z_sc[ii])])
                except Exception:
                    pass
                if (sc_link_artist is not None) and bool(cfg.show_sc_connector):
                    sc_link_artist.set_data([float(x_sc[ii]), float(x_fp[ii])], [float(y_sc[ii]), float(y_fp[ii])])
                    sc_link_artist.set_3d_properties([float(z_sc[ii]), float(z_fp[ii])])

                if bool(cfg.follow_sc) and np.isfinite(lon_sc[ii]) and np.isfinite(lat_sc[ii]):
                    targ_az = float(lon_sc[ii]) + float(cfg.follow_sc_azim_offset_deg)
                    targ_el = float(lat_sc[ii]) + float(cfg.follow_sc_elev_offset_deg)
                    prev_az = _smooth_angle(prev_az, targ_az, float(cfg.follow_sc_smooth))
                    prev_el = float(prev_el) + float(cfg.follow_sc_smooth) * (float(targ_el) - float(prev_el))
                    prev_el = float(max(-89.0, min(89.0, prev_el)))
                    ax3d.view_init(elev=float(prev_el), azim=float(prev_az))

            # Update PFSS marker line
            if (pfss_marker_line is not None) and np.isfinite(br_marker_use[ii]):
                yv = float(br_marker_use[ii])
                pfss_marker_line.set_ydata([yv, yv])

            if (point_marker_line is not None) and (point_vals is not None) and np.isfinite(point_vals[ii]):
                yv = float(point_vals[ii])
                point_marker_line.set_ydata([yv, yv])

            # Update panel cursors + panel colorbar markers
            if vlines:
                tnow = df.index[ii]
                for vl in vlines:
                    if vl is not None:
                        vl.set_xdata([tnow, tnow])

            if panel_cbar_markers:
                for item in panel_cbar_markers:
                    if item is None:
                        continue
                    mk, vv, vmin, vmax = item
                    val = float(vv[ii]) if np.isfinite(vv[ii]) else float(vmin)
                    mk.set_ydata([val, val])

            # Info text
            if bool(cfg.show_hcs_metric):
                info = f"PFSS day={last_day}  |  Br_ss(fp)={br_here:.3g}  |  d_HCS={d_hcs:.2f}°"
                if crossed or (flash_left > 0):
                    info += "  |  HCS crossing"
            else:
                info = f"PFSS day={last_day}  |  Br_ss(fp)={br_here:.3g}"
            info_text.set_text(info)

            # Title
            try:
                ax3d.set_title(f"{cfg.title} | t_obs={pd.Timestamp(df.index[ii])}")
            except Exception:
                pass

            writer.grab_frame()

            if (k % report_every) == 0:
                try:
                    print(f"[3D movie] frame {k+1}/{len(idx_all)} (stride={stride}, surface_stride={s_lat}x{s_lon})")
                except Exception:
                    pass

    plt.close(fig)

    if (not out_mp4.exists()) or (out_mp4.stat().st_size == 0):
        raise RuntimeError(f"3D movie was not created: {out_mp4}")
    return out_mp4


# --------------------------------------------------------------------------------------
# Plotly "carbon-copy" 3D MP4 engine
# --------------------------------------------------------------------------------------

def _load_pfss_products_for_movie3d(
    *,
    date_str: str,
    pfss_out_dir: Union[str, Path],
    N: int,
    rss_rsun: float,
    nr: Optional[int],
    prefer_hhmm: str,
    overwrite_download: bool,
    cache_maps: bool,
) -> Dict[str, Any]:
    """Load PFSS products needed by the Plotly movie engine for a given date.

    Returns:
        {
          'br_photo': 2D array,
          'br_ss': 2D array,
          'neutral_lonlat': (lon_deg, lat_deg) or None
        }
    """
    from .pfss import PFSSConfig, pfss_maps_cached, neutral_line_vertices_lonlat

    cfg = PFSSConfig(
        out_dir=Path(pfss_out_dir),
        date_str=str(date_str),
        prefer_hhmm=str(prefer_hhmm),
        overwrite_download=bool(overwrite_download),
        N=int(N),
        rss_rsun=float(rss_rsun),
        nr=(int(nr) if nr is not None else None),
    )

    maps = pfss_maps_cached(cfg, which=("photosphere", "source_surface"), cache=bool(cache_maps), overwrite=bool(overwrite_download))
    br_photo = np.asarray(maps.get("photosphere"), dtype=float)
    br_ss = np.asarray(maps.get("source_surface"), dtype=float)

    neutral_lonlat = None
    try:
        lon_nl, lat_nl = neutral_line_vertices_lonlat(br_ss, level=0.0, stride=1)
        neutral_lonlat = (np.asarray(lon_nl, dtype=float), np.asarray(lat_nl, dtype=float))
    except Exception:
        neutral_lonlat = None

    return {"br_photo": br_photo, "br_ss": br_ss, "neutral_lonlat": neutral_lonlat}


def _plotly_require_kaleido() -> None:
    import importlib.util as _iu
    if _iu.find_spec("kaleido") is None:
        raise RuntimeError(
            "Plotly frame export requires the 'kaleido' package.\n"
            "Install it with:\n"
            "  python -m pip install -U kaleido\n"
            "Then rerun the movie."
        )


def _ffmpeg_assemble_mp4(*, frames_glob: str, out_mp4: Path, fps: int) -> None:
    """Assemble an MP4 from pre-rendered PNG frames via ffmpeg.

    Notes (Windows):
      - Some ffmpeg builds (notably some conda-forge Windows builds) do NOT support
        libavformat globbing ("-pattern_type glob").
      - We therefore avoid ffmpeg-side globbing and instead use a numeric sequence
        pattern when possible, falling back to the concat demuxer if needed.
      - We still do a Python-side glob first, to fail early with a clear message if
        no frames exist.
    """
    import shutil
    import subprocess
    from glob import glob

    fps_i = int(fps)

    # Fail early with a useful error if no frames exist.
    frames = sorted(glob(frames_glob))
    if not frames:
        raise RuntimeError(
            f"ffmpeg assemble: no frames matched frames_glob={frames_glob!r}.\n"
            f"Looked for: {frames_glob}"
        )

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH.")
    # IMPORTANT (Windows/conda-forge ffmpeg): some builds do NOT support
    # "-pattern_type glob" (libavformat globbing). Therefore we avoid ffmpeg-side
    # globbing entirely and instead:
    #   1) Python-side glob to get the exact frame list (already done), then
    #   2) Prefer a numeric sequence pattern (image2) when frames are sequential,
    #   3) Otherwise fall back to concat-demuxer with explicit per-frame durations.

    from pathlib import Path as _Path
    import re as _re
    import tempfile

    frame_paths = [_Path(fp) for fp in frames]
    out_mp4_ff = str(out_mp4).replace("\\", "/")

    parent = frame_paths[0].parent
    same_parent = all(fp.parent == parent for fp in frame_paths)

    seq_ok = False
    seq_start = 0
    seq_width = 0
    seq_prefix = ""
    seq_suffix = ""

    if same_parent:
        m0 = _re.match(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$", frame_paths[0].name)
        if m0 is not None:
            seq_prefix = m0.group(1)
            seq_suffix = m0.group(3)
            seq_width = len(m0.group(2))
            nums = []
            for fp in frame_paths:
                m = _re.match(r"^(.*?)(\d+)(\.[A-Za-z0-9]+)$", fp.name)
                if (
                    (m is None)
                    or (m.group(1) != seq_prefix)
                    or (m.group(3) != seq_suffix)
                    or (len(m.group(2)) != seq_width)
                ):
                    nums = []
                    break
                nums.append(int(m.group(2)))

            if nums:
                nums_sorted = sorted(nums)
                seq_start = int(nums_sorted[0])
                if nums_sorted == list(range(seq_start, seq_start + len(nums_sorted))):
                    seq_ok = True

    if seq_ok:
        # image2 numeric sequence (most robust across ffmpeg builds)
        in_pattern = (parent / f"{seq_prefix}%0{seq_width}d{seq_suffix}").as_posix()
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps_i),
            "-start_number",
            str(seq_start),
            "-i",
            in_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out_mp4_ff,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
    else:
        # Fallback: concat demuxer with explicit per-frame durations.
        # This works even if filenames are not a strict numeric sequence.
        dt = 1.0 / float(max(1, fps_i))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            list_path = _Path(tf.name)
            for fp in frame_paths[:-1]:
                tf.write(f"file '{fp.as_posix()}'\n")
                tf.write(f"duration {dt:.9f}\n")
            tf.write(f"file '{frame_paths[-1].as_posix()}'\n")
            # Repeat last file when using duration lines.
            tf.write(f"file '{frame_paths[-1].as_posix()}'\n")

        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path.as_posix(),
            "-r",
            str(fps_i),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            out_mp4_ff,
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        finally:
            try:
                list_path.unlink(missing_ok=True)
            except Exception:
                pass

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        msg = (
            f"ffmpeg failed (returncode={proc.returncode}).\n"
            f"cmd: {cmd}\n"
            f"frames: {len(frames)} matched\n"
        )
        if stdout:
            msg += "\n[ffmpeg stdout]\n" + stdout[:4000]
        if stderr:
            msg += "\n[ffmpeg stderr]\n" + stderr[:4000]
        raise RuntimeError(msg)





def _movie3d_plotly_export(
    *,
    data: Union[str, Path],
    cfg: Movie3DConfig,
    pfss_out_dir: Union[str, Path],
    N: int,
    rss_rsun: float,
    nr: Optional[int],
    prefer_hhmm: str,
    overwrite_download: bool,
    cache_maps: bool,
    clim: Optional[Tuple[float, float]],
) -> Path:
    """Render a *true carbon-copy* 3D MP4 by exporting the Plotly HTML figure per frame."""
    from .plotting import plot_source_surface_3d, VAR_SPECS, merge_var_specs, _compute_scalar_limits
    from .pfss import robust_symmetric_clim, angular_distance_to_neutral_line_deg

    _plotly_require_kaleido()

    out_mp4 = Path(cfg.out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    if out_mp4.exists() and (not bool(cfg.overwrite)):
        return out_mp4

    df = pd.read_pickle(Path(data)) if str(data).lower().endswith((".pkl", ".pickle")) else pd.read_parquet(Path(data))
    df = _ensure_columns(df, pfss_date_mode=str(cfg.pfss_date_mode), pfss_date_str=str(cfg.pfss_date_str))
    if df.empty:
        raise RuntimeError("Empty DataFrame.")

    # Determine stride (same logic as mpl path, but re-implemented minimally).
    n = int(len(df))
    t_obs = pd.to_datetime(pd.DatetimeIndex(df.index), utc=True)
    t_ns = t_obs.view("i8")
    dt_ns = np.diff(t_ns)
    dt_ns = dt_ns[np.isfinite(dt_ns)]
    native_dt_s = float(np.nanmedian(dt_ns)) * 1e-9 if dt_ns.size else 0.0
    native_dt_s = float(native_dt_s) if np.isfinite(native_dt_s) else 0.0
    native_dt_s = max(native_dt_s, 0.0)

    user_stride = int(getattr(cfg, "stride", 0) or 0)
    stride = max(1, user_stride) if user_stride > 0 else 1
    max_frames = int(max(2, int(getattr(cfg, "max_frames", 1500))))
    if user_stride <= 0:
        stride_frames = int(np.ceil(n / float(max_frames)))
        stride = max(stride, max(1, stride_frames))
    min_step_s = float(getattr(cfg, "min_step_s", 0.0) or 0.0)
    if (user_stride <= 0) and (n > max_frames) and (native_dt_s > 0.0) and (min_step_s > 0.0):
        stride_min = int(np.ceil(min_step_s / native_dt_s))
        stride = max(stride, max(1, stride_min))

    idx_all = np.arange(n, dtype=int)[::stride]
    if idx_all.size < 2:
        idx_all = np.array([0, n - 1], dtype=int)

    # PFSS product cache per day used in frames
    dates = [str(x) for x in df["pfss_date"].astype(str).to_numpy()]
    def _ok_date_key(s: str) -> bool:
        ss = str(s).strip()
        if not ss:
            return False
        if ss.lower() in {"nan", "nat", "none", "<na>"}:
            return False
        import re as _re
        return bool(_re.match(r"^\d{4}-\d{2}-\d{2}$", ss))

    days_used = [dates[i] for i in idx_all if _ok_date_key(dates[i])]
    if len(days_used) == 0:
        raise RuntimeError("No valid PFSS dates found in data['pfss_date'] for the requested frames.")
    uniq_days = sorted(set(days_used))

    pfss_by_day: Dict[str, Dict[str, Any]] = {}
    for day in uniq_days:
        pfss_by_day[day] = _load_pfss_products_for_movie3d(
            date_str=day,
            pfss_out_dir=pfss_out_dir,
            N=int(N),
            rss_rsun=float(rss_rsun),
            nr=nr,
            prefer_hhmm=str(prefer_hhmm),
            overwrite_download=bool(overwrite_download),
            cache_maps=bool(cache_maps),
        )

    # Stable PFSS clim on the rendered texture map (photosphere by default)
    if clim is None:
        try:
            clim = robust_symmetric_clim([pfss_by_day[d]["br_photo"] for d in uniq_days], percentiles=(2.0, 98.0), fallback=1.0)
        except Exception:
            clim = (-1.0, +1.0)

    # Stable scalar limits for plot_vars (precompute from the full interval so colorbars don't swim)
    plot_vars = tuple(getattr(cfg, "plot_vars", ("polarity", "Vr_bg", "P_ram", "sigma_c")))
    specs = merge_var_specs(VAR_SPECS, None)
    percentiles = (2.0, 98.0)
    for v in plot_vars:
        if v == "polarity":
            continue
        if v not in df.columns:
            continue
        spec = specs.get(v, {})
        if (spec.get("vmin", None) is not None) and (spec.get("vmax", None) is not None):
            continue
        arr = pd.to_numeric(df[v], errors="coerce").to_numpy(dtype=float)
        try:
            lo, hi = _compute_scalar_limits(arr, spec=spec, percentiles=percentiles)
            spec["vmin"] = float(lo)
            spec["vmax"] = float(hi)
            specs[v] = spec
        except Exception:
            pass

    # Conversion: Rsun in AU
    r_sun_au = float(df.attrs.get("r_sun_au", 1.0 / 215.032))  # fallback
    r_ss_au = float(rss_rsun) * float(r_sun_au)

    frames_dir = Path(pfss_out_dir) / "frames_3d_plotly"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # HCS distance metric fallback if absent: compute from neutral line
    have_hcs_dist = str(getattr(cfg, "hcs_dist_var", "pfss_hcs_dist_deg")) in df.columns

    # Br sign at footpoint (for crossing detection)
    pfss_br_col = str(getattr(cfg, "pfss_value_var", "pfss_br"))
    have_pfss_br = pfss_br_col in df.columns

    # Helper to compute Plotly camera from SC position (AU) in the plotting frame
    cam_prev_az = float(getattr(cfg, "follow_sc_azim_offset_deg", 35.0))
    cam_prev_el = float(getattr(cfg, "follow_sc_elev_offset_deg", 20.0))

    def _camera_from_sc(i0: int) -> Optional[Dict[str, Any]]:
        nonlocal cam_prev_az, cam_prev_el
        if not bool(getattr(cfg, "follow_sc", True)):
            return None
        if not {"sc_x_au", "sc_y_au", "sc_z_au"}.issubset(df.columns):
            return None
        x = float(df["sc_x_au"].iloc[i0])
        y = float(df["sc_y_au"].iloc[i0])
        z = float(df["sc_z_au"].iloc[i0])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return None
        # Convert to azimuth/elevation in degrees
        az = float(np.rad2deg(np.arctan2(y, x)))
        rxy = float(np.hypot(x, y))
        el = float(np.rad2deg(np.arctan2(z, rxy)))
        # Smooth
        alpha = float(getattr(cfg, "follow_sc_smooth", 0.18))
        cam_prev_az = _smooth_angle(cam_prev_az, az + float(getattr(cfg, "follow_sc_azim_offset_deg", 35.0)), alpha)
        cam_prev_el = _smooth_angle(cam_prev_el, el + float(getattr(cfg, "follow_sc_elev_offset_deg", 20.0)), alpha)

        # Plotly camera uses an 'eye' vector in scene coordinates; approximate with spherical eye.
        dist = float(getattr(cfg, "camera_distance", 1.75))
        zb = float(getattr(cfg, "camera_z_boost", 0.22))
        azr = np.deg2rad(cam_prev_az)
        elr = np.deg2rad(cam_prev_el)
        ex = dist * np.cos(elr) * np.cos(azr)
        ey = dist * np.cos(elr) * np.sin(azr)
        ez = dist * np.sin(elr) + zb
        return {"eye": {"x": float(ex), "y": float(ey), "z": float(ez)}}

    # Precompute crossing triggers (frame indices) using sign flips + proximity to NL
    crossing_frames: Dict[int, bool] = {}
    if bool(getattr(cfg, "neutral_line", True)):
        last_sign = None
        for k, i0 in enumerate(idx_all):
            day = dates[i0]
            prod = pfss_by_day.get(day, None)
            if prod is None:
                continue
            nl = prod.get("neutral_lonlat", None)
            if nl is None:
                continue
            lon_fp = float(df["phi_src"].iloc[i0])
            lat_fp = float(df["lat_src"].iloc[i0])

            if have_pfss_br:
                br_fp = float(df[pfss_br_col].iloc[i0])
            else:
                br_fp = float(_sample_br_nearest(np.asarray(prod["br_ss"], float), np.array([lon_fp]), np.array([lat_fp]))[0])

            sgn = 0 if (not np.isfinite(br_fp)) else (1 if br_fp > 0 else (-1 if br_fp < 0 else 0))
            if have_hcs_dist:
                dh = float(df[str(getattr(cfg, "hcs_dist_var", "pfss_hcs_dist_deg"))].iloc[i0])
            else:
                dh = float(angular_distance_to_neutral_line_deg(lon_fp, lat_fp, nl[0], nl[1]))
            near = np.isfinite(dh) and (dh <= float(getattr(cfg, "hcs_cross_thresh_deg", 3.0)))
            if (last_sign is not None) and (sgn != 0) and (last_sign != 0) and (sgn != last_sign) and near:
                crossing_frames[int(i0)] = True
            last_sign = sgn

    # Render frames
    for frame_idx, i0 in enumerate(idx_all):
        # Window selection
        if str(getattr(cfg, "draw_mode", "tail")).lower().strip() == "cumulative":
            lo = 0
        else:
            lo = max(0, int(i0) - int(getattr(cfg, "tail", 240)))
        sub = df.iloc[lo : i0 + 1].copy()

        day = dates[i0]
        prod = pfss_by_day.get(day, pfss_by_day[uniq_days[0]])
        br_photo = np.asarray(prod["br_photo"], dtype=float)
        nl = prod.get("neutral_lonlat", None)

        # HCS distance metric
        if bool(getattr(cfg, "show_hcs_metric", True)):
            if have_hcs_dist:
                dh = float(df[str(getattr(cfg, "hcs_dist_var", "pfss_hcs_dist_deg"))].iloc[i0])
            else:
                if nl is None:
                    dh = float("nan")
                else:
                    dh = float(angular_distance_to_neutral_line_deg(float(df["phi_src"].iloc[i0]), float(df["lat_src"].iloc[i0]), nl[0], nl[1]))
        else:
            dh = float("nan")

        tstr = pd.Timestamp(df.index[i0]).strftime("%Y-%m-%d %H:%M:%S")
        title = str(getattr(cfg, "title", "PFSS + backmapping"))
        if bool(getattr(cfg, "show_hcs_metric", True)):
            if np.isfinite(dh):
                title = f"{title}<br>{tstr} UTC | PFSS day {day} | d_HCS={dh:.2f}°"
            else:
                title = f"{title}<br>{tstr} UTC | PFSS day {day}"

        # Colorbar markers: instantaneous values for each panel variable
        cb_mark = {}
        if bool(getattr(cfg, "show_panel_colorbar_markers", True)):
            for v in plot_vars:
                if v == "polarity":
                    continue
                if v in df.columns:
                    try:
                        cb_mark[v] = float(df[v].iloc[i0])
                    except Exception:
                        pass

        # Crossing flash
        flash = False
        if int(getattr(cfg, "crossing_flash_frames", 0)) > 0:
            if crossing_frames.get(int(i0), False):
                flash = True
            else:
                # flash for a few frames after the crossing detection frame
                for j in range(1, int(getattr(cfg, "crossing_flash_frames", 8)) + 1):
                    if crossing_frames.get(int(i0) - j * int(stride), False):
                        flash = True
                        break

        edge_rgba = str(getattr(cfg, "highlight_edge_rgba", "rgba(0,0,0,0.95)"))
        if flash:
            edge_rgba = str(getattr(cfg, "highlight_flash_edge_rgba", "rgba(255,0,0,0.95)"))

        cam = _camera_from_sc(int(i0))

        out_html_dummy = frames_dir / "dummy.html"  # not written when write_html=False
        fig_path, fig = plot_source_surface_3d(
            data=sub,
            out_html=out_html_dummy,
            var_specs=specs,
            r_ss_au=float(r_ss_au),
            r_sun_au=float(r_sun_au),
            frame3d=str(getattr(cfg, "frame3d", "HGC")),
            plot_vars=list(plot_vars),
            percentiles=percentiles,
            ncols_vars=int(getattr(cfg, "ncols_vars", 2)),
            panel_px=int(getattr(cfg, "panel_px", 650)),
            width=(int(getattr(cfg, "export_width")) if getattr(cfg, "export_width", None) is not None else None),
            height=(int(getattr(cfg, "export_height")) if getattr(cfg, "export_height", None) is not None else None),
            decimate=max(1, int(getattr(cfg, "decimate", 1))),
            show_sphere_grid=bool(getattr(cfg, "show_sphere_grid", False)),
            camera_dict=cam,
            title=title,
            show=False,
            write_html=False,
            cb_marker_values=cb_mark if cb_mark else None,
            highlight_last_point=False,
            highlight_size=int(getattr(cfg, "highlight_size", 14)),
            highlight_edge_rgba=edge_rgba,
            highlight_edge_width=int(getattr(cfg, "highlight_edge_width", 5)),
            highlight_fill_rgba="rgba(255,255,255,0.92)",
            highlight_connector=bool(getattr(cfg, "show_sc_connector", True)),
            # PFSS overlay
            pfss_br2d=br_photo,
            pfss_which_br="photosphere",
            pfss_clim=clim,
            pfss_show_colorbar=True,
            pfss_show_in_all_panels=True,
            pfss_neutral_lonlat=(nl if bool(getattr(cfg, "neutral_line", True)) else None),
        )

        frame_png = frames_dir / f"frame_{frame_idx:06d}.png"
        fig.write_image(str(frame_png), scale=float(getattr(cfg, "export_scale", 1.6)))

    # Assemble mp4
    _ffmpeg_assemble_mp4(frames_glob=str(frames_dir / "frame_*.png"), out_mp4=out_mp4, fps=int(getattr(cfg, "fps", 12)))
    return out_mp4
