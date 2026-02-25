# functions/interactive_figs.py

"""Interactive interval viewer for MHDTurbPy downloaded-interval pickles.

This module renders multi-panel timeseries figures and provides an interactive
interval selector (click/drag + key bindings) that writes selected intervals
to disk.

Coordinate frames (explicit):
- Magnetic field panels: inferred per spacecraft from Mag['B_resampled'] columns.
    * RTN if (Br,Bt,Bn) exist
    * GSE if (Bx,By,Bz) exist
  The module computes |B| and stores it in the historical column name 'B_RTN'
  for backward compatibility (it is a magnitude, independent of frame).

- Flow-separation panel (if enabled): separation is computed in **GSE** using
  cached JPL Horizons ephemerides (x_au,y_au,z_au in GSE). The separation vector
  is decomposed relative to an estimated flow direction \hat{v}(t).

  flow_mode options (how \hat{v}(t) is built):
    * 'mean' / 'mean_v' : average of the available V(t) from sc[0] and sc[1]
    * 'sc1_v'           : use V(t) from the first spacecraft in the input list (sc[0])
    * 'sc2_v'           : use V(t) from the second spacecraft in the input list (sc[1])

  flow_dir_gse and vsw_fallback are **fallbacks** only: they are used when the
  required V(t) components are missing/invalid (or at timestamps where |V| is not
  finite/positive). They do not override measured velocities when those exist.

Legend placement:
- Set alternate_legend_sides=True to alternate legend anchors right/left
  panel-by-panel.

Distance panel (automatic, conditional):
- If enable_flow_separation=False (default): show heliocentric distance from the Sun in AU.
  The timeseries is taken from Par['V_resampled']['Dist_au'] when available; if missing it is downloaded
  from JPL Horizons (SunPy) and cached back into final.pkl.
- If enable_flow_separation=True: the *same* distance panel is replaced to show geocentric distance from Earth
  in Earth radii, computed from cached Horizons ephemerides (x_au,y_au,z_au in Geocentric Solar Ecliptic ≈ 'GSE'):
      d_E(t)[R_E] = sqrt(x_au^2 + y_au^2 + z_au^2) * (AU / R_E).
  No extra panel is appended; the existing distance panel is repurposed.

Minimal example (MRE):
    %matplotlib tk
    from pathlib import Path
    import matplotlib.pyplot as plt
    import interactive_figs as ifigs
    import general_functions as func
    ROOT = Path(r'C:\\Users\\nokni\\work\\MHDTurbPy')
    SC = ['WIND','ACE']
    LOAD = {s: ROOT/'examples'/'downloaded_intervals'/s for s in SC}
    fig, events = ifigs.interactive_mhdturbpy_interval(
        sc=SC, which_int=0, load_path=LOAD,
        my_dir=ROOT/'examples', save_path=ROOT/'examples'/'selected_intervals',
        load_files_func=func.load_files,
        enable_flow_separation=True, flow_mode='mean_v', flow_v_smooth_window='2min',
        alternate_legend_sides=True,
    )
    plt.show()
"""

import os
import sys
import time
import io
import re
import math
import contextlib
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import MultiCursor

try:
    import general_functions as func
except Exception:
    func = None

AU_KM = 149597870.7 
R_EARTH_KM = 6378.1363
AU_IN_RE = AU_KM / R_EARTH_KM

# ============================================================
# GLOBAL DEFAULTS
# ============================================================
DEFAULT_PLOT_PARAMS = {
    "figure": {
        "base_width": 0.95 * 30.0,
        "base_height_7pan": 0.95 * 15.0,
        "gridspec_kw": {"wspace": 0.05, "hspace": 0.05},
    },
    "ticks": {"size": "small"},
    "grid": {
        "x_minor": {"which": "minor", "linewidth": 0.1, "ls": ":"},
        "x_major": {"which": "major", "linewidth": 0.1, "ls": ":"},
        "y_minor": {"which": "minor", "linewidth": 0.1, "ls": ":"},
        "y_major": {"which": "major", "linewidth": 0.1, "ls": ":"},
    },
    "legend_presets": {
        "main": {"fontsize": "small", "frameon": False, "bbox_to_anchor": (1.01, 1.0), "loc": 2, "ncol": 1},
        "side": {"fontsize": "small", "frameon": False, "bbox_to_anchor": (1.01, 0.6), "loc": 2, "ncol": 1},
        "main_left": {"fontsize": "small", "frameon": False, "bbox_to_anchor": (-0.01, 1.0), "loc": 1, "ncol": 1},
        "side_left": {"fontsize": "small", "frameon": False, "bbox_to_anchor": (-0.01, 0.6), "loc": 1, "ncol": 1},
    },
    "series_style_defaults": {"lw": 0.8, "ls": "-", "ms": 0},
    "auto_ylims": {"min_factor": 0.95, "max_factor": 1.05},
    "interaction": {
        "status_fontsize": 10,
        "help_fontsize": 10,
        "help_facecolor": "0.96",
        "help_edgecolor": "0.75",
        "help_alpha": 0.95,
    },
}

VALID_SNAP_INDEX_MODES = {"first_sc", "union"}
VALID_NORMALIZE_METHODS = {"zscore", "minmax", "median", "first", "none"}


# ============================================================
# Utilities
# ============================================================
def format_timestamp(ts, fmt="%Y_%m_%d"):
    return pd.Timestamp(ts).strftime(fmt)


def inset_axis_params(size="xx-large"):
    minor_tick_params = dict(which="minor", length=3, width=0.8, labelsize=size, direction="in")
    major_tick_params = dict(which="major", length=6, width=1.0, labelsize=size, direction="in")
    return minor_tick_params, major_tick_params


def load_files(load_path, pattern):
    load_path = str(load_path)
    hits = glob(os.path.join(load_path, "**", pattern), recursive=True)
    return sorted(hits)


def _as_sc_list(sc):
    if isinstance(sc, (list, tuple)):
        out = [str(s) for s in sc]
    else:
        out = [str(sc)]
    out = [s for s in out if len(s) > 0]
    if len(out) == 0:
        raise ValueError("at least one spacecraft name must be provided")
    return out


def _normalize_sc_df_input(value, sc_list, name):
    if isinstance(value, pd.DataFrame):
        return {sc_name: value for sc_name in sc_list}
    if isinstance(value, dict):
        out = {}
        missing = [s for s in sc_list if s not in value]
        if missing:
            raise KeyError(f"{name}: missing spacecraft keys: {missing}")
        for s in sc_list:
            v = value[s]
            if not isinstance(v, pd.DataFrame):
                raise TypeError(f"{name}[{s!r}] expected DataFrame, got {type(v)}")
            out[s] = v
        return out
    raise TypeError(f"{name} must be DataFrame or dict[str, DataFrame], got {type(value)}")


def _sc_tag(sc_list):
    return "-".join(sc_list)


def _normalize_snap_index_mode(snap_index_mode):
    mode = str(snap_index_mode).lower().strip()
    if mode not in VALID_SNAP_INDEX_MODES:
        valid = ", ".join(sorted(VALID_SNAP_INDEX_MODES))
        raise ValueError(f"snap_index_mode must be one of {{{valid}}}, got {snap_index_mode!r}")
    return mode


def _normalize_timeseries_mode(normalize_timeseries):
    if normalize_timeseries is None:
        return None
    mode = str(normalize_timeseries).lower().strip()
    if mode in ("", "none"):
        return None
    if mode not in VALID_NORMALIZE_METHODS:
        valid = ", ".join(sorted(VALID_NORMALIZE_METHODS))
        raise ValueError(f"normalize_timeseries must be one of {{{valid}}} or None, got {normalize_timeseries!r}")
    return mode


def _series_has_units(label):
    if label is None:
        return False
    return re.search(r"[^[^]+\]", str(label)) is not None


def _normalize_series_values(y, mode):
    if mode is None:
        return y
    out = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    vals = out[finite]
    if mode == "zscore":
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        if sigma == 0 or not np.isfinite(sigma):
            return out
        out[finite] = (vals - mu) / sigma
    elif mode == "minmax":
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        span = vmax - vmin
        if span == 0 or not np.isfinite(span):
            return out
        out[finite] = (vals - vmin) / span
    elif mode == "median":
        med = float(np.median(vals))
        if med == 0 or not np.isfinite(med):
            return out
        out[finite] = vals / med
    elif mode == "first":
        base = float(vals[0])
        if base == 0 or not np.isfinite(base):
            return out
        out[finite] = vals / base
    return out


def _ensure_dtindex(df):
    if df is None:
        return df
    if isinstance(df.index, pd.DatetimeIndex):
        out = df if (df.index.is_monotonic_increasing and not df.index.hasnans) else df.copy()
        if out.index.hasnans:
            out = out.loc[~out.index.isna()].copy()
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        return out
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    return out


def _resample_fill_roll(df, resample_rule=None, fill_method=None, rolling=None):
    if df is None or len(df) == 0:
        return df
    if resample_rule is None and fill_method is None and rolling is None:
        return df

    df = _ensure_dtindex(df)

    if resample_rule is not None:
        df = df.resample(resample_rule).mean()

    if fill_method is not None:
        if fill_method in ("ffill", "pad"):
            df = df.ffill()
        elif fill_method in ("bfill", "backfill"):
            df = df.bfill()
        else:
            df = df.fillna(method=fill_method)

    if rolling is not None:
        df = df.rolling(rolling, min_periods=1).mean()

    return df


def _time_reindex_interp(df, idx):
    df = _ensure_dtindex(df)
    idx = pd.DatetimeIndex(idx)
    if df is None or len(df) == 0 or len(idx) == 0:
        return df.reindex(idx)

    if df.index.tz is not None and idx.tz is None:
        idx = idx.tz_localize(df.index.tz)
    elif df.index.tz is None and idx.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(idx.tz)

    union = df.index.union(idx)
    df_u = df.reindex(union).sort_index().interpolate(method="time")
    out = df_u.reindex(idx).ffill().bfill()
    out.index.name = "time"
    return out


# ============================================================
# Gap handling
# ============================================================
def _prep_gap_df(gaps):
    if gaps is None or not isinstance(gaps, pd.DataFrame) or len(gaps) == 0:
        return pd.DataFrame(columns=["Start", "End"])
    if ("Start" not in gaps.columns) or ("End" not in gaps.columns):
        return pd.DataFrame(columns=["Start", "End"])
    g = gaps[["Start", "End"]].copy()
    g["Start"] = pd.to_datetime(g["Start"])
    g["End"] = pd.to_datetime(g["End"])
    g = g[g["End"] > g["Start"]].sort_values("Start").reset_index(drop=True)
    return g


def filter_gaps_by_min_duration(gaps, min_gap):
    g = _prep_gap_df(gaps)
    if len(g) == 0:
        return g
    min_gap = pd.to_timedelta(min_gap)
    dt = g["End"] - g["Start"]
    keep = dt >= min_gap
    return g.loc[keep].reset_index(drop=True)


def merge_gap_intervals(gaps, merge_tol="0s"):
    g = _prep_gap_df(gaps)
    if len(g) == 0:
        return g

    merge_tol = pd.to_timedelta(merge_tol)
    s = g["Start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    e = g["End"].to_numpy(dtype="datetime64[ns]").astype("int64")
    tol = int(merge_tol.value)

    out_s = [int(s[0])]
    out_e = [int(e[0])]

    for i in range(1, len(s)):
        if int(s[i]) <= out_e[-1] + tol:
            if int(e[i]) > out_e[-1]:
                out_e[-1] = int(e[i])
        else:
            out_s.append(int(s[i]))
            out_e.append(int(e[i]))

    return pd.DataFrame(
        {"Start": pd.to_datetime(np.array(out_s, dtype="int64")),
         "End": pd.to_datetime(np.array(out_e, dtype="int64"))}
    )


def mask_df_with_gaps(df, gaps, columns=None):
    if df is None or not isinstance(df, (pd.DataFrame, pd.Series)):
        return df

    out = df.copy()
    g = _prep_gap_df(gaps)
    if len(g) == 0:
        return out

    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return out

    idx_ns = idx.view("int64")
    s_ns = g["Start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    e_ns = g["End"].to_numpy(dtype="datetime64[ns]").astype("int64")

    left = np.searchsorted(idx_ns, s_ns, side="left")
    right = np.searchsorted(idx_ns, e_ns, side="right")

    mask = np.zeros(idx_ns.size, dtype=bool)
    for l, r in zip(left, right):
        if r > l:
            mask[l:r] = True

    if not np.any(mask):
        return out

    if isinstance(out, pd.Series):
        out.iloc[mask] = np.nan
        return out

    if columns is None:
        out.iloc[mask, :] = np.nan
        return out

    out.loc[idx[mask], columns] = np.nan
    return out


def build_large_gap_masks(mag_gaps, qtn_gaps, par_gaps, sc_pot_gaps, gap_thresholds, merge_tol="0s"):
    out = {}
    for key, g in [("mag", mag_gaps), ("qtn", qtn_gaps), ("par", par_gaps), ("sc_pot", sc_pot_gaps)]:
        thr = gap_thresholds.get(key, None)
        if thr is None:
            out[key] = pd.DataFrame(columns=["Start", "End"])
            continue
        gf = filter_gaps_by_min_duration(g, thr)
        out[key] = merge_gap_intervals(gf, merge_tol=merge_tol)
    return out


# ============================================================
# Panel edits
# ============================================================
def apply_panel_edits(panel_cfg, panel_edits):
    if panel_edits is None:
        return panel_cfg

    cfg = panel_cfg

    adds = panel_edits.get("add_series", [])
    if isinstance(adds, dict):
        adds = [adds]
    if isinstance(adds, list):
        for it in adds:
            pi = int(it["panel_idx"])
            ai = int(it["axis_idx"])
            ss = it["series"]
            cfg["panels"][pi]["axes"][ai]["series"].append(ss)

    newp = panel_edits.get("add_panels", [])
    if isinstance(newp, dict):
        newp = [newp]
    if isinstance(newp, list):
        for it in newp:
            where = str(it.get("where", "append")).lower()
            panel = it["panel"]
            if where == "append":
                cfg["panels"].append(panel)
            elif where == "insert":
                cfg["panels"].insert(int(it["index"]), panel)

    return cfg


# ============================================================
# Default panel config
# ============================================================
def default_panel_config(sc, rtn_flag):
    if rtn_flag == 1:
        bcols = ["Br", "Bt", "Bn", "B_RTN"]
        blabs = [r"$B_{r} ~ [nT]$", r"$B_{t} ~ [nT]$", r"$B_{n} ~ [nT]$", r"$|B| ~ [nT]$"]
    else:
        bcols = ["Bx", "By", "Bz", "B_RTN"]
        blabs = [r"$B_{x} ~ [nT]$", r"$B_{y} ~ [nT]$", r"$B_{z} ~ [nT]$", r"$|B| ~ [nT]$"]

    return {
        "panels": [
            {"axes": [{
                "axis_id": "left", "source": "Mag", "scale": "linear",
                "series": [
                    {"kind": "col", "col": bcols[0], "label": blabs[0], "style": {"lw": 0.4, "ls": "-", "ms": 0, "color": "darkblue"}},
                    {"kind": "col", "col": bcols[1], "label": blabs[1], "style": {"lw": 0.4, "ls": "-", "ms": 0, "color": "darkred"}},
                    {"kind": "col", "col": bcols[2], "label": blabs[2], "style": {"lw": 0.4, "ls": "-", "ms": 0, "color": "darkgreen"}},
                    {"kind": "col", "col": bcols[3], "label": blabs[3], "style": {"lw": 0.4, "ls": "-", "ms": 0, "color": "k"}},
                ],
                "legend": "main",
            }]},
            {"axes": [
                {
                    "axis_id": "left", "source": "Par", "scale": "linear",
                    "series": [{"kind": "func", "func": "speed", "name": "Vsw", "label": "$V_{sw} ~[km~s^{-1}]$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "C0"}}],
                    "legend": "main",
                },
                {
                    "axis_id": "right0", "source": "Par", "scale": "linear",
                    "series": [{"kind": "col", "col": "Vth", "label": "$T_{p}~ [eV]$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "k"}}],
                    "legend": "side",
                },
            ]},
            {"axes": [{
                "axis_id": "left", "source": "Par", "scale": "linear",
                "series": [{"kind": "col", "col": "np", "label": "$N_{p}~[cm^{-3}]$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "darkred"}}],
                "legend": "main",
            }]},
            {"axes": [{
                "axis_id": "left", "source": "Sig", "scale": "linear",
                "series": [
                    {"kind": "col", "col": "sigma_c", "label": "$\\sigma_{c}$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "darkblue"}},
                    {"kind": "col", "col": "sigma_r", "label": "$\\sigma_{r}$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "darkred"}},
                ],
                "legend": "main",
            }]},
            {"axes": [
                {
                    "axis_id": "left", "source": "Sig", "scale": "log",
                    "series": [{"kind": "col", "col": "beta", "label": r"$\beta$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "black"}}],
                    "hline": [{"y": 1.0, "ls": ":", "c": "k", "lw": 2}],
                    "legend": "main",
                },
                {
                    "axis_id": "right0", "source": "Sig", "scale": "log",
                    "series": [{"kind": "col", "col": "Ma", "label": "$M_a$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "darkred"}}],
                    "hline": [{"y": 1.0, "ls": ":", "c": "darkred", "lw": 2}],
                    "legend": "side",
                },
            ]},
            {"axes": [{
                "axis_id": "left", "source": "Sig", "scale": "linear",
                "series": [{"kind": "col", "col": "VB", "label": r"$\Theta_{VB} ~[^{\circ}]$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "black"}}],
                "legend": "main",
            }]},
            # Distance panel (Sun distance by default). If flow separation is enabled, the pipeline repurposes
            # this panel to show Earth distance in R_E (no extra panel is appended).
            {"axes": [
                {
                    "axis_id": "left", "source": "Par", "scale": "linear",
                    "series": [{"kind": "col", "col": "Dist_au", "label": r"$R~[AU]$", "style": {"lw": 0.9, "ls": "-", "ms": 0, "color": "black"}}],
                    "legend": "main",
                },
                {
                    "axis_id": "right0", "source": "Par", "scale": "linear",
                    "only_if": {"sc_equals": "PSP"},
                    "series": [{"kind": "col", "col": "carr_lon", "label": "$Carr. long ~ [^{\\circ}]$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "darkred"}}],
                    "legend": "side",
                },
            ]},
        ]
    }
def _resolve_named_func(name):
    if name == "speed":
        def _f(df):
            for trio in (("Vr", "Vt", "Vn"), ("Vx", "Vy", "Vz"), ("vx", "vy", "vz")):
                if all(c in df.columns for c in trio):
                    arr = df[list(trio)].to_numpy(copy=False)
                    v = np.sqrt(np.einsum("ij,ij->i", arr, arr))
                    return pd.Series(v, index=df.index)
            raise KeyError("speed(): missing (Vr,Vt,Vn) and (Vx,Vy,Vz)")
        return _f

    if name == "Rsun_from_au":
        def _f(df):
            if "Dist_au" not in df.columns:
                raise KeyError("Rsun_from_au(): missing Dist_au")
            return pd.Series(215.043 * df["Dist_au"].to_numpy(copy=False), index=df.index)
        return _f

    raise KeyError(f"unknown named func '{name}'")


def _merge_style(style, defaults):
    out = dict(defaults)
    if isinstance(style, dict):
        out.update(style)
    return out



def infer_mag_frame(mag_df):
    if mag_df is None or not isinstance(mag_df, pd.DataFrame):
        return "UNKNOWN"
    if all(c in mag_df.columns for c in ("Br", "Bt", "Bn")):
        return "RTN"
    if all(c in mag_df.columns for c in ("Bx", "By", "Bz")):
        return "GSE"
    return "UNKNOWN"


def infer_v_frame(par_df):
    if par_df is None or not isinstance(par_df, pd.DataFrame):
        return "UNKNOWN"
    if all(c in par_df.columns for c in ("Vx", "Vy", "Vz")) or all(c in par_df.columns for c in ("vx", "vy", "vz")):
        return "GSE"
    if all(c in par_df.columns for c in ("Vr", "Vt", "Vn")):
        return "RTN"
    return "UNKNOWN"


def ensure_left_legend_presets(plot_defaults):
    plot_defaults = dict(plot_defaults) if isinstance(plot_defaults, dict) else {}
    plot_defaults["legend_presets"] = dict(plot_defaults.get("legend_presets", {}))
    plot_defaults["legend_presets"].setdefault(
        "main_left",
        {"fontsize": "small", "frameon": False, "bbox_to_anchor": (-0.01, 1.0), "loc": 1, "ncol": 1},
    )
    plot_defaults["legend_presets"].setdefault(
        "side_left",
        {"fontsize": "small", "frameon": False, "bbox_to_anchor": (-0.01, 0.6), "loc": 1, "ncol": 1},
    )
    return plot_defaults


def alternate_legends_per_panel(panel_config, *, start_side="right"):
    cfg = {**panel_config} if isinstance(panel_config, dict) else {"panels": []}
    panels = [dict(p) for p in cfg.get("panels", [])]

    start_side = str(start_side).lower().strip()
    if start_side not in ("right", "left"):
        start_side = "right"

    def use_left(i):
        return (i % 2 == 1) if start_side == "right" else (i % 2 == 0)

    for i, pan in enumerate(panels):
        if not use_left(i):
            continue
        axes = [dict(a) for a in pan.get("axes", [])]
        for ax in axes:
            leg = ax.get("legend", None)
            if leg == "main":
                ax["legend"] = "main_left"
            elif leg == "side":
                ax["legend"] = "side_left"
        pan["axes"] = axes
        panels[i] = pan

    cfg["panels"] = panels
    return cfg


# Backward-compatible internal aliases
_infer_mag_frame = infer_mag_frame
_infer_v_frame = infer_v_frame

def _panel_uses_left_legend(panel_index, start_side):
    start_side = str(start_side).lower().strip()
    if start_side not in ("right", "left"):
        start_side = "right"
    if start_side == "right":
        return (panel_index % 2) == 1
    return (panel_index % 2) == 0


def _map_legend_preset_for_panel(legend_spec, panel_index, *, alternate, start_side):
    """Map 'main'/'side' presets to left variants panel-by-panel when enabled."""
    if not alternate or not isinstance(legend_spec, str):
        return legend_spec
    if not _panel_uses_left_legend(panel_index, start_side):
        return legend_spec
    if legend_spec == "main":
        return "main_left"
    if legend_spec == "side":
        return "side_left"
    return legend_spec

def _legend_dict(leg, plot_defaults):
    if leg is None:
        return None
    if isinstance(leg, str):
        return dict(plot_defaults["legend_presets"].get(leg, plot_defaults["legend_presets"]["main"]))
    if isinstance(leg, dict):
        return dict(leg)
    return None


def _apply_auto_ylims(ax, y_arrays, scale, plot_defaults):
    if not y_arrays:
        return

    scale = str(scale).lower()
    mn = float(plot_defaults["auto_ylims"]["min_factor"])
    mx = float(plot_defaults["auto_ylims"]["max_factor"])

    have = False
    vmin = np.inf
    vmax = -np.inf

    for y in y_arrays:
        if y is None:
            continue
        arr = np.asarray(y, dtype=float).ravel()
        if arr.size == 0:
            continue
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if scale == "log":
            arr = arr[arr > 0]
            if arr.size == 0:
                continue

        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue

        if not have:
            vmin, vmax = lo, hi
            have = True
        else:
            if lo < vmin:
                vmin = lo
            if hi > vmax:
                vmax = hi

    if not have or not np.isfinite(vmin) or not np.isfinite(vmax):
        return

    if scale == "log":
        lo = mn * vmin
        hi = mx * vmax
        if lo <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or not (lo < hi):
            return
        ax.set_ylim([lo, hi])
        return

    span = vmax - vmin
    if not np.isfinite(span):
        return
    if span == 0:
        pad = 0.1 * (abs(vmin) if vmin != 0 else 1.0)
        ax.set_ylim([vmin - pad, vmax + pad])
        return

    lo = (mx * vmin) if (vmin < 0) else (mn * vmin)
    hi = (mn * vmax) if (vmax < 0) else (mx * vmax)

    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if lo > hi:
        lo, hi = hi, lo
    if lo < hi:
        ax.set_ylim([lo, hi])


def _get_or_create_axis(base_ax, axis_id, created):
    if axis_id == "left":
        created["left"] = base_ax
        return base_ax

    if axis_id in created:
        return created[axis_id]

    if axis_id.startswith("right"):
        if "right0" not in created:
            created["right0"] = base_ax.twinx()
            try:
                created["right0"].patch.set_visible(False)
            except Exception:
                pass
        if axis_id == "right0":
            return created["right0"]

        j = int(axis_id.replace("right", ""))
        ax_new = base_ax.twinx()
        ax_new.spines["right"].set_position(("outward", 60 * j))
        try:
            ax_new.patch.set_visible(False)
        except Exception:
            pass
        created[axis_id] = ax_new
        return ax_new

    raise ValueError(f"unsupported axis_id '{axis_id}'")


@dataclass
class AxesRegistry:
    base_axes: list
    panel_axes: list
    marker_axes: list
    span_axes: list


def _plot_from_panel_config(
    fig,
    axs,
    panel_config,
    data_sources_by_sc,
    sc_list,
    sc_ls_map,
    start_lim,
    end_lim,
    auto_ylims,
    warn_missing,
    plot_defaults,
    normalize_timeseries,
    enforce_sc_linestyle,
    *,
    alternate_legend_sides=False,
    legend_start_side="right",
):
    panels = panel_config.get("panels", None)
    if not isinstance(panels, list) or len(panels) == 0:
        raise ValueError("panel_config must contain a non-empty list under 'panels'")
    if len(axs) != len(panels):
        raise ValueError(f"axes count ({len(axs)}) != panels count ({len(panels)})")

    def _miss(msg):
        if warn_missing:
            print(f"[plot:missing] {msg}")

    per_panel_axes = []

    for i, pan in enumerate(panels):
        base_ax = axs[i]
        created_axes = {"left": base_ax}
        yvals_by_axis = {}

        axes_specs = pan.get("axes", [])
        if not isinstance(axes_specs, list):
            axes_specs = []

        for axspec in axes_specs:
            only_if = axspec.get("only_if", None)

            src = axspec.get("source", None)
            sc_iter = sc_list
            if isinstance(only_if, dict) and only_if.get("sc_equals", None) is not None:
                sc_iter = [sc for sc in sc_list if str(sc) == str(only_if["sc_equals"])]

            sc_data = []
            for sc_name in sc_iter:
                if src not in data_sources_by_sc.get(sc_name, {}):
                    raise KeyError(
                        f"unknown source '{src}' for sc='{sc_name}'. "
                        f"Known: {list(data_sources_by_sc.get(sc_name, {}).keys())}"
                    )
                sc_data.append((sc_name, data_sources_by_sc[sc_name][src]))

            axis_id = str(axspec.get("axis_id", "left"))
            scale = str(axspec.get("scale", "linear")).lower()
            _leg_spec = axspec.get("legend", None)
            _leg_spec = _map_legend_preset_for_panel(
                _leg_spec, i, alternate=bool(alternate_legend_sides), start_side=legend_start_side
            )
            leg = _legend_dict(_leg_spec, plot_defaults)

            ax = None
            labels = []
            any_plotted = False
            y_collector = []

            series_list = axspec.get("series", [])
            if not isinstance(series_list, list):
                series_list = []

            for sc, df in sc_data:
                for s in series_list:
                    if not isinstance(s, dict):
                        continue

                    kind = s.get("kind", None)

                    if kind == "col":
                        col = s.get("col", None)
                        if col not in df.columns:
                            _miss(f"panel={i} sc={sc} axis={axis_id} source={src} col={col} is missing -> omitted")
                            continue
                        x = df.index
                        y = df[col].to_numpy(copy=False)

                    elif kind == "func":
                        fn = s.get("func", None)
                        try:
                            fnc = _resolve_named_func(fn) if isinstance(fn, str) else fn
                            if not callable(fnc):
                                raise TypeError("func is not callable")
                            y_ser = fnc(df)
                            if isinstance(y_ser, pd.Series):
                                x = y_ser.index
                                y = y_ser.to_numpy(copy=False)
                            else:
                                y = np.asarray(y_ser)
                                x = df.index
                                if y.shape[0] != len(x):
                                    raise ValueError("func length mismatch")
                        except Exception as e:
                            _miss(f"panel={i} sc={sc} axis={axis_id} source={src} func={fn} failed ({type(e).__name__}: {e}) -> omitted")
                            continue
                    else:
                        _miss(f"panel={i} sc={sc} axis={axis_id} source={src} kind={kind} unsupported -> omitted")
                        continue

                    if ax is None:
                        ax = _get_or_create_axis(base_ax, axis_id, created_axes)
                        if scale == "log":
                            ax.set_yscale("log")

                    label = s.get("label", str(s.get("col", s.get("name", "series"))))
                    if len(sc_list) > 1:
                        label = f"{label} ({sc})"
                    user_style = s.get("style", {})
                    style = _merge_style(user_style, plot_defaults["series_style_defaults"])
                    if len(sc_list) > 1 and enforce_sc_linestyle:
                        style["ls"] = sc_ls_map.get(sc, style.get("ls", "-"))

                    yarr = np.asarray(y, dtype=float)
                    if _series_has_units(label):
                        yarr = _normalize_series_values(yarr, normalize_timeseries)

                    mask = ~np.isfinite(yarr)
                    if scale == "log":
                        mask = mask | (yarr <= 0)
                    yplot = np.ma.array(yarr, mask=mask, copy=False)

                    ax.plot(x, yplot, **style)
                    labels.append(label)
                    y_collector.append(yarr)
                    any_plotted = True

            if not any_plotted:
                continue

            hlines = axspec.get("hline", [])
            if isinstance(hlines, dict):
                hlines = [hlines]
            if isinstance(hlines, list) and ax is not None:
                for hh in hlines:
                    if not isinstance(hh, dict) or "y" not in hh:
                        continue
                    ax.axhline(
                        y=float(hh["y"]),
                        ls=hh.get("ls", ":"),
                        c=hh.get("c", "k"),
                        lw=float(hh.get("lw", 1.2)),
                    )

            if leg is not None and labels and ax is not None:
                ax.legend(
                    labels,
                    fontsize=leg.get("fontsize", "small"),
                    frameon=bool(leg.get("frameon", False)),
                    bbox_to_anchor=leg.get("bbox_to_anchor", (1.01, 1)),
                    loc=leg.get("loc", 2),
                    ncol=int(leg.get("ncol", 1)),
                )

            yvals_by_axis.setdefault(axis_id, []).extend(y_collector)

        if auto_ylims:
            for axis_id, ax_ in created_axes.items():
                scale = "linear"
                for axspec in axes_specs:
                    if str(axspec.get("axis_id", "left")) == axis_id:
                        scale = str(axspec.get("scale", "linear")).lower()
                        break
                _apply_auto_ylims(ax_, yvals_by_axis.get(axis_id, []), scale=scale, plot_defaults=plot_defaults)

        for ax_ in created_axes.values():
            ax_.set_xlim([start_lim, end_lim])

        per_panel_axes.append(list(dict.fromkeys(created_axes.values())))

    marker_axes = []
    seen = set()
    for pax in per_panel_axes:
        for ax in pax:
            if ax in seen:
                continue
            marker_axes.append(ax)
            seen.add(ax)

    span_axes = list(axs)
    return AxesRegistry(base_axes=list(axs), panel_axes=per_panel_axes, marker_axes=marker_axes, span_axes=span_axes)


# ============================================================
# Dynamic y-lims on zoom/pan (ONE implementation)
# ============================================================
@dataclass
class _DynLineCache:
    line: any
    x: np.ndarray
    y: np.ndarray
    bad: np.ndarray
    monotonic: bool


def _install_dynamic_ylims_on_xzoom(fig, axes_registry, plot_defaults):
    unique_axes = list(axes_registry.marker_axes or [])
    base_axes = list(axes_registry.base_axes or [])
    if not unique_axes or not base_axes:
        return

    old = getattr(fig, "_mhdturbpy_dynylims", None)
    if isinstance(old, dict):
        for ax, cid in old.get("ax_cids", []):
            try:
                ax.callbacks.disconnect(cid)
            except Exception:
                pass
        for cid in old.get("mpl_cids", []):
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        try:
            t = old.get("timer", None)
            if t is not None:
                t.stop()
        except Exception:
            pass

    try:
        full_x0, full_x1 = base_axes[0].get_xlim()
        full_x0 = float(full_x0)
        full_x1 = float(full_x1)
    except Exception:
        return

    full_span = abs(full_x1 - full_x0)
    if not np.isfinite(full_span) or full_span <= 0:
        full_span = None

    base_mn_fac = float(plot_defaults["auto_ylims"]["min_factor"])
    base_mx_fac = float(plot_defaults["auto_ylims"]["max_factor"])

    FILL_ABS_MAX = 1e25

    def _coerce_x_to_float(xdata):
        if xdata is None:
            return None
        try:
            return np.asarray(xdata, dtype=float)
        except Exception:
            try:
                return np.asarray(mdates.date2num(xdata), dtype=float)
            except Exception:
                return None

    def _coerce_y_to_clean_float(ydata):
        if ydata is None:
            return None, None

        if np.ma.isMaskedArray(ydata):
            y = np.asarray(np.ma.getdata(ydata), dtype=float)
            bad = np.asarray(np.ma.getmaskarray(ydata), dtype=bool)
        else:
            y = np.asarray(ydata, dtype=float)
            bad = np.zeros(y.shape, dtype=bool)

        bad |= ~np.isfinite(y)
        bad |= (np.abs(y) > FILL_ABS_MAX)
        return y, bad

    ax_cache = {}
    for ax in unique_axes:
        lines = getattr(ax, "lines", [])
        if not lines:
            continue

        series = []
        for ln in lines:
            try:
                xraw = ln.get_xdata(orig=False)
                yraw = ln.get_ydata(orig=False)
            except Exception:
                continue
            if xraw is None or yraw is None:
                continue

            try:
                if len(xraw) <= 2 or len(yraw) <= 2:
                    continue
            except Exception:
                continue

            x = _coerce_x_to_float(xraw)
            if x is None or x.size == 0:
                continue

            y, bad = _coerce_y_to_clean_float(yraw)
            if y is None or bad is None or y.size != x.size or y.size == 0:
                continue

            dx = np.diff(x)
            mono = bool(np.all(dx >= 0)) if dx.size else True
            series.append(_DynLineCache(line=ln, x=x, y=y, bad=bad, monotonic=mono))

        if series:
            ax_cache[ax] = series

    if not ax_cache:
        return

    state = {"in_cb": False, "last_xkey": None}

    timer = None
    try:
        timer = fig.canvas.new_timer(interval=35)
        timer.single_shot = True
    except Exception:
        timer = None

    def _factors_for_xlim(x0, x1):
        span = abs(float(x1) - float(x0))
        if full_span is None or not np.isfinite(span):
            return 0.9, 1.1
        if span < 0.995 * full_span:
            return 0.9, 1.1
        return base_mn_fac, base_mx_fac

    def _visible_minmax_for_axis(ax, x0, x1):
        if x1 < x0:
            x0, x1 = x1, x0

        scale = str(getattr(ax, "get_yscale", lambda: "linear")()).lower()

        have = False
        vmin = np.inf
        vmax = -np.inf

        for lc in ax_cache.get(ax, []):
            try:
                if not lc.line.get_visible():
                    continue
            except Exception:
                pass

            x = lc.x
            y = lc.y
            bad = lc.bad

            if lc.monotonic:
                i0 = int(np.searchsorted(x, x0, side="left"))
                i1 = int(np.searchsorted(x, x1, side="right"))
                if i1 <= i0:
                    continue
                yy = y[i0:i1]
                bb = bad[i0:i1]
            else:
                m = (x >= x0) & (x <= x1)
                if not np.any(m):
                    continue
                yy = y[m]
                bb = bad[m]

            if yy.size == 0:
                continue
            yy = yy[~bb]
            if yy.size == 0:
                continue

            if scale == "log":
                yy = yy[yy > 0]
                if yy.size == 0:
                    continue

            lo = float(np.min(yy))
            hi = float(np.max(yy))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue

            if not have:
                vmin, vmax = lo, hi
                have = True
            else:
                if lo < vmin:
                    vmin = lo
                if hi > vmax:
                    vmax = hi

        if not have or not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        return vmin, vmax

    def _apply_visible_ylims(ax, vmin, vmax, mn_fac, mx_fac):
        scale = str(getattr(ax, "get_yscale", lambda: "linear")()).lower()

        if scale == "log":
            lo = mn_fac * vmin
            hi = mx_fac * vmax
            if lo <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or not (lo < hi):
                return
            ax.set_ylim([lo, hi])
            return

        span = vmax - vmin
        if not np.isfinite(span):
            return
        if span == 0:
            pad = 0.1 * (abs(vmin) if vmin != 0 else 1.0)
            ax.set_ylim([vmin - pad, vmax + pad])
            return

        lo = (mx_fac * vmin) if (vmin < 0) else (mn_fac * vmin)
        hi = (mn_fac * vmax) if (vmax < 0) else (mx_fac * vmax)

        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        if lo > hi:
            lo, hi = hi, lo
        if lo < hi:
            ax.set_ylim([lo, hi])

    def _do_update(force=False):
        if state["in_cb"]:
            return

        try:
            x0, x1 = base_axes[0].get_xlim()
            x0 = float(x0)
            x1 = float(x1)
        except Exception:
            return

        xkey = (round(x0, 12), round(x1, 12))
        if (not force) and (state["last_xkey"] == xkey):
            return
        state["last_xkey"] = xkey

        mn_fac, mx_fac = _factors_for_xlim(x0, x1)

        state["in_cb"] = True
        try:
            for ax in ax_cache.keys():
                mm = _visible_minmax_for_axis(ax, x0, x1)
                if mm is None:
                    continue
                _apply_visible_ylims(ax, mm[0], mm[1], mn_fac=mn_fac, mx_fac=mx_fac)
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass
        finally:
            state["in_cb"] = False

    def _request_update(force=False):
        if force or timer is None:
            if timer is not None:
                try:
                    timer.stop()
                except Exception:
                    pass
            _do_update(force=True)
            return

        try:
            timer.stop()
        except Exception:
            pass
        try:
            timer.add_callback(_do_update, False)
        except Exception:
            pass
        try:
            timer.start()
        except Exception:
            _do_update(force=False)

    ax_cids = []
    for ax in base_axes:
        try:
            cid = ax.callbacks.connect("xlim_changed", lambda _a: _request_update(force=False))
            ax_cids.append((ax, cid))
        except Exception:
            pass

    mpl_cids = []
    try:
        mpl_cids.append(fig.canvas.mpl_connect("button_release_event", lambda _e: _request_update(force=True)))
    except Exception:
        pass

    _do_update(force=True)
    fig._mhdturbpy_dynylims = {"ax_cids": ax_cids, "mpl_cids": mpl_cids, "timer": timer}


# ============================================================
# Interval selector
# ============================================================
@dataclass
class IntervalArtists:
    start_lines: list
    end_lines: list
    spans: list


def _install_interval_selector(
    fig,
    axes_registry,
    events_file=None,
    autosave=True,
    export_csv=False,
    span_color="0.85",
    span_alpha=0.35,
    snap_index=None,
    snap_to_data=False,
    enable_comments=True,
    debug_interaction=True,
    resume=True,
    dedupe_ms=250,
    dedupe_tol_ns=5_000_000,
    span_zorder=0.8,
    vline_zorder=10.0,
    move_throttle_ms=33,
    use_release_event=False,
):
    if enable_comments:
        events = pd.DataFrame(
            {"t_start": pd.Series(dtype="datetime64[ns]"),
             "t_end": pd.Series(dtype="datetime64[ns]"),
             "comment": pd.Series(dtype="object")}
        )
    else:
        events = pd.DataFrame(
            {"t_start": pd.Series(dtype="datetime64[ns]"),
             "t_end": pd.Series(dtype="datetime64[ns]")}
        )

    def _p(*a):
        if debug_interaction:
            print(*a)
            try:
                sys.stdout.flush()
            except Exception:
                pass

    try:
        old = getattr(fig, "_mhdturbpy_interval_picker", None)
        if isinstance(old, dict):
            for k in ("cid_press", "cid_release", "cid_key", "cid_move"):
                cid = old.get(k, None)
                if isinstance(cid, int):
                    fig.canvas.mpl_disconnect(cid)
    except Exception:
        pass

    try:
        oldb = getattr(fig, "_mhdturbpy_tkbinds", None)
        if isinstance(oldb, list):
            w = fig.canvas.get_tk_widget()
            for seq, fid in oldb:
                try:
                    if fid is not None:
                        w.unbind(seq, fid)
                    else:
                        w.unbind(seq)
                except Exception:
                    pass
        fig._mhdturbpy_tkbinds = []
    except Exception:
        pass

    marker_axes = list(axes_registry.marker_axes)
    span_axes = list(axes_registry.span_axes)

    snap_ns = None
    if snap_index is not None and isinstance(snap_index, pd.DatetimeIndex) and len(snap_index) > 0:
        snap_ns = snap_index.view("int64")

    def _snap(ts):
        if snap_ns is None:
            return ts
        x = int(ts.value)
        i = int(np.searchsorted(snap_ns, x))
        if i <= 0:
            return pd.Timestamp(int(snap_ns[0]))
        if i >= snap_ns.size:
            return pd.Timestamp(int(snap_ns[-1]))
        left = int(snap_ns[i - 1])
        right = int(snap_ns[i])
        return pd.Timestamp(left if (x - left) <= (right - x) else right)

    def _sorted(a, b):
        return (a, b) if a <= b else (b, a)

    def _atomic_pickle_df(df, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_pickle(tmp)
        os.replace(tmp, path)

    def _atomic_csv_df(df, path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def _write():
        if events_file is None:
            return
        try:
            _atomic_pickle_df(events, events_file)
        except Exception as e:
            _p("[picker] save pickle FAILED:", e)
        if export_csv:
            try:
                _atomic_csv_df(events, events_file.with_suffix(".csv"))
            except Exception as e:
                _p("[picker] save csv FAILED:", e)

    def _x_to_ts(xdata):
        if xdata is None:
            return None
        dt = mdates.num2date(float(xdata))
        ts = pd.Timestamp(dt)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts

    class State:
        t0 = None
        snap = bool(snap_to_data)
        hover_xdata = None
        last_move_wall = -1.0

        comment_mode = None
        comment_buffer = ""
        comment_target = None
        armed_comment = ""

        left_select_fallback = (sys.platform.startswith("win") and "tkagg" in matplotlib.get_backend().lower())

    state = State()

    class ClickGate:
        last_wall = -1.0
        last_ts_ns = None
        last_source = None

        def accept(self, ts, source):
            now = time.monotonic()
            ts_ns = int(pd.Timestamp(ts).value)

            if self.last_wall >= 0:
                dt_ms = (now - self.last_wall) * 1000.0

                HARD_ANY_MS = 80.0
                if dt_ms <= HARD_ANY_MS:
                    return False, f"dedupe_hard_any(dt_ms={dt_ms:.1f}, now={source})"

                HARD_CROSS_SOURCE_MS = 120.0
                if (self.last_source is not None) and (source != self.last_source) and (dt_ms <= HARD_CROSS_SOURCE_MS):
                    return False, f"dedupe_cross_source(dt_ms={dt_ms:.1f}, last={self.last_source}, now={source})"

                if self.last_ts_ns is not None:
                    if dt_ms <= float(dedupe_ms) and abs(ts_ns - int(self.last_ts_ns)) <= int(dedupe_tol_ns):
                        return False, f"dedupe(dt_ms={dt_ms:.1f}, source={source})"

            self.last_wall = now
            self.last_ts_ns = ts_ns
            self.last_source = source
            return True, "ok"

    gate = ClickGate()

    def _draw_idle():
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
        try:
            fig.canvas.flush_events()
        except Exception:
            pass

    status = fig.text(
        0.01, 0.005, "Ready. Press 'h' for help.",
        ha="left", va="bottom",
        fontsize=int(DEFAULT_PLOT_PARAMS["interaction"]["status_fontsize"]), color="0.15"
    )
    input_txt = fig.text(
        0.01, 0.03, "",
        ha="left", va="bottom",
        fontsize=int(DEFAULT_PLOT_PARAMS["interaction"]["status_fontsize"]), color="0.05"
    )
    help_box = fig.text(
        0.01, 0.99, "",
        ha="left", va="top",
        fontsize=int(DEFAULT_PLOT_PARAMS["interaction"]["help_fontsize"]),
        family="monospace", color="0.10",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=DEFAULT_PLOT_PARAMS["interaction"]["help_facecolor"],
            edgecolor=DEFAULT_PLOT_PARAMS["interaction"]["help_edgecolor"],
            alpha=float(DEFAULT_PLOT_PARAMS["interaction"]["help_alpha"]),
        ),
    )
    help_box.set_visible(False)

    for _txt in (status, input_txt, help_box):
        try:
            _txt.set_usetex(False)
        except Exception:
            pass

    def _help_text():
        return (
            "MHDTurbPy interval picker\n"
            "------------------------\n"
            "Select:   Right-click OR Shift/Ctrl+Left\n"
            "          (Fallback: plain Left enabled on Windows/TkAgg if needed)\n"
            "Undo:     Ctrl+Z (or u)\n"
            "Delete:   x / Ctrl+D (mouse inside interval)\n"
            "Comment:  c (edit comment under mouse)\n"
            "          m (arm comment for NEXT interval)\n"
            "Snap:     t\n"
            "Save:     s\n"
            "Toggle:   l  (toggle plain-left fallback)\n"
            "Help:     h\n"
            "Cancel:   Esc\n"
        )

    def _set_status(msg, redraw=True):
        status.set_text(str(msg))
        if redraw:
            _draw_idle()

    def _set_input(msg, redraw=True):
        input_txt.set_text(str(msg))
        if redraw:
            _draw_idle()

    def _toggle_help():
        if help_box.get_text() == "":
            help_box.set_text(_help_text())
        help_box.set_visible(not help_box.get_visible())
        _draw_idle()

    pending_start_lines = None
    interval_artists = []
    cache_lo_ns = []
    cache_hi_ns = []

    def _remove_artists(arts):
        for a in arts:
            try:
                a.remove()
            except Exception:
                pass

    def _sync_cache_from_events():
        nonlocal cache_lo_ns, cache_hi_ns
        cache_lo_ns = []
        cache_hi_ns = []
        for i in range(len(events)):
            a = pd.Timestamp(events.iloc[i]["t_start"]).value
            b = pd.Timestamp(events.iloc[i]["t_end"]).value
            cache_lo_ns.append(int(min(a, b)))
            cache_hi_ns.append(int(max(a, b)))

    def _cancel_pending_start(msg=None):
        nonlocal pending_start_lines
        state.t0 = None
        if pending_start_lines is not None:
            _remove_artists(pending_start_lines)
        pending_start_lines = None
        if msg is not None:
            _set_status(msg, redraw=True)

    def _draw_vline_all(t):
        out = []
        for ax in marker_axes:
            out.append(ax.axvline(t, color="0.25", lw=0.9, alpha=0.9, zorder=float(vline_zorder)))
        return out

    def _draw_span_all(a, b):
        if b < a:
            a, b = b, a
        out = []
        for ax in span_axes:
            out.append(ax.axvspan(a, b, color=span_color, alpha=span_alpha, lw=0, zorder=float(span_zorder)))
        return out

    def _render_interval(a, b):
        a, b = _sorted(pd.Timestamp(a), pd.Timestamp(b))
        start_lines = _draw_vline_all(a)
        end_lines = _draw_vline_all(b)
        spans = _draw_span_all(a, b)
        return IntervalArtists(start_lines=start_lines, end_lines=end_lines, spans=spans)

    def _load_resume_into_events():
        if not resume or events_file is None:
            return False
        if not events_file.exists():
            return False
        try:
            loaded = pd.read_pickle(events_file)
            if not (isinstance(loaded, pd.DataFrame) and ("t_start" in loaded.columns) and ("t_end" in loaded.columns)):
                _p("[picker] resume: file exists but not a valid intervals DF")
                return False

            loaded = loaded.copy()
            loaded["t_start"] = pd.to_datetime(loaded["t_start"], errors="coerce")
            loaded["t_end"] = pd.to_datetime(loaded["t_end"], errors="coerce")
            if enable_comments and "comment" not in loaded.columns:
                loaded["comment"] = ""

            keep = ["t_start", "t_end"] + (["comment"] if enable_comments else [])
            loaded = loaded[keep].dropna(subset=["t_start", "t_end"]).reset_index(drop=True)

            events.drop(index=events.index, inplace=True)
            for i in range(len(loaded)):
                events.loc[i, "t_start"] = pd.Timestamp(loaded.loc[i, "t_start"])
                events.loc[i, "t_end"] = pd.Timestamp(loaded.loc[i, "t_end"])
                if enable_comments:
                    cc = loaded.loc[i, "comment"]
                    events.loc[i, "comment"] = "" if pd.isna(cc) else str(cc)

            events.reset_index(drop=True, inplace=True)
            _p(f"[picker] resumed {len(events)} intervals from {events_file}")
            return True
        except Exception as e:
            _p("[picker] resume load FAILED:", e)
            return False

    def _rebuild_artists_from_events():
        nonlocal interval_artists
        _cancel_pending_start()
        for ia in interval_artists:
            _remove_artists(ia.start_lines)
            _remove_artists(ia.end_lines)
            _remove_artists(ia.spans)
        interval_artists = []
        for i in range(len(events)):
            a = pd.Timestamp(events.iloc[i]["t_start"])
            b = pd.Timestamp(events.iloc[i]["t_end"])
            interval_artists.append(_render_interval(a, b))
        _sync_cache_from_events()
        _draw_idle()

    if _load_resume_into_events():
        _rebuild_artists_from_events()
        _set_status(f"Loaded {len(events)} intervals from disk. Press 'h' for help.", redraw=True)

    def _interval_under_hover():
        if state.hover_xdata is None:
            return None
        ts = _x_to_ts(state.hover_xdata)
        if ts is None:
            return None
        if state.snap:
            ts = _snap(ts)
        t_ns = int(pd.Timestamp(ts).value)
        if not cache_lo_ns:
            return None
        lo = np.asarray(cache_lo_ns, dtype=np.int64)
        hi = np.asarray(cache_hi_ns, dtype=np.int64)
        hits = np.where((lo <= t_ns) & (t_ns <= hi))[0]
        if hits.size == 0:
            return None
        return int(hits[0])

    def _undo():
        if state.t0 is not None:
            _cancel_pending_start("Cancelled pending start.")
            _set_input("", redraw=True)
            return
        if len(events) == 0:
            _set_status("Undo: nothing to remove.")
            return
        ia = interval_artists.pop(-1)
        _remove_artists(ia.start_lines)
        _remove_artists(ia.end_lines)
        _remove_artists(ia.spans)
        events.drop(index=events.index[-1], inplace=True)
        events.reset_index(drop=True, inplace=True)
        cache_lo_ns.pop(-1)
        cache_hi_ns.pop(-1)
        _draw_idle()
        _set_status(f"Undo: removed last. N={len(events)}", redraw=True)
        if autosave:
            _write()

    def _delete_hover():
        _cancel_pending_start(None)
        pos = _interval_under_hover()
        if pos is None:
            _set_status("Delete: move mouse inside an interval.")
            return
        ia = interval_artists.pop(pos)
        _remove_artists(ia.start_lines)
        _remove_artists(ia.end_lines)
        _remove_artists(ia.spans)
        events.drop(index=events.index[pos], inplace=True)
        events.reset_index(drop=True, inplace=True)
        cache_lo_ns.pop(pos)
        cache_hi_ns.pop(pos)
        _draw_idle()
        _set_status(f"Deleted interval {pos}. N={len(events)}", redraw=True)
        if autosave:
            _write()

    def _finish_comment(save):
        if state.comment_mode is None:
            return
        if not save:
            state.comment_mode = None
            state.comment_buffer = ""
            state.comment_target = None
            _set_input("", redraw=False)
            _set_status("Comment cancelled.", redraw=True)
            return

        if state.comment_mode == "armed_next":
            state.armed_comment = state.comment_buffer
            state.comment_mode = None
            state.comment_buffer = ""
            state.comment_target = None
            _set_input("", redraw=False)
            _set_status("Comment armed for next interval.", redraw=True)
            return

        if state.comment_mode == "edit_existing":
            pos = state.comment_target
            if pos is None or pos < 0 or pos >= len(events):
                _set_status("Comment target invalid.", redraw=True)
            else:
                if enable_comments:
                    events.at[pos, "comment"] = state.comment_buffer
                    if autosave:
                        _write()
                    _set_status(f"Saved comment for interval {pos}.", redraw=True)
            state.comment_mode = None
            state.comment_buffer = ""
            state.comment_target = None
            _set_input("", redraw=False)

    def _start_comment_edit():
        if not enable_comments:
            _set_status("Comments disabled.")
            return
        pos = _interval_under_hover()
        if pos is None:
            _set_status("Comment: move mouse inside an interval.")
            return
        state.comment_mode = "edit_existing"
        state.comment_target = int(pos)
        cur = events.iloc[pos].get("comment", "")
        cur = "" if pd.isna(cur) else str(cur)
        state.comment_buffer = cur
        _set_status(f"Editing comment for interval {pos}. Enter=save, Esc=cancel.", redraw=False)
        _set_input("Comment: " + state.comment_buffer, redraw=True)

    def _toolbar_mode():
        tb = getattr(fig.canvas, "toolbar", None)
        return str(getattr(tb, "mode", "") or "")

    def _mpl_is_select_click(ev):
        btn = getattr(ev, "button", None)
        key = str(getattr(ev, "key", "") or "").lower()

        if btn == 3:
            return True, "button=3"
        if btn == 2:
            return True, "button=2"

        if btn == 1 and (("ctrl" in key) or ("control" in key) or ("shift" in key)):
            return True, f"button=1 key={key}"

        if btn == 1 and (key == "" or key == "none" or key is None):
            if state.left_select_fallback:
                return True, "button=1 (fallback enabled)"
            return False, "button=1 (fallback disabled)"

        return False, f"btn={btn} key={key}"

    def _select_at_timestamp(ts, source, decoded):
        nonlocal pending_start_lines

        if state.snap:
            ts = _snap(ts)

        ok, why = gate.accept(ts, source=source)
        if not ok:
            if debug_interaction:
                _p(f"[picker:IGNORED] source={source} decoded={decoded} reason={why} ts={ts}")
            return

        if debug_interaction:
            _p(f"[picker:ACCEPT] source={source} decoded={decoded} ts={ts}")

        if state.t0 is None:
            state.t0 = ts
            if pending_start_lines is not None:
                _remove_artists(pending_start_lines)
            pending_start_lines = _draw_vline_all(ts)
            _set_status(f"Start set [{source}]: {ts} (click again to end; Esc cancels)", redraw=True)
            return

        t0 = pd.Timestamp(state.t0)
        state.t0 = None
        if pending_start_lines is not None:
            _remove_artists(pending_start_lines)
        pending_start_lines = None

        a, b = _sorted(t0, pd.Timestamp(ts))
        if int(a.value) == int(b.value):
            state.t0 = a
            pending_start_lines = _draw_vline_all(a)
            _set_status(f"Ignored zero-width end; start retained [{source}]. Click a different time.", redraw=True)
            return

        ia = _render_interval(a, b)
        interval_artists.append(ia)

        i = len(events)
        events.loc[i, "t_start"] = pd.Timestamp(a)
        events.loc[i, "t_end"] = pd.Timestamp(b)
        if enable_comments:
            events.loc[i, "comment"] = state.armed_comment
            state.armed_comment = ""

        cache_lo_ns.append(int(min(a.value, b.value)))
        cache_hi_ns.append(int(max(a.value, b.value)))

        _set_status(f"Saved interval {i}: {a} -> {b} [{source}]", redraw=True)
        _set_input("", redraw=False)

        if autosave:
            _write()

    def _on_press(ev):
        if debug_interaction:
            _p(
                f"[picker:mpl:recv] name={getattr(ev,'name',None)} "
                f"button={getattr(ev,'button',None)} key={getattr(ev,'key',None)} "
                f"inaxes={ev.inaxes is not None} xdata={getattr(ev,'xdata',None)} "
                f"toolbar_mode={_toolbar_mode()!r}"
            )

        if _toolbar_mode() != "":
            if debug_interaction:
                _p("[picker:mpl:ignored] reason=toolbar_mode")
            return

        if ev.inaxes is None or getattr(ev, "xdata", None) is None:
            if debug_interaction:
                _p("[picker:mpl:ignored] reason=inaxes_or_xdata_none")
            return

        if getattr(ev, "dblclick", False):
            if debug_interaction:
                _p("[picker:mpl:ignored] reason=dblclick")
            return

        ok, decoded = _mpl_is_select_click(ev)
        if not ok:
            if debug_interaction:
                _p(f"[picker:mpl:ignored] reason=not_select ({decoded})")
            return

        ts = _x_to_ts(ev.xdata)
        if ts is None:
            if debug_interaction:
                _p("[picker:mpl:ignored] reason=ts_none")
            return

        _select_at_timestamp(ts, source="mpl", decoded=decoded)

    def _on_release(ev):
        if not use_release_event:
            return
        if ev.inaxes is None or getattr(ev, "xdata", None) is None:
            return
        ok, decoded = _mpl_is_select_click(ev)
        if not ok:
            return
        ts = _x_to_ts(ev.xdata)
        if ts is None:
            return
        _select_at_timestamp(ts, source="mpl_release", decoded=decoded)

    def _on_move(ev):
        if ev.inaxes is None or getattr(ev, "xdata", None) is None:
            return
        now = time.monotonic()
        if state.last_move_wall >= 0:
            dt_ms = (now - state.last_move_wall) * 1000.0
            if dt_ms < float(move_throttle_ms):
                return
        state.last_move_wall = now
        try:
            state.hover_xdata = float(ev.xdata)
        except Exception:
            state.hover_xdata = None

    def _on_key(ev):
        k = str(getattr(ev, "key", "")).lower()

        if enable_comments and state.comment_mode is not None:
            if k in ("enter", "return"):
                _finish_comment(True)
                return
            if k == "escape":
                _finish_comment(False)
                return
            if k == "backspace":
                state.comment_buffer = state.comment_buffer[:-1]
                _set_input("Comment: " + state.comment_buffer, redraw=True)
                return
            if k == "space":
                state.comment_buffer += " "
                _set_input("Comment: " + state.comment_buffer, redraw=True)
                return
            if len(k) == 1:
                state.comment_buffer += k
                _set_input("Comment: " + state.comment_buffer, redraw=True)
                return

        if k == "h":
            _toggle_help()
            return
        if k == "escape":
            _cancel_pending_start("Cancelled pending start.")
            _set_input("", redraw=True)
            return
        if k in ("ctrl+z", "control+z") or k == "u":
            _undo()
            return
        if k == "x" or k in ("ctrl+d", "control+d"):
            _delete_hover()
            return
        if k == "c":
            _start_comment_edit()
            return
        if enable_comments and k == "m":
            state.comment_mode = "armed_next"
            state.comment_buffer = ""
            _set_status("Type comment for NEXT interval, then Enter (Esc cancels).", redraw=False)
            _set_input("Comment: ", redraw=True)
            return
        if k == "t":
            state.snap = not state.snap
            _set_status(f"Snap-to-data = {state.snap}", redraw=True)
            return
        if k == "s":
            _write()
            _set_status(f"Saved. N={len(events)}", redraw=True)
            return
        if k == "l":
            state.left_select_fallback = not state.left_select_fallback
            _set_status(f"Plain-left fallback = {state.left_select_fallback}", redraw=True)
            return

    cid_press = fig.canvas.mpl_connect("button_press_event", _on_press)
    cid_move = fig.canvas.mpl_connect("motion_notify_event", _on_move)
    cid_key = fig.canvas.mpl_connect("key_press_event", _on_key)
    cid_release = None
    if use_release_event:
        cid_release = fig.canvas.mpl_connect("button_release_event", _on_release)

    tkbinds = []
    try:
        w = fig.canvas.get_tk_widget()
        H = fig.canvas.get_width_height()[1]

        def _tk_to_ts(tk_event):
            x_disp = float(getattr(tk_event, "x", np.nan))
            y_tk = float(getattr(tk_event, "y", np.nan))
            if not np.isfinite(x_disp) or not np.isfinite(y_tk):
                return None
            y_disp = float(H) - y_tk

            ax_hit = None
            for ax in fig.axes:
                try:
                    if ax.bbox.contains(x_disp, y_disp):
                        ax_hit = ax
                        break
                except Exception:
                    continue
            if ax_hit is None:
                return None

            try:
                xdata = ax_hit.transData.inverted().transform((x_disp, y_disp))[0]
            except Exception:
                return None

            return _x_to_ts(xdata)

        def _tk_select(event, tag):
            if debug_interaction:
                _p(f"[picker:tk:recv] tag={tag} num={getattr(event,'num',None)} state={getattr(event,'state',None)} x={getattr(event,'x',None)} y={getattr(event,'y',None)}")
            if _toolbar_mode() != "":
                if debug_interaction:
                    _p("[picker:tk:ignored] reason=toolbar_mode")
                return
            ts = _tk_to_ts(event)
            if ts is None:
                if debug_interaction:
                    _p("[picker:tk:ignored] reason=ts_none")
                return
            _select_at_timestamp(ts, source=f"tk:{tag}", decoded=tag)

        w.bind("<Button-3>", lambda e: _tk_select(e, "B3"), add="+")
        tkbinds.append(("<Button-3>", None))

        w.bind("<Control-Button-1>", lambda e: _tk_select(e, "CtrlB1"), add="+")
        tkbinds.append(("<Control-Button-1>", None))

        w.bind("<Shift-Button-1>", lambda e: _tk_select(e, "ShiftB1"), add="+")
        tkbinds.append(("<Shift-Button-1>", None))

        fig._mhdturbpy_tkbinds = tkbinds
    except Exception as e:
        fig._mhdturbpy_tkbinds = []
        _p("[picker] tkbind install skipped:", e)

    fig._mhdturbpy_interval_picker = {
        "events": events,
        "cid_press": cid_press,
        "cid_release": cid_release,
        "cid_key": cid_key,
        "cid_move": cid_move,
    }

    _p(f"[picker] installed mpl(cids={cid_press},{cid_release},{cid_move},{cid_key}) + tkbinds={len(getattr(fig,'_mhdturbpy_tkbinds',[]))}")
    _p(f"[picker] left_select_fallback(default)=True  (toggle with 'l')")
    _p(f"[picker] dedupe_ms={dedupe_ms}  dedupe_tol_ns={dedupe_tol_ns}  span_axes={len(span_axes)} marker_axes={len(marker_axes)}  move_throttle_ms={move_throttle_ms}  use_release_event={use_release_event}")

    return events



# ============================================================
# Horizons (GSE) ephemeris: FIX ambiguity by using numeric IDs
# ============================================================
_HORIZONS_SC_ID = {
    "ACE": -92,
    "WIND": -8,
    "IMAP": -43,
    "SWFO-L1": -231,
    "SWIFO-1": -231,
    "SOLAR-1": -231,
    "SOLAR 1": -231,
    "DSCOVR": -78,
    "DISCOVER": -78,
    "DISCOVR": -78,

    "ADITYA": -156,
    "ADIT": -156,
    "ADITYA-L1": -156,
    "ADITYA L1": -156,

    "SOHO": -21,
    "PSP": -96,
    "PARKER SOLAR PROBE": -96,
    "SOLAR ORBITER": -144,
    "SOLO": -144,

    "ULYSSES": -55,
}

_SC_NAME_NORMALIZE = {
    "ACE": "ACE",
    "WIND": "WIND",
    "IMAP": "IMAP",
    "SWFOL1": "SWFO-L1",
    "SWIFOL1": "SWIFO-1",
    "SOLAR1": "SOLAR-1",
    "DSCOVR": "DSCOVR",
    "DISCOVR": "DSCOVR",
    "DISCOVER": "DSCOVR",

    "ADITYA": "ADITYA",
    "ADITYAL1": "ADITYA",
    "ADIT": "ADITYA",
    "AIDTYA": "ADITYA",
    "AIDTYAL1": "ADITYA",

    "SOHO": "SOHO",
    "PSP": "PSP",
    "PARKERSOLARPROBE": "PSP",
    "SOLARORBITER": "SOLO",
    "SOLO": "SOLO",

    "ULYSSES": "ULYSSES",
    "ULY": "ULYSSES",
    "ULYS": "ULYSSES",
}




def _horizons_step_for_index(idx, min_step_s=60, max_points=12000):
    if idx is None or len(idx) < 2:
        return "5m"

    t0 = pd.Timestamp(idx[0]).to_pydatetime()
    t1 = pd.Timestamp(idx[-1]).to_pydatetime()
    dur_s = max(1.0, (t1 - t0).total_seconds())

    dt_s = np.diff(pd.DatetimeIndex(idx).view("int64")) / 1e9
    med = float(np.nanmedian(dt_s[np.isfinite(dt_s)])) if np.any(np.isfinite(dt_s)) else float(min_step_s)

    step_s = max(float(min_step_s), med)
    if dur_s / step_s > max_points:
        step_s = math.ceil(dur_s / float(max_points))

    step_s = int(max(1, round(step_s)))

    if step_s % 86400 == 0:
        return f"{step_s // 86400}d"
    if step_s % 3600 == 0:
        return f"{step_s // 3600}h"
    if step_s % 60 == 0:
        return f"{step_s // 60}m"
    return f"{step_s}s"


def _resolve_horizons_target(sc_name):
    raw = str(sc_name).strip()
    up = raw.upper()
    compact = re.sub(r"[\s_\-]+", "", up)
    canon = _SC_NAME_NORMALIZE.get(compact, _SC_NAME_NORMALIZE.get(up, up))

    if canon in _HORIZONS_SC_ID:
        return int(_HORIZONS_SC_ID[canon]), None

    if up.lstrip("+-").isdigit():
        return int(up), None

    return raw, None



def _ephem_is_usable(df, *, min_finite_frac=0.90):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return False
    need = ["x_au", "y_au", "z_au"]
    if any(c not in df.columns for c in need):
        return False
    arr = df[need].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(arr).all(axis=1)
    if finite.size == 0:
        return False
    return (finite.mean() >= float(min_finite_frac))




def _ensure_ephem_distance_au(df):
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return df
    if "distance_au" in df.columns:
        return df
    need = ("x_au", "y_au", "z_au")
    if any(c not in df.columns for c in need):
        return df
    out = df.copy()
    x = out["x_au"].to_numpy(dtype=float, copy=False)
    y = out["y_au"].to_numpy(dtype=float, copy=False)
    z = out["z_au"].to_numpy(dtype=float, copy=False)
    out["distance_au"] = np.sqrt(x * x + y * y + z * z)
    return out

def _ensure_cached_ephem_gse_in_final(fin, fin_path, sc_name, par_idx, force=False, refresh_if_bad=False):
    """
    Ensure fin['Ephem']['GSE'] exists and is aligned to `par_idx`.

    - If ephem exists and is aligned/usable -> reuse.
    - If missing OR (refresh_if_bad and unusable) -> re-download from Horizons and cache to final.pkl.

    Also ensures a scalar geocentric distance time series is available as column
    'distance_au' (computed from x_au,y_au,z_au in GSE) and cached to final.pkl.
    """
    if "Ephem" not in fin or not isinstance(fin["Ephem"], dict):
        fin["Ephem"] = {}

    have = ("GSE" in fin["Ephem"]) and isinstance(fin["Ephem"]["GSE"], pd.DataFrame)

    if have and (not force):
        df0 = _ensure_dtindex(fin["Ephem"]["GSE"])
        if isinstance(df0.index, pd.DatetimeIndex) and len(df0.index) > 0:
            union = df0.index.union(pd.DatetimeIndex(par_idx))
            df0a = df0.reindex(union).interpolate("time").reindex(par_idx).ffill().bfill()

            before_has_dist = ("distance_au" in df0a.columns)
            df0a = _ensure_ephem_distance_au(df0a)
            after_has_dist = ("distance_au" in df0a.columns)

            if (not refresh_if_bad) or _ephem_is_usable(df0a):
                fin["Ephem"]["GSE"] = df0a

                # If we added distance_au (or meta is missing), persist back to final.pkl
                if (not before_has_dist) and after_has_dist:
                    fin["Ephem"].setdefault("meta_GSE", {})
                    fin["Ephem"]["meta_GSE"].update(
                        {
                            "frame": "GSE",
                            "source": "JPL Horizons via sunpy.coordinates.get_horizons_coord",
                            "cadence": _horizons_step_for_index(par_idx),
                            "aligned_to": "fin['Par']['V_resampled'] index",
                            "columns": [c for c in ["x_au", "y_au", "z_au", "distance_au"] if c in df0a.columns],
                        }
                    )
                    _atomic_pickle(fin, fin_path)
                    return True

                return False

    # Download fresh ephemeris aligned to par_idx
    df = _download_ephem_gse_horizons(sc_name, par_idx)
    df = _ensure_ephem_distance_au(df)

    if refresh_if_bad and (not _ephem_is_usable(df)):
        raise RuntimeError(f"Downloaded ephemeris for {sc_name} is unusable (too many NaNs).")

    fin["Ephem"]["GSE"] = df
    fin["Ephem"]["meta_GSE"] = {
        "frame": "GSE",
        "source": "JPL Horizons via sunpy.coordinates.get_horizons_coord",
        "cadence": _horizons_step_for_index(par_idx),
        "aligned_to": "fin['Par']['V_resampled'] index",
        "columns": [c for c in ["x_au", "y_au", "z_au", "distance_au"] if c in df.columns],
    }

    _atomic_pickle(fin, fin_path)
    return True

def _download_ephem_gse_horizons(target, idx, *, min_step_s=60, max_points=12000):
    """
    Download spacecraft ephemeris via JPL Horizons (SunPy) and return a DataFrame
    indexed on `idx` with GSE x/y/z in AU.

    Fixes:
    - Horizons `step` is a cadence string (e.g. '60s', '5m'), never a raw integer.
    - Cartesian extraction via `SkyCoord.cartesian.x/y/z`.
    - Time interpolation onto `idx`.
    """
    import numpy as np
    import pandas as pd
    import astropy.units as u
    from sunpy.coordinates import get_horizons_coord
    from sunpy.coordinates.frames import GeocentricSolarEcliptic

    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        raise ValueError("Need at least two timestamps to download Horizons ephemeris.")

    # Ensure monotonic, no-NaN index
    idx = idx[~idx.isna()]
    if not idx.is_monotonic_increasing:
        idx = idx.sort_values()
    if len(idx) < 2:
        raise ValueError("Need at least two valid timestamps to download Horizons ephemeris.")

    body, id_type = _resolve_horizons_target(target)

    # Choose a Horizons cadence that won't exceed max_points
    # Start from a cadence implied by idx; then coarsen if needed.
    step = _horizons_step_for_index(idx, min_step_s=min_step_s, max_points=max_points)

    t0 = pd.Timestamp(idx[0]).to_pydatetime()
    t1 = pd.Timestamp(idx[-1]).to_pydatetime()

    time = {
        "start": pd.Timestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
        "stop":  pd.Timestamp(t1).strftime("%Y-%m-%d %H:%M:%S"),
        "step":  step,
    }

    coord0 = get_horizons_coord(body, time=time, id_type=id_type)
    c = coord0.transform_to(GeocentricSolarEcliptic(obstime=coord0.obstime))

    car = c.cartesian
    t = pd.to_datetime(c.obstime.datetime64)

    df = pd.DataFrame(
        {
            "x_au": car.x.to_value(u.AU),
            "y_au": car.y.to_value(u.AU),
            "z_au": car.z.to_value(u.AU),
        },
        index=pd.DatetimeIndex(t),
    )
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.index.name = "time"

    # Interpolate onto the requested idx
    out_idx = pd.DatetimeIndex(idx)
    union = df.index.union(out_idx)
    df_u = df.reindex(union)

    # If Horizons returned too few points (or all NaN), interpolation will fail;
    # let it raise a clear error upstream.
    df_u = df_u.interpolate(method="time", limit_direction="both")
    df_out = df_u.reindex(out_idx).ffill().bfill()

    return df_out




def _download_sun_distance_au_horizons(
    target,
    idx,
    min_step_s=120,
    max_points=5000,
):
    """
    Return heliocentric distance (Sun->target) in AU on the exact DatetimeIndex `idx`,
    using the JPL Horizons *API* directly (robust to the OUT_UNITS=AU-D failure you hit).

    This avoids sunpy.get_horizons_coord(), which (in some stacks) constructs an
    OUT_UNITS=AU-D query that Horizons rejects ("Unknown units specification").
    """
    import numpy as np
    import pandas as pd
    from astropy.time import Time
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    idx = pd.DatetimeIndex(idx)
    if len(idx) == 0:
        return pd.Series(dtype="float64", index=idx, name="Dist_au")

    idx_sorted = idx.sort_values()
    t0 = pd.Timestamp(idx_sorted[0]).to_pydatetime()
    t1 = pd.Timestamp(idx_sorted[-1]).to_pydatetime()

    total_s = max(1.0, (idx_sorted[-1] - idx_sorted[0]).total_seconds())
    step_s = int(np.ceil(total_s / max(1, (max_points - 1))))
    step_s = max(int(min_step_s), step_s)

    def _resolve_command_and_id_type(x):
        if isinstance(x, (int, np.integer)):
            return str(int(x)), "id"
        s = str(x).strip()
        if s and (s[0] in "+-" and s[1:].isdigit() or s.isdigit()):
            return str(int(s)), "id"

        key = s.upper().replace("-", "").replace("_", "").replace(" ", "")
        sc_to_naif = {
            "PSP": "-96",
            "PARKERSOLARPROBE": "-96",
            "SOLO": "-144",
            "SOLARORBITER": "-144",
            "WIND": "-8",
            "ACE": "-92",
            "ULYSSES": "-55",
        }
        if key in sc_to_naif:
            return sc_to_naif[key], "id"

        return s, "name"

    body, id_type = _resolve_command_and_id_type(target)

    def _horizons_vectors_api(body_str, id_type_str, start_dt, stop_dt, step_seconds):
        base = "https://ssd.jpl.nasa.gov/api/horizons.api"

        if id_type_str == "id":
            command = f"'{body_str}'"
        else:
            command = f"'{body_str}'"

        params = {
            "format": "text",
            "MAKE_EPHEM": "'YES'",
            "TABLE_TYPE": "'VECTORS'",
            "COMMAND": command,
            "CENTER": "'500@10'",
            "REF_PLANE": "'ECLIPTIC'",
            "REF_SYSTEM": "'ICRF'",
            "TP_TYPE": "'ABSOLUTE'",
            "VEC_LABELS": "'YES'",
            "VEC_TABLE": "'3'",
            "CSV_FORMAT": "'YES'",
            "VEC_CORR": "'NONE'",
            "VEC_DELTA_T": "'NO'",
            "OBJ_DATA": "'NO'",
            "START_TIME": f"'{pd.Timestamp(start_dt).strftime('%Y-%m-%d %H:%M:%S')}'",
            "STOP_TIME": f"'{pd.Timestamp(stop_dt).strftime('%Y-%m-%d %H:%M:%S')}'",
            "STEP_SIZE": f"'{int(step_seconds)}s'",
            "OUT_UNITS": "'KM-S'",
        }

        url = base + "?" + urlencode(params)
        req = Request(url, headers={"User-Agent": "MHDTurbPy/1.0"})
        with urlopen(req, timeout=60) as r:
            return r.read().decode("utf-8", errors="replace")

    def _parse_vectors_csv(text):
        if "$$SOE" not in text or "$$EOE" not in text:
            raise ValueError(text[:800])

        pre, rest = text.split("$$SOE", 1)
        data_block, _ = rest.split("$$EOE", 1)

        header_line = None
        for line in reversed(pre.splitlines()):
            if "JDTDB" in line and "," in line:
                header_line = line
                break
        if header_line is None:
            raise ValueError("Could not find Horizons CSV header line (JDTDB, ...).")

        cols = [c.strip() for c in header_line.split(",") if c.strip()]
        raw_lines = []
        for line in data_block.splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("*"):
                continue
            raw_lines.append(s)

        import io
        import pandas as pd

        df = pd.read_csv(io.StringIO("\n".join(raw_lines)), header=None)
        if len(cols) == df.shape[1]:
            df.columns = cols
        else:
            df.columns = [f"c{i}" for i in range(df.shape[1])]

        return df

    try:
        txt = _horizons_vectors_api(body, id_type, t0, t1, step_s)
        df = _parse_vectors_csv(txt)

        jd = df["JDTDB"].astype(float).to_numpy()
        t = pd.to_datetime(Time(jd, format="jd", scale="tdb").utc.datetime64)

        if "RG" in df.columns:
            rg_km = df["RG"].astype(float).to_numpy()
            dist_au_grid = rg_km / 149597870.700
        else:
            x = df["X"].astype(float).to_numpy()
            y = df["Y"].astype(float).to_numpy()
            z = df["Z"].astype(float).to_numpy()
            dist_au_grid = np.sqrt(x * x + y * y + z * z) / 149597870.700

        s_grid = pd.Series(dist_au_grid, index=pd.DatetimeIndex(t)).sort_index()
        s_grid = s_grid[~s_grid.index.duplicated(keep="first")]

        all_idx = s_grid.index.union(idx_sorted)
        s_interp = (
            s_grid.reindex(all_idx)
            .sort_index()
            .interpolate(method="time", limit_direction="both")
            .reindex(idx_sorted)
        )

        out = pd.Series(s_interp.to_numpy(), index=idx_sorted, name="Dist_au")
        if not idx_sorted.equals(idx):
            out = out.reindex(idx)

        return out.astype("float64")

    except Exception as e:
        import pandas as pd

        print(f"[WARN] Horizons Sun-distance failed for target={target!r}: {e}")
        return pd.Series(np.nan, index=idx, name="Dist_au", dtype="float64")


def _ensure_cached_sun_distance_au_in_par(fin, *, fin_path, sc_name, par_idx, force=False):
    """
    Ensure Par['V_resampled'] contains Dist_au (Sun->SC distance in AU), aligned to `par_idx`.
    If missing, download from Horizons and cache back into final.pkl.
    """
    if not isinstance(fin, dict):
        return False
    if "Par" not in fin or not isinstance(fin["Par"], dict) or "V_resampled" not in fin["Par"]:
        return False

    par = fin["Par"]["V_resampled"]
    if not isinstance(par, pd.DataFrame):
        return False
    par = _ensure_dtindex(par)

    have = ("Dist_au" in par.columns)
    if have and (not force):
        # Align to par_idx and persist alignment only if index mismatch is severe.
        par_al = _time_reindex_interp(par, pd.DatetimeIndex(par_idx))
        fin["Par"]["V_resampled"] = par_al
        return False

    dist = _download_sun_distance_au_horizons(sc_name, pd.DatetimeIndex(par_idx))
    par_al = _time_reindex_interp(par, pd.DatetimeIndex(par_idx))
    par_al["Dist_au"] = dist.to_numpy(copy=False)

    fin["Par"]["V_resampled"] = par_al
    fin.setdefault("Ephem", {}).setdefault("meta_SUN", {})
    fin["Ephem"]["meta_SUN"].update(
        {
            "quantity": "Dist_au",
            "frame": "Sun-centered (Horizons default)",
            "source": "JPL Horizons via sunpy.coordinates.get_horizons_coord",
            "cadence": _horizons_step_for_index(pd.DatetimeIndex(par_idx)),
            "aligned_to": "fin['Par']['V_resampled'] index",
        }
    )

    _atomic_pickle(fin, Path(fin_path))
    return True


def _replace_distance_panel_with_earth_distance(panel_config, *, sc_list, warn_missing=True):
    """
    Replace the first panel that looks like a 'Sun distance' panel (Dist_au or Rsun_from_au)
    with an Earth-distance panel in R_E using the 'Eph' source (extra_sources_by_sc[sc]['Eph']).

    If no such panel exists, append the Earth-distance panel at the end.
    """
    cfg = dict(panel_config) if isinstance(panel_config, dict) else {"panels": []}
    panels = list(cfg.get("panels", []))

    def is_sun_distance_panel(p):
        for ax in p.get("axes", []) or []:
            for s in ax.get("series", []) or []:
                if s.get("col") == "Dist_au":
                    return True
                if s.get("kind") == "func" and s.get("func") == "Rsun_from_au":
                    return True
        return False

    idx = None
    for i, p in enumerate(panels):
        if isinstance(p, dict) and is_sun_distance_panel(p):
            idx = i
            break

    earth_panel = {
        "axes": [
            {
                "axis_id": "left",
                "source": "Eph",
                "scale": "linear",
                "series": [
                    {"kind": "col", "col": "distance_re", "label": r"$d_{E}~[R_{E}]$",
                     "style": {"lw": 0.9, "ls": "-", "ms": 0, "color": "black"}},
                ],
                "legend": "main",
            }
        ]
    }

    if idx is None:
        if warn_missing:
            print("[plot] No Sun-distance panel found; appending Earth-distance panel.")
        panels.append(earth_panel)
    else:
        panels[idx] = earth_panel

    cfg["panels"] = panels
    return cfg
def _atomic_pickle(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(obj, tmp)
    os.replace(tmp, path)


def _ensure_cached_ephem_gse_in_final(fin, fin_path, sc_name, par_idx, force=False, refresh_if_bad=False):
    """
    Ensure fin['Ephem']['GSE'] exists and is aligned to `par_idx`.

    - If ephem exists and is aligned/usable -> reuse.
    - If missing OR (refresh_if_bad and unusable) -> re-download from Horizons and cache to final.pkl.
    """
    if "Ephem" not in fin or not isinstance(fin["Ephem"], dict):
        fin["Ephem"] = {}

    have = ("GSE" in fin["Ephem"]) and isinstance(fin["Ephem"]["GSE"], pd.DataFrame)

    if have and (not force):
        df0 = _ensure_dtindex(fin["Ephem"]["GSE"])
        if isinstance(df0.index, pd.DatetimeIndex) and len(df0.index) > 0:
            # align to par_idx via time interpolation
            union = df0.index.union(pd.DatetimeIndex(par_idx))
            df0a = df0.reindex(union).interpolate("time").reindex(par_idx).ffill().bfill()

            if (not refresh_if_bad) or _ephem_is_usable(df0a):
                fin["Ephem"]["GSE"] = df0a
                return False

    # Download fresh ephemeris aligned to par_idx
    df = _download_ephem_gse_horizons(sc_name, par_idx)

    if refresh_if_bad and (not _ephem_is_usable(df)):
        raise RuntimeError(f"Downloaded ephemeris for {sc_name} is unusable (too many NaNs).")

    fin["Ephem"]["GSE"] = df
    fin["Ephem"]["meta_GSE"] = {
        "frame": "GSE",
        "source": "JPL Horizons via sunpy.coordinates.get_horizons_coord",
        "cadence": _horizons_step_for_index(par_idx),
        "aligned_to": "fin['Par']['V_resampled'] index",
        "columns": ["x_au", "y_au", "z_au"],
    }

    _atomic_pickle(fin, fin_path)
    return True


def _extract_v_gse(par):
    """Extract a GSE velocity vector from Par (supports Vx/Vy/Vz or vx/vy/vz)."""
    if all(c in par.columns for c in ("Vx", "Vy", "Vz")):
        return par[["Vx", "Vy", "Vz"]].to_numpy(dtype=float, copy=False)
    if all(c in par.columns for c in ("vx", "vy", "vz")):
        return par[["vx", "vy", "vz"]].to_numpy(dtype=float, copy=False)
    return None


def _flow_hat_and_vsw(
    par1,
    par2,
    *,
    mode="mean",
    fallback_dir_gse=(-1.0, 0.0, 0.0),
    smooth_window=None,
    vsw_fallback=400.0,
    alternate_legend_sides=False,
    legend_start_side="right",
    auto_adjust_margins_for_legends=True,
):
    """
    Estimate the solar-wind flow direction (unit vector) and speed time series in GSE.

    Parameters
    ----------
    par1, par2 : pandas.DataFrame
        Plasma parameter DataFrames for sc1 and sc2, already aligned onto the same
        time grid (or at least with comparable indices). If a V vector cannot be
        extracted from one/both, the function falls back to ``fallback_dir_gse``
        and ``vsw_fallback``.
    mode : {"mean","mean_v","first","sc1_v","second","sc2_v",...}
        How to form the flow vector:
        - "mean"/"mean_v": mean of v1 and v2 (when available).
        - "first"/"sc1_v": use v from par1.
        - "second"/"sc2_v": use v from par2.
    fallback_dir_gse : 3-tuple
        Constant direction used when V is not available or invalid.
    smooth_window : str or None
        Optional time-based rolling window (e.g. "30s", "3min") applied to the
        chosen V vector before normalizing.
    vsw_fallback : float
        Fallback solar-wind speed [km/s] when V is missing/invalid.

    Returns
    -------
    vhat : (N,3) ndarray
        Unit flow direction at each timestamp.
    vsw : (N,) ndarray
        Flow speed [km/s] used at each timestamp (after fallback).
    """
    mode = str(mode).lower().strip()

    # Synonyms / user-friendly aliases
    if mode in {"mean_v", "avg", "average", "meanv"}:
        mode = "mean"
    elif mode in {"sc1_v", "first_v", "sc1", "one", "1"}:
        mode = "first"
    elif mode in {"sc2_v", "second_v", "sc2", "two", "2"}:
        mode = "second"

    # Extract V vectors (may fail depending on what Par contains)
    v1 = None
    v2 = None
    try:
        v1 = _extract_v_gse(par1)
    except Exception:
        v1 = None
    try:
        v2 = _extract_v_gse(par2)
    except Exception:
        v2 = None

    # Choose V according to mode (with graceful fallback to whichever exists)
    if mode == "first":
        v = v1 if v1 is not None else v2
    elif mode == "second":
        v = v2 if v2 is not None else v1
    else:  # "mean"
        if (v1 is not None) and (v2 is not None):
            v = 0.5 * (v1 + v2)
        else:
            v = v1 if v1 is not None else v2

    # Reference index for optional smoothing and output length
    idx = None
    if hasattr(par1, "index"):
        idx = par1.index
    elif hasattr(par2, "index"):
        idx = par2.index

    n = len(idx) if idx is not None else (len(v) if v is not None else 0)

    # Normalize fallback direction once
    fhat = np.asarray(fallback_dir_gse, dtype=float)
    fn = np.linalg.norm(fhat)
    if (not np.isfinite(fn)) or (fn <= 0.0):
        fhat = np.array([-1.0, 0.0, 0.0], dtype=float)
        fn = 1.0
    fhat = fhat / fn

    # If no V is available, return constant direction + constant speed
    if (v is None) or (n == 0):
        vhat = np.tile(fhat, (n, 1))
        vsw = np.full(n, float(vsw_fallback), dtype=float)
        return vhat, vsw

    # Optional smoothing (time-based rolling mean)
    if smooth_window not in (None, "", 0):
        try:
            v_df = pd.DataFrame(v, index=idx, columns=["vx", "vy", "vz"])
            v_df = v_df.rolling(str(smooth_window), center=True, min_periods=1).mean()
            v = v_df.to_numpy()
        except Exception:
            # Keep unsmoothed v if rolling fails (e.g. non-monotonic index)
            pass

    vsw = np.sqrt(np.einsum("ij,ij->i", v, v)).astype(float)
    good = np.isfinite(vsw) & (vsw > 0.0)

    vhat = np.empty_like(v, dtype=float)
    vhat[:] = np.nan
    if np.any(good):
        vhat[good] = (v[good].T / vsw[good]).T

    # Fallback for bad rows
    if not np.all(good):
        vhat[~good] = fhat
        vsw[~good] = float(vsw_fallback)

    return vhat, vsw

def _compute_sep_timeseries_two_sc(
    ephem1_gse,
    ephem2_gse,
    par1,
    par2,
    *,
    flow_mode="mean",
    fallback_dir_gse=(-1.0, 0.0, 0.0),
    v_smooth_window="5min",
    vsw_fallback=400.0,
    alternate_legend_sides=False,
    legend_start_side="right",
    auto_adjust_margins_for_legends=True,
):
    """
    Compute parallel/perpendicular separation between two spacecraft as a time series.

    The separation vector is computed in GSE from cached ephemerides (AU). It is then
    decomposed into components parallel/perpendicular to the (estimated) solar-wind
    flow direction.

    Returns a DataFrame indexed like ``ephem1_gse.index`` with columns:
        dpar_au, dperp_au, dpar_re, dperp_re, tau_h, flow_hat_[xyz], vsw_kms_used.
    """
    idx = ephem1_gse.index
    e1 = ephem1_gse.reindex(idx).interpolate(method="time")
    e2 = ephem2_gse.reindex(idx).interpolate(method="time")

    # dr in AU -> km
    dr_au = np.column_stack(
        [
            (e2["x_au"].to_numpy() - e1["x_au"].to_numpy()),
            (e2["y_au"].to_numpy() - e1["y_au"].to_numpy()),
            (e2["z_au"].to_numpy() - e1["z_au"].to_numpy()),
        ]
    )
    dr_km = dr_au * AU_KM
    dr2_km2 = np.einsum("ij,ij->i", dr_km, dr_km)

    # Flow direction + speed (uses measured V when available, with fallback)
    vhat, vsw = _flow_hat_and_vsw(
        par1,
        par2,
        mode=flow_mode,
        fallback_dir_gse=fallback_dir_gse,
        smooth_window=v_smooth_window,
        vsw_fallback=vsw_fallback,
    )

    # Parallel component (signed); perpendicular magnitude (>=0)
    dpar_km = np.einsum("ij,ij->i", dr_km, vhat)
    dperp_km = np.sqrt(np.maximum(dr2_km2 - dpar_km * dpar_km, 0.0))

    # Convert units
    dpar_au = dpar_km / AU_KM
    dperp_au = dperp_km / AU_KM
    dpar_re = dpar_km / R_EARTH_KM
    dperp_re = dperp_km / R_EARTH_KM

    # Equivalent "streaming time" along flow direction
    # vsw is [km/s], so tau_h = (km)/(km/s)/3600
    tau_h = dpar_km / (vsw * 3600.0)

    out = pd.DataFrame(
        index=idx,
        data={
            "dpar_au": dpar_au,
            "dperp_au": dperp_au,
            "dpar_re": dpar_re,
            "dperp_re": dperp_re,
            "tau_h": tau_h,
            "flow_hat_x": vhat[:, 0],
            "flow_hat_y": vhat[:, 1],
            "flow_hat_z": vhat[:, 2],
            "vsw_kms_used": vsw,
        },
    )
    return out

def interactive_visualize_downloaded_intervals(
    sc,
    final_Par,
    final_Mag,
    nn_df,
    my_dir,
    format_2_return="%Y_%m_%d",
    join_path_figs=True,
    save_fig=True,
    fname_tag="",
    save_path=None,
    autosave=True,
    export_csv=False,
    snap_to_data=False,
    enable_comments=True,
    debug_interaction=False,
    panel_config=None,
    panel_edits=None,
    auto_ylims=True,
    warn_missing=True,
    debug_plot_config=False,
    plot_defaults=None,
    span_color="0.85",
    span_alpha=0.35,
    enable_multicursor=True,
    snap_index_mode="first_sc",
    normalize_timeseries=None,
    enforce_sc_linestyle=True,
    extra_sources_by_sc=None,
    enable_flow_separation_panel=False,
    alternate_legend_sides=False,
    legend_start_side="right",
    auto_adjust_margins_for_legends=True,
):
    plot_defaults = DEFAULT_PLOT_PARAMS if plot_defaults is None else plot_defaults

    sc_list = _as_sc_list(sc)
    snap_index_mode_norm = _normalize_snap_index_mode(snap_index_mode)
    normalize_timeseries_mode = _normalize_timeseries_mode(normalize_timeseries)

    par_by_sc = _normalize_sc_df_input(final_Par, sc_list, "final_Par")
    mag_by_sc = _normalize_sc_df_input(final_Mag, sc_list, "final_Mag")
    sig_by_sc = _normalize_sc_df_input(nn_df, sc_list, "nn_df")

    for sc_name in sc_list:
        par_by_sc[sc_name] = _ensure_dtindex(par_by_sc[sc_name])
        mag_by_sc[sc_name] = _ensure_dtindex(mag_by_sc[sc_name])
        sig_by_sc[sc_name] = _ensure_dtindex(sig_by_sc[sc_name])

    first_sc = sc_list[0]

    # ----------------------------
    # Frame bookkeeping (explicit)
    # ----------------------------
    mag_frame_by_sc = {}
    v_frame_by_sc = {}
    rtn_flag_by_sc = {}

    for sc_name in sc_list:
        mag_df = mag_by_sc[sc_name]
        par_df = par_by_sc[sc_name]

        mag_frame = _infer_mag_frame(mag_df)
        v_frame = _infer_v_frame(par_df)
        mag_frame_by_sc[sc_name] = mag_frame
        v_frame_by_sc[sc_name] = v_frame

        # NOTE: 'B_RTN' is a historical MHDTurbPy name for |B| (magnitude), regardless of frame.
        # We also provide a clearer alias 'B_mag'.
        if mag_frame == "RTN":
            arr = mag_df[["Br", "Bt", "Bn"]].to_numpy(copy=False)
            bmag = np.sqrt(np.einsum("ij,ij->i", arr, arr))
            mag_df["B_RTN"] = bmag
            mag_df["B_mag"] = bmag
            rtn_flag_by_sc[sc_name] = 1
        elif mag_frame == "GSE":
            arr = mag_df[["Bx", "By", "Bz"]].to_numpy(copy=False)
            bmag = np.sqrt(np.einsum("ij,ij->i", arr, arr))
            mag_df["B_RTN"] = bmag
            mag_df["B_mag"] = bmag
            rtn_flag_by_sc[sc_name] = 0
        else:
            rtn_flag_by_sc[sc_name] = None
            if warn_missing:
                print(f"[plot:missing] sc={sc_name} cannot infer Mag frame (need Br/Bt/Bn or Bx/By/Bz).")

    # Use the first spacecraft's Mag frame for default panel labeling.
    rtn_flag = rtn_flag_by_sc.get(first_sc, 0) or 0

    if debug_plot_config:
        print(f"[frames] Mag frame by sc: {mag_frame_by_sc}")
        print(f"[frames] Par V frame by sc: {v_frame_by_sc}")

    if panel_config is None:
        panel_config = default_panel_config(sc=str(first_sc), rtn_flag=rtn_flag)

    panel_config = apply_panel_edits(panel_config, panel_edits)

    if enable_flow_separation_panel and len(sc_list) >= 2:
        panel_config = dict(panel_config)
        panel_config["panels"] = list(panel_config.get("panels", []))
        panel_config["panels"].append(
            {
                "axes": [
                    {
                        "axis_id": "left",
                        "source": "Sep",
                        "scale": "linear",
                        "only_if": {"sc_equals": first_sc},
                        "series": [
                            {"kind": "col", "col": "dpar_re", "label": r"$d_{\parallel}~[R_{E}]$",
                             "style": {"lw": 0.9, "ls": "-", "ms": 0, "color": "k"}},
                            {"kind": "col", "col": "dperp_re", "label": r"$d_{\perp}~[R_{E}]$",
                             "style": {"lw": 0.9, "ls": "--", "ms": 0, "color": "darkred"}},
                        ],
                        "legend": "main",
                    }
                ]
            }
        )

    # Repurpose the existing distance panel when flow separation is requested.
    # (Earth distance in R_E; no extra panel appended.)
    if enable_flow_separation_panel and len(sc_list) >= 1:
        panel_config = _replace_distance_panel_with_earth_distance(panel_config, sc_list=sc_list, warn_missing=warn_missing)

    panels = panel_config.get("panels", [])
    n_panels = len(panels)
    if n_panels <= 0:
        raise ValueError("panel_config produced zero panels")

    if debug_plot_config:
        print(f"[plot] n_panels={n_panels}")

    start_lim = min(par_by_sc[sc_name].index[0] for sc_name in sc_list)
    end_lim = max(par_by_sc[sc_name].index[-1] for sc_name in sc_list)

    base_w = float(plot_defaults["figure"]["base_width"])
    base_h7 = float(plot_defaults["figure"]["base_height_7pan"])
    fig_h = base_h7 * (n_panels / 7.0)

    fig, axs = plt.subplots(
        n_panels,
        sharex=True,
        figsize=(base_w, fig_h),
        gridspec_kw=dict(plot_defaults["figure"]["gridspec_kw"]),
    )
    axs = [axs] if n_panels == 1 else list(axs)

    if bool(alternate_legend_sides) and bool(auto_adjust_margins_for_legends):
        try:
            fig.subplots_adjust(left=0.14, right=0.86)
        except Exception:
            pass

    minor_tick_params, major_tick_params = inset_axis_params(size=str(plot_defaults["ticks"]["size"]))

    ls_cycle = ["-", "--", ":", "-."]
    sc_ls_map = {sc_name: ls_cycle[i % len(ls_cycle)] for i, sc_name in enumerate(sc_list)}

    data_sources_by_sc = {
        sc_name: {"Par": par_by_sc[sc_name], "Mag": mag_by_sc[sc_name], "Sig": sig_by_sc[sc_name]}
        for sc_name in sc_list
    }

    if isinstance(extra_sources_by_sc, dict):
        for sc_name, extra in extra_sources_by_sc.items():
            if sc_name in data_sources_by_sc and isinstance(extra, dict):
                data_sources_by_sc[sc_name].update(extra)

    axes_registry = _plot_from_panel_config(
        fig,
        axs,
        panel_config=panel_config,
        data_sources_by_sc=data_sources_by_sc,
        sc_list=sc_list,
        sc_ls_map=sc_ls_map,
        start_lim=pd.Timestamp(start_lim),
        end_lim=pd.Timestamp(end_lim),
        auto_ylims=bool(auto_ylims),
        warn_missing=bool(warn_missing),
        plot_defaults=plot_defaults,
        normalize_timeseries=normalize_timeseries_mode,
        enforce_sc_linestyle=bool(enforce_sc_linestyle),
        alternate_legend_sides=bool(alternate_legend_sides),
        legend_start_side=str(legend_start_side),
    )

    grid = plot_defaults["grid"]
    for ax in axs:
        ax.xaxis.grid(True, **grid["x_minor"])
        ax.xaxis.grid(True, **grid["x_major"])
        ax.yaxis.grid(True, **grid["y_minor"])
        ax.yaxis.grid(True, **grid["y_major"])
        ax.tick_params(**minor_tick_params)
        ax.tick_params(**major_tick_params)
        ax.set_xlim([start_lim, end_lim])

    if auto_ylims:
        _install_dynamic_ylims_on_xzoom(fig, axes_registry, plot_defaults)

    f1 = format_timestamp(min(mag_by_sc[sc_name].index[0] for sc_name in sc_list), format_2_return)
    f2 = format_timestamp(max(mag_by_sc[sc_name].index[-1] for sc_name in sc_list), format_2_return)
    tag = f"_{str(fname_tag).strip().replace(' ', '')}" if str(fname_tag).strip() else ""
    sc_tag = _sc_tag(sc_list)
    figure_name = f"{f1}_{f2}{tag}_{sc_tag}.png"

    my_dir = Path(my_dir)
    final_save_path = my_dir / "figures" if join_path_figs else my_dir
    final_save_path.mkdir(parents=True, exist_ok=True)
    if save_fig:
        fig.savefig(str(final_save_path / figure_name), format="png", dpi=300, bbox_inches="tight")

    if enable_multicursor:
        try:
            fig._mhdturbpy_multicursor = MultiCursor(fig.canvas, axs, color="0.3", lw=0.8, horizOn=False, vertOn=True, useblit=True)
        except Exception:
            try:
                fig._mhdturbpy_multicursor = MultiCursor(fig.canvas, axs, color="0.3", lw=0.8, horizOn=False, vertOn=True, useblit=False)
            except Exception:
                fig._mhdturbpy_multicursor = None
    else:
        fig._mhdturbpy_multicursor = None

    events_file = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        events_file = save_path / f"selected_intervals_{f1}_{f2}{tag}_{sc_tag}.pkl"

    if snap_index_mode_norm == "first_sc":
        snap_index = par_by_sc[sc_list[0]].index
    else:
        snap_index = pd.DatetimeIndex(
            np.unique(np.concatenate([par_by_sc[sc_name].index.view("int64") for sc_name in sc_list]))
        )

    events = _install_interval_selector(
        fig,
        axes_registry,
        events_file=events_file,
        autosave=autosave,
        export_csv=export_csv,
        span_color=span_color,
        span_alpha=span_alpha,
        snap_index=snap_index,
        snap_to_data=snap_to_data,
        enable_comments=enable_comments,
        debug_interaction=debug_interaction,
        resume=True,
        move_throttle_ms=33,
        use_release_event=False,
    )

    try:
        frames_meta = {
            "mag_frame_by_sc": dict(mag_frame_by_sc) if 'mag_frame_by_sc' in locals() else {},
            "par_v_frame_by_sc": dict(v_frame_by_sc) if 'v_frame_by_sc' in locals() else {},
            "separation_frame": "GSE" if bool(enable_flow_separation_panel) else None,
        }
        fig._mhdturbpy_frames = frames_meta
        events.attrs["frames"] = frames_meta
    except Exception:
        pass

    return fig, events


def _extract_interval_bounds_from_final(final_path):
    fin = pd.read_pickle(final_path)
    if not isinstance(fin, dict) or ("Par" not in fin) or ("V_resampled" not in fin["Par"]):
        raise TypeError(f"final.pkl at {final_path} did not match expected Par/V_resampled structure")
    par = _ensure_dtindex(fin["Par"]["V_resampled"])
    if not isinstance(par, pd.DataFrame) or len(par.index) == 0:
        raise ValueError(f"Par/V_resampled in {final_path} is empty or invalid")
    return pd.Timestamp(par.index[0]), pd.Timestamp(par.index[-1])


def _choose_closest_interval_index(final_paths, target_start, target_end):
    if len(final_paths) == 0:
        raise FileNotFoundError("No candidate final.pkl files were provided")
    best_idx = 0
    best_score = None
    for i, fp in enumerate(final_paths):
        try:
            st, en = _extract_interval_bounds_from_final(fp)
        except Exception:
            continue
        score = abs((st - target_start).total_seconds()) + abs((en - target_end).total_seconds())
        if best_score is None or score < best_score:
            best_score = score
            best_idx = i
    return best_idx


# ============================================================
# Wrapper: EXACT loading semantics
# ============================================================
def interactive_mhdturbpy_interval(
    sc,
    which_int,
    load_path,
    save_path,
    my_dir,
    rolling=None,
    resample_rule=None,
    fill_method=None,
    gap_thresholds=None,
    merge_tol="0s",
    load_files_func=None,
    autosave=True,
    resume=True,
    export_csv=False,
    snap_to_data=False,
    enable_comments=True,
    debug_interaction=False,
    panel_config=None,
    panel_edits=None,
    auto_ylims=True,
    warn_missing=True,
    debug_plot_config=False,
    plot_defaults=None,
    span_color="0.85",
    span_alpha=0.35,
    enable_multicursor=True,
    snap_index_mode="first_sc",
    normalize_timeseries=None,
    enforce_sc_linestyle=True,
    align_intervals_to_first_sc=True,
    enable_flow_separation=False,
    flow_dir_gse=(-1.0, 0.0, 0.0),
    flow_mode="mean",
    flow_v_smooth_window="5min",
    vsw_fallback=400.0,
    alternate_legend_sides=False,
    legend_start_side="right",
    auto_adjust_margins_for_legends=True,
):
    if load_files_func is None:
        load_files_func = func.load_files if (func is not None and hasattr(func, "load_files")) else load_files

    sc_list = _as_sc_list(sc)
    snap_index_mode_norm = _normalize_snap_index_mode(snap_index_mode)
    normalize_timeseries_mode = _normalize_timeseries_mode(normalize_timeseries)
    gap_thresholds = gap_thresholds or {}

    if isinstance(load_path, dict):
        load_path_by_sc = {str(k): str(v) for k, v in load_path.items()}
    else:
        load_path_by_sc = {sc_name: str(load_path) for sc_name in sc_list}

    final_par_by_sc = {}
    final_mag_by_sc = {}
    nn_by_sc = {}

    fin_by_sc = {}
    fin_path_by_sc = {}

    files_by_sc = {}
    for sc_name in sc_list:
        if sc_name not in load_path_by_sc:
            raise KeyError(f"load_path missing key for spacecraft '{sc_name}'")
        lp = load_path_by_sc[sc_name]

        _buf_out = io.StringIO()
        _buf_err = io.StringIO()
        with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
            files_by_sc[sc_name] = {
                "fin": load_files_func(lp, "final.pkl"),
                "gen": load_files_func(lp, "general.pkl"),
                "sig": load_files_func(lp, "sig_c_sig_r.pkl"),
                "mag": load_files_func(lp, "mag_gaps.pkl"),
                "qtn": load_files_func(lp, "qtn_gaps.pkl"),
                "par": load_files_func(lp, "par_gaps.pkl"),
                "sc_pot": load_files_func(lp, "sc_pot_gaps.pkl"),
            }

    first_sc = sc_list[0]
    first_n = len(files_by_sc[first_sc]["fin"])
    if first_n == 0:
        raise FileNotFoundError(f"No final.pkl found under {load_path_by_sc[first_sc]}")
    if not (0 <= which_int < first_n):
        raise IndexError(f"which_int={which_int} out of range for sc={first_sc} (found {first_n} intervals)")

    selected_idx_by_sc = {first_sc: int(which_int)}
    target_start, target_end = _extract_interval_bounds_from_final(files_by_sc[first_sc]["fin"][int(which_int)])

    for sc_name in sc_list[1:]:
        finnames = files_by_sc[sc_name]["fin"]
        n = len(finnames)
        if n == 0:
            raise FileNotFoundError(f"No final.pkl found under {load_path_by_sc[sc_name]}")
        if align_intervals_to_first_sc:
            selected_idx_by_sc[sc_name] = _choose_closest_interval_index(finnames, target_start, target_end)
        else:
            if not (0 <= which_int < n):
                raise IndexError(f"which_int={which_int} out of range for sc={sc_name} (found {n} intervals)")
            selected_idx_by_sc[sc_name] = int(which_int)

    for sc_name in sc_list:
        idx = selected_idx_by_sc[sc_name]
        finnames = files_by_sc[sc_name]["fin"]
        gennames = files_by_sc[sc_name]["gen"]
        signames = files_by_sc[sc_name]["sig"]
        maggaps = files_by_sc[sc_name]["mag"]
        qtngaps = files_by_sc[sc_name]["qtn"]
        pargaps = files_by_sc[sc_name]["par"]
        scpotgaps = files_by_sc[sc_name]["sc_pot"]

        fin_path = Path(finnames[idx])
        gen_path = gennames[idx] if idx < len(gennames) else None
        sig_path = signames[idx] if idx < len(signames) else None

        mag_gaps_path = maggaps[idx] if idx < len(maggaps) else None
        qtn_gaps_path = qtngaps[idx] if idx < len(qtngaps) else None
        par_gaps_path = pargaps[idx] if idx < len(pargaps) else None
        sc_pot_gaps_path = scpotgaps[idx] if idx < len(scpotgaps) else None

        fin = pd.read_pickle(fin_path)
        _ = pd.read_pickle(gen_path) if gen_path is not None else None
        sig = pd.read_pickle(sig_path) if sig_path is not None else None

        mag_gaps = pd.read_pickle(mag_gaps_path) if mag_gaps_path is not None else pd.DataFrame(columns=["Start", "End"])
        qtn_gaps = pd.read_pickle(qtn_gaps_path) if qtn_gaps_path is not None else pd.DataFrame(columns=["Start", "End"])
        par_gaps = pd.read_pickle(par_gaps_path) if par_gaps_path is not None else pd.DataFrame(columns=["Start", "End"])
        sc_pot_gaps = pd.read_pickle(sc_pot_gaps_path) if sc_pot_gaps_path is not None else pd.DataFrame(columns=["Start", "End"])

        if not isinstance(fin, dict) or ("Par" not in fin) or ("Mag" not in fin):
            raise TypeError(
                "final.pkl did not match expected dict structure with keys Par/Mag. "
                f"Got: {list(fin.keys()) if isinstance(fin, dict) else type(fin)}"
            )

        final_Par = fin["Par"]["V_resampled"]
        final_Mag = fin["Mag"]["B_resampled"]
        nn_df = sig

        if not isinstance(final_Par, pd.DataFrame):
            raise TypeError(f"fin['Par']['V_resampled'] expected DataFrame, got {type(final_Par)}")
        if not isinstance(final_Mag, pd.DataFrame):
            raise TypeError(f"fin['Mag']['B_resampled'] expected DataFrame, got {type(final_Mag)}")
        if not isinstance(nn_df, pd.DataFrame):
            raise TypeError(f"sig_c_sig_r.pkl expected DataFrame, got {type(nn_df)}")

        final_Par = _ensure_dtindex(final_Par)
        final_Mag = _ensure_dtindex(final_Mag)
        nn_df = _ensure_dtindex(nn_df)

        need_ephem = bool(enable_flow_separation)

        if not enable_flow_separation:
            # Ensure heliocentric distance Dist_au exists for the default distance panel.
            if "Dist_au" not in final_Par.columns:
                _ensure_cached_sun_distance_au_in_par(
                    fin,
                    fin_path=fin_path,
                    sc_name=sc_name,
                    par_idx=final_Par.index,
                    force=False,
                )
                # refresh view in case we cached Dist_au
                final_Par = _ensure_dtindex(fin["Par"]["V_resampled"])

        if need_ephem:
            _ensure_cached_ephem_gse_in_final(
                fin,
                fin_path=fin_path,
                sc_name=sc_name,
                par_idx=final_Par.index,
                force=False,
                refresh_if_bad=True,
            )
        if len(gap_thresholds) > 0:
            masks = build_large_gap_masks(
                mag_gaps=mag_gaps,
                qtn_gaps=qtn_gaps,
                par_gaps=par_gaps,
                sc_pot_gaps=sc_pot_gaps,
                gap_thresholds=gap_thresholds,
                merge_tol=merge_tol,
            )
            final_Mag = mask_df_with_gaps(final_Mag, masks.get("mag"))
            final_Par = mask_df_with_gaps(final_Par, masks.get("par"))
            nn_df = mask_df_with_gaps(nn_df, masks.get("par"))

        final_Mag = _resample_fill_roll(final_Mag, resample_rule=resample_rule, fill_method=fill_method, rolling=rolling)
        final_Par = _resample_fill_roll(final_Par, resample_rule=resample_rule, fill_method=fill_method, rolling=rolling)
        nn_df = _resample_fill_roll(nn_df, resample_rule=resample_rule, fill_method=fill_method, rolling=rolling)

        final_par_by_sc[sc_name] = final_Par
        final_mag_by_sc[sc_name] = final_Mag
        nn_by_sc[sc_name] = nn_df

        fin_by_sc[sc_name] = fin
        fin_path_by_sc[sc_name] = fin_path

    extra_sources_by_sc = {}
    enable_sep_panel = False

    # ------------------------------------------------------------
    # Optional: geocentric distance time series (Earth->SC) in R_E
    # Only enabled when flow separation is requested.
    # ------------------------------------------------------------
    if enable_flow_separation:
        for sc_name in sc_list:
            fin = fin_by_sc.get(sc_name, None)
            par = final_par_by_sc.get(sc_name, None)

            eph = None
            if isinstance(fin, dict):
                eph = fin.get("Ephem", {}).get("GSE", None)

            if isinstance(eph, pd.DataFrame) and isinstance(par, pd.DataFrame):
                eph_al = _time_reindex_interp(eph, par.index)
                eph_al = _ensure_ephem_distance_au(eph_al)
                if "distance_au" in eph_al.columns:
                    out = eph_al[["distance_au"]].copy()
                    out["distance_re"] = out["distance_au"].to_numpy(copy=False) * AU_IN_RE
                    extra_sources_by_sc.setdefault(sc_name, {})["Eph"] = out



    if enable_flow_separation and len(sc_list) >= 2:
        sc1, sc2 = sc_list[0], sc_list[1]

        fin1 = fin_by_sc[sc1]
        fin2 = fin_by_sc[sc2]

        par1 = final_par_by_sc[sc1]
        par2 = final_par_by_sc[sc2]

        eph1 = _time_reindex_interp(fin1["Ephem"]["GSE"], par1.index)
        eph2 = _time_reindex_interp(fin2["Ephem"]["GSE"], par1.index)

        par2_al = _time_reindex_interp(par2, par1.index)

        sep_df = _compute_sep_timeseries_two_sc(
            eph1, eph2, par1, par2_al,
            flow_mode=str(flow_mode).lower(),
            fallback_dir_gse=flow_dir_gse,
            v_smooth_window=flow_v_smooth_window,
            vsw_fallback=vsw_fallback,
        )

        extra_sources_by_sc.setdefault(sc1, {})["Sep"] = sep_df
        enable_sep_panel = True

        print("[flow-sep] Enabled: computing separation in GSE using cached JPL Horizons ephemerides.")
        print(f"[flow-sep] Pair: {sc1} vs {sc2} | flow_mode={str(flow_mode).lower()} | fallback_dir_gse={tuple(flow_dir_gse)}")
        print("[flow-sep] Outputs: dpar_au,dperp_au,dpar_re,dperp_re,tau_h,flow_hat_[xyz],vsw_kms_used.")

        if sep_df[["dpar_re", "dperp_re"]].to_numpy().size > 0:
            if np.all(~np.isfinite(sep_df[["dpar_re", "dperp_re"]].to_numpy(dtype=float))):
                print("[flow-sep:WARN] separation series is all non-finite after alignment. Most likely tz/index mismatch or failed ephemeris fill.")
            else:
                mx = float(np.nanmax(np.abs(sep_df[["dpar_re", "dperp_re"]].to_numpy(dtype=float))))
                if np.isfinite(mx) and mx < 0.5:
                    print(f"[flow-sep:WARN] separation amplitudes are very small (max |d| ~ {mx:.3g} R_E). Check targets, frame, and time-grid.")

    fig, events = interactive_visualize_downloaded_intervals(
        sc=sc_list,
        final_Par=final_par_by_sc if len(sc_list) > 1 else final_par_by_sc[sc_list[0]],
        final_Mag=final_mag_by_sc if len(sc_list) > 1 else final_mag_by_sc[sc_list[0]],
        nn_df=nn_by_sc if len(sc_list) > 1 else nn_by_sc[sc_list[0]],
        my_dir=my_dir,
        save_path=save_path,
        autosave=autosave,
        export_csv=export_csv,
        snap_to_data=snap_to_data,
        enable_comments=enable_comments,
        debug_interaction=debug_interaction,
        panel_config=panel_config,
        panel_edits=panel_edits,
        auto_ylims=auto_ylims,
        warn_missing=warn_missing,
        debug_plot_config=debug_plot_config,
        plot_defaults=plot_defaults,
        span_color=span_color,
        span_alpha=span_alpha,
        enable_multicursor=enable_multicursor,
        snap_index_mode=snap_index_mode_norm,
        normalize_timeseries=normalize_timeseries_mode,
        enforce_sc_linestyle=enforce_sc_linestyle,
        extra_sources_by_sc=extra_sources_by_sc,
        enable_flow_separation_panel=enable_sep_panel,
        alternate_legend_sides=alternate_legend_sides,
        legend_start_side=legend_start_side,
        auto_adjust_margins_for_legends=auto_adjust_margins_for_legends,
    )
    return fig, events