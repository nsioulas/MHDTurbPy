# functions/interactive_figs.py
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import contextlib
import io
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import MultiCursor

try:
    import general_functions as func  # MHDTurbPy/functions/general_functions.py
except Exception:  # pragma: no cover
    func = None


# ============================================================
# GLOBAL DEFAULTS  (UPDATED AS REQUESTED)
# ============================================================
DEFAULT_PLOT_PARAMS: Dict[str, Any] = {
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
def format_timestamp(ts, fmt: str = "%Y_%m_%d") -> str:
    return pd.Timestamp(ts).strftime(fmt)


def inset_axis_params(size: str = "xx-large"):
    minor_tick_params = dict(which="minor", length=3, width=0.8, labelsize=size, direction="in")
    major_tick_params = dict(which="major", length=6, width=1.0, labelsize=size, direction="in")
    return minor_tick_params, major_tick_params


def load_files(load_path: Union[str, Path], pattern: str) -> List[str]:
    load_path = str(load_path)
    hits = glob(os.path.join(load_path, "**", pattern), recursive=True)
    return sorted(hits)


def _as_sc_list(sc: Union[str, List[str], Tuple[str, ...]]) -> List[str]:
    if isinstance(sc, (list, tuple)):
        out = [str(s) for s in sc]
    else:
        out = [str(sc)]
    out = [s for s in out if len(s) > 0]
    if len(out) == 0:
        raise ValueError("at least one spacecraft name must be provided")
    return out


def _normalize_sc_df_input(
    value: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    sc_list: List[str],
    name: str,
) -> Dict[str, pd.DataFrame]:
    if isinstance(value, pd.DataFrame):
        return {sc_name: value for sc_name in sc_list}
    if isinstance(value, dict):
        out: Dict[str, pd.DataFrame] = {}
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


def _sc_tag(sc_list: List[str]) -> str:
    return "-".join(sc_list)


def _normalize_snap_index_mode(snap_index_mode: str) -> str:
    mode = str(snap_index_mode).lower().strip()
    if mode not in VALID_SNAP_INDEX_MODES:
        valid = ", ".join(sorted(VALID_SNAP_INDEX_MODES))
        raise ValueError(f"snap_index_mode must be one of {{{valid}}}, got {snap_index_mode!r}")
    return mode


def _normalize_timeseries_mode(normalize_timeseries: Optional[str]) -> Optional[str]:
    if normalize_timeseries is None:
        return None
    mode = str(normalize_timeseries).lower().strip()
    if mode in ("", "none"):
        return None
    if mode not in VALID_NORMALIZE_METHODS:
        valid = ", ".join(sorted(VALID_NORMALIZE_METHODS))
        raise ValueError(f"normalize_timeseries must be one of {{{valid}}} or None, got {normalize_timeseries!r}")
    return mode


def _series_has_units(label: Any) -> bool:
    if label is None:
        return False
    return re.search(r"\[[^\]]+\]", str(label)) is not None


def _normalize_series_values(y: np.ndarray, mode: Optional[str]) -> np.ndarray:
    if mode is None:
        return y
    out = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out

    vals = out[finite]
    if mode == "zscore":
        mu = float(np.nanmean(vals))
        sigma = float(np.nanstd(vals))
        if sigma == 0 or not np.isfinite(sigma):
            return out
        out[finite] = (vals - mu) / sigma
    elif mode == "minmax":
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        span = vmax - vmin
        if span == 0 or not np.isfinite(span):
            return out
        out[finite] = (vals - vmin) / span
    elif mode == "median":
        med = float(np.nanmedian(vals))
        if med == 0 or not np.isfinite(med):
            return out
        out[finite] = vals / med
    elif mode == "first":
        base = float(vals[0])
        if base == 0 or not np.isfinite(base):
            return out
        out[finite] = vals / base
    return out


def _ensure_dtindex(df: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
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


def _resample_fill_roll(
    df: Union[pd.DataFrame, pd.Series],
    resample_rule: Optional[str],
    fill_method: Optional[str],
    rolling: Optional[str],
) -> Union[pd.DataFrame, pd.Series]:
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


# ============================================================
# Gap handling
# ============================================================
def _prep_gap_df(gaps: pd.DataFrame) -> pd.DataFrame:
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


def merge_gap_intervals(gaps, merge_tol: str = "0s"):
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
    """
    Same behavior as before, but faster:
    - builds an index boolean mask via searchsorted on the DatetimeIndex
    - applies a single vectorized assignment (or column-subset assignment)
    """
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


def build_large_gap_masks(
    mag_gaps: pd.DataFrame,
    qtn_gaps: pd.DataFrame,
    par_gaps: pd.DataFrame,
    sc_pot_gaps: pd.DataFrame,
    gap_thresholds: dict,
    merge_tol: str = "0s",
):
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
def apply_panel_edits(panel_cfg: Dict[str, Any], panel_edits: Optional[Dict[str, Any]]) -> Dict[str, Any]:
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
# Default panel config (baseline)
# ============================================================
def default_panel_config(sc: str, rtn_flag: int) -> Dict[str, Any]:
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
                    "series": [{"kind": "func", "func": "speed", "name": "Vsw", "label": "$V_{sw} ~[km ~s^{-1}$]", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "C0"}}],
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
                "series": [{"kind": "col", "col": "np", "label": "$N_{p}~[(cm^{-3}$]", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "darkred"}}],
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
            {"axes": [
                {
                    "axis_id": "left", "source": "Par", "scale": "linear",
                    "series": [{"kind": "func", "func": "Rsun_from_au", "name": "R", "label": r"$R ~[R_{\odot}]$", "style": {"lw": 0.8, "ls": "-", "ms": 0, "color": "black"}}],
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


# ============================================================
# Plot builder (missing-variable tolerant)
# ============================================================
def _resolve_named_func(name: str):
    if name == "speed":
        def _f(df: pd.DataFrame) -> pd.Series:
            for trio in (("Vr", "Vt", "Vn"), ("Vx", "Vy", "Vz")):
                if all(c in df.columns for c in trio):
                    arr = df[list(trio)].to_numpy(copy=False)
                    v = np.sqrt(np.einsum("ij,ij->i", arr, arr))
                    return pd.Series(v, index=df.index)
            raise KeyError("speed(): missing (Vr,Vt,Vn) and (Vx,Vy,Vz)")
        return _f

    if name == "Rsun_from_au":
        def _f(df: pd.DataFrame) -> pd.Series:
            if "Dist_au" not in df.columns:
                raise KeyError("Rsun_from_au(): missing Dist_au")
            return pd.Series(215.043 * df["Dist_au"].to_numpy(copy=False), index=df.index)
        return _f

    raise KeyError(f"unknown named func '{name}'")


def _merge_style(style: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    if isinstance(style, dict):
        out.update(style)
    return out


def _legend_dict(leg: Any, plot_defaults: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if leg is None:
        return None
    if isinstance(leg, str):
        return dict(plot_defaults["legend_presets"].get(leg, plot_defaults["legend_presets"]["main"]))
    if isinstance(leg, dict):
        return dict(leg)
    return None


def _apply_auto_ylims(ax, y_arrays: List[np.ndarray], scale: str, plot_defaults: Dict[str, Any]) -> None:
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

    # Linear: do NOT scale negatives toward zero (would clip). Expand outward by sign.
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



def _get_or_create_axis(base_ax, axis_id: str, created: Dict[str, Any]) -> Any:
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
    base_axes: List[Any]
    panel_axes: List[List[Any]]
    marker_axes: List[Any]
    span_axes: List[Any]


def _plot_from_panel_config(
    fig,
    axs,
    *,
    panel_config: Dict[str, Any],
    data_sources_by_sc: Dict[str, Dict[str, pd.DataFrame]],
    sc_list: List[str],
    sc_ls_map: Dict[str, str],
    start_lim: pd.Timestamp,
    end_lim: pd.Timestamp,
    auto_ylims: bool,
    warn_missing: bool,
    plot_defaults: Dict[str, Any],
    normalize_timeseries: Optional[str],
    enforce_sc_linestyle: bool,
) -> AxesRegistry:
    panels = panel_config.get("panels", None)
    if not isinstance(panels, list) or len(panels) == 0:
        raise ValueError("panel_config must contain a non-empty list under 'panels'")
    if len(axs) != len(panels):
        raise ValueError(f"axes count ({len(axs)}) != panels count ({len(panels)})")

    def _miss(msg: str):
        if warn_missing:
            print(f"[plot:missing] {msg}")

    per_panel_axes: List[List[Any]] = []

    for i, pan in enumerate(panels):
        base_ax = axs[i]
        created_axes: Dict[str, Any] = {"left": base_ax}
        yvals_by_axis: Dict[str, List[np.ndarray]] = {}

        axes_specs = pan.get("axes", [])
        if not isinstance(axes_specs, list):
            axes_specs = []

        for axspec in axes_specs:
            only_if = axspec.get("only_if", None)

            src = axspec.get("source", None)
            sc_iter = sc_list
            if isinstance(only_if, dict) and only_if.get("sc_equals", None) is not None:
                sc_iter = [sc for sc in sc_list if str(sc) == str(only_if["sc_equals"])]

            sc_data: List[Tuple[str, pd.DataFrame]] = []
            for sc_name in sc_iter:
                if src not in data_sources_by_sc.get(sc_name, {}):
                    raise KeyError(
                        f"unknown source '{src}' for sc='{sc_name}'. "
                        f"Known: {list(data_sources_by_sc.get(sc_name, {}).keys())}"
                    )
                sc_data.append((sc_name, data_sources_by_sc[sc_name][src]))

            axis_id = str(axspec.get("axis_id", "left"))
            scale = str(axspec.get("scale", "linear")).lower()
            leg = _legend_dict(axspec.get("legend", None), plot_defaults)

            ax = None
            labels: List[str] = []
            any_plotted = False
            y_collector: List[np.ndarray] = []

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

    marker_axes: List[Any] = []
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
# Dynamic y-lims on zoom/pan (all panels + twinx)
# ============================================================
@dataclass
class _LineCache:
    x: np.ndarray
    y: np.ndarray
    ymask: Optional[np.ndarray]
    monotonic: bool


def _install_dynamic_ylims_on_xzoom(fig, axes_registry: AxesRegistry, plot_defaults: Dict[str, Any]) -> None:
    """
    When xlim changes (zoom/pan), recompute y-lims for *every* unique axis in the figure
    based on the visible x-window, using the same min/max factor logic as _apply_auto_ylims.

    This includes twinx axes (right axes) since we iterate axes_registry.marker_axes.
    Uses caching + throttling to avoid lag.
    """
    mn_fac = float(plot_defaults["auto_ylims"]["min_factor"])
    mx_fac = float(plot_defaults["auto_ylims"]["max_factor"])

    unique_axes = list(axes_registry.marker_axes)
    if not unique_axes:
        return

    ax_cache: Dict[Any, List[_LineCache]] = {}

    def _coerce_x_to_float(xdata) -> Optional[np.ndarray]:
        if xdata is None:
            return None
        try:
            return np.asarray(xdata, dtype=float)
        except Exception:
            try:
                return np.asarray(mdates.date2num(xdata), dtype=float)
            except Exception:
                return None

    def _coerce_y(ydata) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if ydata is None:
            return None, None
        if np.ma.isMaskedArray(ydata):
            yd = np.asarray(np.ma.getdata(ydata), dtype=float)
            ym = np.asarray(np.ma.getmaskarray(ydata), dtype=bool)
            return yd, ym
        return np.asarray(ydata, dtype=float), None

    # Build cache once (ignore helper artists with <=2 points: vlines/cursor lines/etc)
    for ax in unique_axes:
        lines = getattr(ax, "lines", [])
        if not lines:
            continue
        series: List[_LineCache] = []
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
            y, ymask = _coerce_y(yraw)
            if y is None or y.size != x.size or y.size == 0:
                continue

            # Determine monotonic once (time series should be monotonic)
            dx = np.diff(x)
            mono = bool(np.all(dx >= 0)) if dx.size else True

            series.append(_LineCache(x=x, y=y, ymask=ymask, monotonic=mono))

        if series:
            ax_cache[ax] = series

    if not ax_cache:
        return

    state = {"in_cb": False, "last_wall": -1.0}

    def _visible_minmax_for_axis(ax, x0: float, x1: float) -> Optional[Tuple[float, float]]:
        if x1 < x0:
            x0, x1 = x1, x0

        scale = str(getattr(ax, "get_yscale", lambda: "linear")()).lower()

        have = False
        vmin = np.inf
        vmax = -np.inf

        for lc in ax_cache.get(ax, []):
            x = lc.x
            y = lc.y
            if lc.monotonic:
                i0 = int(np.searchsorted(x, x0, side="left"))
                i1 = int(np.searchsorted(x, x1, side="right"))
                if i1 <= i0:
                    continue
                yy = y[i0:i1]
                mm = lc.ymask[i0:i1] if lc.ymask is not None else None
            else:
                m = (x >= x0) & (x <= x1)
                if not np.any(m):
                    continue
                yy = y[m]
                mm = lc.ymask[m] if lc.ymask is not None else None

            if yy.size == 0:
                continue

            if mm is not None:
                yy = yy[~mm]
                if yy.size == 0:
                    continue

            yy = yy[np.isfinite(yy)]
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

    def _apply_visible_ylims(ax, vmin: float, vmax: float) -> None:
        scale = str(getattr(ax, "get_yscale", lambda: "linear")()).lower()
        if scale == "log":
            lo = mn_fac * vmin
            hi = mx_fac * vmax
            if lo <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or not (lo < hi):
                return
            ax.set_ylim([lo, hi])
            return

        lo = mn_fac * vmin
        hi = mx_fac * vmax
        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            ax.set_ylim([lo, hi])

    def _on_xlim_changed(_ax_trigger):
        if state["in_cb"]:
            return
        now = time.monotonic()
        last = state["last_wall"]
        # throttle: avoid storms of xlim_changed events during interactive drags
        if last >= 0 and (now - last) * 1000.0 < 25.0:
            return
        state["last_wall"] = now

        state["in_cb"] = True
        try:
            # shared-x; get from first base axis
            x0, x1 = axes_registry.base_axes[0].get_xlim()
            x0 = float(x0)
            x1 = float(x1)
            for ax in ax_cache.keys():
                mm = _visible_minmax_for_axis(ax, x0, x1)
                if mm is None:
                    continue
                _apply_visible_ylims(ax, mm[0], mm[1])
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass
        finally:
            state["in_cb"] = False

    # Connect to base axes only (shared x); callback updates *all* axes
    for ax in axes_registry.base_axes:
        try:
            ax.callbacks.connect("xlim_changed", _on_xlim_changed)
        except Exception:
            pass


# ============================================================
# Interval selector (FIXED: 2-click commit + TeX-safe status text)
# ============================================================
@dataclass
class IntervalArtists:
    start_lines: List[Any]
    end_lines: List[Any]
    spans: List[Any]


def _install_interval_selector(
    fig,
    axes_registry: AxesRegistry,
    *,
    events_file: Optional[Path],
    autosave: bool = True,
    export_csv: bool = False,
    span_color: str = "0.85",
    span_alpha: float = 0.35,
    snap_index: Optional[pd.DatetimeIndex] = None,
    snap_to_data: bool = False,
    enable_comments: bool = True,
    debug_interaction: bool = True,
    resume: bool = True,
    dedupe_ms: int = 250,
    dedupe_tol_ns: int = 5_000_000,
    span_zorder: float = 0.8,
    vline_zorder: float = 10.0,
    move_throttle_ms: int = 33,
    use_release_event: bool = False,
) -> pd.DataFrame:
    if enable_comments:
        events = pd.DataFrame(
            {
                "t_start": pd.Series(dtype="datetime64[ns]"),
                "t_end": pd.Series(dtype="datetime64[ns]"),
                "comment": pd.Series(dtype="object"),
            }
        )
    else:
        events = pd.DataFrame(
            {
                "t_start": pd.Series(dtype="datetime64[ns]"),
                "t_end": pd.Series(dtype="datetime64[ns]"),
            }
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

    def _snap(ts: pd.Timestamp) -> pd.Timestamp:
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

    def _sorted(a: pd.Timestamp, b: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return (a, b) if a <= b else (b, a)

    def _atomic_pickle(df: pd.DataFrame, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_pickle(tmp)
        os.replace(tmp, path)

    def _atomic_csv(df: pd.DataFrame, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def _write():
        if events_file is None:
            return
        try:
            _atomic_pickle(events, events_file)
        except Exception as e:
            _p("[picker] save pickle FAILED:", e)
        if export_csv:
            try:
                _atomic_csv(events, events_file.with_suffix(".csv"))
            except Exception as e:
                _p("[picker] save csv FAILED:", e)

    def _x_to_ts(xdata: Any) -> Optional[pd.Timestamp]:
        if xdata is None:
            return None
        dt = mdates.num2date(float(xdata))
        ts = pd.Timestamp(dt)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts

    class State:
        t0: Optional[pd.Timestamp] = None
        snap: bool = bool(snap_to_data)
        hover_xdata: Optional[float] = None
        last_move_wall: float = -1.0

        comment_mode: Optional[str] = None
        comment_buffer: str = ""
        comment_target: Optional[int] = None
        armed_comment: str = ""

        left_select_fallback: bool = (sys.platform.startswith("win") and "tkagg" in matplotlib.get_backend().lower())

    state = State()

    class ClickGate:
        last_wall: float = -1.0
        last_ts_ns: Optional[int] = None
        last_source: Optional[str] = None

        def accept(self, ts: pd.Timestamp, source: str) -> Tuple[bool, str]:
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

    def _set_status(msg: str, redraw: bool = True):
        status.set_text(str(msg))
        if redraw:
            _draw_idle()

    def _set_input(msg: str, redraw: bool = True):
        input_txt.set_text(str(msg))
        if redraw:
            _draw_idle()

    def _toggle_help():
        if help_box.get_text() == "":
            help_box.set_text(_help_text())
        help_box.set_visible(not help_box.get_visible())
        _draw_idle()

    pending_start_lines: Optional[List[Any]] = None
    interval_artists: List[IntervalArtists] = []
    cache_lo_ns: List[int] = []
    cache_hi_ns: List[int] = []

    def _remove_artists(arts: List[Any]):
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

    def _cancel_pending_start(msg: Optional[str] = None):
        nonlocal pending_start_lines
        state.t0 = None
        if pending_start_lines is not None:
            _remove_artists(pending_start_lines)
        pending_start_lines = None
        if msg is not None:
            _set_status(msg, redraw=True)

    def _draw_vline_all(t: pd.Timestamp) -> List[Any]:
        out = []
        for ax in marker_axes:
            out.append(ax.axvline(t, color="0.25", lw=0.9, alpha=0.9, zorder=float(vline_zorder)))
        return out

    def _draw_span_all(a: pd.Timestamp, b: pd.Timestamp) -> List[Any]:
        if b < a:
            a, b = b, a
        out = []
        for ax in span_axes:
            out.append(ax.axvspan(a, b, color=span_color, alpha=span_alpha, lw=0, zorder=float(span_zorder)))
        return out

    def _render_interval(a: pd.Timestamp, b: pd.Timestamp) -> IntervalArtists:
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

    def _interval_under_hover() -> Optional[int]:
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

    def _finish_comment(save: bool):
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

    def _toolbar_mode() -> str:
        tb = getattr(fig.canvas, "toolbar", None)
        return str(getattr(tb, "mode", "") or "")

    def _mpl_is_select_click(ev) -> Tuple[bool, str]:
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

    def _select_at_timestamp(ts: pd.Timestamp, source: str, decoded: str):
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

    tkbinds: List[Tuple[str, Any]] = []
    try:
        w = fig.canvas.get_tk_widget()
        H = fig.canvas.get_width_height()[1]

        def _tk_to_ts(tk_event) -> Optional[pd.Timestamp]:
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

        def _tk_select(event, tag: str):
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



@dataclass
class _LineCache:
    line: Any
    x: np.ndarray
    y: np.ndarray
    bad: np.ndarray  # True where values should be ignored
    monotonic: bool


def _install_dynamic_ylims_on_xzoom(fig, axes_registry: AxesRegistry, plot_defaults: Dict[str, Any]) -> None:
    """
    Robust dynamic y-lims on x-zoom/pan.

    Key fixes vs your current behavior:
      - Idempotent install: disconnects prior hook on same fig (prevents double-callback fights).
      - Debounced updates (timer) to avoid lag during drag.
      - Guaranteed "final" update on mouse button release (toolbar zoom/pan ends here).
      - Uses (0.9, 1.1) factors when x-span is meaningfully smaller than initial full span,
        otherwise uses plot_defaults["auto_ylims"] factors.
      - Linear scaling expands outward correctly for negative minima/maxima (no clipping toward zero).
      - Ignores NaN/inf and huge finite sentinel/fill values.
      - Applies to ALL axes including twinx via axes_registry.marker_axes.
    """
    unique_axes = list(axes_registry.marker_axes or [])
    base_axes = list(axes_registry.base_axes or [])
    if not unique_axes or not base_axes:
        return

    # ---- disconnect any previous installation on this figure
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

    # ---- record "full span" at install time
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

    FILL_ABS_MAX = 1e25  # sentinel/fill guard; keep huge so real data never gets clipped

    def _coerce_x_to_float(xdata) -> Optional[np.ndarray]:
        if xdata is None:
            return None
        try:
            return np.asarray(xdata, dtype=float)
        except Exception:
            try:
                return np.asarray(mdates.date2num(xdata), dtype=float)
            except Exception:
                return None

    def _coerce_y_to_clean_float(ydata) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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

    # ---- cache per axis: only real data lines (ignore helper lines with <=2 points)
    ax_cache: Dict[Any, List[_LineCache]] = {}
    for ax in unique_axes:
        lines = getattr(ax, "lines", [])
        if not lines:
            continue

        series: List[_LineCache] = []
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
            if y is None or y.size != x.size or y.size == 0 or bad is None:
                continue

            dx = np.diff(x)
            mono = bool(np.all(dx >= 0)) if dx.size else True
            series.append(_LineCache(line=ln, x=x, y=y, bad=bad, monotonic=mono))

        if series:
            ax_cache[ax] = series

    if not ax_cache:
        return

    # ---- debounced scheduling
    state = {"in_cb": False, "last_xkey": None}

    timer = None
    try:
        timer = fig.canvas.new_timer(interval=35)
        timer.single_shot = True
    except Exception:
        timer = None

    def _factors_for_xlim(x0: float, x1: float) -> Tuple[float, float]:
        span = abs(float(x1) - float(x0))
        if full_span is None or not np.isfinite(span):
            return 0.9, 1.1
        # treat "near full" as full (prevents annoying factor flip on tiny rounding)
        if span < 0.995 * full_span:
            return 0.9, 1.1
        return base_mn_fac, base_mx_fac

    def _visible_minmax_for_axis(ax, x0: float, x1: float) -> Optional[Tuple[float, float]]:
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

    def _apply_visible_ylims(ax, vmin: float, vmax: float, mn_fac: float, mx_fac: float) -> None:
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

        # expand outward; do NOT shrink negative minima toward zero
        lo = (mx_fac * vmin) if (vmin < 0) else (mn_fac * vmin)
        hi = (mn_fac * vmax) if (vmax < 0) else (mx_fac * vmax)

        if not (np.isfinite(lo) and np.isfinite(hi)):
            return
        if lo > hi:
            lo, hi = hi, lo
        if lo < hi:
            ax.set_ylim([lo, hi])

    def _do_update(force: bool = False):
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

    def _request_update(force: bool = False):
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

    # ---- connect callbacks
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

    # run once at install so the state is consistent immediately
    _do_update(force=True)

    fig._mhdturbpy_dynylims = {"ax_cids": ax_cids, "mpl_cids": mpl_cids, "timer": timer}



# ============================================================
# Main plotting entry
# ============================================================
def interactive_visualize_downloaded_intervals(
    sc: Union[str, List[str], Tuple[str, ...]],
    final_Par: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    final_Mag: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    nn_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    my_dir: Union[str, Path],
    format_2_return: str = "%Y_%m_%d",
    join_path_figs: bool = True,
    save_fig: bool = True,
    fname_tag: str = "",
    save_path: Optional[Union[str, Path]] = None,
    autosave: bool = True,
    export_csv: bool = False,
    snap_to_data: bool = False,
    enable_comments: bool = True,
    debug_interaction: bool = False,
    panel_config: Optional[Dict[str, Any]] = None,
    panel_edits: Optional[Dict[str, Any]] = None,
    auto_ylims: bool = True,
    warn_missing: bool = True,
    debug_plot_config: bool = False,
    plot_defaults: Optional[Dict[str, Any]] = None,
    span_color: str = "0.85",
    span_alpha: float = 0.35,
    enable_multicursor: bool = True,
    snap_index_mode: str = "first_sc",
    normalize_timeseries: Optional[str] = None,
    enforce_sc_linestyle: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
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
    rtn_flag_by_sc: Dict[str, Optional[int]] = {}
    for sc_name in sc_list:
        mag_df = mag_by_sc[sc_name]
        if all(c in mag_df.columns for c in ("Br", "Bt", "Bn")):
            arr = mag_df[["Br", "Bt", "Bn"]].to_numpy(copy=False)
            mag_df["B_RTN"] = np.sqrt(np.einsum("ij,ij->i", arr, arr))
            rtn_flag_by_sc[sc_name] = 1
        elif all(c in mag_df.columns for c in ("Bx", "By", "Bz")):
            arr = mag_df[["Bx", "By", "Bz"]].to_numpy(copy=False)
            mag_df["B_RTN"] = np.sqrt(np.einsum("ij,ij->i", arr, arr))
            rtn_flag_by_sc[sc_name] = 0
        else:
            rtn_flag_by_sc[sc_name] = None
            if warn_missing:
                print(f"[plot:missing] sc={sc_name} cannot compute B_RTN (need Br/Bt/Bn or Bx/By/Bz)")

    rtn_flag = rtn_flag_by_sc.get(first_sc, 0)
    if rtn_flag is None:
        rtn_flag = 0
    if len(sc_list) > 1:
        ref = rtn_flag_by_sc.get(first_sc, None)
        for sc_name in sc_list[1:]:
            if rtn_flag_by_sc.get(sc_name, None) != ref and warn_missing:
                print(f"[plot:missing] sc component frame mismatch: {first_sc}={ref}, {sc_name}={rtn_flag_by_sc.get(sc_name, None)}. Using {first_sc} to build default panel config.")

    if panel_config is None:
        panel_config = default_panel_config(sc=str(first_sc), rtn_flag=rtn_flag)
    panel_config = apply_panel_edits(panel_config, panel_edits)

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

    minor_tick_params, major_tick_params = inset_axis_params(size=str(plot_defaults["ticks"]["size"]))

    ls_cycle = ["-", "--", ":", "-."]
    sc_ls_map = {sc_name: ls_cycle[i % len(ls_cycle)] for i, sc_name in enumerate(sc_list)}
    data_sources_by_sc = {
        sc_name: {"Par": par_by_sc[sc_name], "Mag": mag_by_sc[sc_name], "Sig": sig_by_sc[sc_name]}
        for sc_name in sc_list
    }
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

    # Install dynamic y-lims on zoom/pan across ALL panels + twinx axes
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

    snap_index = (
        par_by_sc[sc_list[0]].index
        if snap_index_mode_norm == "first_sc"
        else pd.DatetimeIndex(
            np.unique(np.concatenate([par_by_sc[sc_name].index.view("int64") for sc_name in sc_list]))
        )
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



    return fig, events


def _extract_interval_bounds_from_final(final_path: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    fin = pd.read_pickle(final_path)
    if not isinstance(fin, dict) or ("Par" not in fin) or ("V_resampled" not in fin["Par"]):
        raise TypeError(f"final.pkl at {final_path} did not match expected Par/V_resampled structure")
    par = _ensure_dtindex(fin["Par"]["V_resampled"])
    if not isinstance(par, pd.DataFrame) or len(par.index) == 0:
        raise ValueError(f"Par/V_resampled in {final_path} is empty or invalid")
    return pd.Timestamp(par.index[0]), pd.Timestamp(par.index[-1])


def _choose_closest_interval_index(final_paths: List[str], target_start: pd.Timestamp, target_end: pd.Timestamp) -> int:
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
    sc: Union[str, List[str], Tuple[str, ...]],
    which_int: int,
    load_path: Union[str, Path, Dict[str, Union[str, Path]]],
    save_path: Union[str, Path],
    my_dir: Union[str, Path],
    rolling: Optional[str] = None,
    resample_rule: Optional[str] = None,
    fill_method: Optional[str] = None,
    gap_thresholds: Optional[Dict[str, str]] = None,
    merge_tol: str = "0s",
    load_files_func=None,
    autosave: bool = True,
    resume: bool = True,
    export_csv: bool = False,
    snap_to_data: bool = False,
    enable_comments: bool = True,
    debug_interaction: bool = False,
    panel_config: Optional[Dict[str, Any]] = None,
    panel_edits: Optional[Dict[str, Any]] = None,
    auto_ylims: bool = True,
    warn_missing: bool = True,
    debug_plot_config: bool = False,
    plot_defaults: Optional[Dict[str, Any]] = None,
    span_color: str = "0.85",
    span_alpha: float = 0.35,
    enable_multicursor: bool = True,
    snap_index_mode: str = "first_sc",
    normalize_timeseries: Optional[str] = None,
    enforce_sc_linestyle: bool = True,
    align_intervals_to_first_sc: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame]:
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

    final_par_by_sc: Dict[str, pd.DataFrame] = {}
    final_mag_by_sc: Dict[str, pd.DataFrame] = {}
    nn_by_sc: Dict[str, pd.DataFrame] = {}

    files_by_sc: Dict[str, Dict[str, List[str]]] = {}
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

    selected_idx_by_sc: Dict[str, int] = {first_sc: int(which_int)}
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

        fin_path = finnames[idx]
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
    )
    return fig, events
