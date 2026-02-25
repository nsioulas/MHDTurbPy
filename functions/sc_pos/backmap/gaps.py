from __future__ import annotations

"""sc_pos.backmap.gaps

Gap-table utilities for MHDTurbPy interval pickles.

This module provides:
- `load_fin_and_padded_gaps`: user-facing convenience function (exact signature requested).
- `build_padded_gaps_from_dfs`: internal helper that avoids re-reading final.pkl.
- `keep_mask_from_padded_gaps`: convert padded gaps -> boolean keep mask for an index.

Design contract
---------------
- Gap tables are interpreted as *invalid-data spans* on the time axis.
- A per-gap symmetric padding is applied: pad = gap_pad_frac * (End-Start), on both sides.
- All timestamps are compared in tz-naive form to avoid pandas tz-mismatch traps.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_fin_and_padded_gaps(
    fin_path,
    *,
    mag_gaps_path=None,
    par_gaps_path=None,
    gap_pad_frac=0.5,
    index=None,
):
    """
    Load `final.pkl` plus optional `mag_gaps.pkl` / `par_gaps.pkl`, standardize them,
    and build the padded (Start, End) union gaps table.

    Parameters
    ----------
    fin_path : str or pathlib.Path
        Path to `final.pkl`.
    mag_gaps_path, par_gaps_path : str or pathlib.Path or None
        Optional paths to gap pickles. Missing/unreadable -> ignored.
    gap_pad_frac : float
        Per-gap padding fraction: pad = gap_pad_frac * (End-Start) applied to BOTH sides.
    index : pandas.DatetimeIndex or array-like of datetimes or None
        If provided, also returns a boolean keep-mask (True = keep) after removing padded gaps.

    Returns
    -------
    fin : object
        Unpickled `final.pkl` object.
    gaps_padded : pandas.DataFrame or None
        Columns: Start, End (tz-naive), already padded, filtered to End > Start.
        None if no usable gaps.
    keep : numpy.ndarray (bool), optional
        Returned only if `index` is not None. True = keep.
    """
    fin_path = Path(fin_path)
    fin = pd.read_pickle(fin_path)

    pad_frac = float(gap_pad_frac)
    if pad_frac < 0.0:
        raise ValueError("gap_pad_frac must be >= 0.")

    def _read_df(p):
        if p is None:
            return None
        try:
            return pd.read_pickle(Path(p))
        except Exception:
            return None

    def _standardize(gaps):
        if gaps is None or (not isinstance(gaps, pd.DataFrame)) or gaps.empty:
            return None

        cols = {c.lower(): c for c in gaps.columns}
        if "start" not in cols or "end" not in cols:
            raise KeyError(f"Gap dataframe must contain Start/End columns. Got: {list(gaps.columns)}")

        g = gaps[[cols["start"], cols["end"]]].copy()
        g.columns = ["Start", "End"]

        g["Start"] = pd.to_datetime(g["Start"], errors="coerce")
        g["End"] = pd.to_datetime(g["End"], errors="coerce")

        # force tz-naive for consistent comparisons
        if getattr(g["Start"].dt, "tz", None) is not None:
            g["Start"] = g["Start"].dt.tz_convert(None)
        if getattr(g["End"].dt, "tz", None) is not None:
            g["End"] = g["End"].dt.tz_convert(None)

        g = g.dropna()
        g = g[g["End"] > g["Start"]]
        if g.empty:
            return None
        return g.reset_index(drop=True)

    def _pad(g):
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

    gaps_padded = None
    if g_mag is not None and g_par is not None:
        gaps_padded = pd.concat([g_mag, g_par], ignore_index=True)
    elif g_mag is not None:
        gaps_padded = g_mag
    elif g_par is not None:
        gaps_padded = g_par

    if gaps_padded is None or gaps_padded.empty:
        if index is None:
            return fin, None
        idx = pd.DatetimeIndex(index)
        return fin, None, np.ones(len(idx), dtype=bool)

    gaps_padded = gaps_padded.sort_values("Start").reset_index(drop=True)

    if index is None:
        return fin, gaps_padded

    idx = pd.DatetimeIndex(index)
    in_gap = np.zeros(len(idx), dtype=bool)
    for row in gaps_padded.itertuples(index=False):
        in_gap |= (idx >= row.Start) & (idx <= row.End)

    keep = ~in_gap
    return fin, gaps_padded, keep


def build_padded_gaps_from_dfs(
    *,
    mag_gaps: Optional[pd.DataFrame],
    par_gaps: Optional[pd.DataFrame],
    gap_pad_frac: float = 0.5,
) -> Optional[pd.DataFrame]:
    """Build the padded union gaps table from already-loaded DataFrames."""
    pad_frac = float(gap_pad_frac)
    if pad_frac < 0.0:
        raise ValueError("gap_pad_frac must be >= 0.")

    def _standardize(gaps: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if gaps is None or (not isinstance(gaps, pd.DataFrame)) or gaps.empty:
            return None
        cols = {c.lower(): c for c in gaps.columns}
        if "start" not in cols or "end" not in cols:
            return None
        g = gaps[[cols["start"], cols["end"]]].copy()
        g.columns = ["Start", "End"]
        g["Start"] = pd.to_datetime(g["Start"], errors="coerce")
        g["End"] = pd.to_datetime(g["End"], errors="coerce")
        if getattr(g["Start"].dt, "tz", None) is not None:
            g["Start"] = g["Start"].dt.tz_convert(None)
        if getattr(g["End"].dt, "tz", None) is not None:
            g["End"] = g["End"].dt.tz_convert(None)
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

    g_mag = _pad(_standardize(mag_gaps))
    g_par = _pad(_standardize(par_gaps))

    gaps_padded = None
    if g_mag is not None and g_par is not None:
        gaps_padded = pd.concat([g_mag, g_par], ignore_index=True)
    elif g_mag is not None:
        gaps_padded = g_mag
    elif g_par is not None:
        gaps_padded = g_par

    if gaps_padded is None or gaps_padded.empty:
        return None
    return gaps_padded.sort_values("Start").reset_index(drop=True)


def keep_mask_from_padded_gaps(
    *,
    index: Union[pd.DatetimeIndex, np.ndarray, list],
    gaps_padded: Optional[pd.DataFrame],
) -> np.ndarray:
    """Return keep mask (True=keep) by removing padded gap spans from `index`."""
    idx = pd.DatetimeIndex(index)

    # compare in tz-naive space to avoid tz-aware vs tz-naive mismatch
    if getattr(idx, "tz", None) is not None:
        idx_cmp = idx.tz_convert(None)
    else:
        idx_cmp = idx

    if gaps_padded is None or (not isinstance(gaps_padded, pd.DataFrame)) or gaps_padded.empty:
        return np.ones(len(idx), dtype=bool)

    g = gaps_padded.copy()
    if "Start" not in g.columns or "End" not in g.columns:
        return np.ones(len(idx), dtype=bool)

    g["Start"] = pd.to_datetime(g["Start"], errors="coerce")
    g["End"] = pd.to_datetime(g["End"], errors="coerce")
    if getattr(g["Start"].dt, "tz", None) is not None:
        g["Start"] = g["Start"].dt.tz_convert(None)
    if getattr(g["End"].dt, "tz", None) is not None:
        g["End"] = g["End"].dt.tz_convert(None)

    g = g.dropna()
    g = g[g["End"] > g["Start"]]
    if g.empty:
        return np.ones(len(idx), dtype=bool)

    in_gap = np.zeros(len(idx_cmp), dtype=bool)
    for row in g.itertuples(index=False):
        in_gap |= (idx_cmp >= row.Start) & (idx_cmp <= row.End)

    return ~in_gap
