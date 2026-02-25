"""
shared_utils.py

Small, lightweight utilities shared across downloading_helpers modules (PSP/SOLO).
Goal:
- enforce DataFrame hygiene everywhere (DatetimeIndex, tz-naive, monotonic, no duplicates)
- provide robust clipping behavior (legacy helper first, pandas fallback)
- provide standard diagnostics dict defaults
- provide optional wheel-noise removal in a controlled, reusable way
- provide a single resolver for MAG wheel-noise configuration across spacecraft

This file is deliberately dependency-light.
"""

from __future__ import annotations

__TONEFIX_VERSION__ = "2026-02-21-final-resolver"

import logging
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Diagnostics defaults
# ============================================================
def diag_default() -> Dict[str, Any]:
    return {
        "Init_dt": np.nan,
        "resampled_df": None,
        "Frac_miss": 100,
        "Large_gaps": 100,
        "Tot_gaps": 100,
        "resol": 100,
    }


def keep_diag_keys(d: Dict[str, Any], keys: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    if keys is None:
        keys = ("Frac_miss", "Large_gaps", "Tot_gaps", "resol")
    out = {}
    for k in keys:
        out[k] = d.get(k, None)
    return out


# ============================================================
# DataFrame hygiene (MANDATORY everywhere)
# ============================================================
def ensure_datetime_index(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Convert index to tz-naive DatetimeIndex (robust to numeric epoch indices),
    enforce sorted unique monotonic index.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return df

    if len(df) == 0:
        df.index = pd.DatetimeIndex([], name="datetime")
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        out = df
    else:
        idx = pd.Index(df.index)

        # Try numeric coercion first (handles object index with epoch floats)
        vals = pd.to_numeric(idx, errors="coerce")
        finite = vals[np.isfinite(vals)]
        if finite.size > 0 and (finite.size / max(1, len(vals))) > 0.8:
            med = float(np.nanmedian(finite[: min(200, finite.size)]))
            if med > 1e15:
                unit = "ns"
            elif med > 1e11:
                unit = "ms"
            else:
                unit = "s"
            dt = pd.to_datetime(vals, unit=unit, utc=True, errors="coerce")
            out = df.copy()
            out.index = pd.DatetimeIndex(dt)
        else:
            dt = pd.to_datetime(idx, errors="coerce")
            out = df.copy()
            out.index = pd.DatetimeIndex(dt)

    out = out.loc[~out.index.isna()]
    out.index = out.index.tz_localize(None)
    out = out.sort_index()
    out = out.loc[~out.index.duplicated(keep="first")]
    out.index.name = "datetime"
    return out


def sanitize_timeseries_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Stronger hygiene wrapper: ensures a proper DatetimeIndex, drops duplicated timestamps,
    sorts, tz-naive, and returns None if empty.
    """
    df = ensure_datetime_index(df)
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None
    return df


def clip_df_pandas(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.loc[(df.index >= start) & (df.index <= end)]


def clip_to_requested(
    df: Optional[pd.DataFrame],
    ind1,
    ind2,
    req_start: pd.Timestamp,
    req_end: pd.Timestamp,
    func_module=None,
) -> Optional[pd.DataFrame]:
    """
    Legacy-first clipping:
    1) try func.use_dates_return_elements_of_df_inbetween(ind1, ind2, df)
    2) fallback to a pure pandas clip
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None

    df = ensure_datetime_index(df)

    if func_module is not None:
        try:
            out = func_module.use_dates_return_elements_of_df_inbetween(ind1, ind2, df)
            if isinstance(out, pd.DataFrame) and len(out) > 0:
                return ensure_datetime_index(out)
        except Exception:
            pass

    out = clip_df_pandas(df, req_start, req_end)
    if isinstance(out, pd.DataFrame) and len(out) > 0:
        return ensure_datetime_index(out)
    return pd.DataFrame()


# ============================================================
# Wheel-noise removal configuration (single resolver)
# ============================================================
import inspect
import numpy as np
import pandas as pd
import logging
from typing import Optional, Callable, Dict, Any


def default_wheel_noise_cfg():
    return {
        "freq_min": 8.0,
        "stft_nperseg": 2048,
        "stft_overlap": 0.5,

        "detect_mode": "per_time",
        "kernel": 301,
        "thresh_db": 3.0,
        "percentile_q": 99.5,
        "whiten_exp": 0.0,

        "remove_half_width_hz": 1.5,
        "dilate_time_bins": 1,
        "atten_db": 100.0,

        "max_mask_frac": 0.20,
        "auto_raise_thresh": True,
        "raise_step_db": 2.0,
        "raise_max_iter": 4,

        "force_fs": None,
        "skip_if_already_done": True,
    }






def resolve_mag_noise_settings(settings: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Single source of truth for whether wheel-noise / tone removal is enabled and its params.

    Supports BOTH:
      - Preferred unified key:
          settings["Mag_SCM"] = {"noise_flag": bool, "noise_removal": {...}}
      - Legacy keys:
          settings["Mag_SCAM_PSP"]  (PSP SCAM pipeline)
          settings["Mag_SCM_SOLO"]  (SOLO merged SCM pipeline)

    Important behavior:
      - If the unified key exists and is enabled (noise_flag=True), it wins.
      - If the unified key exists but is disabled, we still honor a legacy key
        if it is explicitly enabled. This avoids "silent no-op" when defaults
        define Mag_SCM but the user toggles only the legacy block.

    Returns:
      (noise_flag, noise_cfg)
    """
    if not isinstance(settings, dict):
        return False, default_wheel_noise_cfg()

    unified = settings.get("Mag_SCM", None)
    if isinstance(unified, dict):
        flag_u = bool(unified.get("noise_flag", False))
        cfg_u = unified.get("noise_removal", None)
        if not isinstance(cfg_u, dict):
            cfg_u = default_wheel_noise_cfg()

        if flag_u:
            cfg_u = dict(cfg_u)
            cfg_u["__cfg_source"] = "Mag_SCM"
            return True, cfg_u

        # Unified exists but disabled -> allow explicit legacy enable.
        for legacy_key in ("Mag_SCAM_PSP", "Mag_SCM_SOLO"):
            legacy = settings.get(legacy_key, None)
            if isinstance(legacy, dict) and bool(legacy.get("noise_flag", False)):
                cfg_l = legacy.get("noise_removal", None)
                if not isinstance(cfg_l, dict):
                    cfg_l = default_wheel_noise_cfg()
                cfg_l = dict(cfg_l)
                cfg_l["__cfg_source"] = legacy_key
                return True, cfg_l

        return False, dict(cfg_u)

    # No unified key -> legacy fallbacks.
    for legacy_key in ("Mag_SCAM_PSP", "Mag_SCM_SOLO"):
        legacy = settings.get(legacy_key, None)
        if isinstance(legacy, dict):
            flag = bool(legacy.get("noise_flag", False))
            cfg = legacy.get("noise_removal", None)
            if not isinstance(cfg, dict):
                cfg = default_wheel_noise_cfg()
            cfg = dict(cfg)
            cfg["__cfg_source"] = legacy_key
            return flag, cfg

    return False, default_wheel_noise_cfg()

# ============================================================
# Optional wheel-noise removal (used for SCAM or merged SCM)
# ============================================================

import inspect


def apply_optional_wheel_noise_removal(
    resampled_df: Optional[pd.DataFrame],
    cadence_seconds: Optional[float],
    remove_wheel_noise_func: Callable[..., Any],
    noise_cfg: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> Optional[pd.DataFrame]:
    """
    Apply a wheel-noise / tone-removal function column-by-column on a DataFrame.

    Ground-truth behavior:
      - `cadence_seconds` is dt in seconds (not Hz).
      - If the target cleaner accepts **kwargs, we must NOT drop config keys
        based on signature matching (otherwise all knobs become silent no-ops).
      - We pass `return_debug=True` if the function supports it explicitly
        OR if it accepts **kwargs (in which case it is safe to pass).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if resampled_df is None or not isinstance(resampled_df, pd.DataFrame) or len(resampled_df) == 0:
        return resampled_df

    if cadence_seconds is None or (not np.isfinite(cadence_seconds)) or cadence_seconds <= 0:
        return resampled_df

    fs = 1.0 / float(cadence_seconds)
    if not np.isfinite(fs) or fs <= 0:
        return resampled_df

    cfg = default_wheel_noise_cfg()
    if isinstance(noise_cfg, dict):
        cfg.update(noise_cfg)

    # Do not leak internal/private keys into downstream cleaners
    cfg.pop("__cfg_source", None)
    cfg.pop("cfg_source", None)

    # Inspect once
    supports_return_debug = False
    cfg_pass: Dict[str, Any]

    try:
        sig = inspect.signature(remove_wheel_noise_func)
        params = list(sig.parameters.values())
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

        if has_varkw:
            # Cleaner accepts **kwargs -> pass all config keys verbatim.
            cfg_pass = dict(cfg)
            supports_return_debug = True
        else:
            allowed = set(sig.parameters.keys())
            cfg_pass = {k: v for k, v in cfg.items() if k in allowed}
            supports_return_debug = ("return_debug" in allowed)
    except Exception:
        cfg_pass = dict(cfg)
        supports_return_debug = False

    logger.info(
        "Wheel-noise removal: fs=%g Hz | func=%s | pass_n=%d",
        fs,
        getattr(remove_wheel_noise_func, "__name__", str(remove_wheel_noise_func)),
        len(cfg_pass),
    )

    keys = list(resampled_df.columns)
    if len(keys) == 0:
        return resampled_df

    # Process each column
    for key in keys:
        try:
            x = resampled_df[key].to_numpy(dtype=float, copy=False)

            call_kwargs = dict(cfg_pass)
            if supports_return_debug:
                call_kwargs["return_debug"] = True

            out = remove_wheel_noise_func(x, fs, **call_kwargs)

            dbg = None
            if supports_return_debug and isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], dict):
                y, dbg = out
            else:
                y = out

            y = np.asarray(y, float)
            if y.size != len(resampled_df):
                raise ValueError(f"cleaned length {y.size} != df length {len(resampled_df)}")

            # Quick no-op detector (useful for ground-truth audits)
            finite = np.isfinite(x) & np.isfinite(y)
            if np.any(finite):
                max_abs = float(np.nanmax(np.abs(y[finite] - x[finite])))
            else:
                max_abs = float("nan")

            resampled_df[key] = y

            if dbg is not None:
                logger.info(
                    "WheelNoise[%s]: nCand=%d | mask_frac=%g | max_score_db=%g | max|Δ|=%g",
                    key,
                    dbg.get("nCand", -1),
                    dbg.get("mask_frac", np.nan),
                    dbg.get("max_score_db", np.nan),
                    max_abs,
                )
            else:
                logger.info("WheelNoise[%s]: max|Δ|=%g", key, max_abs)

        except Exception:
            logger.exception("Wheel-noise removal failed for column=%s", key)

    return resampled_df

# ============================================================
# Settings normalization (optional, non-breaking)
# ============================================================
def _deep_get(d: Any, path: Sequence[str]) -> Any:
    obj = d
    for k in path:
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        else:
            return None
    return obj


def normalize_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten grouped settings into the top-level dict without overwriting explicit keys."""
    if settings is None:
        return {}

    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    out = dict(settings)

    group_keys = (
        "Paths",
        "Mission",
        "Intervals",
        "Sampling",
        "QC",
        "Gaps",
        "Output",
        "Diagnostics",
        "Coherence",
        "Derived",
        "MAG",
        "Particles",
        "E_field",
        "sc_pot",
        "SOLO",
        "PSP",
    )

    for gk in group_keys:
        grp = out.get(gk, None)
        if isinstance(grp, dict):
            for k, v in grp.items():
                if k not in out:
                    out[k] = v

    if "Data_path" not in out:
        dp = _deep_get(out, ("Paths", "Data_path"))
        if dp is not None:
            out["Data_path"] = dp

    if "save_destination" not in out:
        sd = _deep_get(out, ("Paths", "save_destination"))
        if sd is not None:
            out["save_destination"] = sd

    if "Big_Gaps" not in out:
        bg = _deep_get(out, ("Gaps", "Big_Gaps"))
        if isinstance(bg, dict):
            out["Big_Gaps"] = bg

    if "Mag_SCM" not in out:
        ms = _deep_get(out, ("MAG", "Mag_SCM"))
        if isinstance(ms, dict):
            out["Mag_SCM"] = ms

    return out
