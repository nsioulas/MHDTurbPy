# -*- coding: utf-8 -*-
"""Multi-spacecraft interval loader (analysis-first, no downloads).

Core behavior:
- Discover per-spacecraft interval pickle files via `load_files_func(load_path, pattern)`
  (prefer `func.load_files` when available).
- Select per-spacecraft interval indices:
    * first spacecraft uses `which_int`
    * other spacecraft optionally aligned to the first by closest (start,end) bounds
      inferred from `final.pkl` -> Par['V_resampled'] index
- Load:
    - final.pkl (required)
    - sig_c_sig_r.pkl (optional; may be sourced from another spacecraft per settings)
    - gap pickles (optional)

Particle/sig provenance:
- Some spacecraft may not have usable particle-derived products (including sig_c_sig_r variables).
- `settings['particles']['sig_provider']` controls which spacecraft provides `Sig` for each target spacecraft:
    * "native" (default): Sig comes from the same spacecraft
    * "<SC_NAME>": use that spacecraft for all Sig (e.g. "WIND")
    * {target_sc: "native" or "<SC_NAME>"}: per-spacecraft control
- The loaded output records Sig provenance explicitly per spacecraft.

Provider Sig time-base alignment (important):
- If `provider_sc != target_sc`, by default the provider's `Sig` is reindexed to the target spacecraft
  `Par` time grid using nearest-neighbor matching with a tolerance.
- Optional systematic time shifts can be applied before reindexing.

Controls (all optional, under `settings['particles']`):
- `sig_reindex_to_target` (bool, default True): reindex provider Sig to the target Par index.
- `sig_max_dt` (str, default "60s"): nearest-neighbor tolerance used during reindexing.
- `sig_time_shift_s` (float or {target_sc: float}, default 0): shift applied to provider Sig timestamps
  (in seconds) *before* reindexing when provider != target.

This module only reads local pickles; it does not download or modify upstream products.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import general_functions as func  # type: ignore
except Exception:
    func = None


def _as_sc_list(sc: Any) -> List[str]:
    if isinstance(sc, (list, tuple)):
        out = [str(s) for s in sc]
    else:
        out = [str(sc)]
    out = [s for s in out if s.strip()]
    if not out:
        raise ValueError("spacecraft list is empty")
    return out


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected DataFrame, got {type(df)}")
    if isinstance(df.index, pd.DatetimeIndex):
        out = df if df.index.is_monotonic_increasing else df.sort_index()
        if out.index.hasnans:
            out = out.loc[~out.index.isna()].copy()
        return out
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    return out


def infer_mag_frame(mag_df: Optional[pd.DataFrame]) -> str:
    if mag_df is None or not isinstance(mag_df, pd.DataFrame):
        return "UNKNOWN"
    if all(c in mag_df.columns for c in ("Br", "Bt", "Bn")):
        return "RTN"
    if all(c in mag_df.columns for c in ("Bx", "By", "Bz")):
        return "GSE"
    return "UNKNOWN"


def infer_v_frame(par_df: Optional[pd.DataFrame]) -> str:
    if par_df is None or not isinstance(par_df, pd.DataFrame):
        return "UNKNOWN"
    if all(c in par_df.columns for c in ("Vr", "Vt", "Vn")):
        return "RTN"
    if all(c in par_df.columns for c in ("Vx", "Vy", "Vz")) or all(c in par_df.columns for c in ("vx", "vy", "vz")):
        return "GSE"
    return "UNKNOWN"


def _default_load_files_func() -> Callable[[Union[str, Path], str], List[str]]:
    if func is not None and hasattr(func, "load_files"):
        return func.load_files  # type: ignore[attr-defined]

    import os
    from glob import glob

    def _glob(load_path: Union[str, Path], pattern: str) -> List[str]:
        hits = glob(os.path.join(str(load_path), "**", pattern), recursive=True)
        return sorted(hits)

    return _glob


def _empty_gap_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["Start", "End"])


def _extract_interval_bounds_from_final(final_path: Union[str, Path]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    fin = pd.read_pickle(str(final_path))
    if not isinstance(fin, dict) or ("Par" not in fin) or (not isinstance(fin["Par"], dict)) or ("V_resampled" not in fin["Par"]):
        raise TypeError(f"final.pkl at {final_path} did not match expected Par/V_resampled structure")
    par = _ensure_dtindex(fin["Par"]["V_resampled"])
    if len(par.index) == 0:
        raise ValueError(f"Par/V_resampled in {final_path} is empty")
    return pd.Timestamp(par.index[0]), pd.Timestamp(par.index[-1])


def _choose_closest_interval_index(final_paths: List[str], target_start: pd.Timestamp, target_end: pd.Timestamp) -> int:
    if not final_paths:
        raise FileNotFoundError("no candidate final.pkl files were provided")
    best_idx = 0
    best_score: Optional[float] = None
    for i, fp in enumerate(final_paths):
        try:
            st, en = _extract_interval_bounds_from_final(fp)
        except Exception:
            continue
        score = abs((st - target_start).total_seconds()) + abs((en - target_end).total_seconds())
        if best_score is None or score < best_score:
            best_score = float(score)
            best_idx = int(i)
    return best_idx


@dataclass(frozen=True)
class IntervalFiles:
    fin: List[str]
    gen: List[str]
    sig: List[str]
    mag_gaps: List[str]
    qtn_gaps: List[str]
    par_gaps: List[str]
    sc_pot_gaps: List[str]


def _discover_interval_files(load_root: Union[str, Path], load_files_func: Callable[[Union[str, Path], str], List[str]]) -> IntervalFiles:
    return IntervalFiles(
        fin=load_files_func(load_root, "final.pkl"),
        gen=load_files_func(load_root, "general.pkl"),
        sig=load_files_func(load_root, "sig_c_sig_r.pkl"),
        mag_gaps=load_files_func(load_root, "mag_gaps.pkl"),
        qtn_gaps=load_files_func(load_root, "qtn_gaps.pkl"),
        par_gaps=load_files_func(load_root, "par_gaps.pkl"),
        sc_pot_gaps=load_files_func(load_root, "sc_pot_gaps.pkl"),
    )


def _normalize_load_path(load_path: Any, sc_list: List[str]) -> Dict[str, str]:
    if isinstance(load_path, dict):
        out = {str(k): str(v) for k, v in load_path.items()}
    else:
        out = {sc: str(load_path) for sc in sc_list}
    missing = [sc for sc in sc_list if sc not in out]
    if missing:
        raise KeyError(f"load_path missing keys: {missing}")
    return out


def _normalize_sig_provider(sig_provider: Any, sc_list: List[str]) -> Dict[str, str]:
    if sig_provider is None:
        return {sc: sc for sc in sc_list}

    if isinstance(sig_provider, str):
        mode = sig_provider.strip()
        if mode.lower() == "native":
            return {sc: sc for sc in sc_list}
        if mode not in sc_list:
            raise KeyError(f"sig_provider={mode!r} must be 'native' or one of sc_list={sc_list}")
        return {sc: mode for sc in sc_list}

    if isinstance(sig_provider, dict):
        out: Dict[str, str] = {}
        missing = [sc for sc in sc_list if sc not in sig_provider]
        if missing:
            raise KeyError(f"sig_provider mapping missing keys: {missing}")
        for sc in sc_list:
            v = str(sig_provider[sc]).strip()
            if v.lower() == "native":
                out[sc] = sc
            else:
                if v not in sc_list:
                    raise KeyError(f"sig_provider[{sc!r}]={v!r} must be 'native' or one of sc_list={sc_list}")
                out[sc] = v
        return out

    raise TypeError("particles.sig_provider must be None, str, or dict")


def _normalize_sig_time_shift(sig_time_shift_s: Any, sc_list: List[str]) -> Dict[str, float]:
    """Return per-target time shift in seconds (applied only when provider != target)."""
    if sig_time_shift_s is None:
        return {sc: 0.0 for sc in sc_list}
    if isinstance(sig_time_shift_s, (int, float)):
        return {sc: float(sig_time_shift_s) for sc in sc_list}
    if isinstance(sig_time_shift_s, dict):
        out: Dict[str, float] = {}
        missing = [sc for sc in sc_list if sc not in sig_time_shift_s]
        if missing:
            raise KeyError(f"sig_time_shift_s mapping missing keys: {missing}")
        for sc in sc_list:
            out[sc] = float(sig_time_shift_s[sc])
        return out
    raise TypeError("particles.sig_time_shift_s must be None, float, or dict")


def load_intervals(settings: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")

    sc_cfg = settings.get("spacecraft", {})
    if not isinstance(sc_cfg, dict):
        raise TypeError("settings['spacecraft'] must be a dict")

    sc_list = _as_sc_list(sc_cfg.get("sc_list", []))
    which_int = int(sc_cfg.get("which_int", 0))
    align = bool(sc_cfg.get("align_intervals_to_first_sc", True))

    load_path = sc_cfg.get("load_path", None)
    if load_path is None:
        raise KeyError("settings['spacecraft']['load_path'] is required")
    load_path_by_sc = _normalize_load_path(load_path, sc_list)

    load_cfg = settings.get("load", {}) or {}
    if not isinstance(load_cfg, dict):
        raise TypeError("settings['load'] must be a dict if provided")

    load_files_func = load_cfg.get("load_files_func", None)
    if load_files_func is None:
        load_files_func = _default_load_files_func()
    if not callable(load_files_func):
        raise TypeError("load_files_func must be callable")

    resample_rule = load_cfg.get("resample_rule", None)
    fill_method = load_cfg.get("fill_method", None)
    rolling = load_cfg.get("rolling", None)

    io_cfg = settings.get("io", {}) or {}
    allow_missing_sig = bool(io_cfg.get("allow_missing_sig", False))

    particles_cfg = settings.get("particles", {}) or {}
    if not isinstance(particles_cfg, dict):
        raise TypeError("settings['particles'] must be a dict if provided")
    sig_provider_map = _normalize_sig_provider(particles_cfg.get("sig_provider", None), sc_list)

    sig_reindex_to_target = bool(particles_cfg.get("sig_reindex_to_target", True))
    sig_max_dt = pd.to_timedelta(str(particles_cfg.get("sig_max_dt", "60s")))
    sig_time_shift_map = _normalize_sig_time_shift(particles_cfg.get("sig_time_shift_s", None), sc_list)

    files_by_sc: Dict[str, IntervalFiles] = {sc: _discover_interval_files(load_path_by_sc[sc], load_files_func) for sc in sc_list}

    first_sc = sc_list[0]
    first_fin = files_by_sc[first_sc].fin
    if len(first_fin) == 0:
        raise FileNotFoundError(f"No final.pkl found under {load_path_by_sc[first_sc]}")
    if not (0 <= which_int < len(first_fin)):
        raise IndexError(f"which_int={which_int} out of range for sc={first_sc} (found {len(first_fin)} intervals)")

    selected_idx_by_sc: Dict[str, int] = {first_sc: int(which_int)}
    target_start, target_end = _extract_interval_bounds_from_final(first_fin[int(which_int)])

    for sc in sc_list[1:]:
        fin_paths = files_by_sc[sc].fin
        if len(fin_paths) == 0:
            raise FileNotFoundError(f"No final.pkl found under {load_path_by_sc[sc]}")
        if align:
            selected_idx_by_sc[sc] = _choose_closest_interval_index(fin_paths, target_start, target_end)
        else:
            if not (0 <= which_int < len(fin_paths)):
                raise IndexError(f"which_int={which_int} out of range for sc={sc} (found {len(fin_paths)} intervals)")
            selected_idx_by_sc[sc] = int(which_int)

    out_by_sc: Dict[str, Dict[str, Any]] = {}
    frames = {"mag_frame_by_sc": {}, "par_v_frame_by_sc": {}}

    for sc in sc_list:
        idx = selected_idx_by_sc[sc]
        f = files_by_sc[sc]

        fin_path = Path(f.fin[idx])
        gen_path = Path(f.gen[idx]) if idx < len(f.gen) else None

        mag_gaps_path = Path(f.mag_gaps[idx]) if idx < len(f.mag_gaps) else None
        qtn_gaps_path = Path(f.qtn_gaps[idx]) if idx < len(f.qtn_gaps) else None
        par_gaps_path = Path(f.par_gaps[idx]) if idx < len(f.par_gaps) else None
        sc_pot_gaps_path = Path(f.sc_pot_gaps[idx]) if idx < len(f.sc_pot_gaps) else None

        fin = pd.read_pickle(str(fin_path))
        gen = pd.read_pickle(str(gen_path)) if gen_path is not None else None

        mag_gaps = pd.read_pickle(str(mag_gaps_path)) if mag_gaps_path is not None else _empty_gap_df()
        qtn_gaps = pd.read_pickle(str(qtn_gaps_path)) if qtn_gaps_path is not None else _empty_gap_df()
        par_gaps = pd.read_pickle(str(par_gaps_path)) if par_gaps_path is not None else _empty_gap_df()
        sc_pot_gaps = pd.read_pickle(str(sc_pot_gaps_path)) if sc_pot_gaps_path is not None else _empty_gap_df()

        if not isinstance(fin, dict) or ("Par" not in fin) or ("Mag" not in fin):
            raise TypeError(
                "final.pkl did not match expected dict structure with keys Par/Mag. "
                f"Got: {list(fin.keys()) if isinstance(fin, dict) else type(fin)}"
            )
        if (not isinstance(fin["Par"], dict)) or ("V_resampled" not in fin["Par"]):
            raise TypeError(f"final.pkl missing Par['V_resampled'] for sc={sc}")
        if (not isinstance(fin["Mag"], dict)) or ("B_resampled" not in fin["Mag"]):
            raise TypeError(f"final.pkl missing Mag['B_resampled'] for sc={sc}")

        par = _ensure_dtindex(fin["Par"]["V_resampled"])
        mag = _ensure_dtindex(fin["Mag"]["B_resampled"])

        if resample_rule is not None:
            par = par.resample(str(resample_rule)).mean()
            mag = mag.resample(str(resample_rule)).mean()
        if fill_method is not None:
            fm = str(fill_method).lower()
            if fm in ("ffill", "pad"):
                par = par.ffill()
                mag = mag.ffill()
            elif fm in ("bfill", "backfill"):
                par = par.bfill()
                mag = mag.bfill()
            else:
                par = par.fillna(method=str(fill_method))
                mag = mag.fillna(method=str(fill_method))
        if rolling is not None:
            par = par.rolling(str(rolling), min_periods=1).mean()
            mag = mag.rolling(str(rolling), min_periods=1).mean()

        frames["mag_frame_by_sc"][sc] = infer_mag_frame(mag)
        frames["par_v_frame_by_sc"][sc] = infer_v_frame(par)

        provider_sc = sig_provider_map[sc]
        provider_idx = selected_idx_by_sc[provider_sc]
        provider_files = files_by_sc[provider_sc]
        sig_path = Path(provider_files.sig[provider_idx]) if provider_idx < len(provider_files.sig) else None

        sig = pd.read_pickle(str(sig_path)) if sig_path is not None else None
        if sig is None:
            if not allow_missing_sig:
                raise FileNotFoundError(
                    f"Missing sig_c_sig_r.pkl for target sc={sc} (provider={provider_sc}, idx={provider_idx}). "
                    "Set io.allow_missing_sig=True to proceed."
                )
        else:
            if not isinstance(sig, pd.DataFrame):
                raise TypeError(f"sig_c_sig_r.pkl expected DataFrame, got {type(sig)}")
            sig = _ensure_dtindex(sig)
            if resample_rule is not None:
                sig = sig.resample(str(resample_rule)).mean()
            if fill_method is not None:
                fm = str(fill_method).lower()
                if fm in ("ffill", "pad"):
                    sig = sig.ffill()
                elif fm in ("bfill", "backfill"):
                    sig = sig.bfill()
                else:
                    sig = sig.fillna(method=str(fill_method))
            if rolling is not None:
                sig = sig.rolling(str(rolling), min_periods=1).mean()

            # If provider != target: align the provider Sig time base to the target Par grid.
            # This is required if Sig-derived quantities (e.g., density) are consumed alongside target B/V.
            if provider_sc != sc and sig_reindex_to_target and (not sig.empty):
                shift_s = float(sig_time_shift_map.get(sc, 0.0))
                if abs(shift_s) > 0:
                    sig = sig.copy()
                    sig.index = sig.index + pd.to_timedelta(shift_s, unit="s")
                # nearest-neighbor match to target Par index, with explicit tolerance
                sig = sig.reindex(par.index, method="nearest", tolerance=sig_max_dt)

        out_by_sc[sc] = {
            "paths": {
                "final": str(fin_path),
                "general": (str(gen_path) if gen_path is not None else None),
                "sig": (str(sig_path) if sig_path is not None else None),
                "mag_gaps": (str(mag_gaps_path) if mag_gaps_path is not None else None),
                "qtn_gaps": (str(qtn_gaps_path) if qtn_gaps_path is not None else None),
                "par_gaps": (str(par_gaps_path) if par_gaps_path is not None else None),
                "sc_pot_gaps": (str(sc_pot_gaps_path) if sc_pot_gaps_path is not None else None),
            },
            "Par": par,
            "Mag": mag,
            "Gen": gen,
            "Sig": sig,
            "Sig_meta": {
                "target_sc": sc,
                "provider_sc": provider_sc,
                "provider_idx": int(provider_idx),
                "is_native": bool(provider_sc == sc),
                "reindexed_to_target": bool(provider_sc != sc and sig_reindex_to_target),
                "sig_max_dt": str(sig_max_dt),
                "sig_time_shift_s": float(sig_time_shift_map.get(sc, 0.0)),
            },
            "Gaps": {"mag": mag_gaps, "qtn": qtn_gaps, "par": par_gaps, "sc_pot": sc_pot_gaps},
        }

    meta = {
        "sc_list": list(sc_list),
        "selected_idx_by_sc": dict(selected_idx_by_sc),
        "align_intervals_to_first_sc": bool(align),
        "frames": frames,
        "which_int_requested": int(which_int),
        "target_bounds_first_sc": {"start": str(target_start), "end": str(target_end)},
        "sig_provider_map": dict(sig_provider_map),
    }
    return {"by_sc": out_by_sc, "meta": meta}
