#!/usr/bin/env python3
"""
imap_mag_loader.py

Single-file helper to fetch IMAP MAG data from either:
  (A) IMAP SDC "science files" (CDFs) via `imap-data-access`
  (B) I-ALiRT near-real-time stream via HTTPS JSON

It returns a pandas.DataFrame with a UTC datetime index.

Usage examples:
  python imap_mag_loader.py --source sdc --start 2026-01-01 --end 2026-01-03 --level l1a --data-dir ./imap_data
  python imap_mag_loader.py --source ialirt --start 2025-11-22T05:30:00Z --end 2025-11-22T08:30:00Z

Notes:
  - For SDC: this script *does not* assume variable names; it inspects the CDF and tries to infer
    (i) time variable and (ii) one 3-component magnetic-field vector variable.
    If inference fails, it raises with a diagnostic listing available variables.
  - For I-ALiRT JSON: payload formats can evolve; the parser here handles several common shapes.
    If it can’t parse the response, it raises with a short preview of the payload keys.

Dependencies:
  pip install pandas numpy requests cdflib imap-data-access
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from cdflib import CDF
from cdflib.epochs import CDFepoch


IALIRT_SPACE_WEATHER_URL = "https://ialirt.imap-mission.com/space-weather"


def _to_utc_timestamp(x: str) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True)
    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) != 1:
            raise ValueError(f"Ambiguous datetime input: {x}")
        return ts[0]
    return ts


def _to_yyyymmdd(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y%m%d")


def _cdf_vars(cdf: CDF) -> List[str]:
    info = cdf.cdf_info()
    rvars = list(info.get("rVariables", []) or [])
    zvars = list(info.get("zVariables", []) or [])
    return rvars + zvars


def _try_time_var(cdf: CDF, varname: str) -> Optional[pd.DatetimeIndex]:
    try:
        raw = cdf.varget(varname)
    except Exception:
        return None

    try:
        dt = CDFepoch.to_datetime(raw)
        idx = pd.to_datetime(dt, utc=True)
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            return idx
        if isinstance(idx, pd.Series) and len(idx) > 0:
            return pd.DatetimeIndex(idx)
    except Exception:
        pass

    try:
        idx = pd.to_datetime(raw, utc=True)
        if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
            return idx
    except Exception:
        return None

    return None


def _infer_time_var(cdf: CDF) -> Tuple[str, pd.DatetimeIndex]:
    vars_ = _cdf_vars(cdf)
    ranked = sorted(
        vars_,
        key=lambda v: (
            0 if "EPOCH" in v.upper() else 1,
            0 if v.upper() in {"TIME", "UTC"} else 1,
            len(v),
        ),
    )
    for v in ranked:
        idx = _try_time_var(cdf, v)
        if idx is not None:
            return v, idx

    raise RuntimeError(
        "Could not infer a time variable from CDF. "
        f"Available variables: {vars_}"
    )


def _looks_like_bvec(name: str) -> bool:
    u = name.upper()
    bad = ["FLAG", "QUALITY", "QF", "STATUS", "STD", "SIGMA", "ERR", "ERROR", "RMS"]
    if any(b in u for b in bad):
        return False
    good = ["B", "MAG", "FIELD"]
    return any(g in u for g in good)


def _infer_bvec_var(cdf: CDF, n_time: int) -> Tuple[str, np.ndarray]:
    vars_ = _cdf_vars(cdf)
    candidates: List[Tuple[int, str, np.ndarray]] = []

    for v in vars_:
        if not _looks_like_bvec(v):
            continue
        try:
            arr = cdf.varget(v)
        except Exception:
            continue

        if not isinstance(arr, np.ndarray):
            continue
        if arr.ndim != 2:
            continue

        a = arr
        if a.shape == (n_time, 3):
            score = 0
        elif a.shape == (3, n_time):
            a = a.T
            score = 1
        else:
            continue

        if a.shape[0] != n_time or a.shape[1] != 3:
            continue

        score += 0 if "B" in v.upper() else 2
        score += 0 if "MAG" in v.upper() else 1
        candidates.append((score, v, a))

    if not candidates:
        raise RuntimeError(
            "Could not infer a 3-component magnetic-field variable from CDF. "
            f"Available variables: {vars_}"
        )

    candidates.sort(key=lambda t: (t[0], len(t[1])))
    _, best_name, best_arr = candidates[0]
    return best_name, best_arr


def _cdf_to_df(cdf_path: str) -> pd.DataFrame:
    cdf = CDF(cdf_path)
    tname, tidx = _infer_time_var(cdf)
    bname, bvec = _infer_bvec_var(cdf, len(tidx))

    cols = {f"{bname}_0": bvec[:, 0], f"{bname}_1": bvec[:, 1], f"{bname}_2": bvec[:, 2]}
    df = pd.DataFrame(cols, index=tidx)
    df.index.name = tname
    return df


def load_imap_mag_sdc(
    start: str,
    end: str,
    *,
    level: str = "l1a",
    descriptor: Optional[str] = None,
    version: str = "latest",
    data_dir: str = "./imap_data",
    keep_files: bool = True,
) -> pd.DataFrame:
    start_ts = _to_utc_timestamp(start)
    end_ts = _to_utc_timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be strictly after start")

    try:
        import imap_data_access
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `imap-data-access`. Install with: pip install imap-data-access"
        ) from e

    os.environ["IMAP_DATA_DIR"] = str(Path(data_dir).resolve())

    query_kwargs: Dict[str, Any] = dict(
        instrument="mag",
        data_level=level,
        start_date=_to_yyyymmdd(start_ts),
        end_date=_to_yyyymmdd(end_ts),
        extension="cdf",
        version=version,
    )
    if descriptor:
        query_kwargs["descriptor"] = descriptor

    results = imap_data_access.query(**query_kwargs)
    if not results:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    dfs: List[pd.DataFrame] = []
    downloaded_paths: List[str] = []

    for r in results:
        file_path = r.get("file_path")
        if not file_path:
            continue

        local_path = os.path.join(os.environ["IMAP_DATA_DIR"], file_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(local_path):
            imap_data_access.download(file_path)

        if os.path.exists(local_path):
            downloaded_paths.append(local_path)
            try:
                df = _cdf_to_df(local_path)
                dfs.append(df)
            except Exception as e:
                raise RuntimeError(f"Failed parsing CDF: {local_path}\n{e}") from e

    if not dfs:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    out = pd.concat(dfs, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]

    if not keep_files:
        for p in downloaded_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    return out


def _parse_ialirt_payload(payload: Any) -> pd.DataFrame:
    if isinstance(payload, dict):
        for key in ["data", "results", "timeseries", "records"]:
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break

    if not isinstance(payload, list) or len(payload) == 0:
        raise RuntimeError(f"Unexpected I-ALiRT payload shape: {type(payload)}")

    first = payload[0]
    if not isinstance(first, dict):
        raise RuntimeError("I-ALiRT payload list does not contain dict records")

    time_keys = ["time_utc", "time", "timestamp", "datetime", "epoch"]
    tkey = next((k for k in time_keys if k in first), None)
    if tkey is None:
        raise RuntimeError(f"Could not find a time key in record. Keys: {list(first.keys())}")

    idx = pd.to_datetime([rec.get(tkey) for rec in payload], utc=True)

    b_keys = [
        ("bx", "by", "bz"),
        ("Bx", "By", "Bz"),
        ("b_x", "b_y", "b_z"),
        ("B_x", "B_y", "B_z"),
    ]
    cols: Dict[str, List[float]] = {}

    triple = None
    for trio in b_keys:
        if all(k in first for k in trio):
            triple = trio
            break

    if triple is not None:
        for k in triple:
            cols[k] = [rec.get(k) for rec in payload]
        df = pd.DataFrame(cols, index=idx)
        df.index.name = tkey
        return df

    vec_key = None
    for k in ["b", "B", "mag", "magnetic_field", "B_vec", "b_vec"]:
        if k in first:
            vec_key = k
            break

    if vec_key is not None:
        arr = np.array([rec.get(vec_key) for rec in payload], dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3:
            df = pd.DataFrame(
                {f"{vec_key}_0": arr[:, 0], f"{vec_key}_1": arr[:, 1], f"{vec_key}_2": arr[:, 2]},
                index=idx,
            )
            df.index.name = tkey
            return df

    numeric_keys = [k for k, v in first.items() if isinstance(v, (int, float)) and k != tkey]
    if numeric_keys:
        df = pd.DataFrame({k: [rec.get(k) for rec in payload] for k in numeric_keys}, index=idx)
        df.index.name = tkey
        return df

    raise RuntimeError(f"Could not infer MAG columns from record keys: {list(first.keys())}")


def load_imap_mag_ialirt(
    start: str,
    end: str,
    *,
    url: str = IALIRT_SPACE_WEATHER_URL,
    timeout: int = 60,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    start_ts = pd.to_datetime(start, utc=True)
    end_ts = pd.to_datetime(end, utc=True)
    if end_ts <= start_ts:
        raise ValueError("end must be strictly after start")

    t0 = start_ts.strftime("%Y-%m-%dT%H:%M:%S")
    t1 = end_ts.strftime("%Y-%m-%dT%H:%M:%S")

    headers = {"Accept": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    def _try(query_url: str) -> requests.Response:
        r = requests.get(query_url, headers=headers, timeout=timeout)
        if r.status_code == 400:
            raise RuntimeError(f"400 Bad Request\nURL: {query_url}\nBody:\n{r.text[:500]}")
        r.raise_for_status()
        return r

    url1 = f"{url}?instrument=mag&time_utc_start={t0}&time_utc_end={t1}"
    try:
        r = _try(url1)
    except RuntimeError:
        url2 = f"{url}?instrument=mag&met_in_utc_start={t0}&met_in_utc_end={t1}"
        r = _try(url2)

    payload = r.json()
    df = _parse_ialirt_payload(payload)

    df = df.sort_index()
    df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
    return df



def get_imap_mag(
    start: str,
    end: str,
    *,
    source: str = "sdc",
    level: str = "l1a",
    descriptor: Optional[str] = None,
    version: str = "latest",
    data_dir: str = "./imap_data",
    ialirt_api_key: Optional[str] = None,
) -> pd.DataFrame:
    s = source.strip().lower()
    if s == "sdc":
        return load_imap_mag_sdc(
            start,
            end,
            level=level,
            descriptor=descriptor,
            version=version,
            data_dir=data_dir,
        )
    if s in {"ialirt", "i-alirt"}:
        return load_imap_mag_ialirt(start, end, api_key=ialirt_api_key)
    raise ValueError("source must be either 'sdc' or 'ialirt'")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch IMAP MAG data and return a DataFrame with datetime index.")
    p.add_argument("--source", choices=["sdc", "ialirt"], default="sdc")
    p.add_argument("--start", required=True, help="Start datetime (e.g. 2026-01-01 or 2025-11-22T05:30:00Z)")
    p.add_argument("--end", required=True, help="End datetime (exclusive-ish; will be clipped inclusive in output)")
    p.add_argument("--level", default="l1a", help="SDC data level (SDC only)")
    p.add_argument("--descriptor", default=None, help="SDC descriptor if required (SDC only)")
    p.add_argument("--version", default="latest", help="SDC version (SDC only)")
    p.add_argument("--data-dir", default="./imap_data", help="Local data directory (SDC only)")
    p.add_argument("--ialirt-api-key", default=None, help="API key if required (I-ALiRT only)")
    p.add_argument("--out", default=None, help="Optional output path (csv or parquet). If omitted, prints head/info.")
    args = p.parse_args()

    df = get_imap_mag(
        args.start,
        args.end,
        source=args.source,
        level=args.level,
        descriptor=args.descriptor,
        version=args.version,
        data_dir=args.data_dir,
        ialirt_api_key=args.ialirt_api_key,
    )

    if args.out:
        out = args.out.lower()
        if out.endswith(".csv"):
            df.to_csv(args.out)
        elif out.endswith(".parquet"):
            df.to_parquet(args.out)
        else:
            raise ValueError("Output extension must be .csv or .parquet")
    else:
        print(df.head())
        print()
        print(df.info())


if __name__ == "__main__":
    main()
