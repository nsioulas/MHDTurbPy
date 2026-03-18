from __future__ import annotations

import glob
import pickle
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from three_D_funcs import coefficient_store_to_dataframe

DEFAULT_BUCKET_SCALE = {
    'ell_all': 'l_mag',
    'ell_perp': 'l_lambda',
    'Ell_perp': 'l_xi',
    'ell_par': 'l_ell',
    'ell_par_rest': 'l_ell',
}


def _expand_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    items = [paths] if isinstance(paths, str) else list(paths)
    out: List[str] = []
    for item in items:
        matches = sorted(glob.glob(str(item)))
        out.extend(matches if matches else [str(item)])
    return out


def _load_pickle(path: str):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def load_and_merge_coefficient_stores(paths: Union[str, Sequence[str]]) -> pd.DataFrame:
    """Load one or more saved interval pickles into one long-form dataframe.

    The rev12 coefficient store is written once on ``ell_all``. Directional buckets are
    reconstructed later from the saved local angles.
    """
    frames: List[pd.DataFrame] = []
    for path in _expand_paths(paths):
        payload = _load_pickle(path)
        store = payload.get('CoefficientStore', payload.get('CompactCoefficients'))
        if store is None:
            raise KeyError(f'{path} does not contain CoefficientStore/CompactCoefficients.')
        df = coefficient_store_to_dataframe(store, bucket='ell_all')
        df['source_file'] = path
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    if frames:
        payload0 = _load_pickle(_expand_paths(paths)[0])
        store0 = payload0.get('CoefficientStore', payload0.get('CompactCoefficients'))
        if store0 is not None:
            merged.attrs['bucket_conditions'] = dict(store0.get('bucket_conditions', {}))
    return merged


def _normalize_values(values: np.ndarray, tau: np.ndarray, response_energy: np.ndarray, normalization: str, q: float, absolute_value: bool) -> np.ndarray:
    base = np.abs(values) if absolute_value else values
    if normalization == 'scale_normalized':
        return np.power(base, q) / np.power(tau, 0.5 * q)
    if normalization == 'psd':
        if q != 2.0:
            raise ValueError('PSD normalization is defined only for q=2.')
        return np.square(base) / response_energy
    raise ValueError(f'Unknown normalization {normalization!r}.')


def _bucket_mask_from_angles(theta: np.ndarray, phi: np.ndarray, bucket: str, bucket_conditions: Optional[Mapping[str, Mapping[str, float]]] = None) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    finite_theta = np.isfinite(theta)
    finite_phi = finite_theta & np.isfinite(phi)
    conds = {} if bucket_conditions is None else bucket_conditions
    if bucket in ('ell_all', 'ell_overall'):
        return finite_theta
    cond = conds.get(bucket, {})
    if bucket == 'ell_perp':
        return finite_phi & (theta > float(cond.get('theta', np.nan))) & (phi > float(cond.get('phi', np.nan)))
    if bucket == 'Ell_perp':
        return finite_phi & (theta > float(cond.get('theta', np.nan))) & (phi < float(cond.get('phi', np.nan)))
    if bucket in ('ell_par', 'ell_par_rest'):
        return finite_theta & (theta < float(cond.get('theta', np.nan)))
    raise KeyError(f'Unknown bucket {bucket!r}.')


def reduce_conditional_rows(
    df: pd.DataFrame,
    bucket: str,
    value_keys: Sequence[str],
    cond_var: str,
    qorder: Sequence[float],
    normalization: str,
    scale_bin_edges_di: Sequence[float],
    constraints: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
    min_count: int = 25,
    nquant: int = 10,
    absolute_value: bool = True,
    scale_key: Optional[str] = None,
) -> pd.DataFrame:
    """Pool row-level coefficients across intervals and reduce them after conditioning.

    Parameters
    ----------
    df:
        Long-form coefficient dataframe returned by ``load_and_merge_coefficient_stores``.
    bucket:
        ``ell_all``, ``ell_perp``, ``Ell_perp``, ``ell_par``, or ``ell_par_rest``.
        Directional buckets are reconstructed from saved angles.
    value_keys:
        Columns to reduce, for example ``['W_Zp_mag', 'W_Zm_mag']`` for trace spectra
        or ``['V_par', 'V_perp']`` for projected moments.
    cond_var:
        Row-level conditioning variable such as ``sig_c_ts`` or ``compress_simple``.
    normalization:
        ``'psd'`` for ``|a|^2 / response_energy_integral`` or
        ``'scale_normalized'`` for ``|a|^q / tau_equiv_seconds^(q/2)``.
    """
    work = df.copy()
    if constraints:
        mask = np.ones(len(work), dtype=bool)
        for key, (lo, hi) in constraints.items():
            vals = pd.to_numeric(work[key], errors='coerce').to_numpy(dtype=float)
            local = np.isfinite(vals)
            if lo is not None:
                local &= vals >= float(lo)
            if hi is not None:
                local &= vals <= float(hi)
            mask &= local
        work = work.loc[mask].copy()

    bucket_conditions = None
    if 'bucket_conditions' in work.attrs:
        bucket_conditions = work.attrs['bucket_conditions']
    if bucket not in ('ell_all', 'ell_overall'):
        mask_bucket = _bucket_mask_from_angles(
            pd.to_numeric(work['thetas'], errors='coerce').to_numpy(dtype=float),
            pd.to_numeric(work['phis'], errors='coerce').to_numpy(dtype=float),
            bucket=bucket,
            bucket_conditions=bucket_conditions,
        )
        work = work.loc[mask_bucket].copy()

    scale_key_eff = DEFAULT_BUCKET_SCALE[bucket] if scale_key is None else scale_key
    needed = [scale_key_eff, 'tau_equiv_seconds', 'response_energy_integral', cond_var] + list(value_keys)
    work = work[needed].apply(pd.to_numeric, errors='coerce')
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[scale_key_eff, 'tau_equiv_seconds', 'response_energy_integral', cond_var])

    if work.empty:
        return pd.DataFrame()

    scale_vals = work[scale_key_eff].to_numpy(dtype=float)
    bins = np.digitize(scale_vals, np.asarray(scale_bin_edges_di, dtype=float)) - 1
    n_bins = len(scale_bin_edges_di) - 1
    valid = (bins >= 0) & (bins < n_bins)
    work = work.loc[valid].copy()
    work['scale_bin'] = bins[valid]

    out: List[Dict[str, float]] = []
    for b in range(n_bins):
        sub = work.loc[work['scale_bin'] == b].copy()
        if len(sub) < max(min_count, nquant):
            continue
        sub = sub.sort_values(cond_var, kind='mergesort')
        groups = np.array_split(np.arange(len(sub)), nquant)
        for iq, idx in enumerate(groups):
            if idx.size < min_count:
                continue
            block = sub.iloc[idx]
            tau = block['tau_equiv_seconds'].to_numpy(dtype=float)
            response_energy = block['response_energy_integral'].to_numpy(dtype=float)
            scale_block = block[scale_key_eff].to_numpy(dtype=float)
            cond_block = block[cond_var].to_numpy(dtype=float)
            meta = {
                'bucket': bucket,
                'scale_key': scale_key_eff,
                'scale_bin': int(b),
                'scale_left_di': float(scale_bin_edges_di[b]),
                'scale_right_di': float(scale_bin_edges_di[b + 1]),
                'scale_mean_di': float(np.nanmean(scale_block)),
                'scale_median_di': float(np.nanmedian(scale_block)),
                'cond_var': cond_var,
                'quantile_index': int(iq),
                'cond_mean': float(np.nanmean(cond_block)),
                'cond_median': float(np.nanmedian(cond_block)),
                'count': int(idx.size),
                'normalization': normalization,
            }
            for key in value_keys:
                vals = block[key].to_numpy(dtype=float)
                for q in qorder:
                    obs = _normalize_values(vals, tau, response_energy, normalization, float(q), absolute_value)
                    finite = np.isfinite(obs)
                    if int(finite.sum()) < min_count:
                        continue
                    row = dict(meta)
                    row['value_key'] = key
                    row['q'] = float(q)
                    row['estimate'] = float(np.nanmean(obs[finite]))
                    out.append(row)
    return pd.DataFrame.from_records(out)
