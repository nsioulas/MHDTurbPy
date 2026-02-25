
"""
modwtpy_fast.py

Drop-in compatible MODWT/MRA backend for MHDTurbPy.

Key speed idea
--------------
Avoid building upsampled kernels full of zeros and avoid convolving with
length-N periodized filters when most entries are exactly zero.

We preserve *bitwise identical* results to the original modwtpy.py by:
- computing the same circular sums in the same tap order as the implicit
  SciPy kernel layout would use;
- skipping taps that are *exactly* zero (skipping 0*x additions is exact).

Controls
--------
- Set MODWTPY_FAST_DISABLE_NUMBA=1 to force SciPy fallbacks.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple, List

import numpy as np
import pywt
from scipy.ndimage import convolve1d

_DISABLE_NUMBA = os.environ.get("MODWTPY_FAST_DISABLE_NUMBA", "").strip() == "1"
try:
    if _DISABLE_NUMBA:
        raise ImportError
    from numba import njit, prange  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def _wrap(fn):  # type: ignore
            return fn
        return _wrap

    def prange(*args, **kwargs):  # type: ignore
        return range(*args)


# -----------------------------------------------------------------------------
# Public helpers (API identical)
# -----------------------------------------------------------------------------
def upArrow_op(li, j):
    if j == 0:
        return [1]
    li_arr = np.asarray(li, dtype=float)
    N = li_arr.size
    step = 2 ** (j - 1)
    out = np.zeros(step * (N - 1) + 1, dtype=float)
    out[::step] = li_arr
    return out


def period_list(li, N):
    n = len(li)
    n_app = N - np.mod(n, N)
    li_list = list(li) + [0] * int(n_app)
    if len(li_list) < 2 * N:
        return np.array(li_list, dtype=float)
    li_np = np.array(li_list, dtype=float)
    li_np = np.reshape(li_np, (-1, N))
    return np.sum(li_np, axis=0)


# -----------------------------------------------------------------------------
# Numba kernels (no fastmath; deterministic per-output sums)
# -----------------------------------------------------------------------------
@njit(cache=True)
def _cconv_dilated_minus_1d(x, f, step):
    N = x.shape[0]
    L = f.shape[0]
    y = np.empty(N, dtype=np.float64)
    for i in range(N):
        acc = 0.0
        for t in range(L):
            acc += f[t] * x[(i - t * step) % N]
        y[i] = acc
    return y


@njit(cache=True)
def _cconv_dilated_plus_1d(x, f, step):
    N = x.shape[0]
    L = f.shape[0]
    y = np.empty(N, dtype=np.float64)
    for i in range(N):
        acc = 0.0
        for t in range(L):
            acc += f[t] * x[(i + t * step) % N]
        y[i] = acc
    return y


@njit(cache=True, parallel=True)
def _cconv_dilated_minus_2d(x, f, step):
    N, C = x.shape
    L = f.shape[0]
    y = np.empty((N, C), dtype=np.float64)
    for c in prange(C):
        for i in range(N):
            acc = 0.0
            for t in range(L):
                acc += f[t] * x[(i - t * step) % N, c]
            y[i, c] = acc
    return y


@njit(cache=True, parallel=True)
def _cconv_dilated_plus_2d(x, f, step):
    N, C = x.shape
    L = f.shape[0]
    y = np.empty((N, C), dtype=np.float64)
    for c in prange(C):
        for i in range(N):
            acc = 0.0
            for t in range(L):
                acc += f[t] * x[(i + t * step) % N, c]
            y[i, c] = acc
    return y


@njit(cache=True)
def _cconv_sparse_plus_1d(x, idx, val):
    N = x.shape[0]
    K = idx.shape[0]
    y = np.empty(N, dtype=np.float64)
    for i in range(N):
        acc = 0.0
        for m in range(K):
            acc += val[m] * x[(i + idx[m]) % N]
        y[i] = acc
    return y


@njit(cache=True, parallel=True)
def _cconv_sparse_plus_2d(x, idx, val):
    N, C = x.shape
    K = idx.shape[0]
    y = np.empty((N, C), dtype=np.float64)
    for c in prange(C):
        for i in range(N):
            acc = 0.0
            for m in range(K):
                acc += val[m] * x[(i + idx[m]) % N, c]
            y[i, c] = acc
    return y


# -----------------------------------------------------------------------------
# Public convolution primitives (API identical + one new helper)
# -----------------------------------------------------------------------------
def circular_convolve_d(h_t, v_j_1, j):
    """
    jth level decomposition (identical to original modwtpy.py):

      convolve1d(v, ker, wrap, origin=-len(ker)//2) with ker[::step]=h_t

    For even-length wavelet filters this equals:
      y[i] = sum_t h_t[t] * v[(i - t*step) mod N]
    """
    step = 2 ** (j - 1)
    h = np.asarray(h_t, dtype=np.float64)
    x = np.asarray(v_j_1, dtype=np.float64)

    if not _HAVE_NUMBA:
        ker = np.zeros(h.size * step, dtype=np.float64)
        ker[::step] = h
        return convolve1d(x, ker, mode="wrap", origin=-len(ker) // 2)

    if x.ndim == 1:
        return _cconv_dilated_minus_1d(x, h, step)
    if x.ndim == 2:
        return _cconv_dilated_minus_2d(x, h, step)
    raise ValueError("v_j_1 must be 1D or 2D (N,C)")


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    """
    (j-1)th level synthesis (identical to original modwtpy.py):

      convolve1d(w, flip(h_ker), wrap, origin=(len(h_ker)-1)//2)
    which equals:
      y[i] = sum_t h_t[t] * w[(i + t*step) mod N]
    """
    step = 2 ** (j - 1)
    h = np.asarray(h_t, dtype=np.float64)
    g = np.asarray(g_t, dtype=np.float64)
    w = np.asarray(w_j, dtype=np.float64)
    v = np.asarray(v_j, dtype=np.float64)

    if not _HAVE_NUMBA:
        h_ker = np.zeros(h.size * step, dtype=np.float64)
        g_ker = np.zeros(g.size * step, dtype=np.float64)
        h_ker[::step] = h
        g_ker[::step] = g
        out = convolve1d(w, np.flip(h_ker), mode="wrap", origin=(len(h_ker) - 1) // 2)
        out += convolve1d(v, np.flip(g_ker), mode="wrap", origin=(len(g_ker) - 1) // 2)
        return out

    if w.ndim == 1:
        return _cconv_dilated_plus_1d(w, h, step) + _cconv_dilated_plus_1d(v, g, step)
    if w.ndim == 2:
        return _cconv_dilated_plus_2d(w, h, step) + _cconv_dilated_plus_2d(v, g, step)
    raise ValueError("w_j and v_j must be 1D or 2D (N,C)")


def circular_convolve_mra_sparse(idx, val, w_j):
    """
    Sparse MRA convolution (helper).

    This is exactly:
      y[i] = sum_m val[m] * w[(i + idx[m]) mod N]
    where (idx,val) represent the non-zero entries of the periodized filter h_j_o
    in ascending index order.
    """
    x = np.asarray(w_j, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    val = np.asarray(val, dtype=np.float64)

    if not _HAVE_NUMBA:
        # Build dense kernel with zeros at unspecified locations (slow fallback).
        N = x.shape[0]
        k = np.zeros(N, dtype=np.float64)
        k[idx] = val
        return convolve1d(x, np.flip(k), mode="wrap", origin=(len(k) - 1) // 2)

    if x.ndim == 1:
        return _cconv_sparse_plus_1d(x, idx, val)
    if x.ndim == 2:
        return _cconv_sparse_plus_2d(x, idx, val)
    raise ValueError("w_j must be 1D or 2D (N,C)")


def circular_convolve_mra(h_j_o, w_j):
    """
    Dense signature kept for compatibility.
    Uses sparse backend by extracting exact non-zeros from h_j_o.
    """
    k = np.asarray(h_j_o, dtype=np.float64)
    idx = np.nonzero(k != 0.0)[0].astype(np.int64)
    val = k[idx].astype(np.float64)

    # If not sparse, SciPy can be faster.
    if (not _HAVE_NUMBA) or (idx.size > 0.6 * k.size):
        return convolve1d(np.asarray(w_j, dtype=np.float64), np.flip(k), mode="wrap", origin=(len(k) - 1) // 2)

    return circular_convolve_mra_sparse(idx, val, w_j)


# -----------------------------------------------------------------------------
# Wavelet filter caches
# -----------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _dec_filters(filters: str) -> Tuple[np.ndarray, np.ndarray]:
    w = pywt.Wavelet(filters)
    h_t = np.asarray(w.dec_hi, dtype=np.float64) / np.sqrt(2.0)
    g_t = np.asarray(w.dec_lo, dtype=np.float64) / np.sqrt(2.0)
    return h_t, g_t


@lru_cache(maxsize=32)
def _mra_sparse_filters(filters: str, level: int, N: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Build sparse representations (idx,val) for each MRA detail filter and the final smooth filter.

    Returned:
      h_idx_list[j], h_val_list[j] for j=0..level-1
      g_idx, g_val for smooth
    """
    wavelet = pywt.Wavelet(filters)
    h = np.asarray(wavelet.dec_hi, dtype=np.float64)
    g = np.asarray(wavelet.dec_lo, dtype=np.float64)

    h_idx_list: List[np.ndarray] = []
    h_val_list: List[np.ndarray] = []

    g_j_part = np.array([1.0], dtype=np.float64)

    for j in range(level):
        g_j_up = upArrow_op(g, j)
        g_j_part = np.convolve(g_j_part, g_j_up)

        h_j_up = upArrow_op(h, j + 1)
        h_j = np.convolve(g_j_part, h_j_up)

        h_j_t = h_j / (2.0 ** ((j + 1) / 2.0))
        if j == 0:
            h_j_t = h / np.sqrt(2.0)

        h_j_t_o = period_list(h_j_t, N)
        idx = np.nonzero(h_j_t_o != 0.0)[0].astype(np.int64)
        val = np.asarray(h_j_t_o[idx], dtype=np.float64)
        h_idx_list.append(idx)
        h_val_list.append(val)

    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2.0 ** ((j + 1) / 2.0))
    g_j_t_o = period_list(g_j_t, N)

    g_idx = np.nonzero(g_j_t_o != 0.0)[0].astype(np.int64)
    g_val = np.asarray(g_j_t_o[g_idx], dtype=np.float64)

    return h_idx_list, h_val_list, g_idx, g_val


# -----------------------------------------------------------------------------
# Public MODWT / IMODWT / MRA (API identical)
# -----------------------------------------------------------------------------
def modwt(x, filters, level):
    """
    Original modwtpy.modwt semantics (kept identical):
    each level is computed directly from x (no cascade).
    """
    x = np.asarray(x, dtype=np.float64)
    if level < 0:
        raise ValueError("level must be >= 0")
    if level == 0:
        return np.vstack([x])

    h_t, g_t = _dec_filters(str(filters))

    coeffs = []
    v_last = None
    for j in range(1, level + 1):
        wj = circular_convolve_d(h_t, x, j)
        vj = circular_convolve_d(g_t, x, j)
        coeffs.append(wj)
        v_last = vj
    if v_last is None:
        v_last = x
    coeffs.append(v_last)
    return np.vstack(coeffs)


def imodwt(w, filters):
    w = np.asarray(w, dtype=np.float64)
    level = len(w) - 1
    if level <= 0:
        return w[-1]

    h_t, g_t = _dec_filters(str(filters))
    v_j = w[-1]
    for jp in range(level):
        j = level - jp
        v_j = circular_convolve_s(h_t, g_t, w[j - 1], v_j, j)
    return v_j


def modwtmra(w, filters):
    """
    Multiresolution analysis based on MODWT (API identical output).
    Uses cached sparse filters + sparse circular sums when beneficial.
    """
    w = np.asarray(w, dtype=np.float64)
    level_plus_1, N = w.shape
    level = level_plus_1 - 1

    h_idx_list, h_val_list, g_idx, g_val = _mra_sparse_filters(str(filters), int(level), int(N))

    D = []
    for j in range(level):
        # identical to circular_convolve_mra(h_j_t_o, w[j]) but sparse
        D.append(circular_convolve_mra_sparse(h_idx_list[j], h_val_list[j], w[j]))

    S = circular_convolve_mra_sparse(g_idx, g_val, w[-1])
    D.append(S)
    return np.vstack(D)
