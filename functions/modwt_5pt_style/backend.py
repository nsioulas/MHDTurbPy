"""
Correct, vectorized MODWT/MRA backend for the 5pt-style MHDTurbPy pipeline.

What changed relative to the previous backend
--------------------------------------------
The earlier implementation re-filtered the *original* signal at every level,
which is not the MODWT pyramid algorithm. The correct recursion is

    W_j = H_j V_{j-1},
    V_j = G_j V_{j-1},
    V_0 = X,

with undecimated circular filters at level ``j``. This file implements that
recursion directly while keeping the same circular-convolution convention and
Numba-accelerated kernels.

Notes
-----
* MODWT does not require the input length to be a power of two.
* 1D inputs return arrays with shape ``(J+1, N)``.
* 2D inputs interpreted as ``(N, C)`` return arrays with shape ``(J+1, N, C)``.
* ``modwtmra`` returns the multiresolution reconstruction with the same shape
  convention as ``modwt``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Tuple

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
# Public helpers retained for compatibility
# -----------------------------------------------------------------------------
def upArrow_op(li, j):
    if j == 0:
        return [1]
    li_arr = np.asarray(li, dtype=float)
    n = li_arr.size
    step = 2 ** (j - 1)
    out = np.zeros(step * (n - 1) + 1, dtype=float)
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
# Internal shape helpers
# -----------------------------------------------------------------------------
def _as_2d_signal(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("Signal must be 1D or 2D with shape (N, C).")


def _as_3d_coeffs(w: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(w, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, :, None], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError("Coefficient array must be 2D or 3D with shape (J+1, N[, C]).")


# -----------------------------------------------------------------------------
# Numba kernels (deterministic; no fastmath)
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
# Public convolution primitives
# -----------------------------------------------------------------------------
def circular_convolve_d(h_t, v_j_1, j):
    """
    jth-level undecimated circular decomposition filter.

    For even-length filters this matches the previous backend convention:

        y[i] = sum_t h_t[t] * v[(i - t * 2^(j-1)) mod N].
    """
    step = 2 ** (j - 1)
    h = np.asarray(h_t, dtype=np.float64)
    x = np.asarray(v_j_1, dtype=np.float64)

    if not _HAVE_NUMBA:
        ker = np.zeros(h.size * step, dtype=np.float64)
        ker[::step] = h
        return convolve1d(x, ker, axis=0, mode="wrap", origin=-len(ker) // 2)

    if x.ndim == 1:
        return _cconv_dilated_minus_1d(x, h, step)
    if x.ndim == 2:
        return _cconv_dilated_minus_2d(x, h, step)
    raise ValueError("v_j_1 must be 1D or 2D (N, C)")


def circular_convolve_s(h_t, g_t, w_j, v_j, j):
    """
    Inverse MODWT synthesis step for level ``j``.

    This preserves the same circular synthesis convention as the previous
    backend and is compatible with the recursive forward transform.
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
        out = convolve1d(w, np.flip(h_ker), axis=0, mode="wrap", origin=(len(h_ker) - 1) // 2)
        out += convolve1d(v, np.flip(g_ker), axis=0, mode="wrap", origin=(len(g_ker) - 1) // 2)
        return out

    if w.ndim == 1:
        return _cconv_dilated_plus_1d(w, h, step) + _cconv_dilated_plus_1d(v, g, step)
    if w.ndim == 2:
        return _cconv_dilated_plus_2d(w, h, step) + _cconv_dilated_plus_2d(v, g, step)
    raise ValueError("w_j and v_j must be 1D or 2D (N, C)")


def circular_convolve_mra_sparse(idx, val, w_j):
    """
    Sparse MRA convolution helper.

    Exactly computes

        y[i] = sum_m val[m] * w[(i + idx[m]) mod N]

    for 1D inputs or for each component of a 2D ``(N, C)`` array.
    """
    x = np.asarray(w_j, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    val = np.asarray(val, dtype=np.float64)

    if not _HAVE_NUMBA:
        N = x.shape[0]
        k = np.zeros(N, dtype=np.float64)
        k[idx] = val
        return convolve1d(x, np.flip(k), axis=0, mode="wrap", origin=(len(k) - 1) // 2)

    if x.ndim == 1:
        return _cconv_sparse_plus_1d(x, idx, val)
    if x.ndim == 2:
        return _cconv_sparse_plus_2d(x, idx, val)
    raise ValueError("w_j must be 1D or 2D (N, C)")


def circular_convolve_mra(h_j_o, w_j):
    k = np.asarray(h_j_o, dtype=np.float64)
    idx = np.nonzero(k != 0.0)[0].astype(np.int64)
    val = k[idx].astype(np.float64)

    if (not _HAVE_NUMBA) or (idx.size > 0.6 * k.size):
        return convolve1d(np.asarray(w_j, dtype=np.float64), np.flip(k), axis=0, mode="wrap", origin=(len(k) - 1) // 2)
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
    Build sparse equivalent filters for MODWT MRA detail bands and the final smooth.

    Returned objects represent the exact periodized filters used by ``modwtmra``.
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

    if level <= 0:
        return h_idx_list, h_val_list, np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)

    j = level - 1
    g_j_up = upArrow_op(g, j + 1)
    g_j = np.convolve(g_j_part, g_j_up)
    g_j_t = g_j / (2.0 ** ((j + 1) / 2.0))
    g_j_t_o = period_list(g_j_t, N)

    g_idx = np.nonzero(g_j_t_o != 0.0)[0].astype(np.int64)
    g_val = np.asarray(g_j_t_o[g_idx], dtype=np.float64)
    return h_idx_list, h_val_list, g_idx, g_val


# -----------------------------------------------------------------------------
# Public MODWT / IMODWT / MRA
# -----------------------------------------------------------------------------
def modwt(x, filters, level):
    """
    Correct recursive MODWT pyramid.

    Parameters
    ----------
    x : array_like
        1D signal of shape ``(N,)`` or multicomponent signal of shape ``(N, C)``.
    filters : str
        Wavelet name understood by PyWavelets.
    level : int
        Number of MODWT levels.
    """
    x2, squeeze = _as_2d_signal(x)
    if level < 0:
        raise ValueError("level must be >= 0")
    if level == 0:
        out = x2[None, :, :]
        return out[:, :, 0] if squeeze else out

    h_t, g_t = _dec_filters(str(filters))
    n, c = x2.shape
    coeffs = np.empty((level + 1, n, c), dtype=np.float64)

    v = x2.copy()
    for j in range(1, level + 1):
        w_j = circular_convolve_d(h_t, v, j)
        coeffs[j - 1] = w_j
        v = circular_convolve_d(g_t, v, j)
    coeffs[-1] = v

    return coeffs[:, :, 0] if squeeze else coeffs


def imodwt(w, filters):
    coeffs, squeeze = _as_3d_coeffs(w)
    level = coeffs.shape[0] - 1
    if level <= 0:
        out = coeffs[-1]
        return out[:, 0] if squeeze else out

    h_t, g_t = _dec_filters(str(filters))
    v_j = coeffs[-1]
    for j in range(level, 0, -1):
        v_j = circular_convolve_s(h_t, g_t, coeffs[j - 1], v_j, j)
    return v_j[:, 0] if squeeze else v_j


def modwtmra(w, filters):
    """
    MODWT multiresolution reconstruction.

    Returns the detail reconstructions ``D_1, ..., D_J`` and the final smooth
    reconstruction ``S_J`` stacked along the first axis.
    """
    coeffs, squeeze = _as_3d_coeffs(w)
    level_plus_1, N, C = coeffs.shape
    level = level_plus_1 - 1

    if level <= 0:
        out = coeffs.copy()
        return out[:, :, 0] if squeeze else out

    h_idx_list, h_val_list, g_idx, g_val = _mra_sparse_filters(str(filters), int(level), int(N))

    out = np.empty_like(coeffs)
    for j in range(level):
        out[j] = circular_convolve_mra_sparse(h_idx_list[j], h_val_list[j], coeffs[j])
    out[-1] = circular_convolve_mra_sparse(g_idx, g_val, coeffs[-1])

    return out[:, :, 0] if squeeze else out
