from __future__ import annotations

import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

os.environ.setdefault("MODWTPY_FAST_DISABLE_NUMBA", "1")


def _install_pywt_stub_if_needed() -> None:
    try:
        import pywt  # noqa: F401
        return
    except Exception:
        pass

    pywt = types.ModuleType("pywt")

    class Wavelet:
        def __init__(self, name: str):
            n = str(name).lower()
            if n != "haar":
                raise ValueError(f"PyWavelets is unavailable; stub only supports haar, got {name!r}")
            s = 1.0 / math.sqrt(2.0)
            self.dec_lo = np.array([s, s], dtype=float)
            self.dec_hi = np.array([-s, s], dtype=float)
            self.dec_len = 2

    def dwt_max_level(data_len: int, filter_len: int) -> int:
        if filter_len <= 1:
            return int(max(0, math.floor(math.log2(max(1, data_len)))))
        return int(max(0, math.floor(math.log(max(1, data_len) / float(filter_len - 1), 2.0))))

    pywt.Wavelet = Wavelet
    pywt.dwt_max_level = dwt_max_level
    sys.modules["pywt"] = pywt


_install_pywt_stub_if_needed()

from backend import circular_convolve_d, imodwt, modwt  # noqa: E402
from backend import _dec_filters  # noqa: E402


def modwt_original_shortcut(x: np.ndarray, filters: str, level: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Original shortcut reference implemented here for 1D inputs only.")
    h_t, g_t = _dec_filters(str(filters))
    coeffs = []
    v_last = None
    for j in range(1, level + 1):
        w_j = circular_convolve_d(h_t, arr, j)
        v_j = circular_convolve_d(g_t, arr, j)
        coeffs.append(w_j)
        v_last = v_j
    coeffs.append(arr.copy() if v_last is None else v_last)
    return np.vstack(coeffs)


def reconstruction_error(x: np.ndarray, coeffs: np.ndarray, filters: str) -> float:
    rec = imodwt(coeffs, filters)
    return float(np.max(np.abs(np.asarray(rec) - np.asarray(x))))


def benchmark(fn, *args, repeats: int = 30):
    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn(*args)
    dt = (time.perf_counter() - t0) / float(repeats)
    return dt, out


def main() -> None:
    filters = "haar"
    rng = np.random.default_rng(4)

    print("=" * 88)
    print("MODWT backend validation")
    print("=" * 88)

    x_const = np.ones(64, dtype=float)
    coeffs_const = modwt(x_const, filters, 4)
    max_const_detail = float(np.max(np.abs(coeffs_const[:-1])))
    print(f"constant-signal detail max |W_j| = {max_const_detail:.3e}")

    x = np.arange(8, dtype=float)
    print("\nReconstruction errors on ramp x = [0,1,2,3,4,5,6,7]")
    for J in (1, 2, 3):
        coeffs_wrong = modwt_original_shortcut(x, filters, J)
        coeffs_right = modwt(x, filters, J)
        err_wrong = reconstruction_error(x, coeffs_wrong, filters)
        err_right = reconstruction_error(x, coeffs_right, filters)
        print(f"  J={J}: original-shortcut error = {err_wrong:.6g}, corrected error = {err_right:.6g}")

    x3 = rng.normal(size=(1024, 3))
    dt_corr_3c, coeffs_3c = benchmark(modwt, x3, filters, 6, repeats=8)
    errs_3c = np.max(np.abs(imodwt(coeffs_3c, filters) - x3))
    print("\nVectorized 3-component test")
    print(f"  corrected 3C runtime per call: {dt_corr_3c:.6f} s")
    print(f"  corrected 3C reconstruction max error: {errs_3c:.3e}")

    x1 = rng.normal(size=1024)
    dt_wrong_1c, coeffs_wrong_1c = benchmark(modwt_original_shortcut, x1, filters, 6, repeats=12)
    dt_corr_1c, coeffs_corr_1c = benchmark(modwt, x1, filters, 6, repeats=12)
    err_wrong_1c = reconstruction_error(x1, coeffs_wrong_1c, filters)
    err_corr_1c = reconstruction_error(x1, coeffs_corr_1c, filters)
    print("\n1D timing and reconstruction")
    print(f"  original-shortcut runtime per call: {dt_wrong_1c:.6f} s")
    print(f"  corrected runtime per call:         {dt_corr_1c:.6f} s")
    print(f"  original-shortcut reconstruction error: {err_wrong_1c:.3e}")
    print(f"  corrected reconstruction error:         {err_corr_1c:.3e}")


if __name__ == "__main__":
    main()
