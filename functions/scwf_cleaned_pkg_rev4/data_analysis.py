from __future__ import annotations

"""Public entry points for the self-consistent wavelet-frame anisotropy pipeline.

This module keeps one notebook-facing compatibility alias,
``run_logscale_filterbank_analysis``, but drops the unused MODWT/two-point export
names that did not correspond to distinct implementations in this package.
"""

from pathlib import Path
import importlib
import importlib.util
import sys
from types import ModuleType
from typing import Optional

_HERE = Path(__file__).resolve().parent
_PACKAGE_NAME = _HERE.name
_PARENT_DIR = _HERE.parent


def _ensure_project_paths() -> None:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        path_setup = candidate / "functions" / "path_setup.py"
        if not path_setup.is_file():
            continue
        spec = importlib.util.spec_from_file_location("mhdturbpy_path_setup", path_setup)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ensure = getattr(module, "ensure_project_paths", None)
        if ensure is None:
            continue
        for kwargs in (
            dict(start=here, include_downloading_helpers=True, include_anisotropy_toolbox=True, include_sc_pos=True),
            dict(start=here, include_downloading_helpers=True, include_anisotropy_toolbox=True),
            dict(start=here, include_downloading_helpers=True),
            dict(start=here),
        ):
            try:
                ensure(**kwargs)
                return
            except TypeError:
                continue
            except Exception:
                return
        return


_bootstrap_done = False
_local_impl_cache: Optional[ModuleType] = None


def _bootstrap_once() -> None:
    global _bootstrap_done
    if _bootstrap_done:
        return
    _ensure_project_paths()
    parent_str = str(_PARENT_DIR)
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
    _bootstrap_done = True


def _purge_stale_package_modules() -> None:
    stale_names = [
        _PACKAGE_NAME,
        f"{_PACKAGE_NAME}.three_D_funcs",
        f"{_PACKAGE_NAME}.data_analysis",
    ]
    for name in stale_names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            sys.modules.pop(name, None)
            continue
        try:
            resolved = Path(mod_file).resolve()
        except Exception:
            sys.modules.pop(name, None)
            continue
        if _HERE not in resolved.parents and resolved != _HERE / "__init__.py":
            sys.modules.pop(name, None)


def _load_local_impl() -> ModuleType:
    global _local_impl_cache
    if _local_impl_cache is not None:
        return _local_impl_cache
    _bootstrap_once()
    _purge_stale_package_modules()
    module = importlib.import_module(f"{_PACKAGE_NAME}.three_D_funcs")
    _local_impl_cache = module
    return module


_IMPL = _load_local_impl()

run_filterbank_interval_analysis = _IMPL.run_filterbank_interval_analysis
run_logscale_filterbank_analysis = _IMPL.run_filterbank_interval_analysis
estimate_3D_sfuncs_same_format = _IMPL.estimate_3D_sfuncs_same_format
estimate_3D_sfuncs = _IMPL.estimate_3D_sfuncs_same_format
estimate_filterbank_backgrounds_and_fluctuations = _IMPL.estimate_wavelet_backgrounds_and_fluctuations

__all__ = [
    "run_filterbank_interval_analysis",
    "run_logscale_filterbank_analysis",
    "estimate_3D_sfuncs_same_format",
    "estimate_3D_sfuncs",
    "estimate_filterbank_backgrounds_and_fluctuations",
]
