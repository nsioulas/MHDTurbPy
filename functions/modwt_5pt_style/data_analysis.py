from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sys
import importlib.util
import scipy.io
import os
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[1]

_PATH_SETUP = _REPO_ROOT / "functions" / "path_setup.py"
_spec = importlib.util.spec_from_file_location("mhdturbpy_path_setup", _PATH_SETUP)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load path setup from {_PATH_SETUP}")
_path_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_path_setup)
ensure_project_paths = _path_setup.ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True, include_anisotropy_toolbox=True)
import pickle
import gc
from glob import glob
from datetime import datetime
import traceback
from time import sleep
import matplotlib.dates as mdates
from scipy import interpolate
from scipy.interpolate import interp1d

# Make sure to use the local spedas
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

""" Import manual functions """

import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func
import Figures as figs
from SEA import SEA
try:
    from . import three_D_funcs as threeD
except Exception:
    import three_D_funcs as threeD

try:
    from PSP import download_ephemeris_PSP
except Exception:
    download_ephemeris_PSP = None


def _import_pipeline():
    try:
        from . import pipeline as _pipeline
    except Exception:
        import pipeline as _pipeline
    return _pipeline


def modwt_two_pt_wavelet_analysis(*args, **kwargs):
    return _import_pipeline().modwt_two_pt_wavelet_analysis(*args, **kwargs)


def estimate_3D_sfuncs_same_format(*args, **kwargs):
    return threeD.estimate_3D_sfuncs_same_format(*args, **kwargs)


def estimate_3D_sfuncs_modwt(*args, **kwargs):
    return threeD.estimate_3D_sfuncs_modwt(*args, **kwargs)


__all__ = [
    'modwt_two_pt_wavelet_analysis',
    'estimate_3D_sfuncs_same_format',
    'estimate_3D_sfuncs_modwt',
    'func',
    'turb',
    'threeD',
]
