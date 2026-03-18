import numpy as np
import pandas as pd
from pathlib import Path

try:
    from .path_setup import ensure_project_paths
except ImportError:
    from path_setup import ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
import general_functions as func
import TurbPy as turb


def _import_shared_logic():
    try:
        from . import shared_logic as _shared_logic
    except Exception:
        import shared_logic as _shared_logic
    return _shared_logic


def _import_pipeline():
    try:
        from . import pipeline as _pipeline
    except Exception:
        import pipeline as _pipeline
    return _pipeline


# -----------------------------------------------------------------------------
# 5pt-style public front door.
# Keep the public calls rooted in three_D_funcs, but import the heavy MODWT
# implementation lazily so TurbPy -> three_D_funcs -> TurbPy circular imports do
# not break joblib worker startup.
# -----------------------------------------------------------------------------

def _has_columns(*args, **kwargs):
    return _import_shared_logic()._has_columns(*args, **kwargs)


def infer_frame_from_data(*args, **kwargs):
    return _import_shared_logic().infer_frame_from_data(*args, **kwargs)


def _background_polarity(*args, **kwargs):
    return _import_shared_logic()._background_polarity(*args, **kwargs)


def _mag_from_frame(*args, **kwargs):
    return _import_shared_logic()._mag_from_frame(*args, **kwargs)


def est_alignment_angles(*args, **kwargs):
    return _import_shared_logic().est_alignment_angles(*args, **kwargs)


def fast_unit_vec(*args, **kwargs):
    return _import_shared_logic().fast_unit_vec(*args, **kwargs)


def mag_of_ell_projections_and_angles(*args, **kwargs):
    return _import_shared_logic().mag_of_ell_projections_and_angles(*args, **kwargs)


def structure_functions_3D(*args, **kwargs):
    return _import_shared_logic().structure_functions_3D(*args, **kwargs)


def vars_2_estimate(*args, **kwargs):
    return _import_shared_logic().vars_2_estimate(*args, **kwargs)


def quants_2_estimate(*args, **kwargs):
    return _import_shared_logic().quants_2_estimate(*args, **kwargs)


def save_flucs(*args, **kwargs):
    return _import_shared_logic().save_flucs(*args, **kwargs)


def local_structure_function(*args, **kwargs):
    return _import_pipeline().modwt_local_structure_function(*args, **kwargs)


def estimate_coeffs_background_flucs_MODWT(*args, **kwargs):
    return _import_pipeline().estimate_coeffs_background_flucs_MODWT(*args, **kwargs)


def estimate_approxs(*args, **kwargs):
    return _import_pipeline().estimate_approxs(*args, **kwargs)


def estimate_3D_sfuncs(*args, **kwargs):
    return _import_pipeline().estimate_3D_sfuncs_same_format(*args, **kwargs)


def estimate_3D_sfuncs_same_format(*args, **kwargs):
    return _import_pipeline().estimate_3D_sfuncs_same_format(*args, **kwargs)


def estimate_3D_sfuncs_modwt(*args, **kwargs):
    return _import_pipeline().estimate_3D_sfuncs_same_format(*args, **kwargs)


__all__ = [
    '_has_columns',
    'infer_frame_from_data',
    '_background_polarity',
    '_mag_from_frame',
    'est_alignment_angles',
    'fast_unit_vec',
    'mag_of_ell_projections_and_angles',
    'structure_functions_3D',
    'vars_2_estimate',
    'quants_2_estimate',
    'save_flucs',
    'local_structure_function',
    'estimate_coeffs_background_flucs_MODWT',
    'estimate_approxs',
    'estimate_3D_sfuncs',
    'estimate_3D_sfuncs_same_format',
    'estimate_3D_sfuncs_modwt',
]
