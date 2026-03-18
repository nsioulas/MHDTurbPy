from pathlib import Path

try:
    from .path_setup import ensure_project_paths
except ImportError:
    from path_setup import ensure_project_paths

ensure_project_paths(start=Path(__file__).resolve(), include_downloading_helpers=True)
import general_functions as func
import TurbPy as turb

try:
    from .three_D_funcs import *
    from .data_analysis import *
except Exception:
    from three_D_funcs import *
    from data_analysis import *
