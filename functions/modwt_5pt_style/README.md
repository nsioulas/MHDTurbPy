MODWT 5pt-style folder (revised)
================================

This folder is intended to live inside ``MHDTurbPy/functions``.

What was fixed
--------------
The previous backend re-filtered the original signal at every level. That is
not the MODWT pyramid algorithm. The revised backend now uses the correct
recursive undecimated decomposition,

``W_j = H_j V_{j-1}``, ``V_j = G_j V_{j-1}``, ``V_0 = X``,

with the same circular-convolution convention as the old code.

Main consequences
-----------------
- Exact reconstruction now holds for the forward/inverse backend.
- ``modwt`` supports vectorized multicomponent inputs of shape ``(N, C)`` and
  returns ``(J+1, N, C)``.
- ``estimate_coeffs_background_flucs_MODWT`` now decomposes all components in
  one call instead of looping over them in Python.
- The default ``level_mode='recommended'`` now uses a MODWT-style unbiased
  depth estimate rather than a DWT-style ``pywt.dwt_max_level`` bound.
- MODWT still does **not** require padding the input length to a power of two.

Design choices retained
-----------------------
- ``general_functions`` and ``TurbPy`` are imported in the same way as in the
  5pt pipeline: ``ensure_project_paths(...)`` first, then
  ``import general_functions as func`` and ``import TurbPy as turb``.
- No local copies of ``general_functions.py`` or ``TurbPy.py`` are shipped
  here.
- The folder keeps only the MODWT-specific pieces plus thin 5pt-style front
  doors: ``three_D_funcs.py`` and ``data_analysis.py``.

Main files
----------
- ``three_D_funcs.py``: 5pt-style front door for ``estimate_3D_sfuncs``.
- ``data_analysis.py``: 5pt-style front door for ``modwt_two_pt_wavelet_analysis``.
- ``pipeline.py``: main MODWT implementation.
- ``backend.py``: corrected MODWT/MRA backend.
- ``shared_logic.py``: shared quantity and geometry logic.
- ``path_setup.py``: local path helper used only to reach the parent ``functions`` tree.
- ``revised_pipeline_same_format.py``: compatibility re-export.
- ``validate_modwt_backend.py``: minimal validation/benchmark script comparing the old shortcut with the corrected backend.

Notebook note
-------------
``3D_MODWT_sfuncs.ipynb`` is still a direct copy of the original 3D 5pt
notebook with only the MODWT folder inserted on ``sys.path`` and the analysis
call switched from ``five_pt_two_pt_wavelet_analysis(...)`` to
``modwt_two_pt_wavelet_analysis(...)``.
