Revised MODWT pipeline: key changes
==================================

Files changed
-------------
- `backend.py`
- `pipeline.py`
- `README.md`
- `validate_modwt_backend.py` (new)
- `validation_results.txt` (new)

Backend fixes
-------------
- Replaced the non-MODWT shortcut in `modwt()` with the correct recursive pyramid.
- Added proper multicomponent support:
  - input `(N,)` -> output `(J+1, N)`
  - input `(N, C)` -> output `(J+1, N, C)`
- Kept the same circular-convolution convention and sparse MRA reconstruction.
- Kept `imodwt()` and `modwtmra()` compatible with the corrected transform.

Pipeline fixes
--------------
- `estimate_coeffs_background_flucs_MODWT()` now decomposes all components in one vectorized call instead of looping over components in Python.
- `_recommended_level()` now uses a MODWT-style unbiased depth estimate by default.
- Added metadata entries `level_max_transform`, `level_max_unbiased`, and `level_mode_used`.

Validation
----------
Run `python validate_modwt_backend.py` inside the folder.
The included `validation_results.txt` shows that the old shortcut fails exact reconstruction for `J >= 2`, while the corrected backend reconstructs to machine precision.
