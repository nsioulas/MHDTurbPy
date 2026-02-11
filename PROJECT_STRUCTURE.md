# Project Structure

This repository now follows a more user-friendly top-level layout:

- `assets/`
  - Static media used by documentation and notebooks.
  - `assets/logo/` contains project logos.
- `examples/`
  - Reproducible outputs and demos.
  - `examples/notebooks/` contains all tutorial and workflow notebooks.
  - `examples/figures/` contains generated example figures.
- `functions/`
  - Core analysis and data-download utilities used by notebooks/scripts.
- `requirements/`
  - Python dependency specification for `pip` installs.
- `environment.yml`
  - Conda environment definition.
- `user_paths.example.json`
- `scripts/path_audit.py` (checks `.py`/`.ipynb` path references after repo reorganizations)
  - Template for local machine-specific path settings.
- `pyspedas/`
  - Bundled dependency source tree.

## Where to start

1. Read `README.md` for setup.
2. Copy `user_paths.example.json` to `user_paths.json` and set your paths.
3. Run notebooks from `examples/notebooks/`.
