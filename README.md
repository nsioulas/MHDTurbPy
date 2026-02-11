![MHDTurbPy](assets/logo/final.png "turb")

# MHDTurbPy
MHDTurbPy is a Python toolkit for downloading, cleaning, and analyzing heliophysics spacecraft data from:

- Parker Solar Probe
- Solar Orbiter
- WIND
- Helios A
- Helios B
- Ulysses

Documentation and additional functionality are under active development, and contributions are welcome.



# Installation

## 1) Download the package
```bash
git clone https://github.com/nsioulas/MHDTurbPy/
cd MHDTurbPy
```

## 2) Create and activate a virtual environment
We recommend conda because it handles scientific dependencies (e.g., `spacepy`, `netCDF4`) more reliably across platforms.

```bash
conda env update --file environment.yml --name mhdturbpy
conda activate mhdturbpy
```

If you prefer `venv` + `pip`, you can use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
```

> Note: The requirements file contains the core dependencies used across the download/clean/analyze pipeline, including Solar Orbiter retrieval support (`sunpy`, `sunpy-soar`) and the plotting colormap package (`colormaps`). Notebook/Jupyter tooling is intentionally left out so you can install it only when needed.


## 3) Configure paths once (optional but recommended)
Create a single `user_paths.json` file in the repository root (copy from `user_paths.example.json`) and set your machine-specific paths once:

```json
{
  "Data_path": "/absolute/path/to/data/root",
  "save_destination": "/absolute/path/to/MHDTurbPy/examples",
  "cdf_lib_path": "/absolute/path/to/cdf/lib",
  "analysis_data_path": "/optional/path/for-custom-analysis-inputs",
  "solo_dist_path": "/optional/path/to/MHDTurbPy/examples/SOLO/solo_dist.pkl"
}
```

All updated Python modules and example notebooks now read from this shared config, so you do not need to edit paths in multiple places. Optional keys (for notebook-specific datasets) can also be kept here to avoid ad-hoc edits.

### Where to input base paths (important)
Use **exactly one** of the following methods:

1. **Preferred:** create `user_paths.json` in the repository root (same directory as `README.md`).
2. **Alternative:** set environment variables before running notebooks/scripts:
   - `MHDTURBPY_DATA_PATH`
   - `MHDTURBPY_SAVE_DESTINATION`
   - `MHDTURBPY_CDF_LIB_PATH`

`user_paths.json` keys:
- `Data_path`: base directory where raw downloaded mission folders (e.g. `psp_data`, `solar_orbiter_data`) are created.
- `save_destination`: base directory for processed interval outputs (`final.pkl`, `general.pkl`, etc.).
- `cdf_lib_path`: local CDF library location (optional unless CDF-dependent loaders are used).
- `analysis_data_path` (optional): override input location used by analysis notebooks.

To validate path references after setup, run:

```bash
python scripts/path_audit.py --root .
```

This checks `.py` files and notebook code cells for unresolved repo/base path references.

# Troubleshooting (IPython/Jupyter on Windows)

If you see logs like:
- `ModuleNotFoundError: No module named 'autoreload'`
- `IPython.extensions.deduperreload ... UnicodeDecodeError ... cp1252`

this is usually an **IPython extension/config issue**, not a MHDTurbPy runtime bug.

Recommended fixes:

1. Ensure UTF-8 mode is enabled before launching Python/Jupyter:

```bash
set PYTHONUTF8=1
```

(Or persist this in your shell/environment settings.)

2. In IPython/Jupyter, use the built-in extension:

```python
%load_ext autoreload
%autoreload 2
```

3. If your environment tries to auto-load `deduperreload` and fails, disable that startup hook or update/remove the extension in the environment.

4. If extension behavior remains inconsistent, recreate the conda env from `environment.yml` and test a clean IPython session with no custom startup scripts.

5. Apply the repo-provided automatic patch for IPython config (recommended on Windows):

```bash
python scripts/fix_ipython_windows.py
```

This writes/updates `~/.ipython/profile_default/ipython_config.py` to:
- remove `deduperreload` from auto-loaded extensions,
- add `IPython.extensions.autoreload`,
- set `PYTHONUTF8=1` for session startup.

If your tracebacks mention stdlib files like `functools.py`, `urllib/parse.py`, `shlex.py`, `re/_casefix.py`, or `pydoc_data/topics.py`, that confirms the decode failure is happening inside extension scanning of the Python installation, not inside MHDTurbPy project files.


# Usage

Example notebooks that demonstrate how to download and visualize data are available in `examples/notebooks`.
Start with the notebooks in `examples/notebooks/` if you want end-to-end data download and analysis examples.

# Contact
If you have any questions, please don't hesitate to reach out to nsioulas@g.ucla.edu.

# Citation

If you use this work, please cite:

```
@software{nikos_sioulas_2023_7572468,
  author       = {Nikos Sioulas},
  title        = {MHDTurbPy},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.7572468},
  url          = {https://doi.org/10.5281/zenodo.7572468}
}
```
