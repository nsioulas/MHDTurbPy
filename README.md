![MHDTurbPy](logo/final.png "turb")

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

# Usage

Example notebooks that demonstrate how to download and visualize data are available in `Notebooks_Examples`.
Start with the notebooks in `Notebooks_Examples/` if you want end-to-end data download and analysis examples.

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
