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
Install virtualenv using pip:
```bash
pip install virtualenv
```
Create a new virtual environment:
```bash
virtualenv MHDTurbPy
```
Activate the virtual environment:
```bash
source MHDTurbPy/bin/activate
```

## 3) Install dependencies
Install the required packages from the repo requirements file:
```bash
pip install -r requirements/requirements.txt
```

If you need to continue installing packages even when some fail, you can try a bash loop that falls back to conda:

```bash
while read p; do
    pip install "$p" || (echo "Trying to install $p with conda" && conda install "$p" -y || echo "Failed to install $p with both pip and conda")
done < requirements/requirements.txt
```

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

