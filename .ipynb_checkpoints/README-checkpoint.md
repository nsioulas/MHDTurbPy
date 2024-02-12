![MHDTurbpy](logo/final.png "turb")


# MHDTurbPy
A collection of functions for downloading, cleaning, and analyzing data from:

 - Parker Solar Probe
 - Solar Orbiter 
 - WIND
 - Helios A
 - Helios B
 - Ulysses

Additional functions and documentation will be added soon.


 I welcome contributions to this repository!



# Installation


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

Install the required packages from your environment file. Don't forget to change the path to your downloaded .txt file: 
```bash
pip install -r path/to/file/filtered_requirements.txt
 ```
 
 To continue installing packages even if some fail, you can use a bash loop to try installing each package individually. This way, even if one package fails to install, the loop will proceed to the next package in the list. 
 
```bash

while read p; do
    pip install "$p" || (echo "Trying to install $p with conda" && conda install "$p" -y || echo "Failed to install $p with both pip and conda")
done < /path/to/file/filtered_requirements.txt
 ```

 - Download the package
``` bash
git clone https://github.com/nsioulas/MHDTurbPy/
```

# Usage

Some examples of how to download, visualize data can be found in the folder ```Notebooks_Examples```

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



