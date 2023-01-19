
## Equator-S
The routines in this module can be used to load data from the Equator-S mission. 

### Instruments
- Fluxgate magnetometer (MAM)
- Electron beam sensing instrument (EDI)
- Electrostatic analyzer (3DA)
- Solid state detector (EPI)
- Time-of-fight spectrometer (ICI)
- Ion emitter (PCD)
- Scintillating fiber detector (SFD)

### Examples
Get started by importing pyspedas and tplot; these are required to load and plot the data:

```python
import pyspedas
from pytplot import tplot
```

#### Fluxgate magnetometer (MAM)

```python
mam_vars = pyspedas.equator_s.mam(trange=['1998-04-06', '1998-04-07'])

tplot('B_xyz_gse%eq_pp_mam')
```

#### Electron beam sensing instrument (EDI)

```python
edi_vars = pyspedas.equator_s.edi(trange=['1998-04-06', '1998-04-07'])

tplot('E_xyz_gse%eq_pp_edi')
```

#### Solid state detector (EPI)

```python
epi_vars = pyspedas.equator_s.epi(trange=['1998-04-06', '1998-04-30'])

tplot(['J_e_1%eq_pp_epi', 'J_e_2%eq_pp_epi', 'J_e_3%eq_pp_epi'])
```

#### Time-of-fight spectrometer (ICI)

```python
ici_vars = pyspedas.equator_s.ici(trange=['1998-04-06', '1998-04-07'])

tplot('V_p_xyz_gse%eq_pp_ici')
```

#### Ion emitter (PCD)

```python
pcd_vars = pyspedas.equator_s.pcd(trange=['1998-04-06', '1998-04-07'])

tplot('I_ion%eq_pp_pcd')
```

#### Scintillating fiber detector (SFD)

```python
sfd_vars = pyspedas.equator_s.sfd()

tplot('F_e>0.26%eq_sp_sfd')
```
