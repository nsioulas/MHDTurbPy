import numpy as np

from scipy import constants
from astropy import units as u
from plasmapy.formulary import beta, magnetic_pressure, thermal_pressure, ion_sound_speed

mu_0            = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu0             = constants.mu_0   #
m_p             = constants.m_p    # Proton mass [kg]
kb              = constants.k      # Boltzman's constant     [j/K]
au_to_km        = 1.496e8
T_to_Gauss      = 1e4
km2m            = 1e3
nT2T            = 1e-9
cm2m            = 1e-2    


def estimate_Vth(Vth):
    Vth[Vth < 0]                     = np.nan
    return Vth, np.nanmean(Vth), np.nanmedian(Vth), np.nanstd(Vth)

def estimate_Vsw(Vsw):
    Vsw[(np.abs(Vsw) > 1e5)]         = np.nan
    return Vsw, np.nanmean(Vsw), np.nanmedian(Vsw), np.nanstd(Vsw)

def estimate_Np(Np):
    return Np, np.nanmean(Np), np.nanmedian(Np), np.nanstd(Np)

def estimate_di(Np):
    di                               = 228 / np.sqrt(Np)
    di[di < 1e-3]                    = np.nan
    return di, np.nanmean(di), np.nanmedian(di), np.nanstd(di)


def estimate_beta(magnetic_field_nT, density_cm3, temperature_eV):
    """
    Calculate the plasma beta given density (in cm^-3), temperature (in eV), 
    and magnetic field strength (in nT).

    Parameters:
        density_cm3 (float): Density in cm^-3.
        temperature_eV (float): Temperature in electronvolts (eV).
        magnetic_field_nT (float): Magnetic field strength in nanotesla (nT).

    Returns:
        float: The plasma beta (dimensionless).
    """
    # Convert density from cm^-3 to m^-3
    density    = density_cm3 * u.cm**-3
    density_si = density.to(u.m**-3)
    
    # Convert temperature from eV to Joules 
    temperature = temperature_eV * u.eV
    temperature_J = temperature.to(u.J)
    
    # Calculate plasma pressure: p = n * (energy per particle)
    p = density_si * temperature_J

    # Convert magnetic field from nT to Tesla
    B = magnetic_field_nT * u.nT
    B_T = B.to(u.T)
    
    # Calculate magnetic pressure: (B^2) / (2 * mu0)
    magnetic_pressure = B_T**2 / (2 * mu0)
    
    # Compute plasma beta (dimensionless)
    beta = (p / magnetic_pressure).decompose().value

    beta[beta < 0]                   = np.nan
    beta[np.abs(np.log10(beta)) > 4] = np.nan
    
    return beta, np.nanmean(beta), np.nanmedian(beta), np.nanstd(beta)

def estimate_rho_i(di, beta):
    rho_i                           = di*np.sqrt(beta)
    rho_i[rho_i < 0]                = np.nan
    rho_i[np.log10(rho_i) < -3]     = np.nan
    return rho_i, np.nanmean(rho_i), np.nanmedian(rho_i), np.nanstd(rho_i)

