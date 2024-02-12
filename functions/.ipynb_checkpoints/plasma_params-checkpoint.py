import numpy as np

from scipy import constants
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

def estimate_beta(Bmag, Vth, Np):
    B_mag                            = Bmag * nT2T
    temp                             = 0.5 * m_p * (Vth * km2m) ** 2
    dens                             = Np / (cm2m ** 3)
    beta                             = (dens * temp) / ((B_mag ** 2) / (2 * mu_0))
    beta[beta < 0]                   = np.nan
    beta[np.abs(np.log10(beta)) > 4] = np.nan
    return beta, temp,  np.nanmean(beta), np.nanmedian(beta), np.nanstd(beta)

def estimate_rho_ci(Vth, B_mag):
    rho_ci                           = 10.43968491 * Vth / B_mag
    rho_ci[rho_ci < 0]               = np.nan
    rho_ci[np.log10(rho_ci) < -3]    = np.nan
    return rho_ci, np.nanmean(rho_ci), np.nanmedian(rho_ci), np.nanstd(rho_ci)
