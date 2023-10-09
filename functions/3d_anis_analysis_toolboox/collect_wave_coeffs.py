import numpy as np
import pandas as pd
import sys
import scipy.io
import os
import sys
from pathlib import Path
import pickle
from gc import collect
from glob import glob
from datetime import datetime
import traceback
from time import sleep
import matplotlib.dates as mdates



#from numba import prange, jit
#@jit( parallel =True, nopython=True)
def choose_wavelet_coeffs(indices, Wx, Wy, Wz, Vsw_di, taus, freqs):


    # Initialize dictionary
    collect_dict = {
                    "Wx"            : Wx[indices],
                    "Wy"            : Wy[indices],
                    "Wz"            : Wz[indices],
                    "taus"          : taus[indices], 
                    "Vsw_over_di"   : Vsw_di[indices],
                    'freqs'         : freqs[indices],
                   }
    print('Minvals', min(taus[indices]))
    return collect_dict



def keep_conditioned_coeffs(phis, thetas, coeffs, conditions):
    
    #initialize conditions
    ell_perp_conds     = conditions['ell_perp']
    Ell_perp_conds     = conditions['Ell_perp']
    ell_par_conds      = conditions['ell_par']
    ell_par_rest_conds = conditions['ell_par_rest']
    
    # Create 1 d arrays with needed data
    Vsw_di  = np.hstack(np.ones(np.shape(coeffs['Wx']))*(coeffs['Vsw_minus_Vsc']/coeffs['di']))

    taus    = np.hstack(np.ones(np.shape(coeffs['Wx']))*((coeffs['phys_scales'])[:, np.newaxis]))
    freqs   = np.hstack(np.ones(np.shape(coeffs['Wx']))*(coeffs['freqs'][:, np.newaxis]))
    #print(taus)


    # Turn them into 1d array
    Wx      = np.hstack(coeffs['Wx'])
    Wy      = np.hstack(coeffs['Wy'])   
    Wz      = np.hstack(coeffs['Wz']) 
    
    del coeffs
    
    # Collect coeffs corresponding to ell_perp
    indices                      = np.where((thetas>ell_perp_conds['theta']) & (phis>ell_perp_conds['phi']))[0]
    ell_perp_dict = choose_wavelet_coeffs(
                                          indices,
                                          Wx,
                                          Wy,
                                          Wz,
                                          Vsw_di,
                                          taus,
                                          freqs,

    )

    # Collect coeffs corresponding to Ell_perp
    indices                      = np.where((thetas>Ell_perp_conds['theta']) & (phis<Ell_perp_conds['phi']))[0]
    Ell_perp_dict = choose_wavelet_coeffs(
                                          indices,
                                          Wx,
                                          Wy,
                                          Wz,
                                          Vsw_di,
                                          taus,
                                          freqs,

    )

    # Collect coeffs corresponding to Ell_par
    indices                      = np.where((thetas<ell_par_conds['theta']) & (phis<ell_par_conds['phi']))[0]
    ell_par_dict = choose_wavelet_coeffs(
                                          indices,
                                          Wx,
                                          Wy,
                                          Wz,
                                          Vsw_di,
                                          taus,
                                          freqs,

    )

    # Collect coeffs corresponding to Ell_par_restricted
    indices                      = np.where((thetas<ell_par_rest_conds['theta']) & (phis<ell_par_rest_conds['phi']))[0]
    ell_par_rest_dict = choose_wavelet_coeffs(
                                          indices,
                                          Wx,
                                          Wy,
                                          Wz,
                                          Vsw_di,
                                          taus,
                                          freqs,

    )
    
    return ell_perp_dict, Ell_perp_dict, ell_par_dict, ell_par_rest_dict