import numpy as np
import pandas as pd
import sys
import gc
import time
from numba import jit, njit, prange, objmode 
import os
from pathlib import Path
from glob import glob
from gc import collect
import traceback
import datetime


# SPEDAS API

# SPEDAS API
# make sure to use the local spedas
sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

# Import TurbPy
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import TurbPy as turb
import general_functions as func
import LoadData
import plasma_params as plasma



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




def align_dataframes_based_on_frequency(df_higher_freq, df_lower_freq, settings):
    """
    Align two dataframes based on their frequency, upsample lower frequency dataframe if specified.
    
    Args:
    - df_higher_freq: DataFrame with higher frequency data.
    - df_lower_freq: DataFrame with lower frequency data.
    - settings: Dictionary containing settings including 'upsample_low_freq_ts'.
    
    Returns:
    - Tuple of aligned DataFrames.
    """
    if settings.get('upsample_low_freq_ts', False):
        # Upsample the lower frequency dataframe to match the higher frequency one
        aligned_lower_freq = func.newindex(df_lower_freq, df_higher_freq.index)
        return df_higher_freq, aligned_lower_freq
    else:
        # Attempt to align the higher frequency dataframe to the lower frequency one
        try:
            aligned_higher_freq = func.newindex(df_higher_freq, df_lower_freq.index)
            return aligned_higher_freq, df_lower_freq
        except Exception as e:  # Consider specifying the exact exception if known
            print(f'Error aligning dataframes: {e}')
            # Optionally, handle the error or return the original dataframes
            return df_higher_freq, df_lower_freq

def apply_rolling_mean(f_df, settings, coord_type='RTN'):
    """
    Apply rolling mean to specified columns based on coordinate type.
    
    Parameters:
    - f_df: DataFrame to process.
    - settings: Dictionary containing settings like 'rol_window'.
    - coord_type: Type of coordinates, 'RTN' for R,T,N and 'XYZ' for x,y,z.
    """
    # Define columns based on coordinate type
    if coord_type == 'RTN':
        columns = [['Br', 'Bt', 'Bn'], ['Vr', 'Vt', 'Vn']]
    else:  # Assume 'XYZ' if not 'RTN'
        columns = [['Bx', 'By', 'Bz'], ['Vx', 'Vy', 'Vz']]
    
    # Apply rolling mean and interpolate
    for c in columns:
        f_df[[f"{col}_mean" for col in c]] = f_df[c].rolling(
            settings['rol_window'], center=True).mean().interpolate()
    
    return f_df

def apply_rolling_mean_and_get_columns(f_df, settings):
    """Attempt to apply rolling mean with RTN coordinates, fallback to XYZ if fails."""
    try:
        f_df = apply_rolling_mean(f_df, settings, 'RTN')
        columns_b = ["Br", "Bt", "Bn"]
        columns_v = ["Vr", "Vt", "Vn"]
    except KeyError:  # Assuming KeyError is the relevant exception if columns are missing
        f_df = apply_rolling_mean(f_df, settings, 'XYZ')
        columns_b = ["Bx", "By", "Bz"]
        columns_v = ["Vx", "Vy", "Vz"]
    return f_df, columns_b, columns_v


def calculate_signB(f_df):
    """Calculate the sign of B based on available column."""
    if 'Br_mean' in f_df.columns:
        return -np.sign(f_df['Br_mean'])
    elif 'Bx_mean' in f_df.columns:
        return np.abs(-np.sign(f_df['Bx_mean']))
    else:
        raise ValueError("Required column is missing in DataFrame.")

def calculate_components(dv, dva, signB):
    """Calculate Zp and Zm components."""
    
    # Calculate Zp and Zm components in a vectorized manner
    Zpr, Zmr           = dv[0] + signB * dva[0], dv[0] - signB * dva[0]
    Zpt, Zmt           = dv[1] + signB * dva[1], dv[1] - signB * dva[1]
    Zpn, Zmn           = dv[2] + signB * dva[2], dv[2] - signB * dva[2]
    
    return np.array([Zpr, Zpt, Zpn]), np.array([Zmr, Zmt, Zmn])

def calculate_energies_sigmas(Zp, Zm, dv, dva):
    """Calculate energies and normalized residual energies."""
    Z_plus_squared  = np.sum(Zp**2, axis=0)
    Z_minus_squared = np.sum(Zm**2, axis=0)
    Ek              = np.sum(dv**2, axis=0)
    Eb              = np.sum(dva**2, axis=0)
    
    sigma_r         = (Ek - Eb) / (Ek + Eb)
    sigma_c         = (Z_plus_squared - Z_minus_squared) / (Z_plus_squared + Z_minus_squared)
    
    # Apply threshold
    sigma_r[np.abs(sigma_r) > 1e5] = np.nan
    sigma_c[np.abs(sigma_c) > 1e5] = np.nan
    
    return sigma_r, sigma_c


def estimate_psds(sigs_df, component_keys, dtb, settings):
    """
    Estimate the power spectral density (PSD) for given signal components.

    Args:
    - sigs_df (DataFrame): DataFrame containing signal data.
    - component_keys (list): List of keys for the components to be processed.
    - dtb (float): Time step or other relevant parameter for PSD calculation.
    - settings (dict): Settings dict that includes 'est_PSD_components'.

    Returns:
    - Tuple containing frequency and PSD values, including components if requested.
    """
    values = [sigs_df[key].values for key in component_keys]
    return turb.TracePSD(*values, dtb, return_components=settings['est_PSD_components'])

def estimate_B_psd_and_smooth(mag_resampled, dtb, settings):
    """
    Estimate the Power Spectral Density (PSD) of the magnetic field and optionally smooth it.
    """
    # Initialize variables to None in case they are not set due to skipping the estimation
    psd_B_R, psd_B_T, psd_B_N, f_B, psd_B, f_B_mid, f_B_mean, psd_B_smooth = (None,) * 8
    
    if settings['estimate_psd_b']:
        try:
            # Estimate PSD of the magnetic field
            f_B, psd_B, psd_B_R, psd_B_T, psd_B_N = turb.TracePSD(
                mag_resampled.values.T[0],
                mag_resampled.values.T[1],
                mag_resampled.values.T[2],
                dtb,
                return_components=settings['est_PSD_components'])
            
            # Smooth PSD of the magnetic field if required
            if settings['smooth_psd']:
                f_B_mid, f_B_mean, psd_B_smooth = func.smoothing_function(f_B, psd_B, window=2, pad=1)
                
        except Exception as e:
            traceback.print_exc()  # Log the error for debugging
            # Variables are already initialized to None, so they can be returned as is in case of an error
    
    # Return all relevant variables, regardless of whether PSD estimation was performed or not
    return f_B, psd_B, psd_B_R, psd_B_T, psd_B_N, f_B_mid, f_B_mean, psd_B_smooth



def calculate_diagnostics(
                          mag_resampled,
                          df_part,
                          dist, 
                          settings,
                          diagnostics
                         ):     
            
    """ Interpolate gaps"""  
    mag_resampled       = mag_resampled.dropna().interpolate()
    df_part             = df_part.dropna().interpolate()
    
    # Reindex magnetic field data to particle data index
    dtv                 = func.find_cadence(df_part)
    dtb                 = func.find_cadence(mag_resampled)

    # Estimate the Power Spectral Density (PSD) of the magnetic field and optionally smooth it
    f_B, psd_B, psd_B_R, psd_B_T, psd_B_N, f_B_mid, f_B_mean, psd_B_smooth = estimate_B_psd_and_smooth(mag_resampled, dtb, settings)
    
    # Determine which dataframe has higher frequency based on dtv and dtb comparison
    if dtv > dtb:
        mag_resampled, df_part_aligned = align_dataframes_based_on_frequency(mag_resampled, df_part, settings)
    else:
        df_part_aligned, mag_resampled = align_dataframes_based_on_frequency(df_part, mag_resampled, settings)

    # Create final dataframe by joining and cleaning up the data
    f_df = mag_resampled.join(df_part_aligned).dropna().interpolate()
    
    if settings.get('rol_mean'):
        f_df, columns_b, columns_v = apply_rolling_mean_and_get_columns(f_df, settings)

        # Extracting values more cleanly
        vx, vy, vz      = f_df[columns_v].to_numpy().T
        bx, by, bz      = f_df[columns_b].to_numpy().T
        
        # Const to normalize mag field in vel units
        f_df['np_mean'] = f_df['np'].rolling('10min', center=True).mean().interpolate()
        kinet_normal    = 1e-15 / np.sqrt(mu0 * f_df['np_mean'].values * m_p)

        # Estimate Alfvén speed and SW speed
        Va_ts           = np.vstack([bx, by, bz]) * kinet_normal
        V_ts            = np.vstack([vx, vy, vz])  # Fixed to use vy instead of repeating vx

        # Estimate fluctuations
        mean_columns_b  = [f"{col}_mean" for col in columns_b]
        mean_columns_v  = [f"{col}_mean" for col in columns_v]
 
        dva             = Va_ts - f_df[mean_columns_b].to_numpy().T * kinet_normal
        dv              = V_ts  - f_df[mean_columns_v].to_numpy().T
    
    # Calculate magnetic field magnitude and solar wind speed
    Bmag                = np.sqrt(bx**2 + by**2 + bz**2)
    Vsw                 = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Estimate Vth
    Vth, Vth_mean, Vth_median, Vth_std             = plasma.estimate_Vth(f_df.Vth.values)

    # Estimate Vsw
    Vsw, Vsw_mean, Vsw_median, Vsw_std             = plasma.estimate_Vsw(Vsw)

    # Estimate Np
    Np, Np_mean, Np_median, Np_std                 = plasma.estimate_Np(f_df.np.values)

    # Estimate di
    di, di_mean, di_median, di_std                 = plasma.estimate_di(Np)

    # Estimate beta
    beta,  temp, beta_mean, beta_median, beta_std  = plasma.estimate_beta(Bmag, Vth, Np)

    # Estimate rho_ci
    rho_ci, rho_ci_mean, rho_ci_median, rho_ci_std = plasma.estimate_rho_ci(Vth, Bmag)

    # Calculate the Alfvén speed
    alfv_speed = np.sqrt(np.sum(Va_ts**2, axis=0))
    
    # Estiamte MA
    Ma_ts      = Vsw/alfv_speed
    
    # Estimate VB angle
    vbang      = func.angle_between_vectors(Va_ts.T,V_ts.T)
    
    #End its mean
    VBangle_mean, VBangle_std = np.nanmean(vbang), np.nanstd(vbang)

    # Estimate sigmas and elssaser variable fluctuations
    signB            = calculate_signB(f_df)                         # Calculate sign of B
    Zp, Zm           = calculate_components(dv, dva, signB)          # Calculate Zp and Zm components
    sigma_r, sigma_c = calculate_energies_sigmas(Zp, Zm, dv, dva)    # Calculate energies and normalized residual energies

    
    sigs_df              = pd.DataFrame({'DateTime': f_df.index.values,
                                         'Zpr'      : Zp[0],   'Zpt'  : Zp[1],  'Zpn'    : Zp[2],
                                         'Zmr'      : Zm[0],   'Zmt'  : Zm[1],  'Zmn'    : Zm[2], 
                                         'va_r'     : dva[0],  'va_t' : dva[1], 'va_n'   : dva[2],
                                         'v_r'      : dv[0],   'v_t'  : dv[1],  'v_n'    : dv[2],
                                         'beta'     : beta,    'np'   : Np,     'Tp'     : temp,
                                         'VB'       : vbang,   'd_i'  : di,     'Ma'     : Ma_ts,
                                         'sigma_c'  : sigma_c,                  'sigma_r': sigma_r}).set_index('DateTime')
    sigs_df              = sigs_df.dropna().interpolate()

    

    #Estimate  z+, z-, v, b PSD
    if settings['estimate_psd_v']:
        # Define component keys for each signal
        components = {
            'zp': ['Zpr', 'Zpt', 'Zpn'],
            'zm': ['Zmr', 'Zmt', 'Zmn'],
            'vv': ['v_r', 'v_t', 'v_n'],
            'bb': ['va_r', 'va_t', 'va_n']
        }

        # Estimate PSD for each set of components
        results = {key: estimate_psds(sigs_df, components[key], dtb, settings) for key in components}

        # Unpack results into individual variables
        f_zp, psd_zp, psd_zp_R, psd_zp_T, psd_zp_N = results['zp']
        f_zm, psd_zm, psd_zm_R, psd_zm_T, psd_zm_N = results['zm']
        f_vv, psd_vv, psd_vv_R, psd_vv_T, psd_vv_N = results['vv']
        f_bb, psd_bb, psd_bb_R, psd_bb_T, psd_bb_N = results['bb']
        
    else:
        psd_zp_R, psd_zp_T, psd_zp_N = None, None, None
        psd_zm_R, psd_zm_T, psd_zm_N = None, None, None
        psd_bb_R, psd_bb_T, psd_bb_N = None, None, None
        psd_vv_R, psd_vv_T, psd_vv_N = None, None, None
        f_zp, psd_zp                 = None, None
        f_zm, psd_zm                 = None, None 
        f_vv, psd_vv                 = None, None
        f_bb, psd_bb                 = None, None


    # Also keep a dict containing psd_vv, psd_bb, psd_zp, psd_zm
    dict_psd = {
                "f_zpm"     : f_zp,
                "f_vb"      : f_vv,
        
                'psd_v'     : psd_vv,
                "psd_v_R"   : psd_vv_R,
                "psd_v_T"   : psd_vv_T,
                "psd_v_N"   : psd_vv_N,
        
                "psd_b"     : psd_bb,
                "psd_b_R"   : psd_bb_R,
                "psd_b_T"   : psd_bb_T,
                "psd_b_N"   : psd_bb_N,
        

                "psd_zp"    : psd_zp,
                "psd_zp_R"  : psd_zp_R,
                "psd_zp_T"  : psd_zp_T,
                "psd_zp_N"  : psd_zp_N,
        
                "psd_zm"    : psd_zm,
                "psd_zm_R"  : psd_zm_R,
                "psd_zm_T"  : psd_zm_T,
                "psd_zm_N"  : psd_zm_N,
            } if settings['estimate_psd_v'] else {}
    

    mag_dict = {
                        "B_resampled"          :  mag_resampled,
                        "PSD_f_orig"           :  psd_B,
                        "PSD_f_orig_R"         :  psd_B_R,
                        "PSD_f_orig_T"         :  psd_B_T,
                        "PSD_f_orig_N"         :  psd_B_N,
                        "f_B_orig"             :  f_B,
                        "PSD_f_smoothed"       :  psd_B_smooth,
                        "f_B_mid"              :  f_B_mid,
                        "f_B_mean"             :  f_B_mean,
                        "Fraction_missing"     :  diagnostics['Mag']["Frac_miss"],
                        "resolution"           :  diagnostics['Mag']["resol"]
        

    }
    
    part_dict =  {

                    'dict_psd'          : dict_psd,
        
                    'median_sw_speed'   : Vsw_mean,
                    'mean_sw_speed'     : Vsw_median,
                    'std_sw_speed'      : Vsw_std,
        
                    'median_alfv_speed' : np.nanmedian(alfv_speed),
                    'mean_alfv_speed'   : np.nanmean(alfv_speed),
                    'std_alfv_speed'    : np.nanstd(alfv_speed),
        
                    'beta_mean'         : beta_mean,
                    'beta_std'          : beta_std,
        
                    'sigma_r_mean'      : np.nanmean(sigma_r),
                    'sigma_r_median'    : np.nanmedian(sigma_r),
                    'sigma_r_std'       : np.nanstd(sigma_r),
        
                    'sigma_c_median'    : np.nanmedian(np.abs(sigma_c)),
                    'sigma_c_mean'      : np.nanmean(np.abs(sigma_c)),
                    'sigma_c_std'       : np.nanstd(np.abs(sigma_c)),
        
                    'Vth_mean'          : Vth_mean,
                    'Vth_std'           : Vth_std,
        
                    'Vsw_mean'          : Vsw_mean,
                    'Vsw_std'           : Vsw_std,
                    'Np_mean'           : Np_mean,
                    'Np_std'            : Np_std,
                    'di_mean'           : di_mean,
                    'di_std'            : di_std,
                    'rho_ci_mean'       : rho_ci_mean,
                    'rho_ci_std'        : rho_ci_std,
                    'VBangle_mean'      : VBangle_mean,
                    'VBangle_std'       : VBangle_std
                }


  
    return mag_dict, part_dict, sigs_df


def general_dict_func(diagnostics,
                 mag_resampled,
                 df_par,
                 dist,
                 ):
    
    
    """ Make distance dataframe to begin at the same time as magnetic field timeseries """ 
    try:
        r_psp                = np.nanmean(func.use_dates_return_elements_of_df_inbetween(mag_resampled.index[0], mag_resampled.index[-1], dist['Dist_au']))
    except:
        try:
            r_psp            = np.nanmean(func.use_dates_return_elements_of_df_inbetween(mag_resampled.index[0], mag_resampled.index[-1], df_par['Dist_au']))
        except:
            traceback.print_exc()

    if not 'qtn_flag' in diagnostics:
        diagnostics['qtn_flag'] = None
        
    if not 'part_flag' in diagnostics:
        diagnostics['part_flag'] = None
        
    general_dict = {
                        "Start_Time"           :  mag_resampled.index[0],
                        "End_Time"             :  mag_resampled.index[-1],  
                        "d"                    :  r_psp,
                        "Fraction_missing_MAG" :  diagnostics['Mag']["Frac_miss"],
                        "Fract_large_gaps"     :  diagnostics['Mag']["Large_gaps"],
                        "Resolution_MAG"       :  diagnostics['Mag']["resol"],
                        'part_flag'            :  diagnostics['part_flag'],
                        'qtn_flag'             :  diagnostics['qtn_flag']
    }

    
    return    general_dict 



