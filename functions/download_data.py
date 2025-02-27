from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import sys
import scipy.io
import os
import sys
from pathlib import Path
import pickle
import gc
from glob import glob
from datetime import datetime
import traceback
from time import sleep
import matplotlib.dates as mdates
from scipy import interpolate
import gc
from scipy.interpolate import interp1d

# Make sure to use the local spedas
sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

import logging

# Change color of info!
BG_WHITE = '\033[47m'
RESET    = '\033[0m'  # Reset the color

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
    datefmt='%Y-%m-%d %H:%M:%S',  # Format for the timestamp
)

""" Import manual functions """

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import general_functions as func
import TurbPy as turb
import polarization_analysis
import calibrate_efield as efield
import calibrate_sc_potential as sc_potential

sys.path.insert(1, os.path.join(os.getcwd(), 'functions/downloading_helpers'))
from  PSP import  LoadTimeSeriesPSP, download_ephemeris_PSP
from  SOLO import LoadTimeSeriesSOLO
from  WIND import LoadTimeSeriesWIND
from  HELIOS_A import LoadTimeSeriesHELIOS_A
from  HELIOS_B import LoadTimeSeriesHELIOS_B
from  Ulysses import LoadTimeSeriesUlysses


from scipy import constants
mu_0            = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu0             = constants.mu_0   #
m_p             = constants.m_p    # Proton mass [kg]
kb              = constants.k      # Boltzman's constant     [j/K]

def download_files( ok,
                    df,
                    settings,
                    vars_2_downnload,
                    cdf_lib_path,
                    credentials,
                    save_path):
    
    try:

        t0 = df['Start'][ok]
        t1 = df['End'][ok]

        """Setup for main function"""
        path0  = Path(save_path)
        tfmt   = "%Y-%m-%d_%H-%M-%S"
                                                             
        

        start_time  = df['Start'][ok]
        end_time    = df['End'][ok]
        
        
        # Define folder name
        foldername  = "%s_%s_sc_%d" %(str(start_time.strftime(tfmt)), str(end_time.strftime(tfmt)), 0)

        #if not os.path.exists(path0.joinpath(foldername).joinpath('final_data.pkl')):
        if (not os.path.exists(path0.joinpath(foldername)))  | (settings['overwrite_files']):
            if (not os.path.exists(path0.joinpath(foldername))):
                logging.info(BG_WHITE +'Creating new folder  %s'+ RESET, path0.joinpath(foldername))
            else:
                
                logging.info(BG_WHITE +'Overwriting folder %s'+ RESET, path0.joinpath(foldername))

                
            
              
            # Running the main function
            big_gaps_SC_pot, big_gaps, big_gaps_par, big_gaps_elec, big_gaps_qtn, flag_good, final, general, sig_c_sig_r_timeseries, dfdis, diagnostics = main_function(
                                                                                                    start_time         , 
                                                                                                    end_time           , 
                                                                                                    settings           , 
                                                                                                    vars_2_downnload   ,
                                                                                                    cdf_lib_path       ,
                                                                                                    credentials)

            try:
                final['Par']['V_resampled'] = final['Par']['V_resampled'].join(func.newindex(dfdis[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']],
                                                                                             final['Par']['V_resampled'].index))

            except:
                pass


            if flag_good == 1:
                
                
                #Create a folder to save the data
                os.makedirs(path0.joinpath(foldername), exist_ok=True)
                
                
                """ Calibrate e-field data if requested"""
                if (settings.get('E_field', {}).get('flag', False)) & (settings['in_rtn']== False) & (settings['sc']== 'PSP'):
                    
                    try:
                    
                        print('Calibrating E-field data')
                        # Estimate coefficients
                        coeffs_low_res = efield.process_data(
                                                        final['Mag']['B_resampled'].copy(),
                                                        final['Par']['V_resampled'][['Vx', 'Vy', 'Vz']].copy(), 
                                                        final['E'].copy(),
                                                        cadence_seconds      = settings.get('E_field', {}).get('cadence_seconds', 6),      # Block-averaging cadence in second
                                                        fit_interval_minutes = settings.get('E_field', {}).get('fit_interval_minutes', 4), # Length of each fitting interval in minutes
                                                        stride_minutes       = settings.get('E_field', {}).get('stride_minutes', 0.1),     # Stride length between intervals in minutes
                                                        min_correlation      = settings.get('E_field', {}).get('min_correlation', 0.8)    # Minimum acceptable cross-correlation value
                        )

                        # Use coefficients to calibrate the df                              
                        final['E'] =  efield.calibrate_data(final['E'], coeffs_low_res) 
                    except:
                        traceback.print_exc()

                """ Calibrate sc_pot data if requested"""
                if (settings.get('sc_pot', {}).get('flag', False)) & (settings['sc']== 'PSP'):
                    
                    try:

                        print('Calibrating sc potential data')
                        cal_res, roll_qtn, save_a, save_b, save_c, save_err_a, save_err_b, save_err_c, df_highfreq = sc_potential.calibrate_highfreq_in_intervals(
                                    pd.DataFrame(final['SC_pot']).copy(),
                                    pd.DataFrame(final['Par']['V_resampled']['np']).copy() ,
                                    interval_size = settings.get('sc_pot', {}).get('fit_interval_minutes', '20min'), # Length of each fitting interval in minutes,
                                    rol_med_wind  = settings.get('sc_pot', {}).get('roll_wind_minutes', '4min'),
                                    est_roll_med  = settings.get('sc_pot', {}).get('est_roll_med', False),
                                    n_sigma       = settings.get('sc_pot', {}).get('n_sigma_outliers', 3))

                       
                        final['np_sc_pot'] =  {'dens_df'     : pd.DataFrame(cal_res["sc_pot_dens"]), 
                                            'fit_params'  : { 'a'      : save_a,
                                                              'b'      : save_b,
                                                              'c'      : save_c,
                                                              'err_a'  : save_err_a,
                                                              'err_b'  : save_err_b,
                                                              'err_c'  : save_err_c}
                                            }
                         
                    except:
                        traceback.print_exc()          
                
                # In case we want smaller windows
                if settings['cut_in_small_windows']['flag']:
  
                    generated_list         = generate_intervals(func.format_date_to_str(str(final['Mag']['B_resampled'].index[0])),
                                                                func.format_date_to_str(str(final['Mag']['B_resampled'].index[-1])),
                                                                         settings['cut_in_small_windows']['flag'],
                                                                         data_path           = None,
                                                                         settings = settings['cut_in_small_windows'])
    
    
                    combined_dict = small_sub_intervals_parallel_process(
                                                      generated_list         = generated_list,
                                                      sig_c_sig_r_timeseries = sig_c_sig_r_timeseries,
                                                      final                  = final,
                                                      settings               = settings,
                                                      diagnostics            = diagnostics,
                                                      general                = general,
                                                      mu0                    = mu0,
                                                      m_p                    = m_p,
                                                      n_jobs                 = settings['cut_in_small_windows']['njobs'])

                    # Save data!
                    func.savepickle(combined_dict, str(path0.joinpath(foldername)), "overall.pkl" )
                else:
                
                    if settings['save_all']== True:
                        func.savepickle(final, str(path0.joinpath(foldername)), "final.pkl" )
                        func.savepickle(general, str(path0.joinpath(foldername)), "general.pkl" )
                        func.savepickle(sig_c_sig_r_timeseries, str(path0.joinpath(foldername)), "sig_c_sig_r.pkl" )
                        func.savepickle(big_gaps, str(path0.joinpath(foldername)), "mag_gaps.pkl" )
                        func.savepickle(big_gaps_qtn, str(path0.joinpath(foldername)), "qtn_gaps.pkl" )
                        func.savepickle(big_gaps_elec, str(path0.joinpath(foldername)), "elec_gaps.pkl" )
                        func.savepickle(big_gaps_SC_pot, str(path0.joinpath(foldername)), "sc_pot_gaps.pkl" )
                        
                        func.savepickle(big_gaps_par, str(path0.joinpath(foldername)), "par_gaps.pkl" )

                    else:
                        print('Deleting memory intensive timeseries per your commands. Only keeping derived quantities!')

                        del final["Mag"]['B_resampled'], final["Par"]['V_resampled']
                        func.savepickle(final, str(path0.joinpath(foldername)), "final.pkl" )
                        func.savepickle(general, str(path0.joinpath(foldername)), "general.pkl" )

                    gc.collect()

                    print("%d out of %d finished" %(ok, len(df)))
            else:
                os.makedirs(path0.joinpath(foldername), exist_ok=True)
                print("%s - %s failed!" %(ok, len(df)))   
    except Exception as e:
        os.makedirs(path0.joinpath(foldername), exist_ok=True)
        print('failed at index', ok, 'with error:', str(e))
        traceback.print_exc()

        
   
from copy import deepcopy
def small_sub_intervals_parallel_process(
    generated_list,
    sig_c_sig_r_timeseries,
    final,
    settings,
    diagnostics,
    general,
    mu0,
    m_p,
    n_jobs=-1,  # Use all available cores by default
    backend='loky'  # Default backend; you can choose 'multiprocessing' or others
):
    """
    Processes the generated_list in parallel and returns a combined dictionary.
    """
    

    

    # Assuming necessary modules and variables are already imported and defined:
    # func, turb, calc, polarization_analysis, generated_list, sig_c_sig_r_timeseries,
    # final, settings, diagnostics, general, mu0, m_p

    def process_jj(jj, generated_list, sig_c_sig_r_timeseries, final, phi_ts, settings, diagnostics, general, mu0, m_p):
        try:
            # Initialize the time format if needed elsewhere
            tfmt = "%Y-%m-%d_%H-%M-%S"

            # Extract start and end times
            t0_time = generated_list['Start'][jj]
            tf_time = generated_list['End'][jj]

            # Convert string times to datetime indices
            ind1 = func.string_to_datetime_index(t0_time)
            ind2 = func.string_to_datetime_index(tf_time)

            # Extract data for the sub-interval
            sig = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, sig_c_sig_r_timeseries.copy())
            Vdf = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, final['Par']['V_resampled'].copy())
            Bdf = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, final['Mag']['B_resampled'].copy())
            
            if settings['E_field']['flag']:
                Edf = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, final['E'].copy())
            else:
                Edf = None
                
            phi = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, phi_ts.copy())
            

            # Check window size
            wanted_dt = (tf_time - t0_time).total_seconds()
            ts_dt = (Bdf.index[-1] - Bdf.index[0]).total_seconds()
            dt = func.find_cadence(Bdf)

            if ts_dt > 0.95 * wanted_dt:
                # Initialize dictionaries
                sf_dict    = None
                mag_dict   = None
                dict_psd   = None
                addit_dict = None
                coh_dict   = {}

                # Structural Functions
                if settings['npt_struc_funcs']['flag']:
                    sf_dict = turb.est_5_pt_sfuncs(
                        Bdf,
                        dt,
                        func_params=settings['npt_struc_funcs']
                    )

                # Power Spectral Density (PSD)
                if settings['PSDs']['flag']:
                    mag_dict = calc.estimate_magnetic_field_psd(
                        Bdf,
                        None,
                        dt,
                        settings,
                        diagnostics,
                        return_mag_df=False
                    )

                # Estimate PSD Dictionary
                dict_psd, addit_dict = calc.estimate_psd_dict(
                    settings,
                    sig,
                    func.find_cadence(sig),
                    estimate_means=True
                )

                # Coherence Analysis
                if settings['coherence_analysis']['flag']:
                    
                    coh_dicts = {}
                    

                    Bdf, Vdf     = func.synchronize_dfs(Bdf, Vdf, True)
                    

                    # To take the alfven velocity into account
                    np_mean      = V_df['np'].rolling('30s', center=True).mean().interpolate()
                    kinet_normal = pd.DataFrame(1e-15 / np.sqrt(mu0 * np_mean * m_p))
                    _, kinet_normal = func.synchronize_dfs(B_df, kinet_normal, True)
                    kinet_normal    = kinet_normal.values


                    if ('Vr' in Vdf) and (settings['sc'] == 'PSP'):

                        sc_V         = Vdf[['Vr', 'Vt', 'Vn']] - Vdf[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].values
                        Vfin         = func.newindex(sc_V, Bdf.index) + (Bdf[['Br', 'Bt', 'Bn']].values.T * kinet_normal).T

                    elif ('Vx' in Vdf) and (settings['sc'] == 'PSP'):

                        Vfin         = Vdf[['Vx', 'Vy', 'Vz']] + (Bdf[['Bx', 'By', 'Bz']].values.T * kinet_normal).T

                    else:
                        Vfin         = Vdf
                    
                    
                    vb_ang_va = np.arccos(
                                            (Vfin.values.T[0] * Bdf.values.T[0] + 
                                             Vfin.values.T[1] * Bdf.values.T[1] + 
                                             Vfin.values.T[2] * Bdf.values.T[2]) /
                                             (np.sqrt(Vfin.values.T[0]**2 + Vfin.values.T[1]**2 + Vfin.values.T[2]**2) * 
                                              np.sqrt(Bdf.values.T[0]**2  + Bdf.values.T[1]**2  + Bdf.values.T[2]**2))
                                        )*180/np.pi



                    # Save additional coherence data
                    coh_dicts['P_spiral_mean']  = np.nanmean(phi)
                    coh_dicts['P_spiral_std']   = np.nanstd(phi)
                    coh_dicts['sig_c']          = np.nanmean(np.abs(sig['sigma_c'])) if 'sigma_c' in sig else np.nan
                    coh_dicts['sig_r']          = np.nanmean(sig['sigma_r']) if 'sigma_r' in sig else np.nan
                    coh_dicts['di']             = np.nanmedian(sig['d_i']) if 'd_i' in sig else np.nan
                    coh_dicts['Tp']             = np.nanmean(Vdf['TEMP']) if 'TEMP' in Vdf else np.nan
                    coh_dicts['Vsw_with_Va']    = np.nanmedian(np.sqrt(Vfin.values.T[0]**2 + Vfin.values.T[1]**2 + Vfin.values.T[2]**2))
                    coh_dicts['VB_wth_Va']      = np.nanmean(vb_ang_va)
                    
                    
                    for method in list(settings['coherence_analysis']['method'].keys()):
                        try:

                            for mm, field in enumerate(settings['coherence_analysis']['method'][method]):
                                #print('Working on field', field, 'with coh method', method)
                                #print(Edf)
                                # Perform coherence analysis
                                coh_dict = polarization_analysis.anisotropy_coherence(
                                    Bdf,
                                    Vfin,
                                    E_df        = Edf if field=='E' else None,
                                    method      = method,
                                    func_params = settings['coherence_analysis'],
                                    f_dict      = None if mm==0 else coh_dict
                                )
                        

                        except Exception as e:
                            traceback.print_exc()
                            coh_dict = None
                            
                        coh_dicts[method] = coh_dict
                            
                            
                        

                # Prepare the general_final dictionary
                general_final = deepcopy(general)
                if settings.get('sc') == 'PSP':
                    general_final['d'] = np.nanmedian(Vdf['Dist_au'].values)

                general_final["Start_Time"] = Bdf.index[0]
                general_final["End_Time"] = Bdf.index[-1]

                # Finalize the additional dictionary
                try:
                    if addit_dict is not None:
                        addit_dict['dict_psd'] = dict_psd
                        addit_dict['mag_dict'] = mag_dict
                    else:
                        addit_dict = {
                            'dict_psd': dict_psd,
                            'mag_dict': mag_dict
                        }
                    
                except Exception as e:
                    addit_dict = None
                    traceback.print_exc()

                # Compile the combined dictionary entry
                entry = {
                    'g'      : general_final,
                    'f'      : addit_dict,
                    'coh'    : coh_dicts,
                    'sf_dict': sf_dict
                }
                print(f"Working on jj={jj} out of N ={len(generated_list['Start'])}")

                return (jj, entry)

            else:
                # If ts_dt is not sufficient, skip processing
                return (jj, None)

        except Exception as e:
            traceback.print_exc()
            return (jj, None)

        
    phi_ts = func.calculate_parker_spiral(final['Mag']['B_resampled'].rolling('90s', center=True).mean())

    print('Working on', len(generated_list), 'intervals')
    # Parallel execution
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(process_jj)(
            jj,
            generated_list,
            sig_c_sig_r_timeseries,
            final,
            phi_ts,
            settings,
            diagnostics,
            general,
            mu0,
            m_p
        )
        for jj in range(len(generated_list))
    )

    # Assemble the combined_dict from results
    combined_dict = {jj: entry for jj, entry in results if entry is not None}

    return combined_dict
        

def main_function(start_time, end_time, settings, vars_2_downnload, cdf_lib_path, credentials):
    import traceback

    # Initialize variables
    dfqtn           = None
    df_sc_pot       = None
    mag_flag        = None
    df_elec         = None
    big_gaps_elec   = None
    big_gaps        = None
    big_gaps_qtn    = None
    big_gaps_par    = None
    big_gaps_SC_pot = None
    mis             = None
    df_e_field      = None  # needed later for PSP and SOLO
    qtn_flag        = None

    sc = settings.get('sc', None)

    # Load data based on spacecraft type
    if sc == 'PSP':
        print("Working on PSP data")
        try:
            (
                dfqtn,
                dfmag,
                dfpar,
                df_e_field,
                df_sc_pot,
                dfdis,
                big_gaps,
                big_gaps_qtn,
                big_gaps_par,
                big_gaps_SC_pot,
                misc
            ) = LoadTimeSeriesPSP(
                start_time,
                end_time,
                settings,
                vars_2_downnload,
                cdf_lib_path,
                credentials    = credentials,
                time_amount    = settings['addit_time_around'],
                time_unit      = 'h'
            )
        except Exception:
            traceback.print_exc()

        # Delete "na" entry in dfpar if it exists
        try:
            if dfpar is not None:
                del dfpar['na']
        except Exception:
            print("No na")

    elif sc == 'SOLO':
        print("Working on SOLO data")
        try:
            (
                dfmag,
                mag_flag,
                dfpar,
                dfdis,
                big_gaps,
                big_gaps_qtn,
                big_gaps_par,
                misc
            ) = LoadTimeSeriesSOLO(
                start_time,
                end_time,
                settings,
                vars_2_downnload,
                cdf_lib_path,
                credentials   = credentials,
                time_amount   = settings['addit_time_around'],
                time_unit     = 'h'
            )
            if settings.get('SOLO_use_burst') and (mag_flag != 'Burst'):
                dfmag, dfapr = None, None  # dfapr is set to None as in the original code
                print("We dont have Burst Data, thus we wont consider this interval")
        except Exception:
            traceback.print_exc()

    elif sc == 'HELIOS_A':
        print("Working on HELIOS_A data")
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesHELIOS_A(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()

    elif sc == 'HELIOS_B':
        print("Working on HELIOS_B data")
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesHELIOS_B(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()

    elif sc == 'WIND':
        print("Working on WIND data")
        try:
            (
                dfmag,
                dfpar,
                df_elec,
                dfdis,
                big_gaps,
                big_gaps_qtn,
                big_gaps_par,
                big_gaps_elec,
                misc,
                qtn_flag
            ) = LoadTimeSeriesWIND(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()

    elif sc == 'Ulysses':
        print("Working on Ulysses data")
        try:
            (dfmag, dfpar, dfdis, big_gaps, misc) = LoadTimeSeriesUlysses(start_time, end_time, settings)
        except Exception:
            traceback.print_exc()

    # Process and validate particle data
    if dfpar is not None:
        # Check if the fraction of missing particles is acceptable
        if misc.get('Par', {}).get('Frac_miss', 1) < settings.get('Max_par_missing', 0):
            try:

         
                # Synchronize the df's
                if settings.get('upsample_low_freq_ts'):
                    B_df, V_df                       = func.synchronize_dfs(dfmag, dfpar, True)
                else:
                    B_df_low, V_df                    = func.synchronize_dfs(dfmag, dfpar, False)
                    B_df    , _                       = func.synchronize_dfs(dfmag, B_df_low, True)
                    
                # Compute general dictionary
                try:
                    general_dict = calc.general_dict_func(misc, dfmag.copy(), dfpar.copy(), dfdis)
                    
                    if sc == 'WIND':
                        general_dict['qtn_flag'] = qtn_flag

                    if settings.get('estimate_derived_param'):
                        part_dict, sig_c_sig_r_timeseries = calc.calculate_diagnostics(
                            B_df if settings.get('upsample_low_freq_ts') else  B_df_low, 
                            V_df, 
                            dfdis,
                            settings, 
                            misc)
                        
                        # Estimate the Power Spectral Density (PSD) of the magnetic field and optionally smooth it
                        mag_dict = calc.estimate_magnetic_field_psd(
                                                                    B_df,
                                                                    func.find_cadence(B_df),
                                                                    settings,
                                                                    misc)
                    else:
                        mag_dict, part_dict, sig_c_sig_r_timeseries = {}, {}, {}


                    # Keep final timeseries
                    part_dict['V_resampled']          = V_df
                    mag_dict['B_resampled']           = B_df
                    
                except Exception:
                    traceback.print_exc()


                # Build final output dictionary
                final_dict = {
                    "Mag"      : mag_dict,
                    "Par"      : part_dict,
                    "Elec"     : df_elec,
                    "Mag_flag" : mag_flag,
                }

                # Update general_dict with particle information
                general_dict["Fraction_missing_part"] = misc['Par']['Frac_miss']
                general_dict["Resolution_part"]       = misc['Par']["resol"]

                try:
                    general_dict["Fraction_missing_elec"] = misc['Elec']['Frac_miss']
                    general_dict["Resolution_elec"]       = misc['Elec']["resol"]
                except Exception:
                    general_dict["Fraction_missing_elec"] = None
                    general_dict["Resolution_elec"]       = None

                if sc in ('PSP', 'SOLO'):
                    try:
                        general_dict["Fraction_missing_qtn"] = misc['QTN']['Frac_miss']
                        general_dict["Resolution_qtn"]       = misc['QTN']["resol"]
                    except Exception:
                        general_dict["Fraction_missing_qtn"] = None
                        general_dict["Resolution_qtn"]       = None

                    try:
                        final_dict['E'] = df_e_field
                        general_dict["Fraction_missing_E"]    = misc['E']['Frac_miss']
                        general_dict["Resolution_E"]          = misc['E']["resol"]
                    except Exception:
                        general_dict["Fraction_missing_E"]    = None
                        general_dict["Resolution_E"]          = None

                    try:
                        final_dict['SC_pot']                    = df_sc_pot
                        general_dict["Fraction_missing_SC_pot"] = misc['SC_pot']['Frac_miss']
                        general_dict["Resolution_SC_pot"]       = misc['SC_pot']["resol"]
                    except Exception:
                        general_dict["Fraction_missing_SC_pot"] = None
                        general_dict["Resolution_SC_pot"]       = None

                general_dict["sc"] = sc
                flag_good          = 1

            except Exception:
                traceback.print_exc()
                print("No MAG data!")
                flag_good = 0
                big_gaps = None
                final_dict = None
                general_dict = None
                sig_c_sig_r_timeseries = None
        else:
            traceback.print_exc()
            print("No particle data!")
            final_dict = None
            flag_good = 0
            big_gaps = None
            general_dict = None
            sig_c_sig_r_timeseries = None
    else:
        traceback.print_exc()
        print("No particle data!")
        final_dict = None
        flag_good = 0
        big_gaps = None
        general_dict = None
        sig_c_sig_r_timeseries = None

    return (
        big_gaps_SC_pot,
        big_gaps,
        big_gaps_par,
        big_gaps_elec,
        big_gaps_qtn,
        flag_good,
        final_dict,
        general_dict,
        sig_c_sig_r_timeseries,
        dfdis,
        misc
    )



        
import datetime
import logging

def generate_intervals(start_time_str,
                       end_time_str,
                       generate_1_interval,
                       data_path = None, 
                       settings  = None):
    """
    Generates time intervals between start_time and end_time.

    Parameters:
    - start_time_str (str): Start time in the format '%Y-%m-%d %H:%M'.
    - end_time_str (str): End time in the format '%Y-%m-%d %H:%M'.
    - generate_1_interval (bool): Flag to generate a single interval or multiple.
    - data_path (str): Path to the data (unused in current function).
    - settings (dict): Dictionary containing 'Step' and 'duration' keys.

    Returns:
    - pd.DataFrame: DataFrame with 'Start' and 'End' columns for each interval.
    """
    
    # Parse the start and end times
    try:
        start_date = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
        end_date   = datetime.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
    except ValueError as ve:
        logging.error("Incorrect time format. Please use 'YYYY-MM-DD HH:MM'. Error: %s", ve)
        raise

    if not generate_1_interval:
        # Generate only one interval
        start_times = [start_date]
        end_times   = [end_date]
        logging.info("Generating only one interval based on the provided start and end times.")
    else:
        # Validate 'Step' and 'duration' in settings
        step     = settings.get('Step')
        duration = settings.get('duration')
        
        if not step or not duration:
            logging.error("'Step' and 'duration' must be specified in settings.")
            raise ValueError("'Step' and 'duration' must be specified in settings.")

        # Generate the start times using the specified frequency
        start_times = pd.date_range(start=start_date, end=end_date, freq=step)
        
        # Convert 'duration' to a Timedelta
        try:
            delta = pd.to_timedelta(duration)
        except ValueError as ve:
            logging.error("Invalid duration format: %s. Error: %s", duration, ve)
            raise
        
        # Generate end times by adding the duration to each start time
        end_times = start_times + delta

        # Ensure that end_times do not exceed the specified end_date
        valid_indices = end_times <= end_date
        start_times  = start_times[valid_indices]
        end_times    = end_times[valid_indices]
        
        logging.info("Generated intervals with Step: %s and Duration: %s", step, duration)
        logging.info("Number of Intervals: %d", len(start_times))

    # Create a DataFrame
    df = pd.DataFrame({'Start': start_times, 'End': end_times})

    # Log the start and end times
    logging.info("Start Time: %s", start_date)
    logging.info("End Time: %s", end_date)

    if not generate_1_interval:
        logging.info("Considering a single interval spanning: %s to %s", start_date, end_date)
    else:
        logging.debug("Intervals DataFrame:\n%s", df)

    return df
