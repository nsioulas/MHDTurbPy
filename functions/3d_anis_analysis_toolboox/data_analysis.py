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


""" Import manual functions """

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func
import Figures as figs
from   SEA import SEA
import three_D_funcs as threeD

sys.path.insert(1, os.path.join(os.getcwd(), 'functions', 'downloading_helpers'))
from PSP import  download_ephemeris_PSP








def compute_sf_overall(tau_value, theta_arrs, phi_arrs, qorder, B, V, Np, dt, return_unit_vecs, five_points_sfuncs, estimate_alignment_angle, return_mag_align_correl):
    results = {}

    dB, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm = threeD.local_structure_function(
                                                                                                                                    B,
                                                                                                                                    V,
                                                                                                                                    Np,
                                                                                                                                    int(tau_value),
                                                                                                                                    dt,
                                                                                                                                    return_unit_vecs=return_unit_vecs,
                                                                                                                                    five_points_sfunc=five_points_sfuncs,
                                                                                                                                    estimate_alignment_angle=estimate_alignment_angle,
                                                                                                                                    return_mag_align_correl=return_mag_align_correl,
                                                                                                                                )

    qorder = np.array([2])
    for mm in range(1, len(theta_arrs)):
        for nn in range(1, len(phi_arrs)):
            indices = np.where(((VBangle > theta_arrs[mm - 1]) & (VBangle <= theta_arrs[mm])) & (
                        (Phiangle > phi_arrs[nn - 1]) & (Phiangle <= phi_arrs[nn])))[0]

            sf_overall = threeD.structure_functions_3D(indices, qorder, dB)
            tag = 'theta_' + str(theta_arrs[mm - 1]) + '_' + str(theta_arrs[mm]) + '_phi_' + str(phi_arrs[nn - 1]) + '_' + str(phi_arrs[nn])
            results[tag] = sf_overall

    return results



def five_pt_two_pt_wavelet_analysis(i,
                                    fnames,
                                    credentials,
                                    conditions,
                                    return_flucs,
                                    consider_Vsc,
                                    Estimate_5point,
                                    keep_wave_coeefs,
                                    strict_thresh,
                                    max_hours,
                                    qorder,
                                    estimate_alignment_angle,
                                    return_mag_align_correl,
                                    only_general,
                                    phi_thresh_gen,
                                    theta_thresh_gen,
                                    sc                       ='PSP', 
                                    extra_conditions         = False,
                                    ts_list                  = None,
                                    overwrite_existing_files = False,
                                    thetas_phis_step         = 10, 
                                    return_B_in_vel_units    = False, 
                                    max_interval_dur         =  240,
                                    estimate_dzp_dzm          = False,
                                    use_low_resol_data        = False, 
                                    use_local_polarity       = True,
                                    dt_step                  = 0.25 ):
    
    import warnings
    
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    
    try:
    
        # Print progress
        func.progress_bar(i, len(fnames))

        # Load files
        res = pd.read_pickle(fnames[i])
        
        gen_name = fnames[i].replace('final.pkl', 'general.pkl')
        gen      = pd.read_pickle(gen_name)
        
        # Duration of interval in hours
        dts = (gen['End_Time']- gen['Start_Time']).total_seconds()/3600
        
        if dts<max_interval_dur:
            print('Considering an interval of duration:', dts,'[hrs]')
                    
            if strict_thresh==1:
                strict_suffix = '5deg_'
            elif strict_thresh==2:
                strict_suffix = '2deg_'
            else:
                strict_suffix = ''


            if extra_conditions:
                conditions_suffix = 'extra_conditions_'
            else:
                conditions_suffix = ''

            if Estimate_5point:

                npoints_suffix = '5pt_'
            else:
                npoints_suffix = '2pt_'

            if return_flucs:
                sfuncs_suffix = ''
            else:
                sfuncs_suffix = 'sfuncs_estimated'


            if only_general==1:
                general_suffix = 'general_SF_'
            else:
                general_suffix = ''

            if consider_Vsc:
                vsc_suffix = 'Vsc_removed_'
            else:
                vsc_suffix = ''  


            if only_general==1:
                fname          = f"{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}theta_{theta_thresh_gen}_phi_{phi_thresh_gen}.pkl"
                align_name     = f"{npoints_suffix}{vsc_suffix}.pkl"
            elif only_general==2:
                fname          = f"_all_bins_{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}_step_{str(thetas_phis_step)}.pkl"
                
            else:  
                fname          = f"{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}_final.pkl"
            
                
            # Check wether file alredy exists
            check_file     = str(Path(gen_name.replace('general.pkl', '')).joinpath('final_vs8').joinpath(fname))
                                 
            # Now work on data!
            if (not os.path.exists(check_file)) or (overwrite_existing_files):

                                 
                if (overwrite_existing_files) & (os.path.exists(check_file)):
                    print('Overwriting', check_file ,'per your commands ')
                else:
                    print('Working on new file: ', check_file)

                # Choose V, B dataframes
                if use_low_resol_data:
                    try:
                        B  = res['Mag']['B_resampled_part_res'][['Br', 'Bt', 'Bn']]
                    except:
                        B  = res['Mag']['B_resampled'][['Br', 'Bt', 'Bn']]
                else: 
                    B  = res['Mag']['B_resampled'][['Br', 'Bt', 'Bn']]
                V  = res['Par']['V_resampled'][['Vr', 'Vt', 'Vn']]
                Np = res['Par']['V_resampled'][['np']]


                Np = Np[~Np.index.duplicated()]

                V  = V[~V.index.duplicated()]
                B  = B[~B.index.duplicated()]
                
                
                # Reindex V timeseries
                try:
                    V   = func.newindex(V, B.index)
                except:
                    print(func.find_cadence(V))

                try:
                    Np = func.newindex(Np, B.index)
                except:
                    print(Np.index[0])
                    print(V)
                    print(func.find_cadence(Np))

                if (consider_Vsc) & (sc=='PSP'):
                    # download ephemeris data
                    ephem                 = download_ephemeris_PSP(str(gen['Start_Time']-pd.Timedelta('10min')), 
                                                                   str(gen['End_Time']  +pd.Timedelta('10min')),
                                                                   credentials,
                                                                   ['position','velocity']
                                                                  )

                    # Only keep data needed
                    ephem                 = ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']]

                    # Reindex them to V df index
                    ephem                 = func.newindex(ephem, V.index)

                    # Subract

                    V_sc                  = ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].interpolate()#.values

                    # Keep Vsw to normalize
                    Vsw_norm              = np.nanmean(np.sqrt((V['Vr']-ephem.values.T[0])**2 + (V['Vt']-ephem.values.T[1])**2 + (V['Vn']-ephem.values.T[2])**2))
                else:
                    V_sc                  = 0*res['Par']['V_resampled'][['Vr', 'Vt', 'Vn']]#.values 
                    Vsw_norm              = np.nanmean(np.sqrt(res['Par']['V_resampled']['Vr']**2 + res['Par']['V_resampled']['Vt']**2 + res['Par']['V_resampled']['Vn']**2))

                if sc =='WIND':
                    fix_sign =False
                else:
                    fix_sign =True           

                # Vsw Mean, di mean
                di  = res['Par']['di_mean']
                Vsw = res['Par']['Vsw_mean']


                # Estimate lags
                dt  = func.find_cadence(B)

                # Try CWT method
                if keep_wave_coeefs:

                    Wx, Wy, Wz, freqs, PSD,  scales = turb.trace_PSD_wavelet(B.Br.values,
                                                                                          B.Bt.values,
                                                                                          B.Bn.values, 
                                                                                          dt,
                                                                                          dj_wave)
                    phys_scales =  1 / (freqs)

                    # Create dictionary
                    keep_wave_coeff = {'Wx': Wx, 'Wy': Wy, 'Wz': Wz, 'freqs':freqs, 'PSD':PSD, 'phys_scales':phys_scales, 'scales': scales, 'di':di, 'Vsw_RTN':Vsw, 'Vsw_minus_Vsc':Vsw_norm, 'dt':dt }

                    del Wx, Wy, Wz

                    # Save dictionary
                    if strict_thresh:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs_5deg.pkl')               
                    else:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs.pkl')

                    del keep_wave_coeff
                else:
                    max_lag     = int((max_hours*3600)/dt)
                    tau_values  = 2**np.arange(0, 1000, dt_step)
                    max_ind     = (tau_values<max_lag) & (tau_values>0)
                    phys_scales = np.unique(tau_values[max_ind].astype(int))
                    
                    
                    
                    
                if sc =='WIND':
                    use_np_factor            = 1.16
                else:
                    use_np_factor            = 1
                    

                # Create an empty list to store the final results
                _,_, _, _, thetas, phis, flucts, ell_di, Sfunctions, PDFs, overall_align_angles = threeD.estimate_3D_sfuncs(B,
                                                                                                                     V,
                                                                                                                     V_sc, 
                                                                                                                     Np,
                                                                                                                     dt,
                                                                                                                    # Vsw_norm, 
                                                                                                                     di, 
                                                                                                                     conditions,
                                                                                                                     qorder, 
                                                                                                                     phys_scales, 
                                                                                                                     estimate_PDFS            = False,
                                                                                                                     return_unit_vecs         = False,
                                                                                                                     five_points_sfuncs       = Estimate_5point,
                                                                                                                     estimate_alignment_angle = estimate_alignment_angle,
                                                                                                                     return_mag_align_correl  = return_mag_align_correl,
                                                                                                                     return_coefs             = return_flucs,
                                                                                                                     only_general             = only_general,
                                                                                                                     theta_thresh_gen         = theta_thresh_gen,
                                                                                                                     phi_thresh_gen           = phi_thresh_gen,
                                                                                                                     extra_conditions         = extra_conditions,
                                                                                                                     fix_sign                 = fix_sign,
                                                                                                                     ts_list                  = ts_list,
                                                                                                                     thetas_phis_step         = thetas_phis_step, 
                                                                                                                     return_B_in_vel_units    = return_B_in_vel_units,
                                                                                                                     estimate_dzp_dzm         = estimate_dzp_dzm,
                                                                                                                     use_np_factor            = use_np_factor

                                                                                                                    )


                keep_sfuncs_final = {'di':di, 'Vsw':Vsw, 'Vsw_norm': Vsw_norm, 'ell_di':ell_di,'Sfuncs':Sfunctions, 'flucts':flucts}

                #Now save file
                func.savepickle(keep_sfuncs_final, str(Path(gen_name.replace('general.pkl', '')).joinpath('final_vs8')), fname)

                # also Save alignment abnles
                if  estimate_alignment_angle:
                    align_name   = f"alignment_angles_{npoints_suffix}{vsc_suffix}.pkl"
                    func.savepickle(overall_align_angles, str(Path(gen_name.replace('general.pkl', '')).joinpath('final_vs8')), align_name)


                # Save some space for ram
                del keep_sfuncs_final

                if keep_wave_coeefs:

                    # We want to prevent saving tons of data (phis, thetas). For this reason we will load the 'wave_coeffs.pkl' file again.
                    # We will once again use the conditionals for the angles. Keep the wavelet coefficients we want for each category
                    # Then we will overwrite the previous version of 'wave_coeffs.pkl'

                    # First load
                    if strict_thresh:
                        wave_coeffs =  pd.read_pickle(str(os.path.join(fnames[i][:-9], 'wave_coeffs_5deg.pkl') ))                
                    else:
                        wave_coeffs =  pd.read_pickle(str(os.path.join(fnames[i][:-9], 'wave_coeffs.pkl') ))

                    ell_perp_dict, Ell_perp_dict, ell_par_dict, ell_par_rest_dict = collect_wave_coeffs.keep_conditioned_coeffs(
                                                                                                                                np.hstack(list(phis.values())),
                                                                                                                                np.hstack(list(thetas.values())),
                                                                                                                                wave_coeffs,
                                                                                                                                conditions)
                    keep_wave_coeff = {'ell_perp'           : ell_perp_dict,
                                       'Ell_perp'           : Ell_perp_dict,
                                       'ell_par'            : ell_par_dict,
                                       'ell_par_rest_dict'  : ell_par_rest_dict,
                                       'freqs'              : freqs,
                                       'PSD'                : PSD,  
                                       'di'                 : di,
                                       'Vsw_RTN'            : Vsw, 
                                       'Vsw_minus_Vsc'      : Vsw_norm,
                                       'dt'                 : dt }
                    # Save dictionary
                    if strict_thresh:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs_5deg.pkl')               
                    else:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs.pkl')

                    del thetas, phis, wave_coeffs, ell_perp_dict, Ell_perp_dict, ell_par_dict, ell_par_rest_dict

                # Check if it's time to trigger garbage collection
                if i % 2 == 0:
                    gc.collect()
            else:
                if (os.path.exists(check_file)):
                    print('Omitted file because it already exists, and overwrite option is False !')
    except:
        traceback.print_exc()
        pass
    
    
def five_pt_two_pt_wavelet_analysis_E1_only(i,
                                    fnames,
                                    scam_names,
                                    credentials,
                                    conditions,
                                    gen_names,
                                    return_flucs,
                                    consider_Vsc,
                                    only_E1,
                                    Estimate_5point,
                                    keep_wave_coeefs,
                                    strict_thresh,
                                    max_hours,
                                    qorder,
                                    estimate_alignment_angle,
                                    return_mag_align_correl,
                                    only_general,
                                    phi_thresh_gen,
                                    theta_thresh_gen,
                                    sc                       ='PSP', 
                                    extra_conditions         = False,
                                    ts_list                  = None,
                                    overwrite_existing_files = False,
                                    thetas_phis_step         = 10, 
                                    return_B_in_vel_units    = False, 
                                    max_interval_dur         =  240,
                                   estimate_dzp_dzm          = False):
    
    import warnings
    
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    
    try:
    
        # Print progress
        func.progress_bar(i, len(fnames))

        # Load files
        res  = pd.read_pickle(fnames[i])
        gen  = pd.read_pickle(gen_names[i])
        scam = pd.read_pickle(scam_names[i])
        
        # Duration of interval in hours
        dts = (gen['End_Time']- gen['Start_Time']).total_seconds()/3600
        
        if dts<max_interval_dur:
            print('Considering an interval of duration:', dts,'[hrs]')
                    
            if strict_thresh==1:
                strict_suffix = '5deg_'
            elif strict_thresh==2:
                strict_suffix = '2deg_'
            else:
                strict_suffix = ''


            if extra_conditions:
                conditions_suffix = 'extra_conditions_'
            else:
                conditions_suffix = ''

            if Estimate_5point:

                npoints_suffix = '5pt_'
            else:
                npoints_suffix = '2pt_'

            if return_flucs:
                sfuncs_suffix = ''
            else:
                sfuncs_suffix = 'sfuncs_estimated'


            if only_general==1:
                general_suffix = 'general_SF_'
            else:
                general_suffix = ''

            if consider_Vsc:
                vsc_suffix = 'Vsc_removed_'
            else:
                vsc_suffix = ''  


            if only_general==1:
                fname          = f"{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}theta_{theta_thresh_gen}_phi_{phi_thresh_gen}.pkl"
                align_name     = f"{npoints_suffix}{vsc_suffix}.pkl"
            elif only_general==2:
                fname          = f"_all_bins_{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}_step_{str(thetas_phis_step)}.pkl"
                
            else:  
                fname          = f"{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}_final.pkl"
            
                
            # Check wether file alredy exists
            check_file     = str(Path(gen_names[i][:-11]).joinpath('final').joinpath(fname))
                                 
            # Now work on data!
            if (not os.path.exists(check_file)) or (overwrite_existing_files):

                                 
                if (overwrite_existing_files) & (os.path.exists(check_file)):
                    print('Overwriting', check_file ,'per your commands ')
                else:
                    print('Working on new file: ', check_file)

                # Choose V, B dataframes
                B  = scam["resampled_df"][['Br', 'Bt', 'Bn']]
                V  = res['Par']['V_resampled']()[['Vr', 'Vt', 'Vn']]
                Np = res['Par']['V_resampled']()[['np']]


                Np = Np[~Np.index.duplicated()]

                V  = V[~V.index.duplicated()]
                B  = B[~B.index.duplicated()]
                try:
                    Np = func.newindex(Np, B.index)
                except:
                    print(Np.index[0])
                    print(V)
                    print(func.find_cadence(Np))

                if (consider_Vsc) & (sc=='PSP'):
                    # download ephemeris data
                    ephem                 = download_ephemeris_PSP(str(gen['Start_Time']-pd.Timedelta('10min')), 
                                                                   str(gen['End_Time']  +pd.Timedelta('10min')),
                                                                   credentials,
                                                                   ['position','velocity']
                                                                  )

                    # Only keep data needed
                    ephem                 = ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']]

                    # Reindex them to V df index
                    ephem                 = func.newindex(ephem, V.index)

                    # Subract
                    V[['Vr', 'Vt', 'Vn']] = res['Par']['V_resampled'][['Vr', 'Vt', 'Vn']]().values - ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].interpolate().values

                    # Keep Vsw to normalize
                    Vsw_norm              = np.nanmean(np.sqrt(V['Vr']**2 + V['Vt']**2 + V['Vn']**2))
                else:
                    Vsw_norm              = np.nanmean(np.sqrt(res['Par']['V_resampled']()['Vr']**2 + res['Par']['V_resampled']()['Vt']**2 + res['Par']['V_resampled']()['Vn']**2))

                if sc =='WIND':
                    fix_sign =False
                else:
                    fix_sign =True           

                # Vsw Mean, di mean
                di  = res['Par']['di_mean']
                Vsw = res['Par']['Vsw_mean']

                # Reindex V timeseries
                try:
                    V   = func.newindex(V, B.index)
                except:
                    print(func.find_cadence(V))

                # Estimate lags
                dt  = func.find_cadence(B)

                # Try CWT method
                if keep_wave_coeefs:

                    Wx, Wy, Wz, freqs, PSD,  scales = turb.trace_PSD_wavelet(B.Br.values,
                                                                                          B.Bt.values,
                                                                                          B.Bn.values, 
                                                                                          dt,
                                                                                          dj_wave)
                    phys_scales =  1 / (freqs)

                    # Create dictionary
                    keep_wave_coeff = {'Wx': Wx, 'Wy': Wy, 'Wz': Wz, 'freqs':freqs, 'PSD':PSD, 'phys_scales':phys_scales, 'scales': scales, 'di':di, 'Vsw_RTN':Vsw, 'Vsw_minus_Vsc':Vsw_norm, 'dt':dt }

                    del Wx, Wy, Wz

                    # Save dictionary
                    if strict_thresh:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs_5deg.pkl')               
                    else:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs.pkl')

                    del keep_wave_coeff
                else:
                    max_lag     = int((max_hours*3600)/dt)
                    tau_values  = 1.2**np.arange(0, 1000)
                    max_ind     = (tau_values<max_lag) & (tau_values>0)
                    phys_scales = np.unique(tau_values[max_ind].astype(int))


                # Create an empty list to store the final results
                l_mag, l_lambda, l_xi, l_ell, thetas, phis, flucts, ell_di, Sfunctions, PDFs, overall_align_angles = threeD.estimate_3D_sfuncs(B,
                                                                                                                 V, 
                                                                                                                 Np,
                                                                                                                 dt,
                                                                                                                 Vsw_norm, 
                                                                                                                 di, 
                                                                                                                 conditions,
                                                                                                                 qorder, 
                                                                                                                 phys_scales, 
                                                                                                                 estimate_PDFS            = False,
                                                                                                                 return_unit_vecs         = False,
                                                                                                                 five_points_sfuncs       = Estimate_5point,
                                                                                                                 estimate_alignment_angle = estimate_alignment_angle,
                                                                                                                 return_mag_align_correl  = return_mag_align_correl,
                                                                                                                 return_coefs             = return_flucs,
                                                                                                                 only_general             = only_general,
                                                                                                                 theta_thresh_gen         = theta_thresh_gen,
                                                                                                                 phi_thresh_gen           = phi_thresh_gen,
                                                                                                                 extra_conditions         = extra_conditions,
                                                                                                                 fix_sign                 = fix_sign,
                                                                                                                 ts_list                  = ts_list,
                                                                                                                 thetas_phis_step         = thetas_phis_step, 
                                                                                                                 return_B_in_vel_units    = return_B_in_vel_units,
                                                                                                                 estimate_dzp_dzm          = estimate_dzp_dzm
                                                                                                                 
                                                                                                            )


                keep_sfuncs_final = {'di':di, 'Vsw':Vsw, 'Vsw_norm': Vsw_norm, 'ell_di':ell_di,'Sfuncs':Sfunctions, 'flucts':flucts}

                #Now save file
                func.savepickle(keep_sfuncs_final, str(Path(gen_names[i][:-11]).joinpath('final')), fname)

                # also Save alignment abnles
                if  estimate_alignment_angle:
                    align_name   = f"alignment_angles_{npoints_suffix}{vsc_suffix}.pkl"
                    func.savepickle(overall_align_angles, str(Path(gen_names[i][:-11]).joinpath('final')), align_name)


                # Save some space for ram
                del keep_sfuncs_final

                if keep_wave_coeefs:

                    # We want to prevent saving tons of data (phis, thetas). For this reason we will load the 'wave_coeffs.pkl' file again.
                    # We will once again use the conditionals for the angles. Keep the wavelet coefficients we want for each category
                    # Then we will overwrite the previous version of 'wave_coeffs.pkl'

                    # First load
                    if strict_thresh:
                        wave_coeffs =  pd.read_pickle(str(os.path.join(fnames[i][:-9], 'wave_coeffs_5deg.pkl') ))                
                    else:
                        wave_coeffs =  pd.read_pickle(str(os.path.join(fnames[i][:-9], 'wave_coeffs.pkl') ))

                    ell_perp_dict, Ell_perp_dict, ell_par_dict, ell_par_rest_dict = collect_wave_coeffs.keep_conditioned_coeffs(
                                                                                                                                np.hstack(list(phis.values())),
                                                                                                                                np.hstack(list(thetas.values())),
                                                                                                                                wave_coeffs,
                                                                                                                                conditions)
                    keep_wave_coeff = {'ell_perp'           : ell_perp_dict,
                                       'Ell_perp'           : Ell_perp_dict,
                                       'ell_par'            : ell_par_dict,
                                       'ell_par_rest_dict'  : ell_par_rest_dict,
                                       'freqs'              : freqs,
                                       'PSD'                : PSD,  
                                       'di'                 : di,
                                       'Vsw_RTN'            : Vsw, 
                                       'Vsw_minus_Vsc'      : Vsw_norm,
                                       'dt'                 : dt }
                    # Save dictionary
                    if strict_thresh:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs_5deg.pkl')               
                    else:
                        func.savepickle(keep_wave_coeff, fnames[i][:-9], 'wave_coeffs.pkl')

                    del thetas, phis, wave_coeffs, ell_perp_dict, Ell_perp_dict, ell_par_dict, ell_par_rest_dict

                # Check if it's time to trigger garbage collection
                if i % 2 == 0:
                    gc.collect()
            else:
                if (os.path.exists(check_file)):
                    print('Omitted file because it already exists, and overwrite option is False !')
    except:
        traceback.print_exc()
        pass


from numba import prange, jit
def digitize_flucs(xvalues, yvalues, what, std_or_std_mean, nbins, loglog):

    xpar = np.array(xvalues)
    ypar = np.array(yvalues)
    ind = (xpar > -1e9) & (ypar > -1e9)

    xpar = xpar[ind]
    ypar = ypar[ind]

    if loglog:
        bins = np.logspace(np.log10(np.nanmin(xpar)), np.log10(np.nanmax(xpar)), nbins)
    else:
        bins = np.linspace(np.nanmin(xpar), np.nanmax(xpar), nbins)

    dig_indices = np.digitize(xpar, bins)

    bin_counts = np.bincount(dig_indices)

    return xpar, ypar, dig_indices, bin_counts

@jit(nopython=True, parallel=True)
def est_quants(kk, ypar, dig_indices, qorders, bin_count):
    yvals  = np.zeros(len(qorders))
    ervals = np.zeros( len(qorders))

    for jj in prange(len(qorders)):
        yvals[jj]   = np.nanmean(ypar[np.where(dig_indices==kk)[0]]**qorders[jj])
        ervals[jj]  = np.nanstd(ypar[np.where(dig_indices==kk)[0]]**qorders[jj])/np.sqrt(bin_count)

    return yvals, ervals


@jit(nopython=True)
def sfunc_statistics(xpar, ypar, dig_indices, bin_counts, nbins, qorders):

    xvals  = []
    yvals  = np.zeros((nbins, len(qorders)))
    ervals = np.zeros((nbins, len(qorders)))

    for kk in prange(nbins):

        xvals.append(np.nanmean(xpar[np.where(dig_indices==kk)[0]]))
        yvalues, ervalues = est_quants(kk, ypar, dig_indices, qorders, bin_counts[kk])

        yvals[kk, :]  = yvalues
        ervals[kk, :] = ervalues

    return xvals, yvals, ervals


def process_data( data_file, data_file_gen, data_file_fin, data_file_align, max_d, min_d, qorders, general_sf):
    
    res       = pd.read_pickle(data_file)
    res_gen   = pd.read_pickle(data_file_gen)
    res_fin   = pd.read_pickle(data_file_fin)
    res_align = pd.read_pickle(data_file_align)

    # Define parameters
    Vsw      = res['Vsw']
    di       = res['di']
    sigma_c  = res_fin['Par']['sigma_c_mean']
    sigma_r  = res_fin['Par']['sigma_r_mean']
    VB_sin   = res_align['VB']['reg']
    B_mean   = np.nanmean(np.linalg.norm(res_fin["Mag"]["B_resampled"].values, axis=1)) ** 2
    duration = (pd.to_datetime(res_fin['Mag']['B_resampled'].index)[-1] - pd.to_datetime(res_fin['Mag']['B_resampled'].index)[0]) / np.timedelta64(1, 's')



    keep_sfunc_ell_perp             = np.nan*np.ones((len(qorders), len(res['flucts']['ell_perp']['T'])))
    keep_sfunc_Ell_perpendicular    = np.nan*np.ones((len(qorders), len(res['flucts']['Ell_perp']['T'])))
    keep_sfunc_ell_par              = np.nan*np.ones((len(qorders), len(res['flucts']['ell_par']['T'])))
    keep_sfunc_ell_par_rest         = np.nan*np.ones((len(qorders), len(res['flucts']['ell_par_rest']['T'])))
    
    keep_sdk_ell_perp               = np.nan*np.ones(len(res['flucts']['ell_perp']['T']))
    keep_sdk_Ell_perpendicular      = np.nan*np.ones(len(res['flucts']['Ell_perp']['T']))
    keep_sdk_ell_par                = np.nan*np.ones(len(res['flucts']['ell_par']['T']))
    keep_sdk_ell_par_rest           = np.nan*np.ones(len(res['flucts']['ell_par_rest']['T']))

    keep_sfunc_ell_perp_di          = np.nan*np.ones(len(res['flucts']['ell_perp']['T']))
    keep_sfunc_Ell_perpendicular_di = np.nan*np.ones(len(res['flucts']['Ell_perp']['T']))
    keep_sfunc_ell_par_di           = np.nan*np.ones(len(res['flucts']['ell_par']['T']))
    keep_sfunc_ell_par_rest_di      = np.nan*np.ones(len(res['flucts']['ell_par_rest']['T']))
    
    keep_align                      = []
    
    for j, ell_di in enumerate(res['ell_di']):
        for qorder in qorders:
            if general_sf:
                final_dict_slow['flucts_ell_perp_R'][slow_ind] = res['flucts']['ell_perp']['R'][str(j)] 
                final_dict_slow['flucts_ell_perp_T'][slow_ind] = res['flucts']['ell_perp']['T'][str(j)]
                final_dict_slow['flucts_ell_perp_N'][slow_ind] = res['flucts']['ell_perp']['N'][str(j)]

                final_dict_slow['ell_all_di'][slow_ind]        = ell_di*np.ones(len(final_dict_slow['flucts_ell_all'][slow_ind]))
            else:
                keep_sfunc_ell_perp[qorder-1, j]               = np.nanmean(np.abs(res['flucts']['ell_perp']['R'][str(j)])**qorder)     + np.nanmean(np.abs(res['flucts']['ell_perp']['T'][str(j)])**qorder)     +  np.nanmean(np.abs(res['flucts']['ell_perp']['N'][str(j)])**qorder)
                keep_sfunc_Ell_perpendicular[qorder-1, j]      = np.nanmean(np.abs(res['flucts']['Ell_perp']['R'][str(j)])**qorder)     + np.nanmean(np.abs(res['flucts']['Ell_perp']['T'][str(j)])**qorder)     +  np.nanmean(np.abs(res['flucts']['Ell_perp']['N'][str(j)])**qorder) 
                keep_sfunc_ell_par[qorder-1, j]                = np.nanmean(np.abs(res['flucts']['ell_par']['R'][str(j)])**qorder)      + np.nanmean(np.abs(res['flucts']['ell_par']['T'][str(j)])**qorder)      +  np.nanmean(np.abs(res['flucts']['ell_par']['N'][str(j)])**qorder)
                keep_sfunc_ell_par_rest[qorder-1, j]           = np.nanmean(np.abs(res['flucts']['ell_par_rest']['R'][str(j)])**qorder) + np.nanmean(np.abs(res['flucts']['ell_par_rest']['T'][str(j)])**qorder) +  np.nanmean(np.abs(res['flucts']['ell_par_rest']['N'][str(j)])**qorder)

                
                if  qorder ==5:
                    # Also save sdk's
                    keep_sdk_ell_perp[j]                = keep_sfunc_ell_perp[3, j] /(np.nanmean(np.abs(res['flucts']['ell_perp']['R'][str(j)])**2)**2      + np.nanmean(np.abs(res['flucts']['ell_perp']['T'][str(j)])**2)**2      +  np.nanmean(np.abs(res['flucts']['ell_perp']['N'][str(j)])**2)**2)
                    keep_sdk_Ell_perpendicular[j]       = keep_sfunc_Ell_perpendicular[3, j]/(np.nanmean(np.abs(res['flucts']['Ell_perp']['R'][str(j)])**2)**2      + np.nanmean(np.abs(res['flucts']['Ell_perp']['T'][str(j)])**2)**2      +  np.nanmean(np.abs(res['flucts']['Ell_perp']['N'][str(j)])**2)**2 )
                    keep_sdk_ell_par[ j]                = keep_sfunc_ell_par[3, j]/( np.nanmean(np.abs(res['flucts']['ell_par']['R'][str(j)])**2)**2       + np.nanmean(np.abs(res['flucts']['ell_par']['T'][str(j)])**2)**2       +  np.nanmean(np.abs(res['flucts']['ell_par']['N'][str(j)])**2)**2 )
                    keep_sdk_ell_par_rest[j]            = keep_sfunc_ell_par_rest[3, j] /(np.nanmean(np.abs(res['flucts']['ell_par_rest']['R'][str(j)])**2)**2  + np.nanmean(np.abs(res['flucts']['ell_par_rest']['T'][str(j)])**2)**2  +  np.nanmean(np.abs(res['flucts']['ell_par_rest']['N'][str(j)])**2)**2) 



                    # Also save ell's
                    keep_sfunc_ell_perp_di[j]                      = np.nanmean(np.array(res['flucts']['ell_perp']['lambdas'][str(j)])/res['di'])
                    keep_sfunc_Ell_perpendicular_di[j]             = np.nanmean(np.array(res['flucts']['Ell_perp']['xis'][str(j)])/res['di'])               
                    keep_sfunc_ell_par_di[j]                       = np.nanmean(np.array(res['flucts']['ell_par']['ells'][str(j)])/res['di'] )   
                    keep_sfunc_ell_par_rest_di[j]                  = np.nanmean(np.array(res['flucts']['ell_par_rest']['ells_rest'][str(j)])/res['di'] ) 
                    
                    # Also save align angles
                    
                    
    
    SF_ell_final = pd.DataFrame({'sfuncs_ell_perp'          : [keep_sfunc_ell_perp],
                                  'ells_ell_perp'           : [keep_sfunc_ell_perp_di],
                                  'VB_reg_sin'              : [VB_sin],
                                  'sdk_ell_perp'            : [keep_sdk_ell_perp],
                                  'sfuncs_Ell_perpendicular': [keep_sfunc_Ell_perpendicular],
                                  'ells_Ell_perpendicular'  : [keep_sfunc_Ell_perpendicular_di],
                                  'sdk_Ell_perpendicular'   : [keep_sdk_Ell_perpendicular],
                                  'sfuncs_ell_par'          : [keep_sfunc_ell_par],
                                  'sdk_ell_par'             : [keep_sdk_ell_par],
                                  'ells_ell_par'            : [keep_sfunc_ell_par_di],
                                  'sfuncs_ell_par_rest'     : [keep_sfunc_ell_par_rest],
                                  'sdk_ell_par_rest'        : [keep_sdk_ell_par_rest],
                                  'ells_ell_par_rest'       : [keep_sfunc_ell_par_rest_di],
                                  'sig_c'                   : sigma_c,
                                  'sig_r'                   : sigma_r,
                                  'd'                       : res_gen['d'],
                                  'Vsw'                     : Vsw,
                                  'B_mean_sq'               : B_mean,
                                  'duration'                : duration,
                                  'miss_frac_mag'           : res_gen['Fraction_missing_MAG'],
                                  'miss_frac_par'           : res_gen['Fraction_missing_part']})
    
    return SF_ell_final

def process_data_new( data_file,
                     data_file_gen,
                     data_file_fin,
                     data_file_align,
                     max_d, 
                     min_d,
                     qorders,
                     general_sf):
    
    res       = pd.read_pickle(data_file)
    res_gen   = pd.read_pickle(data_file_gen)
    res_fin   = pd.read_pickle(data_file_fin)
    res_align = pd.read_pickle(data_file_align)

    # Define parameters
    Vsw      = res['Vsw']
    di       = res['di']
    sigma_c  = res_fin['Par']['sigma_c_mean']
    sigma_r  = res_fin['Par']['sigma_r_mean']
    VB_sin   = res_align['VB']['reg']
    B_mean   = np.nanmean(np.linalg.norm(res_fin["Mag"]["B_resampled"].values, axis=1)) ** 2
    duration = (pd.to_datetime(res_fin['Mag']['B_resampled'].index)[-1] - pd.to_datetime(res_fin['Mag']['B_resampled'].index)[0]) / np.timedelta64(1, 's')


    keep_sfunc_ell_perp             = np.nan*np.ones((len(qorders), len(res['Sfuncs']['B']['ell_perp'][0])))
    keep_sfunc_Ell_perpendicular    = np.nan*np.ones((len(qorders), len(res['Sfuncs']['B']['ell_perp'][0])))
    keep_sfunc_ell_par              = np.nan*np.ones((len(qorders), len(res['Sfuncs']['B']['ell_perp'][0])))
    keep_sfunc_ell_par_rest         = np.nan*np.ones((len(qorders), len(res['Sfuncs']['B']['ell_perp'][0])))

    keep_sdk_ell_perp               = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))
    keep_sdk_Ell_perpendicular      = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))
    keep_sdk_ell_par                = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))
    keep_sdk_ell_par_rest           = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))

    keep_sfunc_ell_perp_di          = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))
    keep_sfunc_Ell_perpendicular_di = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))
    keep_sfunc_ell_par_di           = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))
    keep_sfunc_ell_par_rest_di      = np.nan*np.ones(len(res['Sfuncs']['B']['ell_perp'][0]))        
    keep_align                      = []
    

    for qorder in qorders:

        keep_sfunc_ell_perp[qorder-1, :]               = res['Sfuncs']['B']['ell_perp'][qorder-1]
        keep_sfunc_Ell_perpendicular[qorder-1, :]      = res['Sfuncs']['B']['Ell_perp'][qorder-1]
        keep_sfunc_ell_par[qorder-1, :]                = res['Sfuncs']['B']['ell_par'][qorder-1]
        keep_sfunc_ell_par_rest[qorder-1, :]           = res['Sfuncs']['B']['ell_par_rest'][qorder-1]

        if  qorder ==5:

            keep_sfunc_ell_perp_di[:]                      = res['ell_di']
            keep_sfunc_Ell_perpendicular_di[:]             = res['ell_di']            
            keep_sfunc_ell_par_di[:]                       = res['ell_di']   
            keep_sfunc_ell_par_rest_di[:]                  = res['ell_di'] 


                    
    
    SF_ell_final = pd.DataFrame({'sfuncs_ell_perp'          : [keep_sfunc_ell_perp],
                                  'ells_ell_perp'           : [keep_sfunc_ell_perp_di],
                                  'VB_reg_sin'              : [VB_sin],
                                  'sdk_ell_perp'            : [keep_sdk_ell_perp],
                                  'sfuncs_Ell_perpendicular': [keep_sfunc_Ell_perpendicular],
                                  'ells_Ell_perpendicular'  : [keep_sfunc_Ell_perpendicular_di],
                                  'sdk_Ell_perpendicular'   : [keep_sdk_Ell_perpendicular],
                                  'sfuncs_ell_par'          : [keep_sfunc_ell_par],
                                  'sdk_ell_par'             : [keep_sdk_ell_par],
                                  'ells_ell_par'            : [keep_sfunc_ell_par_di],
                                  'sfuncs_ell_par_rest'     : [keep_sfunc_ell_par_rest],
                                  'sdk_ell_par_rest'        : [keep_sdk_ell_par_rest],
                                  'ells_ell_par_rest'       : [keep_sfunc_ell_par_rest_di],
                                  'sig_c'                   : sigma_c,
                                  'sig_r'                   : sigma_r,
                                  'd'                       : res_gen['d'],
                                  'Vsw'                     : Vsw,
                                  'B_mean_sq'               : B_mean,
                                  'duration'                : duration,
                                  'miss_frac_mag'           : res_gen['Fraction_missing_MAG'],
                                  'miss_frac_par'           : res_gen['Fraction_missing_part']})
    
    return SF_ell_final


                     

#Define function to process each file in parallel
def process_file(data_file, data_file_gen, data_file_fin, data_file_align, max_d, min_d, qorders, general_sf, old_way = True):
    # Call process_data function on each file
    if old_way:
        result = process_data(data_file, data_file_gen, data_file_fin, data_file_align,  max_d, min_d, qorders, general_sf)
    else:
        result = process_data_new(data_file, data_file_gen, data_file_fin, data_file_align,  max_d, min_d, qorders, general_sf)                     
    return result

def calculate_sfuncs(which_ones,
                     step,
                     duration,
                     what,
                     std,
                     wind,
                     loglog,
                     alfvenic,
                     E1_only,
                     five_point,
                     strict_thresh,
                     min_d,
                     max_d,
                     sig_c_non_alfv,
                     sig_c_alfv,
                     min_norm,
                     max_norm,
                     max_Vsw,
                     min_mag_mis_frac,
                     min_par_mis_frac,
                     min_dur,
                     max_qorder,
                     path, 
                     old_way          = False,
                     normalize_sfuncs ='with_mean'):
    if E1_only:
        if old_way:
            if five_point:
                prefix = '5pt'
            else:
                prefix = '2pt'
            if strict_thresh:
                res = pd.read_pickle(path+str(duration)+'_'+str(step)+'_binned_data/trace_SF/final_sfuncs/E1_'+str(prefix)+'_alfvenic_SFuncs_final_slow_5deg.pkl')
            else:
                res = pd.read_pickle(path+str(duration)+'_'+str(step)+'_binned_data/trace_SF/final_sfuncs/E1_'+str(prefix)+'_alfvenic_SFuncs_final_slow.pkl')

                
        print(len(res))
    else:
        if alfvenic:
            res = pd.read_pickle(path+str(duration)+'_'+str(step)+'_all_binned_data/trace_SF/0.1_1.2/final_sfuncs/5pt_alfvenic_SFuncs_final_slow.pkl')
        else:
            res = pd.read_pickle(path+str(duration)+'_'+str(step)+'_all_binned_data/trace_SF/0.1_1.2/final_sfuncs/5pt_non_alfven_SFuncs_final_slow.pkl')

    res = res.sort_values(by='sig_c').reset_index(drop=True)

    col = ['C{}'.format(i) for i in range(len(res))]

    keep_all = {}
    for which_one in which_ones:
        all_x     = []
        all_y     = []
        all_err   = []
        
        all_sdk_x   = []
        all_sdk_y   = []
        all_sdk_err = []
        for qorder in range(max_qorder):

            xvals                        = []
            yvals                        = []
            sdks                         = []
            sig_c                        = []
            chi_array_db_over_B0         = [] 
            chi_array_db_over_B0_sin     = []
            chi_array_sqrt_db_sq_over_B0 = []
            B_0                          = []
            for N in range(len(res)):
                
                try:
                    keep_row = res[int(N):int(N+1)]

                    if E1_only:
                        conditions = (keep_row['sig_c'].values[0] > sig_c_alfv) & \
                                     (keep_row['d'].values[0] > min_d) & \
                                     (keep_row['d'].values[0] < max_d) & \
                                     (keep_row['Vsw'].values[0] < max_Vsw) &\
                                     (keep_row['miss_frac_mag'].values[0] < min_mag_mis_frac) & \
                                     (keep_row['miss_frac_par'].values[0] < min_par_mis_frac) & \
                                     (keep_row['duration'].values[0] > min_dur)
                    else:
                        if alfvenic:
                            conditions = (keep_row['sig_c'].values[0] > sig_c_alfv) & \
                                         (keep_row['d'].values[0] > min_d) & \
                                         (keep_row['d'].values[0] < max_d) & \
                                         (keep_row['miss_frac_mag'].values[0] < min_mag_mis_frac) & \
                                         (keep_row['miss_frac_par'].values[0] < min_par_mis_frac) & \
                                         (keep_row['duration'].values[0] > min_dur)
                        else:
                            conditions = (keep_row['sig_c'].values[0] < sig_c_non_alfv) & \
                                         (keep_row['d'].values[0] > min_d) & \
                                         (keep_row['d'].values[0] < max_d) & \
                                         (keep_row['miss_frac_mag'].values[0] < min_mag_mis_frac) & \
                                         (keep_row['miss_frac_par'].values[0] < min_par_mis_frac) & \
                                         (keep_row['duration'].values[0] > min_dur)

                    if (conditions) or (E1_only) :
                        sig_c.append(keep_row['sig_c'].values[0])
                        try:
                            B_0.append(np.sqrt(keep_row['B_mean_sq'].values[0]))
                        except:
                            B_0.append(np.nan)
                            #print(keep_row.keys())
                        xx         = keep_row['ells_'+str(which_one)].values[0]
                        yvals_keep = keep_row['sfuncs_'+str(which_one)].values[0][qorder][(xx>min_norm) & (xx<max_norm)]
                        if normalize_sfuncs=='with_mean':
                            yvals.append(keep_row['sfuncs_'+str(which_one)].values[0][qorder]/np.nanmean(yvals_keep))
                        elif normalize_sfuncs=='B_sq':
                            try:
                                yvals.append(keep_row['sfuncs_'+str(which_one)].values[0][qorder]/keep_row['B_mean_sq'].values[0])
                            except:
                                yvals.append(np.nan)
                        else:
                            yvals.append(keep_row['sfuncs_'+str(which_one)].values[0][qorder])                        
                        xvals.append(keep_row['ells_'+str(which_one)].values[0])
                        sdks.append(keep_row['sdk_'+str(which_one)].values[0])
                        if qorder==0:
                            try:
                                chi_array_db_over_B0.append((keep_row['sfuncs_'+str(which_one)].values[0][qorder])/np.sqrt(keep_row['B_mean_sq']).values[0])
                            except:
                                chi_array_db_over_B0.append(np.nan)
                except:
                    traceback.print_exc()

            if qorder ==0:
                wave_chi_array_db_over_B0 = chi_array_db_over_B0
                wave_chi_array_db_over_B0_sin     = chi_array_db_over_B0_sin
            if qorder==1:
                wave_xvals = xvals
                wave_yvals = yvals
                wave_sig_c = sig_c
                wave_B0    = B_0
                wave_chi_array_sqrt_db_sq_over_B0 = chi_array_sqrt_db_sq_over_B0
               
                
            #print(np.shape(xvals))
            xvals_new = np.hstack(xvals)
            yvals_new = np.hstack(yvals)
            sdks_new  = np.hstack(sdks)

            results = func.binned_quantity(xvals_new[(yvals_new>0) & (xvals_new>0)], yvals_new[(yvals_new>0) & (xvals_new>0)], what, std, wind, loglog)
            all_x.append(results[0])
            all_y.append(results[1])
            all_err.append(results[2]) 
            
            results = func.binned_quantity(xvals_new[(sdks_new>0) & (xvals_new>0)], sdks_new[(sdks_new>0) & (xvals_new>0)], what, std, wind, loglog)
            all_sdk_x.append(results[0])
            all_sdk_y.append(results[1])  
            all_sdk_err.append(results[2])  

        keep_all[which_one] = {
                               'anis_anal' : {'xvals'                : wave_xvals,
                                              'yvals'                : wave_yvals,
                                              'sig_c'                : wave_sig_c,
                                              'B_0'                  : wave_B0,
                                              'db_over_B0'           : wave_chi_array_db_over_B0,
                                              'db_over_B0_sin'       : wave_chi_array_db_over_B0_sin,
                                              'sqrt_db_sq_over_B0'   : wave_chi_array_sqrt_db_sq_over_B0
                                             },
                               'xvals'     : all_x,
                               'yvals'     : all_y,
                               'err'       : all_err,
                               'sdk_xvals' : all_sdk_x,
                               'sdk_yvals' : all_sdk_y,
                               'sdk_err'   : all_sdk_err
                              }

    return keep_all



def calculate_sfuncs_WIND_sc(
                             interval_type,
                             sfunc_type,
                             which_ones,
                             what,
                             std,
                             window_size,
                             loglog,
                             strict_thresh,
                             min_sig_c,
                             max_sig_c,
                             min_sig_r,
                             max_sig_r,
                             min_Vsw,
                             max_Vsw,
                             min_norm_di,
                             max_norm_di,
                             min_mag_mis_frac,
                             min_par_mis_frac,
                             min_dur,
                             max_qorder,
                             path, 
                             normalize_sfuncs ='with_mean'):
    # Load df!
    res = pd.read_pickle(f'{path}/{interval_type}/final_sfuncs/{sfunc_type}.pkl')
    
    # Sort df by sigma_c value
    res = res.sort_values(by='sig_c').reset_index(drop=True)

    col = ['C{}'.format(i) for i in range(len(res))]

    keep_all = {}
    for which_one in which_ones:
        all_x     = []
        all_y     = []
        all_err   = []
        
        all_sdk_x   = []
        all_sdk_y   = []
        all_sdk_err = []
        
        
        for qorder in range(max_qorder):

            xvals                        = []
            yvals                        = []
            sdks                         = []
            sig_c                        = []
            chi_array_db_over_B0         = [] 
            chi_array_db_over_B0_sin     = []
            chi_array_sqrt_db_sq_over_B0 = []
            B_0                          = []
            
            counter=0
            for N in range(len(res)):
                keep_row = res[int(N):int(N+1)]



                conditions = (keep_row['sig_c'].values[0] > min_sig_c) & \
                             (keep_row['sig_c'].values[0] < max_sig_c) & \
                             (keep_row['sig_r'].values[0] > min_sig_r) & \
                             (keep_row['sig_r'].values[0] < max_sig_r) & \
                             (keep_row['miss_frac_mag'].values[0] < min_mag_mis_frac) & \
                             (keep_row['miss_frac_par'].values[0] < min_par_mis_frac) & \
                             (keep_row['duration'].values[0] > min_dur) & \
                             (keep_row['Vsw'].values[0] < max_Vsw)      & \
                             (keep_row['Vsw'].values[0] > min_Vsw)      

        
                if conditions:
                    counter+=1
                    sig_c.append(keep_row['sig_c'].values[0])
                    B_0.append(np.sqrt(keep_row['B_mean_sq'].values[0]))
                    xx         = keep_row['ells_'+str(which_one)].values[0]
                    yvals_keep = keep_row['sfuncs_'+str(which_one)].values[0][qorder][(xx>min_norm_di) & (xx<max_norm_di)]
                    
                    if normalize_sfuncs=='with_mean':
                        yvals.append(keep_row['sfuncs_'+str(which_one)].values[0][qorder]/np.nanmean(yvals_keep))
                    elif normalize_sfuncs=='B_sq':
                        yvals.append(keep_row['sfuncs_'+str(which_one)].values[0][qorder]/keep_row['B_mean_sq'].values[0])
                    else:
                        yvals.append(keep_row['sfuncs_'+str(which_one)].values[0][qorder])
                        
                    xvals.append(keep_row['ells_'+str(which_one)].values[0])
                    sdks.append(keep_row['sdk_'+str(which_one)].values[0])
                    
                    
                    if qorder==0:
                        chi_array_db_over_B0.append((keep_row['sfuncs_'+str(which_one)].values[0][qorder])/np.sqrt(keep_row['B_mean_sq']).values[0])
 

            if qorder ==0:
                wave_chi_array_db_over_B0         = chi_array_db_over_B0
                wave_chi_array_db_over_B0_sin     = chi_array_db_over_B0_sin
            if qorder==1:
                wave_xvals = xvals
                wave_yvals = yvals
                wave_sig_c = sig_c
                wave_B0    = B_0
                wave_chi_array_sqrt_db_sq_over_B0 = chi_array_sqrt_db_sq_over_B0
               

            xvals_new = np.hstack(xvals)
            yvals_new = np.hstack(yvals)
            sdks_new  = np.hstack(sdks)

            results = func.binned_quantity(xvals_new[(yvals_new>0) & (xvals_new>0)], yvals_new[(yvals_new>0) & (xvals_new>0)], what, std, window_size, loglog)
            all_x.append(results[0])
            all_y.append(results[1])
            all_err.append(results[2]) 
            
            results = func.binned_quantity(xvals_new[(sdks_new>0) & (xvals_new>0)], sdks_new[(sdks_new>0) & (xvals_new>0)], what, std, window_size, loglog)
            all_sdk_x.append(results[0])
            all_sdk_y.append(results[1])  
            all_sdk_err.append(results[2])  

        keep_all[which_one] = {
                               'anis_anal' : {'xvals'                : wave_xvals,
                                              'yvals'                : wave_yvals,
                                              'sig_c'                : wave_sig_c,
                                              'B_0'                  : wave_B0,
                                              'db_over_B0'           : wave_chi_array_db_over_B0,
                                              'db_over_B0_sin'       : wave_chi_array_db_over_B0_sin,
                                              'sqrt_db_sq_over_B0'   : wave_chi_array_sqrt_db_sq_over_B0
                                             },
                               'xvals'     : all_x,
                               'yvals'     : all_y,
                               'err'       : all_err,
                               'sdk_xvals' : all_sdk_x,
                               'sdk_yvals' : all_sdk_y,
                               'sdk_err'   : all_sdk_err
                              }

    print('Considering', counter, 'Intervals in total')
    return keep_all



def estimate_non_lin_parameter(load_path,
                               sf_name, 
                               fin_name,
                               al_name,
                               aling_x_min,
                               sig_c_min):
    
    from scipy import constants
    mu0          = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
    m_p          = constants.m_p    # Proton mass [kg]

    # Load files
    sf_files    = func.load_files(load_path, sf_name, 'final')
    final_files = func.load_files(load_path, fin_name, '')
    align_files = func.load_files(load_path, al_name, 'final')
    
    # Initialize lists
    xvm, yvm =[],[]
    xvp, yvp =[],[]

    yvm_xi   =[]
    yvp_xi   =[]

    for ww, ( sf_file, fin_file, align_file)  in enumerate(zip(sf_files, final_files, align_files)):
        sf = pd.read_pickle(sf_file)
        fin = pd.read_pickle(fin_file)
        al  = pd.read_pickle(align_file)


        if fin['Par']['sigma_c_median']> sig_c_min:

            kinet_normal = np.nanmedian(1e-15 / np.sqrt(mu0 * fin['Par']['V_resampled']['np'].values * m_p))

            Var          = kinet_normal*fin['Mag']['B_resampled']['Br']
            Vat          = kinet_normal*fin['Mag']['B_resampled']['Bt']
            Van          = kinet_normal*fin['Mag']['B_resampled']['Bn']
            Va_ts        = np.sqrt(Var**2 + Vat**2 + Van**2)
            Va           = np.nanmedian(Va_ts)

            d_zp_lambda, d_zp_xi, d_zp_ell = sf['Sfuncs']['Zp']['ell_perp'][0], sf['Sfuncs']['Zp']['Ell_perp'][0], sf['Sfuncs']['Zp']['ell_par_rest'][0]
            d_zm_lambda, d_zm_xi, d_zm_ell = sf['Sfuncs']['Zm']['ell_perp'][0], sf['Sfuncs']['Zm']['Ell_perp'][0], sf['Sfuncs']['Zm']['ell_par_rest'][0]

            zp_lambda  , zp_xi  , zp_ell   = sf['Sfuncs']['l_ell_perp']/sf['di'], sf['Sfuncs']['l_Ell_perp']/sf['di'], sf['Sfuncs']['l_ell_par']/sf['di']
            zm_lambda  , zm_xi  , zm_ell   = sf['Sfuncs']['l_ell_perp']/sf['di'], sf['Sfuncs']['l_Ell_perp']/sf['di'], sf['Sfuncs']['l_ell_par']/sf['di']

            align_angle = np.array(al['Zpm']['reg'])
            index       = zp_lambda >aling_x_min
            
            try:
 
                # Estimate non-lin parameter
                lambdas, chi_m_lambda, chi_m_xi, chi_p_lambda, chi_p_xi = turb.calculate_non_linearity_parameter(d_zp_lambda[index], d_zp_xi[index], d_zp_ell[index],
                                                                                                                  d_zm_lambda[index], d_zm_xi[index], d_zm_ell[index],
                                                                                                                  zp_lambda[index], zp_xi[index], zp_ell[index], zm_lambda[index],
                                                                                                                  zm_xi[index], zm_ell[index], align_angle[index], Va)
                xvm.append(lambdas)
                xvp.append(lambdas)

                yvm.append(chi_m_lambda)
                yvp.append(chi_p_lambda)

                yvm_xi.append(chi_m_xi)
                yvp_xi.append(chi_p_xi)
            except:
                print('bad')
                pass

    xvm, yvm = np.hstack(xvm), np.hstack(yvm)
    xvp, yvp = np.hstack(xvp), np.hstack(yvp) 

    yvm_xi =  np.hstack(yvm_xi)
    yvp_xi =  np.hstack(yvp_xi)

    
    return xvm, xvp, yvm, yvp, yvm_xi, yvp_xi





def estimate_non_lin_parameter_WIND(load_path,
                                       sf_name, 
                                       fin_name,
                                       al_name,
                                    
                                       aling_x_min,
                                       sig_c_max,
                                       sig_c_min = 0,
                                       dur_min   = 0):
    
    from scipy import constants
    mu0          = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
    m_p          = constants.m_p    # Proton mass [kg]

    # Load files
    sf_files    = func.load_files(load_path, sf_name, 'final')

    
    # Initialize lists
    xvm, yvm =[],[]
    xvp, yvp =[],[]

    yvm_xi   =[]
    yvp_xi   =[]
    
    counts   = []

    for ww, sf_file  in enumerate(sf_files):
        
        
        
        sf  = pd.read_pickle(sf_file)
        fin = pd.read_pickle(sf_file.replace(str('final/')+sf_name, fin_name))
        al  = pd.read_pickle(sf_file.replace(sf_name, al_name))
        gen = pd.read_pickle(sf_file.replace(str('final/')+sf_name, 'general.pkl'))

        dur  = (gen['End_Time'] - gen['Start_Time'])/ pd.Timedelta(hours=1)
        
        if (fin['Par']['sigma_c_median']> sig_c_max) &(fin['Par']['sigma_c_median']> sig_c_min) & (dur>dur_min):

            kinet_normal = np.nanmedian(1e-15 / np.sqrt(1.16*mu0 * fin['Par']['V_resampled']['np'].values * m_p))

            Var          = kinet_normal*fin['Mag']['B_resampled']['Br']
            Vat          = kinet_normal*fin['Mag']['B_resampled']['Bt']
            Van          = kinet_normal*fin['Mag']['B_resampled']['Bn']
            Va_ts        = np.sqrt(Var**2 + Vat**2 + Van**2)
            Va           = np.nanmean(Va_ts)


            
            length                         = sum(np.array(al['Zpm']['reg'])>0)
            
            #count           =np.array(sf['Sfuncs']['counts_ell_overall'])[0:length]
            
            
            d_zp_lambda      = np.array(sf['Sfuncs']['Zp']['ell_perp'][0])[0:length] ## 0 bevause we want fluctuations!
            d_zp_xi          =  np.array(sf['Sfuncs']['Zp']['Ell_perp'][0])[0:length]
            d_zp_ell         =  np.array(sf['Sfuncs']['Zp']['ell_par'][0])[0:length]
            
            d_zm_lambda      = np.array(sf['Sfuncs']['Zm']['ell_perp'][0])[0:length]
            d_zm_xi          = np.array(sf['Sfuncs']['Zm']['Ell_perp'][0])[0:length]
            d_zm_ell         = np.array(sf['Sfuncs']['Zm']['ell_par'][0])[0:length]


#             d_zp_lambda      = np.sqrt(sf['Sfuncs']['Zp']['ell_perp'][1])[0:length] ## 0 bevause we want fluctuations!
#             d_zp_xi          =  np.sqrt(sf['Sfuncs']['Zp']['Ell_perp'][1])[0:length]
#             d_zp_ell         =  np.sqrt(sf['Sfuncs']['Zp']['ell_par'][1])[0:length]
            
#             d_zm_lambda      = np.sqrt(sf['Sfuncs']['Zm']['ell_perp'][1])[0:length]
#             d_zm_xi          = np.sqrt(sf['Sfuncs']['Zm']['Ell_perp'][1])[0:length]
#             d_zm_ell         = np.sqrt(sf['Sfuncs']['Zm']['ell_par'][1])[0:length]

            zp_lambda = np.array(sf['Sfuncs']['l_ell_perp']/sf['di'])[0:length]
            zp_xi     = np.array(sf['Sfuncs']['l_Ell_perp']/sf['di'])[0:length]
            zp_ell    = np.array(sf['Sfuncs']['l_ell_par']/sf['di'])[0:length]

            zm_lambda  = np.array(sf['Sfuncs']['l_ell_perp']/sf['di'])[0:length]
            zm_xi      = np.array(sf['Sfuncs']['l_Ell_perp']/sf['di'])[0:length]
            zm_ell     = np.array(sf['Sfuncs']['l_ell_par']/sf['di'])[0:length]
            
            
            #print(length, len(zp_lambda), len(zm_ell), len(np.array(al['Zpm']['reg'])[0:length]))
#

            align_angle = np.array(al['Zpm']['reg'])
            index       = zp_lambda >aling_x_min
            
            try:
 
                # Estimate non-lin parameter
                lambdas, chi_m_lambda, chi_m_xi, chi_p_lambda, chi_p_xi = turb.calculate_non_linearity_parameter(d_zp_lambda[index], d_zp_xi[index], d_zp_ell[index],
                                                                                                                  d_zm_lambda[index], d_zm_xi[index], d_zm_ell[index],
                                                                                                                  zp_lambda[index], zp_xi[index], zp_ell[index], zm_lambda[index],
                                                                                                                  zm_xi[index], zm_ell[index], align_angle[index], Va)
                xvm.append(lambdas)
                xvp.append(lambdas)

                yvm.append(chi_m_lambda)
                yvp.append(chi_p_lambda)

                yvm_xi.append(chi_m_xi)
                yvp_xi.append(chi_p_xi)
                
                counts.append(np.ones(len(lambdas))*dur)
            except:
                traceback.print_exc()
                print('bad')
                pass

    xvm, yvm = np.hstack(xvm), np.hstack(yvm)
    xvp, yvp = np.hstack(xvp), np.hstack(yvp) 

    yvm_xi =  np.hstack(yvm_xi)
    yvp_xi =  np.hstack(yvp_xi)

    
    return xvm, xvp, yvm, yvp, yvm_xi, yvp_xi, np.hstack(counts)





def alignment_anlges(
                     what,
                     std,
                     al_names,
                     wind,
                     loglog,
                     sig_c_min,
                     data_path    = '',
                     fnames_sf    ='',
                     fnames_align ='',
                     base_path    ='',
                     connect2     = ''):
    




    data_filess_align = func.load_files(data_path,al_names, '')
    fin_filess        = func.load_files(data_path,'final.pkl', '')
    

    xvals       = []
    zpm_ang_w   = []
    VB_ang_w    = []
    zpm_ang_reg = []
    VB_ang_reg  = []   
    zpm_ang_pol = []
    VB_ang_pol  = []  
    VB_sin_reg  = []
    
    sig_r_mean   = []
    sig_r_median = []
    sig_c_mean   = []
    sig_c_median = []
    for  data_files_align, fin_file in zip( data_filess_align, fin_filess):
        
        res = pd.read_pickle(data_files_align)
        fin  = pd.read_pickle(fin_file)
        
        
        if  fin['Par']['sigma_c_median']> sig_c_min:
            vv = np.array(res['Zpm']['weighted'])
            xvals.append(np.array(res['l_di'])[0:len(vv)])

            zpm_ang_w.append(np.arcsin(res['Zpm']['weighted'])*180/np.pi)
            VB_ang_w.append(np.arcsin(res['VB']['weighted'])*180/np.pi)

            zpm_ang_reg.append(np.arcsin(res['Zpm']['reg'])*180/np.pi)
            VB_ang_reg.append(np.arcsin(res['VB']['reg'])*180/np.pi)
            VB_sin_reg.append(np.array(res['VB']['reg']))

            zpm_ang_pol.append(np.arcsin(res['Zpm']['polar'])*180/np.pi)
            VB_ang_pol.append(np.arcsin(res['VB']['polar'])*180/np.pi)  

            sig_r_mean.append(np.array(res['VB']['sig_r_mean']))
            sig_r_median.append(np.array(res['VB']['sig_r_median']))
            sig_c_mean.append(np.array(res['Zpm']['sig_c_mean']))
            sig_c_median.append(np.array(res['Zpm']['sig_c_median']))

    print(len(sig_r_mean))
    keep_all  = {
                  'xvals'         : np.hstack(xvals),
                  'Zpm_reg'       : np.hstack(zpm_ang_reg),        
                  'VB_reg'        : np.hstack(VB_ang_reg),  
                  'Zpm_pol'       : np.hstack(zpm_ang_pol),        
                  'VB_pol'        : np.hstack(VB_ang_pol), 
                  'Zpm_w'         : np.hstack(zpm_ang_w),        
                  'VB_w'          : np.hstack(VB_ang_w),
                  'VB_sin_reg'    : np.hstack(VB_sin_reg),
                  'sig_c_mean'    : np.hstack(sig_c_mean),        
                  'sig_c_median'  : np.hstack(sig_c_median),  
                  'sig_r_mean'    : np.hstack(sig_r_mean),        
                  'sig_r_median'  : np.hstack(sig_r_median),  
    }
        
        
    keys = list(keep_all.keys())[1:]
    
    for key in keys:
        results = func.binned_quantity(keep_all['xvals'], keep_all[key], what, std, wind, loglog)

        if key =='VB_sin_reg':
            keep_all[key] = {  'all'  : VB_sin_reg,
                              'xvals' : results[0],
                              'yvals' : results[1],
                              'err'   : results[2],
                            }          
        else:
            keep_all[key] = {
                              'xvals' : results[0],
                              'yvals' : results[1],
                              'err'   : results[2],
                            }

    return keep_all




def alignment_anlges2(
                     what,
                     std,
                     al_names,
                     wind,
                     loglog,
                     sig_c_min,
                     data_path    = '',
                     fnames_sf    ='',
                     fnames_align ='',
                     base_path    ='',
                     connect2     = ''):
    




    data_filess_align = func.load_files(data_path,al_names, '')
    fin_filess        = func.load_files(data_path,'final.pkl', '')
    

    xvals       = []
    zpm_ang_w   = []
    VB_ang_w    = []
    zpm_ang_reg = []
    VB_ang_reg  = []   
    zpm_ang_pol = []
    VB_ang_pol  = []  
    VB_sin_reg  = []
    
    sig_r_mean   = []
    sig_r_median = []
    sig_c_mean   = []
    sig_c_median = []
    for  data_files_align, fin_file in zip( data_filess_align, fin_filess):
        
        res = pd.read_pickle(data_files_align)
        fin  = pd.read_pickle(fin_file)
        
        
        if  fin['Par']['sigma_c_median']> sig_c_min:
            vv = np.array(res['Zpm']['weighted'])
            xvals.append(np.array(res['l_di'])[0:len(vv)])

            zpm_ang_w.append((res['Zpm']['weighted']))
            VB_ang_w.append((res['VB']['weighted']))

            zpm_ang_reg.append((res['Zpm']['reg']))
            VB_ang_reg.append((res['VB']['reg']))
            VB_sin_reg.append(np.array(res['VB']['reg']))

            zpm_ang_pol.append((res['Zpm']['polar']))
            VB_ang_pol.append((res['VB']['polar']))  

            sig_r_mean.append(np.array(res['VB']['sig_r_mean']))
            sig_r_median.append(np.array(res['VB']['sig_r_median']))
            sig_c_mean.append(np.array(res['Zpm']['sig_c_mean']))
            sig_c_median.append(np.array(res['Zpm']['sig_c_median']))

    print(len(sig_r_mean))
    keep_all  = {
                  'xvals'         : np.hstack(xvals),
                  'Zpm_reg'       : np.hstack(zpm_ang_reg),        
                  'VB_reg'        : np.hstack(VB_ang_reg),  
                  'Zpm_pol'       : np.hstack(zpm_ang_pol),        
                  'VB_pol'        : np.hstack(VB_ang_pol), 
                  'Zpm_w'         : np.hstack(zpm_ang_w),        
                  'VB_w'          : np.hstack(VB_ang_w),
                  'VB_sin_reg'    : np.hstack(VB_sin_reg),
                  'sig_c_mean'    : np.hstack(sig_c_mean),        
                  'sig_c_median'  : np.hstack(sig_c_median),  
                  'sig_r_mean'    : np.hstack(sig_r_mean),        
                  'sig_r_median'  : np.hstack(sig_r_median),  
    }
        
        
    keys = list(keep_all.keys())[1:]
    
    for key in keys:
        results = func.binned_quantity(keep_all['xvals'], keep_all[key], what, std, wind, loglog)

        if key =='VB_sin_reg':
            keep_all[key] = {  'all'  : VB_sin_reg,
                              'xvals' : results[0],
                              'yvals' : results[1],
                              'err'   : results[2],
                            }          
        else:
            keep_all[key] = {
                              'xvals' : results[0],
                              'yvals' : results[1],
                              'err'   : results[2],
                            }

    return keep_all




from scipy.interpolate import interp1d, PchipInterpolator

def calculate_wavenumber_aniso(d_b_lambda, d_b_xi, d_b_ell, 
                               b_lambda, b_xi, b_ell,
                               return_dbs  = False,
                               smooth      = False,
                               method      = 'linear',
                               window      = 1.2):
    # Helper function to clean and sort data
    def clean_data(x, y):
        valid = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[valid], y[valid]
        sorted_indices = np.argsort(x)
        x, y = x[sorted_indices], y[sorted_indices]
        return x, y

    # Optionally smooth the data (ensure you have a proper smoothing function)
    if smooth:
        # Assuming smooth_filter is a valid function that returns smoothed x and y
        b_lambda, d_b_lambda  = smooth_filter(b_lambda, d_b_lambda, window)
        b_xi, d_b_xi          = smooth_filter(b_xi, d_b_xi, window)
        b_ell, d_b_ell        = smooth_filter(b_ell, d_b_ell, window)

    # Clean and sort data
    b_lambda, d_b_lambda = clean_data(b_lambda, d_b_lambda)
    b_xi, d_b_xi         = clean_data(b_xi, d_b_xi)
    b_ell, d_b_ell       = clean_data(b_ell, d_b_ell)

    # Create interpolation functions for δB vs. scale
    interp_d_b_lambda  = interp1d(b_lambda, d_b_lambda, kind=method, fill_value="extrapolate", assume_sorted=True)
    interp_d_b_xi      = interp1d(b_xi, d_b_xi, kind=method, fill_value="extrapolate", assume_sorted=True)
    interp_d_b_ell     = interp1d(b_ell, d_b_ell, kind=method, fill_value="extrapolate", assume_sorted=True)

    # Create inverse interpolation functions (scale vs. δB)
    def create_inverse_interp(y, x):
        y, x = clean_data(y, x)
        if np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0):
            return interp1d(y, x, kind=method, fill_value="extrapolate", assume_sorted=True)
        else:
            # Use PchipInterpolator for non-monotonic data
            return PchipInterpolator(y, x, extrapolate=True)

    interp_b_lambda = create_inverse_interp(d_b_lambda, b_lambda)
    interp_b_xi     = create_inverse_interp(d_b_xi, b_xi)
    interp_b_ell    = create_inverse_interp(d_b_ell, b_ell)

    # Determine overlapping δB range across all directions
    d_b_min = max(np.min(d_b_lambda), np.min(d_b_xi), np.min(d_b_ell))
    d_b_max = min(np.max(d_b_lambda), np.max(d_b_xi), np.max(d_b_ell))

    # Generate δB values within the overlapping range
    d_b_values = np.logspace(np.log10(d_b_min), np.log10(d_b_max), num=1000)

    lambdas = []
    xis     = []
    ells    = []
    dbs_lambda = []
    dbs_xi     = []
    dbs_ell    = []

    # Loop over δB values to find corresponding scales
    for d_b in d_b_values:
        try:
            b_lambda_val = interp_b_lambda(d_b)
            b_xi_val     = interp_b_xi(d_b)
            b_ell_val    = interp_b_ell(d_b)
        except ValueError:
            # Skip if interpolation is not possible
            continue

        lambdas.append(b_lambda_val)
        xis.append(b_xi_val)
        ells.append(b_ell_val)

        if return_dbs:
            dbs_lambda.append(d_b)
            dbs_xi.append(interp_d_b_xi(b_xi_val))
            dbs_ell.append(interp_d_b_ell(b_ell_val))
            
            
    # Sorting based on lambdas
    sorted_indices = np.argsort(lambdas)
    lambdas        = np.array(lambdas)[sorted_indices]
    xis            = np.array(xis)[sorted_indices]
    ells           = np.array(ells)[sorted_indices]

    if return_dbs:
        return (np.array(ells), np.array(xis), np.array(lambdas),
                np.array(dbs_lambda), np.array(dbs_xi), np.array(dbs_ell))
    else:
        return np.array(ells), np.array(xis), np.array(lambdas)



def find_nearest(array, value):
    """Find the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

from scipy.interpolate import PchipInterpolator
def calculate_wavenumber_aniso_old(d_b_lambda, d_b_xi, d_b_ell, 
                               b_lambda, b_xi, b_ell,
                               return_dbs  = False,
                               smooth      = False,
                               method      = 'linear',
                               window      = 1.2
                              ):
    # Interpolation functions
    
    if smooth:

        b_lambda,_, d_b_lambda = func.smooth_filter(b_lambda, d_b_lambda, window)
        b_xi, _, d_b_xi         = func.smooth_filter(b_xi, d_b_xi, window)
        b_ell,_,  d_b_ell       = func.smooth_filter(b_ell,d_b_ell, window)


    interp_d_b_lambda = interp1d(b_lambda, d_b_lambda, kind=method, bounds_error=False)#, fill_value=(d_b_lambda[0], d_b_lambda[-1]))
    interp_d_b_ell    = interp1d(b_ell, d_b_ell, kind=method, bounds_error=False)#, fill_value=(d_b_ell[0], d_b_ell[-1]))
    interp_d_b_xi     = interp1d(b_xi, d_b_xi, kind=method, bounds_error=False)#, fill_value=(d_b_xi[0], d_b_xi[-1]))

    interp_b_lambda   = interp1d(d_b_lambda, b_lambda, kind=method, bounds_error=False)#, fill_value=(b_lambda[0], b_lambda[-1]))
    interp_b_ell      = interp1d(d_b_ell, b_ell, kind=method, bounds_error=False)#, fill_value=(b_ell[0], b_ell[-1]))   
    interp_b_xi       = interp1d(d_b_xi, b_xi, kind=method, bounds_error=False)#, fill_value=(b_xi[0], b_xi[-1])) 
        
    # Calculating chi_p_lambda_ast
    lambdas = []
    xis     = []
    ells    = []
    
    dbs_lambda = []
    dbs_xi     = []
    dbs_ell    = []
   # new_lambdas =np.logspace(np.log10(np.nanmin(b_lambda)), np.log10(np.nanmax(b_lambda)), int(Npoints))
    for b_lambda_ast in b_lambda:
        d_b_lambda_ast = interp_d_b_lambda(b_lambda_ast)
        
        interp_ell  =interp_b_ell(d_b_lambda_ast)
        interp_xi  = interp_b_xi(d_b_lambda_ast)
 
        ells.append( interp_ell)
        xis.append(interp_xi)
        lambdas.append(b_lambda_ast)  
        
        if return_dbs:
            dbs_lambda.append(d_b_lambda_ast)
            dbs_xi.append(interp_d_b_xi( interp_xi))
            dbs_ell.append(interp_d_b_ell( interp_ell))   
            #ratio_perp.append()

    if return_dbs:
        return np.array(ells), np.array(xis),  np.array(lambdas), np.array(dbs_lambda), np.array(dbs_xi), np.array(dbs_ell)
        
    else:
        return np.array(ells), np.array(xis),  np.array(lambdas) 




def estimate_wave_power_anisotropy_updated(path,
                                           sf_name,
                                           gen_name,
                                           final_fname,
                                           q_order,
                                           what,
                                           sigma_c_min,
                                           Vsw_max,
                                           mag_mis_min,
                                           fit_windows,
                                           which_var,
                                           which_comps,
                                           x01, xf1,
                                           x02, xf2,
                                           smooth      = False,
                                           method      = 'linear',
                                           window      = 1.2
                                          # Npoints =int(3e2)
                                          ):
    
    
    ells_yvm    = []
    xis_yvm     = []
    lambdas_yvm = []
    fit_x_xis   = []
    fit_y_xis   = []
    fit_x_ells  = []
    fit_y_ells  = []
    
    power_aniso_xis   = []
    power_aniso_ells  = []
    power_aniso_xvals = []

    fit1_w_xi_dict  = []
    fit1_w_ell_dict = []  
    fit2_w_xi_dict  = [] 
    fit2_w_ell_dict = []      

    fit1_p_xi_dict  = []
    fit1_p_ell_dict = []  
    fit2_p_xi_dict  = [] 
    fit2_p_ell_dict = [] 
    
    sigma_c =[]
    sigma_r =[]   
    
    fit_x_xis_dict_wave   = {}
    fit_y_xis_dict_wave   = {}
    fit_x_ells_dict_wave  = {}
    fit_y_ells_dict_wave  = {}

    fit_x_xis_dict_power   = {}
    fit_y_xis_dict_power   = {}
    fit_x_ells_dict_power  = {}
    fit_y_ells_dict_power  = {}


    #locate files
    sf_names    = func.load_files(path, sf_name, 'final')
    gen         = func.load_files(path, gen_name, '')
    final       = func.load_files(path, final_fname, '')


    bad =0
    counts =0
    for ww in range(len(sf_names)):
        func.progress_bar(ww, len(sf_names))
        try:
            res = pd.read_pickle(sf_names[ww])
            fin = pd.read_pickle(final[ww])
            gg  = pd.read_pickle(gen[ww])

            if (fin['Par']['Vsw_mean']< Vsw_max) &(fin['Par']['sigma_c_median']> sigma_c_min) & ((gg['Fraction_missing_MAG']< mag_mis_min)) :

                # Load data
                d_b_lambda, d_b_xi, d_b_ell    = res['Sfuncs'][which_var][which_comps[0]][q_order], res['Sfuncs'][which_var][which_comps[1]][q_order], res['Sfuncs'][which_var][which_comps[2]][q_order]
                #d_b_lambda2, d_b_xi2, d_b_ell2 = res['Sfuncs'][which_var][which_comps[0]][1], res['Sfuncs'][which_var][which_comps[1]][q_order], res['Sfuncs'][which_var][which_comps[2]][q_order]

                b_lambda  , b_xi  , b_ell   = res['Sfuncs']['l_ell_perp']/res['di'], res['Sfuncs']['l_Ell_perp']/res['di'], res['Sfuncs']['l_ell_par']/res['di']

                # Do wavevector anisotropy analysis
                ells, xis,  lambdas  = calculate_wavenumber_aniso(d_b_lambda, d_b_xi, d_b_ell, 
                                                                  b_lambda,   b_xi,   b_ell,
                                                                   smooth      = smooth,
                                                                   method      = method,
                                                                   window      = window
                                                                 )

                
                
                # Save wavenumber anisotropy

                
                # Estimate power anisotropy
                p_aniso_ell = np.array(d_b_lambda)/ np.array(d_b_ell)
                p_aniso_xi  = np.array(d_b_lambda)/ np.array(d_b_xi)
 
                
                
                # Estimate fits over ranges for wavenumber aniso
                fit1_w_xi   = func.find_fit(lambdas, xis, x01, xf1)
                fit1_w_ell  = func.find_fit(lambdas, ells, x01, xf1)
                fit2_w_xi   = func.find_fit(lambdas, xis, x02, xf2)
                fit2_w_ell  = func.find_fit(lambdas, ells, x02, xf2)
                
                 
                # Estimate fits over ranges for power aniso
                fit1_p_xi   = func.find_fit(b_lambda, p_aniso_xi, x01, xf1)
                fit1_p_ell  = func.find_fit(b_lambda, p_aniso_ell, x01, xf1)
                fit2_p_xi   = func.find_fit(b_lambda, p_aniso_xi, x02, xf2)
                fit2_p_ell  = func.find_fit(b_lambda, p_aniso_ell, x02, xf2)
                
                # Example combined check before appending
                if (not np.isnan(fit1_w_xi[0][0][1]) and fit1_w_xi[0][0][1] is not None and
                    not np.isnan(fit1_w_ell[0][0][1]) and fit1_w_ell[0][0][1] is not None and
                    not np.isnan(fit2_w_xi[0][0][1]) and fit2_w_xi[0][0][1] is not None and
                    not np.isnan(fit2_w_ell[0][0][1]) and fit2_w_ell[0][0][1] is not None and
                    not np.isnan(fit1_p_xi[0][0][1]) and fit1_p_xi[0][0][1] is not None and
                    not np.isnan(fit1_p_ell[0][0][1]) and fit1_p_ell[0][0][1] is not None and
                    not np.isnan(fit2_p_xi[0][0][1]) and fit2_p_xi[0][0][1] is not None and
                    not np.isnan(fit2_p_ell[0][0][1]) and fit2_p_ell[0][0][1] is not None):


                    fit1_w_xi_dict.append( fit1_w_xi[0][0][1])
                    fit1_w_ell_dict.append(fit1_w_ell[0][0][1])  
                    fit2_w_xi_dict.append( fit2_w_xi[0][0][1])
                    fit2_w_ell_dict.append( fit2_w_ell[0][0][1]) 

                    fit1_p_xi_dict.append( fit1_p_xi[0][0][1])
                    fit1_p_ell_dict.append(fit1_p_ell[0][0][1])  
                    fit2_p_xi_dict.append( fit2_p_xi[0][0][1])
                    fit2_p_ell_dict.append( fit2_p_ell[0][0][1])  


                
                    ells_yvm.append(ells)
                    xis_yvm.append(xis)
                    lambdas_yvm.append(lambdas)

                    sigma_c.append(fin['Par']['sigma_c_median'])
                    sigma_r.append(fin['Par']['sigma_r_median'])

                    power_aniso_xis.append(p_aniso_xi)
                    power_aniso_ells.append(p_aniso_ell)
                    power_aniso_xvals.append(b_lambda)
                

                try:
                    for  fit_window in fit_windows:

                        f_xis  = func.mov_fit_func(lambdas, xis, fit_window, 1e-3, 5e6)
                        f_ells = func.mov_fit_func(lambdas, ells, fit_window, 1e-3, 5e6)

                        p_xis  = func.mov_fit_func(b_lambda, p_aniso_xi, fit_window, 1e-3, 5e6)
                        p_ells = func.mov_fit_func(b_lambda, p_aniso_ell, fit_window, 1e-3, 5e6)

                        if counts==0:

                            fit_x_xis_dict_wave[str(fit_window)]   = list(f_xis['xvals'])
                            fit_y_xis_dict_wave[str(fit_window)]   = list(f_xis['plaw'])
                            fit_x_ells_dict_wave[str(fit_window)]  = list(f_ells['xvals'])
                            fit_y_ells_dict_wave[str(fit_window)]  = list(f_ells['plaw']) 


                            fit_x_xis_dict_power[str(fit_window)]   = list(p_xis['xvals'])
                            fit_y_xis_dict_power[str(fit_window)]   = list(p_xis['plaw'])
                            fit_x_ells_dict_power[str(fit_window)]  = list(p_ells['xvals'])
                            fit_y_ells_dict_power[str(fit_window)]  = list(p_ells['plaw'])  

                        else:

                            fit_x_xis_dict_wave[str(fit_window)]   = fit_x_xis_dict_wave[str(fit_window)]  + list(f_xis['xvals'])
                            fit_y_xis_dict_wave[str(fit_window)]   = fit_y_xis_dict_wave[str(fit_window)]  + list(f_xis['plaw'])
                            fit_x_ells_dict_wave[str(fit_window)]  = fit_x_ells_dict_wave[str(fit_window)] + list(f_ells['xvals'])
                            fit_y_ells_dict_wave[str(fit_window)]  = fit_y_ells_dict_wave[str(fit_window)] + list(f_ells['plaw'])

                            fit_x_xis_dict_power[str(fit_window)]   = fit_x_xis_dict_power[str(fit_window)]  + list(p_xis['xvals'])
                            fit_y_xis_dict_power[str(fit_window)]   = fit_y_xis_dict_power[str(fit_window)]  + list(p_xis['plaw'])
                            fit_x_ells_dict_power[str(fit_window)]  = fit_x_ells_dict_power[str(fit_window)] + list(p_ells['xvals'])
                            fit_y_ells_dict_power[str(fit_window)]  = fit_y_ells_dict_power[str(fit_window)] + list(p_ells['plaw'])


                except:
                    pass
                counts+=1
        except:
            
            ells_yvm.append(np.nan)
            xis_yvm.append(np.nan)
            lambdas_yvm.append(np.nan)

            sigma_c.append(np.nan)
            sigma_r.append(np.nan)


                
            power_aniso_xis.append(np.nan)
            power_aniso_ells.append(np.nan)
            power_aniso_xvals.append(np.nan)
            
            fit1_w_xi_dict.append(np.nan)
            fit1_w_ell_dict.append(np.nan)  
            fit2_w_xi_dict.append( np.nan)
            fit2_w_ell_dict.append( np.nan)
            
            
            fit1_p_xi_dict.append(np.nan)
            fit1_p_ell_dict.append(np.nan)  
            fit2_p_xi_dict.append( np.nan)
            fit2_p_ell_dict.append( np.nan)  
            traceback.print_exc()

    

    # Power anisotropy, p-laws
    final_plaw_power = {}
    final_plaw_wave  = {}
    for  fit_window in fit_windows:
        final_plaw_power[str(fit_window)] = {
                                                'xis_xv' : np.hstack(fit_x_xis_dict_power[str(fit_window)]),
                                                'xis_yv' : np.hstack(fit_y_xis_dict_power[str(fit_window)]),
                                                'ells_xv': np.hstack(fit_x_ells_dict_power[str(fit_window)]),
                                                'ells_yv': np.hstack(fit_y_ells_dict_power[str(fit_window)]),
                                            }
    
        final_plaw_wave[str(fit_window)] = {
                                                'xis_xv' : np.hstack(fit_x_xis_dict_wave[str(fit_window)]),
                                                'xis_yv' : np.hstack(fit_y_xis_dict_wave[str(fit_window)]),
                                                'ells_xv': np.hstack(fit_x_ells_dict_wave[str(fit_window)]),
                                                'ells_yv': np.hstack(fit_y_ells_dict_wave[str(fit_window)]),
                                            }   


    
    final_dict = {
                    'Power_aniso': {
                                        'xv'     : np.hstack(power_aniso_xvals),
                                    'xis_yv'     : np.hstack(power_aniso_xis),
                                    'ells_yv'    : np.hstack(power_aniso_ells),
                                    'plaws'      : final_plaw_power,
                        
                                    'xi_1_plaw'  : np.hstack(fit1_p_xi_dict),
                                    'ell_1_plaw' : np.hstack(fit1_p_ell_dict),  
                                    'xi_2_plaw'  : np.hstack(fit2_p_xi_dict),
                                    'ell_2_plaw' : np.hstack(fit2_p_ell_dict),  
                                   },
                    'sigma_c'   : np.hstack(sigma_c),
                    'sigma_r'   : np.hstack(sigma_r),
        
                    'Wave_aniso': {
                                    'lambdas'    : np.hstack(lambdas_yvm),
                                    'xis'        : np.hstack(xis_yvm),
                                    'ells'       : np.hstack(ells_yvm),
                                    'plaws'      : final_plaw_wave,
                                    'xi_1_plaw'  : np.hstack(fit1_w_xi_dict),
                                    'ell_1_plaw' : np.hstack(fit1_w_ell_dict),  
                                    'xi_2_plaw'  : np.hstack(fit2_w_xi_dict),
                                    'ell_2_plaw' : np.hstack(fit2_w_ell_dict),  
                                   },    
    
                  }
    
    
    return final_dict


def scanning_variance_analysis(index,
                                    df,
                                    av_hours,
                                    thresh_value,
                                    hours_needed,
                                    min_toler,
                                    save_path,
                                    three_sec_resol = True,
                                    sc              ='WIND',
                                    credentials     = None):
    
    os.path.join(os.getcwd(), 'functions','downloading_helpers' )
    from WIND import  LoadHighResMagWind
    from PSP  import  download_MAG_FIELD_PSP
    from SOLO import  download_MAG_SOLO
    
    
    def load_interval(start_date, 
                      end_date,
                      N,
                      sc ='WIND'):

        
        if sc == 'WIND':
            print('Considering WIND data!')
            # Define function to load data
            B, _,_ =LoadHighResMagWind( start_date,
                                   end_date,
                                   three_sec_resol,
                                   verbose  = True)
        elif sc =='PSP':
            print('Considering PSP data!')
            # Define function to load data
            start_date, end_date = func.ensure_time_format(start_date, end_date)
            B = download_MAG_FIELD_PSP(start_date,
                                       end_date,
                                       mag_resolution= 300,
                                       credentials   = credentials,
                                       varnames      = ['B_RTN'], 
                                       download_SCAM = False)
        elif sc =='SOLO':
            print('Considering SOLO data!')
            start_date, end_date = func.ensure_time_format(start_date, end_date)
            B = download_MAG_SOLO(  start_date,
                                   end_date,
                                   mag_resolution= 300,
                                   varnames      = ['B_RTN'], 
                                   download_SCAM = False)
            
        # Resample timeseries and estimate gaps

        resampled_df = func.resample_timeseries_estimate_gaps(B, 10)

        return resampled_df


    start_date     = df['Start'][index]
    end_date       = df['End'][index]
    
    
    # Define file name
    start_date_str = str(start_date)
    end_date_str   = str(end_date)
    filename       = start_date_str[0:4] + '_' + start_date_str[5:7] + '_' + start_date_str[8:10] + '__' + end_date_str[0:4] + '_' + end_date_str[5:7] + '_' + end_date_str[8:10] + '.pkl'
    
    
    fname2 = str(Path(save_path).joinpath(filename))
    print(fname2)
    E = pd.DataFrame(); selected_dates =pd.DataFrame()
    if not os.path.exists(fname2):

        try:
            print('Start:', start_date)
            print('End date:', end_date)

            # Load Magnetic field data from WIND
            resampled_df = load_interval(start_date,
                                         end_date,
                                         three_sec_resol, sc=sc)

            # Do variance ansisotropy analysis
            E = turb.variance_anisotropy_verdini_spec(int(av_hours * 3600 / resampled_df['Init_dt']),
                                                 resampled_df['resampled_df'][['Br', 'Bt', 'Bn']],
                                                 av_hours     = av_hours,
                                                 return_df    = True)

            # Find the intervals 
            selected_dates = turb.select_intervals_WIND_analysis(E.copy(), thresh_value, hours_needed, min_toler=min_toler)


            # save Selected intervals
            func.savepickle(selected_dates, save_path, filename)
        except:
            traceback.print_exc()
            pass
    return E, selected_dates



    
    
def select_files(fname,
                 gname,
                 final_df_name,
                 alname     = None,
                 lvals      = [8e3, 2e4],
                 conditions = None):

    res = pd.read_pickle(gname)
    fin = pd.read_pickle(final_df_name)
    try:
        al = pd.read_pickle(alname)
    except:
        pass

    d_min, d_max          = conditions['d_min'], conditions['d_max']
    sig_c_min, sig_c_max  = conditions['sigma_c_min'], conditions['sigma_c_max']
    p_f_mis, m_f_mis      = conditions['part_max_frac_mis'], conditions['mag_max_frac_mis']
    
    if 'min_dur' in conditions:
        min_dur = conditions['min_dur']
    
        # Estimate interval duration
        int_dur = (res['End_Time'] - res['Start_Time']).total_seconds() / 3600


    # Define quantities to be checked
    d     = res['d']
    if alname     == None:
        sig_c = fin['Par']['sigma_c_mean']
    else:
        #print('Trying new mwthod')
        try:
             sig_c    = np.nanmean(np.abs(((np.array(al['Zpm']['sig_c_mean'])[(al['l_di']>lvals[0]) & (al['l_di']<lvals[1])]))))
        except:
            traceback.print_exc()
            sig_c     = np.nan     
    p_mis = res['Fraction_missing_part']
    m_mis = res['Fraction_missing_MAG']
    
    
    conditions_final = (d > d_min) and (d < d_max) and (sig_c> sig_c_min) and (sig_c < sig_c_max)  and (p_mis< p_f_mis) and (m_mis< m_f_mis) 
    if 'min_dur' in conditions:
        if (conditions_final) &  (int_dur>min_dur) :
            return fname
    else:
        if conditions_final:
            return fname
    return None

def create_averaged_SF_df(load_path,
                          sf_name_path,
                          gen_name_path,
                          final_df_name_path,
                          norm_scales_di,
                          min_int_duration,
                          max_q_order,
                          find_corressponding_files = True,
                          conditions                = None,
                          load_path_2               = None,
                          self_norm                 = False):
    """
    Create an averaged structure function DataFrame.

    Args:
        load_path (str): Path to load files.
        sf_name_path (str): Path to structure function name files.
        gen_name_path (str): Path to generator name files.
        norm_scales_di (list): List containing min and max normalization scales.
        min_int_duration (float): Minimum interval duration.
        max_q_order (int): Maximum q order.

    Returns:
        pd.DataFrame: A DataFrame containing averaged structure function data.
    """
    import warnings
    
    # Ignore all warnings
    warnings.filterwarnings("ignore")

    # Load files
    f_names         = func.load_files(load_path, sf_name_path)

    print('Initial number of files considered', len(f_names))
    
    if load_path_2 is not None:
        print('Also considering 2 mission data')
        f_names2        = func.load_files(load_path_2, sf_name_path)
        f_names         = list(f_names)   + list(f_names2)

        
    print('Number of files considered after thresholds are imposed', len(f_names))
    

    # Extract normalization scales
    min_norm_di, max_norm_di = norm_scales_di

    # Initialize data containers
    sf_data = {key: {'Counts': [], 'xvals': []} for key in ['ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall']}
    
    sf_data_B = {key: {'yvals': {q: [] for q in range(1, max_q_order + 1)}}
                 for key in ['ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall']}
    sf_data_V = {key: {'yvals': {q: [] for q in range(1, max_q_order + 1)}}
                 for key in ['ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall']}
    
    count_intervals = 0
    
    for j, fname in enumerate(f_names):
        
        
        try:
            # Load file
            sf_dict = pd.read_pickle(fname)
            gen_dict = pd.read_pickle(fname.replace(sf_name_path, gen_name_path))

            if j == 0:
                max_qorder = np.shape(sf_dict['Sfuncs']['B']['ell_perp'])[0]
                qorders = np.arange(1, max_qorder + 1)

            # Estimate interval duration
            int_dur = (gen_dict['End_Time'] - gen_dict['Start_Time']).total_seconds() / 3600

            # Check minimum interval size
            if int_dur > min_int_duration:


                # Assign ell values
                ell_vals = sf_dict['ell_di']

                for key in sf_data:
                    for qorder in qorders:
                        # Assign  sfunctions
                        SF_q_B   = sf_dict['Sfuncs']['B'][key][qorder - 1]
                        SF_q_V   = sf_dict['Sfuncs']['V'][key][qorder - 1]

                        # Find normalization values for q order
                        mask = (ell_vals > min_norm_di) & (ell_vals < max_norm_di)

                        # Normalize and store the y values
                        if self_norm ==1 :

                            fB = interpolate.interp1d(ell_vals, SF_q_B)
                            fV = interpolate.interp1d(ell_vals, SF_q_V)

                            Bnorm_val_di_q  =  fB(min_norm_di)
                            Vnorm_val_di_q  =  fV(min_norm_di)

                            if (qorder==2) & (key =='ell_perp'):
                                if Bnorm_val_di_q>0:
                                    count_intervals += 1

                            if Bnorm_val_di_q>0:
                                sf_data_B[key]['yvals'][qorder].append(SF_q_B / Bnorm_val_di_q)
                                sf_data_V[key]['yvals'][qorder].append(SF_q_V / Vnorm_val_di_q) 

                                if qorder==2:      

                                    sf_data[key]['xvals'].append(ell_vals)
                                    sf_data[key]['Counts'].append(sf_dict['Sfuncs'][f'counts_{key}'])
                                    
                                    
                        elif self_norm ==2:

                            if (qorder==2) & (key =='ell_perp'):
                                #if Bnorm_val_di_q>0:
                                count_intervals += 1

                            if qorder>1:
                                sf_data_B[key]['yvals'][qorder].append(SF_q_B / sf_dict['Sfuncs']['B'][key][0])
                                sf_data_V[key]['yvals'][qorder].append(SF_q_V / sf_dict['Sfuncs']['V'][key][0]) 
                            else:
                                
                                SF_q_B_ov = sf_dict['Sfuncs']['B']['ell_overall'][qorder - 1]
                                SF_q_V_ov = sf_dict['Sfuncs']['V']['ell_overall'][qorder - 1]
                                

                                fB = interpolate.interp1d(ell_vals, SF_q_B_ov)
                                fV = interpolate.interp1d(ell_vals, SF_q_V_ov)

                                Bnorm_val_di_q  =  fB(min_norm_di)
                                Vnorm_val_di_q  =  fV(min_norm_di)
                                
                                if Bnorm_val_di_q>0:

                                    sf_data_B[key]['yvals'][qorder].append(SF_q_B / Bnorm_val_di_q)
                                    sf_data_V[key]['yvals'][qorder].append(SF_q_V / Vnorm_val_di_q)                                
                                
                            if qorder==2:       
                                sf_data[key]['xvals'].append(ell_vals)
                                sf_data[key]['Counts'].append(sf_dict['Sfuncs'][f'counts_{key}'])
                        else:

                            # Create an interpolation function using linear interpolation


                            SF_q_B_ov = sf_dict['Sfuncs']['B']['ell_overall'][qorder - 1]
                            SF_q_V_ov = sf_dict['Sfuncs']['V']['ell_overall'][qorder - 1]


                            fB = interpolate.interp1d(ell_vals, SF_q_B_ov)
                            fV = interpolate.interp1d(ell_vals, SF_q_V_ov)

                            Bnorm_val_di_q  =  fB(min_norm_di)
                            Vnorm_val_di_q  =  fV(min_norm_di)

                            if (qorder==2) & (key =='ell_perp'):
                                if Bnorm_val_di_q>0:
                                    count_intervals += 1


                            if Bnorm_val_di_q>0:

                                sf_data_B[key]['yvals'][qorder].append(SF_q_B / Bnorm_val_di_q)
                                sf_data_V[key]['yvals'][qorder].append(SF_q_V / Vnorm_val_di_q)

                                if qorder==2:

                                    sf_data[key]['xvals'].append(ell_vals)
                                    sf_data[key]['Counts'].append(sf_dict['Sfuncs'][f'counts_{key}'])
                                    
        except:
            traceback.print_exc()
            pass

    print(f'Considering {count_intervals} Intervals in total')

    # Convert the collected data into a DataFrame
    sf_df_data = {
        'B': {
            key: {
                'Counts': sf_data[key]['Counts'],
                'xvals': sf_data[key]['xvals'],
                **{f'yvals{q}': np.array(sf_data_B[key]['yvals'][q]) for q in qorders}
            }
            for key in sf_data
        },
        'V': {
            key: {
                'Counts': sf_data[key]['Counts'],
                'xvals': sf_data[key]['xvals'],
                **{f'yvals{q}': np.array(sf_data_V[key]['yvals'][q]) for q in qorders}
            }
            for key in sf_data
        }
    }

    sf_df = pd.DataFrame(sf_df_data)

    return sf_df, f_names, gen_names, final_df_names


def estimate_averaged_sfunc(df,
                            field,
                            component,
                            max_qorder, 
                            max_std=12, 
                            nbins=100):

    xvals  = np.concatenate(df[field][component]['xvals'])
    counts = np.concatenate(df[field][component]['Counts'])
    yvals  = np.concatenate(df[field][component]['yvals1'])

    
    print(np.shape(xvals), np.shape(counts), np.shape(yvals))
    mask = (xvals > 0) & (counts > 0) & (yvals > 0) & (~np.isinf(xvals) ) & (~np.isinf(yvals) )  & (~np.isinf(counts) ) 
    xvals, counts, yvals = xvals[mask], counts[mask], yvals[mask]

    lbins          = np.logspace(np.log10(np.nanmin(xvals)), np.log10(np.nanmax(xvals)), nbins)
    indices        = np.digitize(xvals, lbins)
    unique_indices = np.unique(indices)

    num_bins = len(lbins) - 1
    yvals = np.zeros((num_bins, max_qorder))

    for qorder in range(1, max_qorder + 1):
        yvals_q = np.concatenate(df[field][component][f'yvals{qorder}'])[mask]

        for unique_idx in unique_indices:
            index = np.where(indices == unique_idx)[0]
            n_counts, n_yvals = counts[index], yvals_q[index]
            std_y             = np.nanstd(n_yvals)
 
            # if  qorder ==1:
            #     fin_index = np.ones(len(n_yvals)).astype(int)
            # else:
            fin_index = n_yvals < max_std * std_y
      
            if 0 <= unique_idx - 1 < num_bins:  # Check index bounds
                yvals[unique_idx - 1, qorder - 1] = np.nansum(n_yvals[fin_index] * (n_counts[fin_index])) / np.nansum(n_counts[fin_index])

                #print(qorder)
    bin_centersout = lbins[:-1] + np.log10(0.5) * (lbins[1:] - lbins[:-1])

    return bin_centersout, yvals


def merge_all_intervals( var_2_keep, fname, j):
    
    try:
        #fnames =6
        if np.mod(j,10)==0:
            func.progress_bar(j, 600)

        res = pd.read_pickle(fname)

        r = res['flucts']['ell_all']['kinet_normal']*res['flucts']['ell_all']['R']
        t = res['flucts']['ell_all']['kinet_normal']*res['flucts']['ell_all']['T']
        n = res['flucts']['ell_all']['kinet_normal']*res['flucts']['ell_all']['N']

        rv = res['flucts']['ell_all']['V_R']
        tv = res['flucts']['ell_all']['V_T']
        nv = res['flucts']['ell_all']['V_N']

        
        maskv = (rv>1e-6) &(tv>1e-6) &(nv>1e-6)
        maskr = (r>1e-16)# &(t>1e-6) &(n>1e-6)

        del r, t, n, rv, tv, nv


        temp_df            = pd.DataFrame(res['flucts']['ell_all'])[var_2_keep][maskv&maskr]

        temp_df['lambda']  = temp_df['lambda'] / res['di']
        temp_df['sins']    = temp_df['sins_zpm_num'] / temp_df['sins_zpm_den']
        temp_df['sig_c']   = np.abs(temp_df['sig_c'])

        del_cols           = [ 'thetas', 'phis']
        all_l              =  temp_df[(temp_df['compress_simple_V'] < 10) &(temp_df['compress_simple'] < 10)   ]
        #print('len', len(all_l))
        return all_l
    except:
        traceback.print_exc()
        pass