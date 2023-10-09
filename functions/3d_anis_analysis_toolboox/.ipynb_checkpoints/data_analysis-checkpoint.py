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

os.path.join(os.getcwd(), 'functions','downloading_helpers' )
from PSP import  download_ephemeris_PSP

os.path.join(os.getcwd(), 'functions','3d_anis_analysis_toolboox' )
import collect_wave_coeffs 




def download_files( ok,
                    df,
                    final_path,
                    only_one_interval,
                    step,
                    duration,
                    addit_time_around,
                    settings,
                    vars_2_downnload,
                    cdf_lib_path,
                    credentials,
                    gap_time_threshold,
                    estimate_PSD_V,
                    subtract_rol_mean,
                    rolling_window,
                    f_min_spec,
                    f_max_spec,
                    estimate_PSD,
                    sc,
                    high_resol_data,
                    in_RTN,
                    three_sec_resol= False):
    
    try:
        t0 = df['Start'][ok]
        t1 = df['End'][ok]

        """Setup for main function"""
        tstarts, tends, tfmt, path0  = calc.set_up_main_loop(final_path, only_one_interval,t0, t1, step, duration)
        
       

        start_time  = df['Start'][ok]
        end_time    = df['End'][ok]
        

        # Define folder name
        foldername  = "%s_%s_sc_%d" %(str(start_time.strftime(tfmt)), str(end_time.strftime(tfmt)), 0)

        if not os.path.exists(path0.joinpath(foldername).joinpath('final_data.pkl')):
            print('Folder name', path0.joinpath(foldername))
            # Running the main function
            big_gaps, flag_good, final, general, sig_c_sig_r_timeseries, dfdis = calc.final_func(
                                                                                            start_time         , 
                                                                                            end_time           , 
                                                                                            addit_time_around  ,
                                                                                            settings           , 
                                                                                            vars_2_downnload   ,
                                                                                            cdf_lib_path       ,
                                                                                            credentials        ,
                                                                                            gap_time_threshold ,
                                                                                            estimate_PSD_V     ,
                                                                                            subtract_rol_mean  ,
                                                                                            rolling_window     ,
                                                                                            f_min_spec         ,
                                                                                            f_max_spec         ,  
                                                                                            estimate_PSD       , 
                                                                                            sc                 , 
                                                                                            high_resol_data    ,
                                                                                            in_RTN             ,
                                                                                            three_sec_resol = three_sec_resol
                                                                                          )
            try:
                final['Par']['V_resampled'] = final['Par']['V_resampled'].join(func.newindex(dfdis[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']],
                                                                                             final['Par']['V_resampled'].index))

            except:
                pass


            if flag_good == 1:
                os.makedirs(path0.joinpath(foldername), exist_ok=True)

                pickle.dump(final,open(path0.joinpath(foldername).joinpath("final_data.pkl"),'wb'))
                pickle.dump(general,open(path0.joinpath(foldername).joinpath("general.pkl"),'wb'))
                pickle.dump(sig_c_sig_r_timeseries,open(path0.joinpath(foldername).joinpath("sig_c_sig_r.pkl"),'wb'))

                print("%d out of %d finished" %(ok, len(df)))
            else:
                os.makedirs(path0.joinpath(foldername), exist_ok=True)
                print("%s - %s failed!" %(ok, len(df)))   
    except Exception as e:
        print('failed at index', ok, 'with error:', str(e))
        traceback.print_exc()



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
                                    max_interval_dur         =  240):
    
    import warnings
    
    # Ignore all warnings
    warnings.filterwarnings("ignore")
    
    try:
    
        # Print progress
        func.progress_bar(i, len(fnames))

        # Load files
        res = pd.read_pickle(fnames[i])
        gen = pd.read_pickle(gen_names[i])
        
        # Duration of interval in hours
        dts = (gen['End_Time']- gen['Start_Time']).total_seconds()/3600
        
        if dts<max_interval_dur:
            print('Considering an interval of duration:', dts,'[hrs]')
                    
            if strict_thresh:
                strict_suffix = '5deg_'
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
                fname          = f"{general_suffix}{npoints_suffix}{strict_suffix}{conditions_suffix}{vsc_suffix}{sfuncs_suffix}.pkl"
            
                
            # Check wether file alredy exists
            check_file     = str(Path(gen_names[i][:-11]).joinpath('final').joinpath(fname))
                                 
            # Now work on data!
            if (not os.path.exists(check_file)) or (overwrite_existing_files):

                                 
                if (overwrite_existing_files) & (os.path.exists(check_file)):
                    print('Overwriting', check_file ,'per your commands ')
                else:
                    print('Working on new file: ', check_file)

                # Choose V, B dataframes
                B  = res['Mag']['B_resampled'][['Br', 'Bt', 'Bn']]
                V  = res['Par']['V_resampled'][['Vr', 'Vt', 'Vn']]
                Np = res['Par']['V_resampled'][['np']]


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
                    V[['Vr', 'Vt', 'Vn']] = res['Par']['V_resampled'][['Vr', 'Vt', 'Vn']].values - ephem[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']].interpolate().values

                    # Keep Vsw to normalize
                    Vsw_norm              = np.nanmean(np.sqrt(V['Vr']**2 + V['Vt']**2 + V['Vn']**2))
                else:
                    Vsw_norm              = np.nanmean(np.sqrt(res['Par']['V_resampled']['Vr']**2 + res['Par']['V_resampled']['Vt']**2 + res['Par']['V_resampled']['Vn']**2))

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
                thetas, phis, flucts, ell_di, Sfunctions, PDFs, overall_align_angles = threeD.estimate_3D_sfuncs(B,
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
                                                                                                                 return_B_in_vel_units    = return_B_in_vel_units
                                                                                                                 
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


#Define function to process each file in parallel
def process_file(data_file, data_file_gen, data_file_fin, data_file_align, max_d, min_d, qorders, general_sf):
    # Call process_data function on each file
    result = process_data(data_file, data_file_gen, data_file_fin, data_file_align,  max_d, min_d, qorders, general_sf)
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
                     normalize_sfuncs ='with_mean'):
    if E1_only:
        if five_point:
            prefix = '5pt'
        else:
            prefix = '2pt'
        if strict_thresh:
            res = pd.read_pickle(path+str(duration)+'_'+str(step)+'_binned_data/trace_SF/final_sfuncs/E1_'+str(prefix)+'_alfvenic_SFuncs_final_slow_5deg.pkl')
        else:
            res = pd.read_pickle(path+str(duration)+'_'+str(step)+'_binned_data/trace_SF/final_sfuncs/E1_'+str(prefix)+'_alfvenic_SFuncs_final_slow.pkl')
            


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
                keep_row = res[int(N):int(N+1)]

                if E1_only:
                    conditions = 0
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

                if conditions or (E1_only and keep_row['Vsw'].values[0] < max_Vsw):
                    sig_c.append(keep_row['sig_c'].values[0])
                    B_0.append(np.sqrt(keep_row['B_mean_sq'].values[0]))
                    xx         = keep_row['ells_'+str(which_one)].values[0]
                    yvals_keep = keep_row['sfuncs_'+str(which_one)].values[0][qorder][(xx>min_norm) & (xx<max_norm)]
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
                        #chi_array_db_over_B0_sin.append((keep_row['sfuncs_'+str(which_one)].values[0][qorder]* #keep_row['VB_reg_sin'].values[0])/np.sqrt(keep_row['B_mean_sq']).values[0])
#                    elif qorder==1:
#                        chi_array_sqrt_db_sq_over_B0.append((np.sqrt(keep_row['sfuncs_'+str(which_one)].values[0][qorder]* #keep_row['VB_reg_sin'].values[0]))/np.sqrt(keep_row['B_mean_sq']).values[0])                        
                    
            if qorder ==0:
                wave_chi_array_db_over_B0 = chi_array_db_over_B0
                wave_chi_array_db_over_B0_sin     = chi_array_db_over_B0_sin
            if qorder==1:
                wave_xvals = xvals
                wave_yvals = yvals
                wave_sig_c = sig_c
                wave_B0    = B_0
                wave_chi_array_sqrt_db_sq_over_B0 = chi_array_sqrt_db_sq_over_B0
               
                
            #print(xvals)
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
                wave_chi_array_db_over_B0 = chi_array_db_over_B0
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



def alignment_anlges(duration,
                     step,
                     what,
                     std,
                     wind,
                     loglog,
                     zesen_ints= 0):
    
    if zesen_ints==0:
        base_path = Path('/Users/nokni/work/3d_anisotropy/structure_functions_E1/data')
        data_path = base_path / f'{duration}_{step}_final_fixed'
        
        data_filess = np.sort(list(data_path.glob('*/5point_sfuncs_5deg.pkl')))
        data_filess_align = np.sort(list(data_path.glob('*/alignment_angles.pkl')))
    else:
        data_path = Path('/Users/nokni/work/3d_anisotropy/structure_functions_E1/3plaw_data_zesen/intervals')
        
        data_filess = np.sort(list(data_path.glob('*/final/5point_sfuncs.pkl')))
        data_filess_align = np.sort(list(data_path.glob('*/final/alignment_angles.pkl')))


    
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
    for data_files, data_files_align in zip(data_filess, data_filess_align):
        res = pd.read_pickle(data_files_align)
        res1 = pd.read_pickle(data_files)
        
        xvals.append(res1['ell_di'])
        
        zpm_ang_w.append(np.arcsin(res['Zpm']['weighted'])*180/np.pi)
        VB_ang_w.append(np.arcsin(res['VB']['weighted'])*180/np.pi)
        
        zpm_ang_reg.append(np.arcsin(res['Zpm']['reg'])*180/np.pi)
        VB_ang_reg.append(np.arcsin(res['VB']['reg'])*180/np.pi)
        VB_sin_reg.append(res['VB']['reg'])
        
        zpm_ang_pol.append(np.arcsin(res['Zpm']['polar'])*180/np.pi)
        VB_ang_pol.append(np.arcsin(res['VB']['polar'])*180/np.pi)  
        
        sig_r_mean.append(res['VB']['sig_r_mean'])
        sig_r_median.append(res['VB']['sig_r_median'])
        sig_c_mean.append(res['Zpm']['sig_c_mean'])
        sig_c_median.append(res['Zpm']['sig_c_median'])
        
        
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
    gen_names       = func.load_files(load_path, gen_name_path)
    final_df_names  = func.load_files(load_path, final_df_name_path)
    
    print('Initial number of files considered', len(final_df_names))
    
    if load_path_2 is not None:
        print('Also considering 2 mission data')
        f_names2        = func.load_files(load_path_2, sf_name_path)
        gen_names2      = func.load_files(load_path_2, gen_name_path)
        final_df_names2 = func.load_files(load_path_2, final_df_name_path)
        
        f_names         = list(f_names)   + list(f_names2)
        gen_names       = list(gen_names) + list(gen_names2 ) 
        final_df_names  = list(final_df_names) + list(final_df_names2 ) 
    

    
    if find_corressponding_files:
    
    
        f_names, gen_names          = func.find_matching_files_with_common_parent(f_names,  sf_name_path , gen_names, gen_name_path,  num_parents_f=2, num_parents_g=1)
        final_df_names, gen_names   = func.find_matching_files_with_common_parent(final_df_names,  final_df_name_path , gen_names, gen_name_path,  num_parents_f=1, num_parents_g=1)


        
        # Parallelize the processing of fnames and gnames using joblib
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(select_files)(fname,
                                                 gname,
                                                 final_df_name,
                                                 conditions =conditions)
            for fname, gname, final_df_name in zip(f_names, gen_names, final_df_names)
        )

        # Filter out the None values from the results (if any)
        f_names = [f_name for f_name in results if f_name is not None]
        
    
        f_names, gen_names          = func.find_matching_files_with_common_parent(f_names,  sf_name_path , gen_names, gen_name_path,  num_parents_f=2, num_parents_g=1)
        final_df_names, gen_names   = func.find_matching_files_with_common_parent(final_df_names,  final_df_name_path , gen_names, gen_name_path,  num_parents_f=1, num_parents_g=1)
        
        
        print('Number of files considered after thresholds are imposed', len(final_df_names))
    

    # Extract normalization scales
    min_norm_di, max_norm_di = norm_scales_di

    # Initialize data containers
    sf_data = {key: {'Counts': [], 'xvals': []} for key in ['ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall']}
    
    sf_data_B = {key: {'yvals': {q: [] for q in range(1, max_q_order + 1)}}
                 for key in ['ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall']}
    sf_data_V = {key: {'yvals': {q: [] for q in range(1, max_q_order + 1)}}
                 for key in ['ell_perp', 'Ell_perp', 'ell_par', 'ell_par_rest', 'ell_overall']}
    
    count_intervals = 0
    
    for j, (fname, gen_name) in enumerate(zip(f_names, gen_names)):
        try:
            # Load file
            sf_dict = pd.read_pickle(fname)
            gen_dict = pd.read_pickle(gen_name)

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