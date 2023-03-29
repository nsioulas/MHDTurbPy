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

sys.path.insert(1, os.path.join(os.getcwd(), 'functions/downloading_helpers'))
from  PSP import  LoadTimeSeriesPSP
from  SOLO import LoadTimeSeriesSOLO


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


def compute_sigma(psd_A, psd_B, f, f_min, f_max):
    from scipy.integrate import trapz

    mask = (f>f_min) & (f<f_max) & (~np.isnan(psd_A)) & (~np.isnan(psd_B))
    integral_A = trapz(psd_A[mask], f[mask])
    integral_B = trapz(psd_B[mask], f[mask])
    
    return (integral_A - integral_B) / (integral_A + integral_B)


def estimate_quants_particle_data(estimate_PSDv, rolling_window, f_min_spec, f_max_spec,  in_rtn, df, mag_resampled, subtract_rol_mean, smoothed = True):
    # Particle data remove nan values
    df_part = df.dropna()
   
    # Reindex magnetic field data to particle data index
    dtv                 = func.find_cadence(df_part)
    dtb                 = func.find_cadence(mag_resampled)
    
    # We need plasma timeseries cadence
    freq_final          = str(int(dtv*1e3))+'ms'

    # Combine magnetic field and particle data, resampled to final frequency
    if dtv>dtb:
        mag_resampled = func.newindex(mag_resampled, df_part.index)
    else:
        df_part = func.newindex(df_part, mag_resampled.index)
    f_df          = mag_resampled.join(df_part).interpolate()

    # Calculate magnetic field magnitude
    Bx, By, Bz = f_df.values.T[:3]
    Bmag       = np.sqrt(Bx**2 + By**2 + Bz**2)

    if subtract_rol_mean:
        try:
            columns = [['Br', 'Bt', 'Bn'], ['Vr', 'Vt', 'Vn', 'np']]
            for c in columns:
                f_df[[f"{col}_mean" for col in c]] = f_df[c].rolling(rolling_window, center=True).mean().interpolate()
        except:
            columns = [['Bx', 'By', 'Bz'], ['Vx', 'Vy', 'Vz', 'np']]
            for c in columns:
                f_df[[f"{col}_mean" for col in c]] = f_df[c].rolling(rolling_window, center=True).mean().interpolate()       


    #Estimate median solar wind speed   
    Vth                     = f_df.Vth.values
    Vth[Vth < 0]            = np.nan
    Vth_mean                = np.nanmedian(Vth)
    Vth_std                 = np.nanstd(Vth);
    
    #Estimate median solar wind speed  

    Vx                       = f_df["Vr"].values if in_rtn else f_df["Vx"].values
    Vy                       = f_df["Vt"].values if in_rtn else f_df["Vy"].values
    Vz                       = f_df["Vn"].values if in_rtn else f_df["Vz"].values

    Vsw                      = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    Vsw[(np.abs(Vsw) > 1e5)] = np.nan
    Vsw_mean                 = np.nanmedian(Vsw)
    Vsw_std                  = np.nanstd(Vsw)

    # estimate mean number density
    Np                       = f_df['np'].values 
    Np_mean                  = np.nanmedian(Np)
    Np_std                   = np.nanstd(Np);
        
    # Estimate Ion inertial length di in [Km]
    di                       = 228/np.sqrt(Np)
    di[di< 1e-3]             = np.nan
    di_mean                  = np.nanmedian(di) 
    di_std                   = np.nanstd(di);

    # Estimate plasma Beta
    B_mag       = Bmag * nT2T                              # |B| units:      [T]
    temp        = 1./2 * m_p * (Vth*km2m)**2               # in [J] = [kg] * [m]^2 * [s]^-2
    dens        = Np/(cm2m**3)                             # number density: [m^-3] 
    beta        = (dens*temp)/((B_mag**2)/(2*mu_0))        # plasma beta 
    beta[beta < 0] = np.nan
    beta[np.abs(np.log10(beta))>4] = np.nan # delete some weird data
    beta_mean   = np.nanmedian(beta); beta_std   = np.nanstd(beta);
    
    
    # ion gyro radius
    rho_ci = 10.43968491 * Vth/B_mag #in [km]
    rho_ci[rho_ci < 0] = np.nan
    rho_ci[np.log10(rho_ci) < -3] = np.nan
    rho_ci_mean =np.nanmedian(rho_ci); rho_ci_std =np.nanstd(rho_ci);
 
    ### Define b and v ###
    if in_rtn:
        columns_v = ["Vr", "Vt", "Vn"] if "Vr" in f_df.columns else ["Vx", "Vy", "Vz"]
        columns_b = ["Br", "Bt", "Bn"] if "Br" in f_df.columns else ["Bx", "By", "Bz"]
    else:
        columns_v = ["Vx", "Vy", "Vz"]
        columns_b = ["Bx", "By", "Bz"]

    # Assign values
    vr, vt, vn = f_df[columns_v].values.T
    br, bt, bn = f_df[columns_b].values.T


    # Const to normalize mag field in vel units
    kinet_normal = 1e-15 / np.sqrt(mu0 * f_df['np_mean'].values * m_p)

    # Estimate Alfv speed
    Va_ts = np.array([br, bt, bn]) * kinet_normal
    
    # Estimate SW speed
    V_ts = np.array([vr, vt, vn])
    
    # Estimate mean values of both for interval
    alfv_speed, sw_speed = [np.nanmean(x, axis=0) for x in (Va_ts, V_ts)]
    
    # Estimate VB angle
    vbang = func.angle_between_vectors(Va_ts.T,V_ts.T)
    
    #End its mean
    VBangle_mean, VBangle_std = np.nanmean(vbang), np.nanstd(vbang)
    
    #Sign of Br forrolling window
    try:
        signB = - np.sign(f_df['Br_mean'])
    except:
        signB = np.abs(- np.sign(f_df['Bx_mean']))

    # Estimate fluctuations
    if subtract_rol_mean:
        try:
            dva    = Va_ts  -  f_df[['Br_mean', 'Bt_mean', 'Bn_mean']].values.T * kinet_normal
            dv     = V_ts   -  f_df[['Vr_mean', 'Vt_mean', 'Vn_mean']].values.T
        except:
            dva    = Va_ts  -  f_df[['Bx_mean', 'By_mean', 'Bz_mean']].values.T * kinet_normal
            dv     = V_ts   -  f_df[['Vx_mean', 'Vy_mean', 'Vz_mean']].values.T
    else:
        dva    = Va_ts  -  np.nanmean(alfv_speed, axis=0)
        dv     = V_ts   -  np.nanmean(sw_speed, axis=0)

    # Estimate Zp, Zm components  
    Zpr, Zmr = dv[0] + signB * dva[0], dv[0] - signB * dva[0]
    Zpt, Zmt = dv[1] + signB * dva[1], dv[1] - signB * dva[1]
    Zpn, Zmn = dv[2] + signB * dva[2], dv[2] - signB * dva[2]
    
    # Estimate energy in Zp, Zm
    Z_plus_squared     = Zpr**2 + Zpt**2 + Zpn**2
    Z_minus_squared    = Zmr**2 + Zmt**2 + Zmn**2
    
    # Estimate amplitude of fluctuations
    Z_amplitude        = np.sqrt((Z_plus_squared + Z_minus_squared) / 2)
    Z_amplitude_mean   = np.nanmedian(Z_amplitude)
    Z_amplitude_std    = np.nanstd(Z_amplitude)


    # Kin, mag energy
    Ek           = dv[0]**2  + dv[1]**2  + dv[2]**2
    Eb           = dva[0]**2 + dva[1]**2 + dva[2]**2
    
    
    #Estimate normalized residual energy
    sigma_r      = (Ek-Eb)/(Ek+Eb);                                                         sigma_r[np.abs(sigma_r) > 1e5] = np.nan;
    sigma_c      = (Z_plus_squared - Z_minus_squared)/( Z_plus_squared + Z_minus_squared);  sigma_c[np.abs(sigma_c) > 1e5] = np.nan
    
    nn_df       = pd.DataFrame({'DateTime': f_df.index.values,
                                'Zpr'     : Zpr,     'Zpt'  : Zpt,   'Zpn' : Zpn,
                                'Zmr'     : Zmr,     'Zmt'  : Zmt,   'Zmn' : Zmn, 
                                'va_r'    : dva[0],  'va_t' : dva[1],'va_n': dva[2],
                                'v_r'     : dv[0],   'v_t'  : dv[1], 'v_n' : dv[2],
                                'beta'    : beta,    'np'   : Np,    'Tp'  : temp, 'VB': vbang,
                                'sigma_c' : sigma_c,              'sigma_r': sigma_r}).set_index('DateTime')
    nn_df       = nn_df.mask(np.isinf(nn_df)).dropna().interpolate(method='linear')

    
    # Estimate mean, median,... of  normalized residual energy
    sigma_r_mean = np.nanmean(sigma_r); sigma_r_median =np.nanmedian(sigma_r); sigma_r_std =np.nanstd(sigma_r);
    
    # Estimate mean, median,... of  normalized cross helicity
    sigma_c_mean = np.nanmean(np.abs(sigma_c)); sigma_c_median = np.nanmedian(np.abs(sigma_c)); sigma_c_std = np.nanstd(np.abs(sigma_c));     sigma_c_median_no_abs  = np.nanmedian(sigma_c);     sigma_c_mean_no_abs    = np.nanmean(sigma_c);   

    #Estimate  z+, z- PSD
    if estimate_PSDv:
        f_Zplus, psd_Zplus    = turb.TracePSD(nn_df['Zpr'].values, nn_df['Zpt'].values, nn_df['Zpn'].values, 0,  part_resolution)
        f_Zminus, psd_Zminus  = turb.TracePSD(nn_df['Zmr'].values, nn_df['Zmt'].values, nn_df['Zmn'].values, 0,  part_resolution)
    else:
        f_Zplus, psd_Zplus    = None, None
        f_Zminus, psd_Zminus  = None, None 

    #Estimate  v,b PSD
    if estimate_PSDv==0:
        f_vv, psd_vv         = None, None
        f_bb, psd_bb         = None, None
    else:
        f_vv, psd_vv         = turb.TracePSD(nn_df['v_r'].values, nn_df['v_t'].values, nn_df['v_n'].values, 0,  part_resolution)
        f_bb, psd_bb         = turb.TracePSD(nn_df['va_r'].values, nn_df['va_t'].values, nn_df['va_n'].values, 0,  part_resolution)      
    

    # Only keep indices within the range of frequencies specified
    if estimate_PSDv:
        sigma_c_spec = compute_sigma(psd_Zplus, psd_Zminus, f_Zplus, f_min_spec, f_max_spec)
        sigma_r_spec = compute_sigma(psd_vv, psd_bb, f_Zplus, f_min_spec, f_max_spec)
    else:
    
        sigma_c_spec    = None
        sigma_r_spec    = None
        
    #  Estimate normalized cross helicity and normalized residual energy spectraly
    sigma_c_spec = compute_sigma(psd_Zplus, psd_Zminus, f_Zplus, f_min_spec, f_max_spec) if estimate_PSDv else None
    sigma_r_spec = compute_sigma(psd_vv, psd_bb, f_Zplus, f_min_spec, f_max_spec) if estimate_PSDv else None
    
    # Also keep a dict containing psd_vv, psd_bb, psd_Zplus, psd_Zminus
    dict_psd = {"f_vb": f_vv, 'psd_v': psd_vv, "psd_b": psd_bb, "f_zpm":f_Zplus, "psd_zp":psd_Zplus , "psd_zm":psd_Zminus} if estimate_PSDv else {}

  
    return  dict_psd, nn_df, sw_speed, alfv_speed,  f_Zplus, psd_Zplus, f_Zminus, psd_Zminus, sigma_r_spec, sigma_c_spec, beta_mean, beta_std, Z_amplitude_mean, Z_amplitude_std, sigma_r_mean, sigma_r_median, sigma_r_std, sigma_c_median, sigma_c_mean, sigma_c_median_no_abs, sigma_c_mean_no_abs, sigma_c_std, Vth_mean, Vth_std , Vsw_mean, Vsw_std, Np_mean, Np_std, di_mean, di_std, rho_ci_mean, rho_ci_std, VBangle_mean, VBangle_std





def calc_mag_diagnostics(diagnostics, gap_time_threshold, dist, estimate_SPD, mag_data, mag_resolution):


    """ Make distance dataframe to begin at the same time as magnetic field timeseries """ 
    
    r_psp            = np.nanmean(func.use_dates_return_elements_of_df_inbetween(mag_data.index[0], mag_data.index[-1], dist['Dist_au']))

    """ Interpolate gaps"""  
    mag_interpolated = mag_data.interpolate(method = 'linear').dropna()

    try:
        if estimate_SPD:
            """Estimate PSD of  magnetic field"""
            f_B, psd_B                         = turb.TracePSD(
                                                                mag_interpolated.values.T[0],
                                                                mag_interpolated.values.T[1],
                                                                mag_interpolated.values.T[2],
                                                                True,
                                                                diagnostics["resol"]*1e-3
            )

            """Smooth PSD of  magnetic field"""    
            f_B_mid, f_B_mean, psd_B_smooth    =  func.smoothing_function(
                                                                           f_B, 
                                                                           psd_B,
                                                                           window=2,
                                                                           pad = 1
            )
        else:
            f_B, psd_B                         = None, None  
            f_B_mid, f_B_mean, psd_B_smooth    = None, None, None 

    except:
        traceback.print_exc()  


    general_dict = {
            "Start_Time"           :  mag_data.index[0],
            "End_Time"             :  mag_data.index[-1],  
            "d"                    :  r_psp,
            "Fraction_missing_MAG" :  diagnostics["Frac_miss"],
            "Fract_large_gaps"     :  diagnostics["Large_gaps"],
            "Resolution_MAG"       :  diagnostics["resol"] 
                   }

    mag_dict = {
                "B_resampled"      :  mag_interpolated,
                "PSD_f_orig"       :  psd_B,
                "f_B_orig"         :  f_B,
                "PSD_f_smoothed"   :  psd_B_smooth,
                "f_B_mid"          :  f_B_mid,
                "f_B_mean"         :  f_B_mean,
                "Fraction_missing" :  diagnostics["Frac_miss"],
                "resolution"       :  diagnostics["resol"]
                }
    
    return general_dict, mag_dict



def calc_particle_diagnostics(dfpar, dfmag, misc_par, estimate_PSD_V, subtract_rol_mean, rolling_window,  f_min_spec, f_max_spec, in_rtn, smoothed = True):


    """Define Span velocity field components"""
    dfpar_interp   = dfpar.interpolate(method='linear').dropna()
    cols           = ['Vr', 'Vt', 'Vn'] if in_rtn else ['Vx', 'Vy', 'Vz'] 
    Vx, Vy, Vz     = dfpar_interp[cols].values.T

    """Estimate PSD of  Velocity field"""
    f_V, psd_V     = (turb.TracePSD(Vx, Vy, Vz, True , misc_par["resol"]*1e-3) if estimate_PSD_V else (None, None))

    """Estimate derived plasma quantities"""
    part_quants = estimate_quants_particle_data(estimate_PSD_V, rolling_window, f_min_spec, f_max_spec,  in_rtn,  dfpar_interp, dfmag, subtract_rol_mean, smoothed)
    dict_psd, nn_df, sw_speed, alfv_speed, f_Zplus, psd_Zplus, f_Zminus, psd_Zminus, sigma_r_spec, sigma_c_spec, beta_mean, beta_std, Z_amplitude_mean, Z_amplitude_std, sigma_r_mean, sigma_r_median, sigma_r_std, sigma_c_median, sigma_c_mean, sigma_c_median_no_abs, sigma_c_mean_no_abs, sigma_c_std, Vth_mean, Vth_std , Vsw_mean, Vsw_std, Np_mean, Np_std, di_mean, di_std, rho_ci_mean, rho_ci_std, VBangle_mean, VBangle_std = part_quants  
    

    partdict = { 
                 "V_resampled"           : dfpar_interp.interpolate(),        # dataframe containing all particle data
                 "dict_psd"              : dict_psd,            # dictionary containing psd's of Zp, Zm, V, B
                 "f_V"                   : f_V,             # array containing spacecraft frequency for power spectrum
                 "psd_V"                 : psd_V,           # power spectrum  V field
                 "Va"                    : alfv_speed,          # alfven speed 
                 "Sw_speed"              : sw_speed,            # solar wind speed
                 "f_Zplus"               : f_Zplus,             # array containing spacecraft frequency for z+ power spectrum
                 "PSD_Zplus"             : psd_Zplus,           # power spectrum of z+
                 "f_Zminus"              : f_Zminus,            #  array containing spacecraft frequency for z- power spectrum
                 "PSD_Zminus"            : psd_Zminus,          # power spectrum of z-
                 'di_mean'               : di_mean,         # mean value of ion inertial lenght for the interval
                 'di_std'                : di_std,          # standard deviation of -//-
                 'rho_ci_mean'           : rho_ci_mean,     # mean ion gyroradius
                 'rho_ci_std'            : rho_ci_std,      # std ion gyroradius
                 'sigma_c_mean'          : sigma_c_mean,    # normalized cross-helicity mean 
                 'sigma_c_std'           : sigma_c_std,     # std cross helicity
                 'sigma_c_median'        : sigma_c_median, 
                 'sigma_c_mean_no_abs'   : sigma_c_mean_no_abs,
                 'sigma_c_median_no_abs' : sigma_c_median_no_abs, 
                 'sigma_r_median'        : sigma_r_median,  # normalized residual energy
                 'sigma_c_spec'          : sigma_c_spec,
                 'sigma_r_spec'          : sigma_r_spec,
                 'sigma_r_mean'          : sigma_r_mean,
                 'sigma_r_std'           : sigma_r_std,
                 'Vsw_mean'              : Vsw_mean,         # bulk solar wind speed
                 'Vsw_std'               : Vsw_std,
                 'VBangle_mean'          : VBangle_mean,     # angle between backround magnetic field and solar wind flow
                 'VBangle_std'           : VBangle_std,
                 'beta_mean'             : beta_mean,        # plasma b parameter
                 'beta_std'              : beta_std,
                 'Vth_mean'              : Vth_mean,         # thermal velocity of ions
                 'Vth_std'               : Vth_std,
                 'Np_mean'               : Np_mean,          # number density of ions
                 'Np_std'                : Np_std,
                 'Z_mean'                : Z_amplitude_mean, # amplitude of fluctuations
                 'Z_std'                 : Z_amplitude_std,
                 "Fraction_missing"      : misc_par["Frac_miss"], # fraction of timeseries Nan
                 "Fract_large_gaps"      : misc_par["Large_gaps"],
                 "resolution"            : misc_par["resol"]        # resolution of timeseries
              }
                                                    
    return partdict, nn_df

def set_up_main_loop(final_path, only_one_interval, t0, t1, step, duration):

    # Define lists with dates
    tstarts = []
    tends   = []
    if only_one_interval:
        tstarts.append(t0)
        tends.append(t1)
        
    else:
        i1 = 0
        while True:
            tstart  = t0+i1*pd.Timedelta(step)
            tend    = tstart + pd.Timedelta(duration)
            if tend > t1:
                break
            tstarts.append(tstart)
            tends.append(tend)
            i1 += 1


    path0 = Path(final_path)
    tfmt  = "%Y-%m-%d_%H-%M-%S"
    return tstarts, tends, tfmt, path0 

def final_func( 
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

              ):
    # Parker Solar Probe
    if sc==0:
        
        dfmag, dfpar, dfdis, big_gaps, misc          = LoadTimeSeriesPSP(
                                                                          start_time, 
                                                                          end_time, 
                                                                          settings, 
                                                                          vars_2_downnload,
                                                                          cdf_lib_path,
                                                                          credentials        = credentials,
                                                                          download_SCAM      = high_resol_data,
                                                                          gap_time_threshold = gap_time_threshold,
                                                                          time_amount        = addit_time_around,
                                                                          time_unit          = 'h'
        )


    # Solar Orbiter
    elif sc ==1:
 
        dfmag, dfpar, dfdis, big_gaps, misc           =  LoadTimeSeriesSOLO(
                                                                          start_time, 
                                                                          end_time, 
                                                                          settings, 
                                                                          vars_2_downnload,
                                                                          cdf_lib_path,
                                                                          credentials        = credentials,
                                                                          download_SCAM      = high_resol_data,
                                                                          gap_time_threshold = gap_time_threshold,
                                                                          time_amount        = addit_time_around,
                                                                          time_unit          = 'h'
        )
    elif sc ==3:
        # print('HELIOS')
        final_dataframe = LoadData.LoadTimeSeriesWrapper(
        sc, start_time, end_time,
        settings = {}, credentials = None)
       
        final_dataframe =  final_dataframe[0]
        dfmag           =  final_df[['Br','Bt','Bn','Bx','By','Bz']]
        dfpar           =  final_df[['Vr','Vt','Vn','np','Tp','Vth']]
        dist_df         =  final_df[['Dist_au','lon','lat']]
        misc            =  final_dataframe[1]


    if dfpar is not None:
        if len(dfpar.dropna()) > 0 :
            try:
                
                """ Make sure both correspond to the same interval """ 
                dfpar                                             = func.use_dates_return_elements_of_df_inbetween(dfmag.index[0], dfmag.index[-1], dfpar) 

                try:
                    general_dict, mag_dict                        = calc_mag_diagnostics(
                                                                                          misc['Mag'],
                                                                                          gap_time_threshold,
                                                                                          dfdis,
                                                                                          estimate_PSD,
                                                                                          dfmag,
                                                                                          settings['MAG_resol'])
  
                except:
                    traceback.print_exc()   

                """Now calculates quantities related to particle timeseries"""
          
                res_particles, sig_c_sig_r_timeseries              = calc_particle_diagnostics(
                                                                                                dfpar, 
                                                                                                dfmag,
                                                                                                misc['Par'],
                                                                                                estimate_PSD_V,
                                                                                                subtract_rol_mean,
                                                                                                rolling_window,
                                                                                                f_min_spec,
                                                                                                f_max_spec,
                                                                                                in_RTN,
                                                                                                smoothed = True
                 )

                # Now save everything in final_dict as a dictionary
                final_dict = { 
                               "Mag"          : mag_dict,
                               "Par"          : res_particles
                              }
                
                # also save a general dict with basic info (add what is missing from particletimeseries)
                general_dict["Fraction_missing_part"] = res_particles["Fraction_missing"]
                general_dict["Resolution_part"]       = res_particles["resolution"]

                flag_good = 1

            except:
                traceback.print_exc()

                flag_good = 0
                print('No MAG data!')
                big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
        else:
            final_dict = None
            print('No particle data!')

            flag_good = 0
            big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
    else:
        final_dict = None
        traceback.print_exc()
        print('No particle data!')

        flag_good = 0
        big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None

    return big_gaps, flag_good, final_dict, general_dict, sig_c_sig_r_timeseries,  dfdis
    





def create_dfs_SCaM( dfmag               ,
                     dfpar               ,
                     dist_df             , 
                     settings
                   ):
    
    if dfpar is not None:
        if len(dfpar.dropna()) > 0 :
            
            diagnostics_PAR  = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'], large_gaps=10)
            diagnostics_MAG  = func.resample_timeseries_estimate_gaps(dfmag, settings['Mag_resol'], large_gaps=10)

            
            keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
            misc = {
                    'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
                    'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep)
            }

            try:
                
                """ Make sure both correspond to the same interval """ 
                dfpar                                   = func.use_dates_return_elements_of_df_inbetween(dfmag.index[0], dfmag.index[-1], dfpar) 


                general_dict, mag_dict                  = calc_mag_diagnostics(
                                                                                          misc['Mag'],
                                                                                          settings['gap_time_thresh'],
                                                                                          dist_df,
                                                                                          settings['est_PSD'],
                                                                                          dfmag,
                                                                                          settings['Mag_resol'])
    

                """Now calculates quantities related to particle timeseries"""
                res_particles, sig_c_sig_r_timeseries     = calc_particle_diagnostics(
                                                                                                dfpar, 
                                                                                                dfmag,
                                                                                                misc['Par'],
                                                                                                settings['est_PSD_V'],
                                                                                                settings['sub_rol_mean'],
                                                                                                settings['roll_window'],
                                                                                                settings['f_min_spec'],
                                                                                                settings['f_max_spec'],
                                                                                                settings['in_RTN'],
                                                                                                smoothed = True
                 )

                # Now save everything in final_dict as a dictionary
                final_dict = { 
                               "Mag"          : mag_dict,
                               "Par"          : res_particles
                              }
                
                # also save a general dict with basic info (add what is missing from particletimeseries)
                general_dict["Fraction_missing_part"] = res_particles["Fraction_missing"]
                general_dict["Resolution_part"]       = res_particles["resolution"]

                flag_good = 1

            except:
                traceback.print_exc()

                flag_good = 0
                print('No MAG data!')
                big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
        else:
            final_dict = None
            print('No particle data!')

            flag_good = 0
            big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
    else:
        final_dict = None
        traceback.print_exc()
        print('No particle data!')

        flag_good = 0
        big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None

    return  flag_good, final_dict, general_dict, sig_c_sig_r_timeseries,  dist_df
         



def prepare_particle_data_for_visualization( df_part, mag_resampled, rolling_window,  subtract_rol_mean=True, smoothed = True, in_rtn=True):

    # Particle data remove nan values
    df_part = df_part.dropna()
    
    # Reindex magnetic field data to particle data index
    dtv                 = (df_part.dropna().index.to_series().diff()/np.timedelta64(1, 's'))[1]
    dtb                 = (mag_resampled.dropna().index.to_series().diff()/np.timedelta64(1, 's'))[1]
    
    # We need plasma timeseries cadence
    freq_final          = str(int(dtv*1e3))+'ms'


    # Combine magnetic field and particle data, resampled to final frequency
    f_df = mag_resampled.resample(freq_final).mean().join(
         df_part.resample(freq_final).mean().interpolate()
    )

    # Calculate magnetic field magnitude
    Bx, By, Bz = f_df.values.T[:3]
    Bmag       = np.sqrt(Bx**2 + By**2 + Bz**2)

    if subtract_rol_mean:
        columns = [['Br', 'Bt', 'Bn'], ['Vr', 'Vt', 'Vn', 'np']]
        for c in columns:
            f_df[[f"{col}_mean" for col in c]] = f_df[c].rolling(rolling_window, center=True).mean().interpolate()

    #Estimate median solar wind speed   
    Vth                     = f_df.Vth.values
    Vth                     = Vth[~np.isnan(Vth)]
    Vth[Vth < 0]            = np.nan
    Vth_mean                = np.nanmedian(Vth)
    Vth_std                 = np.nanstd(Vth);
    
    #Estimate median solar wind speed  
 
    Vx                       = f_df["Vr"].values if in_rtn else f_df["Vx"].values
    Vy                       = f_df["Vt"].values if in_rtn else f_df["Vy"].values
    Vz                       = f_df["Vn"].values if in_rtn else f_df["Vz"].values
    Vsw                      = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    Vsw                      = Vsw[~np.isnan(Vsw)]
    Vsw_mean                 = np.nanmedian(Vsw)
    Vsw_std                  = np.nanstd(Vsw)
    Vsw[(np.abs(Vsw) > 1e5)] = np.nan


    # estimate mean number density
    Np                       = f_df['np'].values 
    Np                       = Np[~np.isnan(Np)]
    Np_mean                  = np.nanmedian(Np)
    Np_std                   = np.nanstd(Np);
        
    # Estimate Ion inertial length di in [Km]
    di                       = 228/np.sqrt(Np)
    di[di< 1e-3]             = np.nan
    di_mean                  = np.nanmedian(di) 
    di_std                   = np.nanstd(di);
    
    # Estimate plasma Beta
    B_mag       = Bmag * nT2T                              # |B| units:      [T]
    temp        = 1./2 * m_p * (Vth*km2m)**2               # in [J] = [kg] * [m]^2 * [s]^-2
    dens        = Np/(cm2m**3)                             # number density: [m^-3] 
    beta        = (dens*temp)/((B_mag**2)/(2*mu_0))        # plasma beta 
    beta[beta < 0] = np.nan
    beta[np.abs(np.log10(beta))>4] = np.nan # delete some weird data
    beta_mean   = np.nanmedian(beta); beta_std   = np.nanstd(beta);
    
    
    # ion gyro radius
    rho_ci = 10.43968491 * Vth/B_mag #in [km]
    rho_ci[rho_ci < 0] = np.nan
    rho_ci[np.log10(rho_ci) < -3] = np.nan
    rho_ci_mean =np.nanmedian(rho_ci); rho_ci_std =np.nanstd(rho_ci);
 
    ### Define b and v ###
    if in_rtn:
        columns_v = ["Vr", "Vt", "Vn"] if "Vr" in f_df.columns else ["Vx", "Vy", "Vz"]
        columns_b = ["Br", "Bt", "Bn"] if "Br" in f_df.columns else ["Bx", "By", "Bz"]
    else:
        columns_v = ["Vx", "Vy", "Vz"]
        columns_b = ["Bx", "By", "Bz"]

    # Assign values
    vr, vt, vn = f_df[columns_v].values.T
    br, bt, bn = f_df[columns_b].values.T


    # Const to normalize mag field in vel units
    kinet_normal = 1e-15 / np.sqrt(mu0 * f_df['np_mean'].values * m_p)

    # Estimate Alfv speed
    Va_ts = np.array([br, bt, bn]) * kinet_normal
    
    # Estimate SW speed
    V_ts = np.array([vr, vt, vn])
    
    # Estimate mean values of both for interval
    alfv_speed, sw_speed = [np.nanmean(x, axis=0) for x in (Va_ts, V_ts)]
    
    # Estimate VB angle
    vbang = func.angle_between_vectors(Va_ts.T,V_ts.T)
    
    #End its mean
    VBangle_mean, VBangle_std = np.nanmean(vbang ), np.nanstd(vbang)
    
    #Sign of Br forrolling window
    signB = - np.sign(f_df['Br_mean'])

    # Estimate fluctuations
    if subtract_rol_mean:
        dva    = Va_ts  -  f_df[['Br_mean', 'Bt_mean', 'Bn_mean']].values.T *kinet_normal
        dv     = V_ts   -  f_df[['Vr_mean', 'Vt_mean', 'Vn_mean']].values.T
    else:
        dva    = Va_ts  -  np.nanmean(alfv_speed, axis=0)
        dv     = V_ts   -  np.nanmean(sw_speed, axis=0)

    # Estimate Zp, Zm components  
    Zpr, Zmr = dv[0] + signB * dva[0], dv[0] - signB * dva[0]
    Zpt, Zmt = dv[1] + signB * dva[1], dv[1] - signB * dva[1]
    Zpn, Zmn = dv[2] + signB * dva[2], dv[2] - signB * dva[2]
    
    # Estimate energy in Zp, Zm
    Z_plus_squared     = Zpr**2 + Zpt**2 + Zpn**2
    Z_minus_squared    = Zmr**2 + Zmt**2 + Zmn**2
    
    # Estimate amplitude of fluctuations
    Z_amplitude        = np.sqrt((Z_plus_squared + Z_minus_squared) / 2)
    Z_amplitude_mean   = np.nanmedian(Z_amplitude)
    Z_amplitude_std    = np.nanstd(Z_amplitude)


    # Kin, mag energy
    Ek           =  dv[0]**2  + dv[1]**2  + dv[2]**2
    Eb           =  dva[0]**2 + dva[1]**2 + dva[2]**2
    
    
    #Estimate normalized residual energy
    sigma_r      = (Ek-Eb)/(Ek+Eb);                                                         sigma_r[np.abs(sigma_r) > 1e5] = np.nan;
    sigma_c      = (Z_plus_squared - Z_minus_squared)/( Z_plus_squared + Z_minus_squared);  sigma_c[np.abs(sigma_c) > 1e5] = np.nan


    #Save in DF format to estimate spectraly
    nn_df       = pd.DataFrame({'DateTime': f_df.index.values,
                                'Zpr'     : Zpr,     'Zpt'  : Zpt,   'Zpn' : Zpn,
                                'Zmr'     : Zmr,     'Zmt'  : Zmt,   'Zmn' : Zmn, 
                                'va_r'    : dva[0],  'va_t' : dva[1],'va_n': dva[2],
                                'v_r'     : dv[0],   'v_t'  : dv[1], 'v_n' : dv[2],
                                'beta'    : beta,    'np'   : Np,    'Tp'  : temp, 'VB': vbang,
                                'sigma_c' : sigma_c,              'sigma_r': sigma_r}).set_index('DateTime')
    nn_df       = nn_df.mask(np.isinf(nn_df)).interpolate(method='linear').dropna()

    
    return   nn_df