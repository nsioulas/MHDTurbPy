from matplotlib import pyplot as plt
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
from mpl_toolkits.axes_grid1 import make_axes_locatable

BG_WHITE = '\033[47m'
RESET    = '\033[0m'  # Reset the color
BG_RED = '\033[41m'
BG_GREEN = '\033[42m'
BG_YELLOW = '\033[43m'
BG_BLUE = '\033[44m'
BG_MAGENTA = '\033[45m'
BG_CYAN = '\033[46m'

import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp, log level, and message
    datefmt='%Y-%m-%d %H:%M:%S',  # Format for the timestamp
)

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO)


# Add the directory containing the local PySPEDAS to the front of sys.path
sys.path.insert(0, "/pyspedas")

# Now, import PySPEDAS and other modules
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

# Optionally, print the path of the PySPEDAS module to confirm its source
print(pyspedas.__file__)


""" Import manual functions """
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func
import TurbPy as turb


# Some constants
from scipy import constants
au_to_km       = 1.496e8  # Conversion factor
rsun            = 696340   # Sun radius in units of  [km]
mu0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu_0            =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
m_p             =  constants.m_p   # Proton mass [kg]
k               = constants.k                  # Boltzman's constant     [j/K]
au_to_rsun      = 215.032
T_to_Gauss      = 1e4


def default_variables_to_download_PSP(vars_2_downnload):
    
    if vars_2_downnload['mag'] is None:
        varnames_MAG          = ['B_RTN']
    else:
        varnames_MAG          = vars_2_downnload['mag']
        
    if vars_2_downnload['qtn'] is None:
        varnames_QTN          = ['electron_density','electron_core_temperature']
    else:
        varnames_QTN          = vars_2_downnload['qtn']
        
    if vars_2_downnload['span'] is None:
        varnames_SPAN          = ['DENS',  'VEL_RTN_SUN', 'TEMP' , 'SUN_DIST', 'SC_VEL_RTN_SUN']
    else:
        varnames_SPAN          = vars_2_downnload['span']        
    
    if vars_2_downnload['spc'] is None:
        varnames_SPC          = ['np_moment','wp_moment','vp_moment_RTN', 'sc_pos_HCI','carr_longitude','na_fit']
    else:
        varnames_SPC          = vars_2_downnload['spc']   
        
    if vars_2_downnload['span-a'] is None:
        varnames_SPAN_alpha          = ['DENS']
    else:
        varnames_SPAN_alpha          = vars_2_downnload['span-a']  

    if vars_2_downnload['ephem'] is None:
        varnames_EPHEM         = ['position','velocity']  
    else:
        varnames_EPHEM          = vars_2_downnload['ephem']
        
    if vars_2_downnload.get('E_field', False):
        varnames_E_field          = ['psp_fld_l2_dfb_wf_dVdc_sc']
    else:
        varnames_E_field          =  None


    if vars_2_downnload.get('sc_pot', False):
        varnames_SC_pot          = ['dfb_wf_vdc']
    else:
        varnames_SC_pot          =  None

        
    return varnames_MAG, varnames_QTN, varnames_SPAN, varnames_SPC,  varnames_SPAN_alpha,varnames_EPHEM, varnames_E_field, varnames_SC_pot


def map_col_names_PSP(instrument, varnames):
    
    # Mapping between variable names and column names for FIELDS
    fields_MAG_cols = {
        'mag_RTN_4_Sa_per_Cyc'         : ['Br', 'Bt', 'Bn'],
        'mag_SC_4_Sa_per_Cyc'          : ['Bx', 'By', 'Bz'],
        'mag_rtn_4_per_cycle'          : ['Br', 'Bt', 'Bn'],
        'mag_sc_4_per_cycle'           : ['Bx', 'By', 'Bz'],
        'mag_RTN'                      : ['Br', 'Bt', 'Bn'],
        'mag_SC'                       : ['Bx', 'By', 'Bz'],
        'mag_rtn'                      : ['Br', 'Bt', 'Bn'],
        'mag_sc'                       : ['Bx', 'By', 'Bz'],
        'psp_fld_l2_dfb_wf_dVdc_sc'    : ['dvx', 'dvy'],
        'dfb_wf_vdc'                   : ['psp_fld_l2_dfb_wf_V1dc',
                                          'psp_fld_l2_dfb_wf_V2dc',
                                          'psp_fld_l2_dfb_wf_V3dc',
                                          'psp_fld_l2_dfb_wf_V4dc']
    }
    # Mapping between variable names and column names for QTN
    fields_QTN_cols = {
        'electron_density'              : ['ne_qtn'],
        'electron_core_temperature'     : ['Te_qtn'],

    }    

    # Mapping between variable names and column names for SPC
    spc_cols = {
        'np_moment'     : ['np'],
        'wp_moment'     : ['Vth'],
        'vp_moment_RTN' : ['Vr', 'Vt', 'Vn'],
        'vp_moment_SC'  : ['Vx', 'Vy', 'Vz'],
        'sc_pos_HCI'    : ['sc_x', 'sc_y', 'sc_z'],
        'sc_vel_HCI'    : ['sc_vel_x', 'sc_vel_y', 'sc_vel_z'],
        'carr_latitude' : ['carr_lat'],
        'carr_longitude': ['carr_lon'],
        'na_fit'        : ['na']
    }

    # Mapping between variable names and column names for SPAN
    span_cols = {
        'DENS'            : ['np'],
        'VEL_SC'          : ['Vx', 'Vy', 'Vz'],
        'VEL_RTN_SUN'     : ['Vr', 'Vt', 'Vn'],
        'TEMP'            : ['TEMP'],
        'SUN_DIST'        : ['Dist_au'],
        'SC_VEL_RTN_SUN'  : ['sc_vel_r','sc_vel_t','sc_vel_n']

    }
    # Mapping between variable names and column names for SPAN
    span_alpha_cols = {
        'DENS'            : ['na']
    }
    # Mapping between variable names and column names for EPHEMERIS
    ephem_cols = {
        'position'            : ['sc_pos_r','sc_pos_t','sc_pos_n'],
        'velocity'            : ['sc_vel_r','sc_vel_t','sc_vel_n'],
    }    
  
    
    if instrument == 'SPC':
        return [spc_cols[var] for var in varnames if var in spc_cols]
    elif instrument == 'FIELDS-MAG':
        return [fields_MAG_cols[var] for var in varnames if var in fields_MAG_cols]
    elif instrument =='QTN':
        return [fields_QTN_cols[var] for var in varnames if var in fields_QTN_cols]
    elif instrument == 'SPAN':
         return [span_cols[var] for var in varnames if var in span_cols]
    elif instrument =='SPAN-alpha':
         return [span_alpha_cols[var] for var in varnames if var in span_alpha_cols]
    elif instrument =='EPHEMERIS':
         return [ephem_cols[var] for var in varnames if var in ephem_cols]
    else:
        return []
    
    

def download_MAG_FIELD_PSP(t0, 
                           t1, 
                           credentials,
                           varnames, 
                           settings):
    try:
        for j, varname in enumerate(varnames): 

            try:
                       
                traceback.print_exc()
                print('Using private mag data')
                if varname == 'B_RTN':
                    print('Using RTN frame mag data.')
                    if settings['MAG_resol']> 230:               # It's better to use lower resol if you want to resample to SPC, SPAN cadence. 
                        datatype = 'mag_RTN_4_Sa_per_Cyc'
                        
                    else:
                        datatype = 'mag_RTN'
                else:
                    print('Using SC frame mag data.')
                    if settings['MAG_resol']> 230:
                        datatype = 'mag_SC_4_Sa_per_Cyc'
                    else:
                        datatype = 'mag_SC'

                username = credentials['psp']['fields']['username']
                password = credentials['psp']['fields']['password']
                MAGdata = pyspedas.psp.fields(trange=[t0, t1], datatype=datatype, level='l2', 
                                              time_clip=True, username=username, password=password)#, no_update=np.invert(settings['use_local_data']))

            except:
                
                print('Using public mag data')
                if varname == 'B_RTN':
                    print('Using RTN frame mag data.')
                    if settings['MAG_resol']> 230:
                        datatype = 'mag_rtn_4_per_cycle'
                    else:
                        datatype = 'mag_rtn'
                else:
                    print('Using SC frame mag data.')
                    if settings['MAG_resol']> 230:
                        datatype = 'mag_sc_4_per_cycle'
                    else:
                        datatype = 'mag_sc'
                MAGdata = pyspedas.psp.fields(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True)#, no_update=np.invert(settings['use_local_data']))           


            if j == 0:
                col_names = map_col_names_PSP('FIELDS-MAG', [datatype])
                if settings['MAG_resol'] < 230:
                    dfs = [pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=col_names[i]) 
                           for i, data in enumerate([MAGdata[0]])]
                    dfmag = dfs[0].join(dfs[1:])
                else:

                    dfmag = pd.DataFrame(
                        index = get_data(MAGdata[0])[0],
                        data = get_data(MAGdata[0])[1]
                    )
                    dfmag.columns = col_names[0]
                    
            else:
                col_names = map_col_names_PSP('FIELDS-MAG', [datatype])
                if settings['MAG_resol'] < 230:
                    dfs1 = [pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=col_names[i]) 
                            for i, data in enumerate([MAGdata[0]])]
                    dfMAG1 = dfs1[0].join(dfs1[1:])  
                    dfmag = dfmag.join(dfMAG1)
                else:
                    dfmag = pd.DataFrame(
                        index = get_data(MAGdata[0])[0],
                        data = get_data(MAGdata[0])[1]
                    )
                    dfmag.columns = col_names[0]                    

        
        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)
     
          
        return dfmag.sort_index()
    except Exception as e:
        print(f'Error occurred while retrieving MAG data: {e}')
        return None
        

        
        
def process_mag_field_data(t0, 
                           t1, 
                           settings,
                           credentials,
                           varnames_MAG, 
                           ind1, 
                           ind2):
    try:
        
        if settings['Mag_SCAM_PSP']['flag']:
            
            print('Working on SCAM  mag data')
            
            dfmag = LoadSCAMFromSPEDAS_PSP(t0,
                                           t1,
                                           credentials,
                                           settings)
            
        else:
            print('Working on fluxgate mag data')
            dfmag = download_MAG_FIELD_PSP(t0,
                                           t1,
                                           credentials, 
                                           varnames_MAG,
                                           settings)
            
      
        try:
            dfmag = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
        except Exception:
            # Adjust index format and retry processing
            dfmag.index = pd.to_datetime(dfmag.index, format='%Y-%m-%d %H:%M:%S.%f')
            dfmag       = func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(ind1), pd.to_numeric(ind2), dfmag)
        
        # Identify big gaps in timeseries
        big_gaps = func.find_big_gaps(dfmag, settings['Big_Gaps']['Mag_big_gaps'], str(ind1), str(ind2))
        # Resample the input dataframes
        diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings['MAG_resol'], large_gaps=10)
        
    
        
        # Remove wheel noise from scam data
        if  settings['Mag_SCAM_PSP']['noise_flag']:
            
            print('Removing wheel noise!')
            
            dt   = func.find_cadence(diagnostics_MAG["resampled_df"])
            keys = list(diagnostics_MAG["resampled_df"].keys())
            
            for key in keys:
                
                signal_noise_removed = turb.remove_wheel_noise(diagnostics_MAG["resampled_df"][key].values,
                                                               1/dt, 
                                                               window_size      = settings['Mag_SCAM_PSP']['noise_removal']['window_size'], 
                                                               avg_length       = settings['Mag_SCAM_PSP']['noise_removal']['avg_length'],
                                                               power_threshold  = settings['Mag_SCAM_PSP']['noise_removal']['power_threshold'],
                                                               freq_min         = settings['Mag_SCAM_PSP']['noise_removal']['freq_min'])

                
               
                #replace with clean data
                diagnostics_MAG["resampled_df"][key] = signal_noise_removed

        
        return dfmag, big_gaps, diagnostics_MAG
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        diagnostics_MAG_default = {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}
        return None, None, diagnostics_MAG_default


def download_SPC_PSP(t0, t1, credentials, varnames, settings):
    
    print('Spc Variables', varnames)
    
    try:
        try:
            
            
            username = credentials['psp']['sweap']['username']
            password = credentials['psp']['sweap']['password']

            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='L3', 
                                        varnames=varnames, time_clip=True, 
                                        username=username, password=password)#, no_update=np.invert(settings['use_local_data']))
           # print('spc', spcdata)
            if len(spcdata)==0:
                print("No data available for this interval.")
                return None, None
            

        except:
            
            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")

            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='l3', 
                                        varnames=varnames, time_clip=True)#, no_update=np.invert(settings['use_local_data']))
              

        col_names = map_col_names_PSP('SPC', varnames)
        dfs = [pd.DataFrame(index=get_data(data).times, 
                            data=get_data(data).y, 
                            columns=col_names[i]) for i, data in enumerate(spcdata)]
        dfspc = dfs[0].join(dfs[1:])
        dfspc['Dist_au'] = (dfspc[['sc_x', 'sc_y', 'sc_z']]**2).sum(axis=1)**0.5 / au_to_km
        dfspc.drop(['sc_x', 'sc_y', 'sc_z'], axis=1, inplace=True)
        
        # Fix datetime index
        dfspc.index = time_string.time_datetime(time=dfspc.index)
        dfspc.index = dfspc.index.tz_localize(None)
        dfspc.index.name = 'datetime'
        
        return dfspc
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        return None, None
                    
def process_spc_data(t0, t1, credentials, varnames_SPC, settings, ind1, ind2):
    try:
        # Download SPC data
        dfspc = download_SPC_PSP(t0, t1, credentials, varnames_SPC, settings)
        
        
        # Trim data to the originally requested interval
        dfspc = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfspc)

        # Apply Hampel filter if required
        if settings['apply_hampel']:
            # Determine which velocity components to filter based on data presence
            columns_for_hampel = ['Vr', 'Vt', 'Vn', 'np', 'Vth'] if 'Vr' in dfspc.keys() else ['Vx', 'Vy', 'Vz', 'np', 'Vth']
            ws_hampel, n_hampel = settings['hampel_params']['w'], settings['hampel_params']['std']

            for column in columns_for_hampel:
                try:
                    outliers_indices = func.hampel(dfspc[column], window_size=ws_hampel, n=n_hampel)
                    dfspc.loc[dfspc.index[outliers_indices], column] = np.nan
                except Exception as e:
                    logging.exception("An error occurred while filtering %s: %s", column, e)

            print(f'Applied Hampel filter to SPC columns: {columns_for_hampel}')

        # Estimate Tp[eV]
        from astropy.constants import m_p, k_B 
        from astropy import units as u
        dfspc['Tp'] = np.array(((m_p * ((dfspc['Vth'].values * u.km/u.s).to(u.m/u.s)**2)) / (2 * k_B)).to(u.eV, equivalencies=u.temperature_energy()))

        # Identify big gaps in timeseries
        big_gaps_spc = func.find_big_gaps(dfspc, settings['Big_Gaps']['Par_big_gaps'], str(ind1), str(ind2))
        
        # Calculate diagnostics
        diagnostics_SPC = func.resample_timeseries_estimate_gaps(dfspc, settings['part_resol'], large_gaps=10)
        spc_flag = 'SPC'
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Return None for dfspc and default values for diagnostics_SPC on error
        dfspc, diagnostics_SPC, spc_flag, big_gaps_spc = None, {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}, 'No SPC', None

    return dfspc, diagnostics_SPC, spc_flag, big_gaps_spc


    

def download_SPAN_PSP(t0, t1, credentials, varnames, varnames_alpha, settings):
    """
    Downloads SPAN data for PSP within the given time range.

    This function first attempts to retrieve the data using credentials. If that fails
    (due to missing credentials, an error, or no data returned), it will attempt a 
    second approach that does not require credentials.

    Parameters:
        t0, t1: Start and end times for the data download.
        credentials: A dictionary containing user credentials.
        varnames: List of variable names for SPAN protons.
        varnames_alpha: List of variable names for SPAN alphas (currently not used).
        settings: Additional settings (not used in this example).

    Returns:
        dfspan: A pandas DataFrame containing the SPAN proton data.
                Returns None if data retrieval or processing fails.
    """
    print("Span Variables:", varnames)
    spandata = None

    # First approach: Retrieve data using credentials.
    try:
        username = credentials['psp']['sweap']['username']
        password = credentials['psp']['sweap']['password']

        spandata = pyspedas.psp.spi(
            trange=[t0, t1],
            datatype='spi_sf00',
            level='L3',
            varnames=varnames,
            time_clip=True,
            username=username,
            password=password
        )

        # If no data is returned, force fallback to the second approach.
        if not spandata or len(spandata) == 0:
            raise ValueError("No data available for this interval using the first approach.")

    except Exception as e:
        print("First approach failed (credentials missing, error occurred, or no data):", e)
        # Second approach: Retrieve data without credentials.
        try:
            spandata = pyspedas.psp.spi(
                trange=[t0, t1],
                datatype='spi_sf00_l3_mom',
                level='l3',
                varnames=varnames,
                time_clip=True
            )
            if not spandata or len(spandata) == 0:
                print("No data available for this interval using the second approach.")
                return None
        except Exception as e2:
            print("Second approach also failed:", e2)
            return None

    # Process the retrieved data into a pandas DataFrame.
    try:
        # Map column names for SPAN proton data.
        col_names = map_col_names_PSP('SPAN', varnames)
        # Create a list of DataFrames from each data segment.
        dfs = [
            pd.DataFrame(
                index=get_data(data).times,
                data=get_data(data).y,
                columns=col_names[i]
            )
            for i, data in enumerate(spandata)
        ]
        # Join the individual DataFrames into one.
        dfspan = dfs[0].join(dfs[1:])

        # Convert the index to datetime and remove timezone info.
        dfspan.index = time_string.time_datetime(time=dfspan.index)
        dfspan.index = dfspan.index.tz_localize(None)
        dfspan.index.name = 'datetime'

        # Convert distance from kilometers to astronomical units (assuming au_to_km is defined).
        dfspan['Dist_au'] = dfspan['Dist_au'] / au_to_km

        # Rename the temperature column and compute thermal speed.
        dfspan['Tp']   = dfspan.pop('TEMP')
        dfspan['Vth']  = 13.84112218 * np.sqrt(dfspan['Tp'])
        # Adjust thermal speed by a factor of sqrt(3) for SPAN.
        dfspan['Vth']  = dfspan['Vth'] / np.sqrt(3)

        print("SPAN DataFrame:", dfspan)

    except Exception as proc_err:
        print("Error processing SPAN data:", proc_err)
        return None

    return dfspan

    
def process_span_data(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1, ind2):
    """
    Downloads and processes SPAN data for the given time interval.

    The function downloads SPAN data using download_SPAN_PSP, applies a Hampel filter if specified,
    trims the data to the interval between ind1 and ind2, identifies large gaps, and calculates diagnostics.

    Parameters:
        t0, t1: Start and end times for the data download.
        credentials: Dictionary with user credentials.
        varnames_SPAN: List of variable names for SPAN protons.
        varnames_SPAN_alpha: List of variable names for SPAN alphas (currently unused).
        settings: Dictionary of settings including:
            - 'apply_hampel': bool indicating whether to apply the Hampel filter.
            - 'hampel_params': dict with keys 'w' (window size) and 'std' (threshold).
            - 'Big_Gaps': dict with key 'Par_big_gaps'.
            - 'part_resol': resolution for diagnostic resampling.
        ind1, ind2: The boundaries for trimming the data.

    Returns:
        tuple: (dfspan, diagnostics_SPAN, span_flag, big_gaps_span)
            - dfspan: Processed pandas DataFrame or None if an error occurred.
            - diagnostics_SPAN: Dictionary with diagnostics.
            - span_flag: 'SPAN' if successful, otherwise 'No SPAN'.
            - big_gaps_span: Gaps information or None.
    """
    try:
        dfspan = download_SPAN_PSP(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings)
        if dfspan is None:
            raise ValueError("No SPAN data returned from download_SPAN_PSP.")
    except Exception as e:
        logging.exception("Failed to download SPAN data: %s", e)
        diagnostics_SPAN = {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}
        return None, diagnostics_SPAN, 'No SPAN', None

    # Apply the Hampel filter if specified in settings.
    if settings.get('apply_hampel', False):
        # Choose columns based on available velocity component names.
        if 'Vr' in dfspan.columns:
            columns_for_hampel = ['Vr', 'Vt', 'Vn', 'np', 'Vth', 'Tp']
        else:
            columns_for_hampel = ['Vx', 'Vy', 'Vz', 'np', 'Vth', 'Tp']
            
        ws_hampel = settings['hampel_params']['w']
        n_hampel = settings['hampel_params']['std']

        for column in columns_for_hampel:
            try:
                outliers_indices = func.hampel(dfspan[column], window_size=ws_hampel, n=n_hampel)
                dfspan.loc[dfspan.index[outliers_indices], column] = np.nan
            except Exception as e:
                logging.exception("Error filtering column %s: %s", column, e)
        print(f"Applied Hampel filter to SPAN columns: {columns_for_hampel} with window size: {ws_hampel}")

    # Trim data to the requested interval.
    dfspan = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfspan)
    
    # Identify big gaps in the time series.
    big_gaps_span = func.find_big_gaps(dfspan, settings['Big_Gaps']['Par_big_gaps'], str(ind1), str(ind2))
    
    # Calculate diagnostics.
    diagnostics_SPAN = func.resample_timeseries_estimate_gaps(dfspan, settings['part_resol'], large_gaps=10)
    span_flag = 'SPAN'

    print("Final length:", len(dfspan))
    return dfspan, diagnostics_SPAN, span_flag, big_gaps_span


def download_QTN_PSP(t0, t1, credentials, varnames, settings):
    
    print('QTN', varnames)
    try:
        try:
            
            
            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']

            qtndata = pyspedas.psp.fields(trange     = [t0, t1],
                                          datatype   = 'sqtn_rfs_V1V2', level='l3',
                                          varnames   = varnames,
                                          time_clip  = True, 
                                          username   = username,
                                          password   = password)#,
            if qtndata ==[]:

                print('Using other qtn version')
                username = credentials['psp']['fields']['username']
                password = credentials['psp']['fields']['password']
                qtndata  = pyspedas.psp.fields(trange    = [t0, t1], 
                                               datatype  = 'rfs_lfr_qtn', level='l2', 
                                               time_clip = True,
                                               username  = username,
                                               password  = password)


            
        except:

            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")
            

            qtndata = pyspedas.psp.fields(trange      = [t0, t1], 
                                          datatype    ='sqtn_rfs_v1v2',
                                          level       ='l3',
                                          varnames    = varnames,
                                          time_clip   = True)
            
            if len(qtndata)==0:
                print("No data available for this interval.")
                return None
            
        col_names = map_col_names_PSP('QTN', varnames)
        dfs       = [pd.DataFrame(index=get_data(data).times, 
                                data=get_data(data).y, 
                                columns=col_names[i]) for i, data in enumerate(qtndata)]
        dfqtn = dfs[0].join(dfs[1:])
        
        dfqtn['np_qtn'] = dfqtn['ne_qtn']*0.96  # 4% of alpha particle

        dfqtn.index = time_string.time_datetime(time=dfqtn.index)
        dfqtn.index = dfqtn.index.tz_localize(None)
        dfqtn.index.name = 'datetime'
        
        # Identify big gaps in timeseries

        return dfqtn
    
    except Exception as e:
        logging.exception(f'Error occurred while retrieving QTN data: {e}')
        return None, None

    
def process_qtn_data(t0, t1, credentials, varnames_QTN, ind1, ind2, settings):
    # Attempt to download QTN data


    try:
        print("Attempting to load Orlando's QTN data...")
        dfqtn = pd.read_pickle(settings['orlandos_QTN'])
    
        # Ensure the index is a DateTimeIndex for proper slicing (if needed)
        if not isinstance(dfqtn.index, pd.DatetimeIndex):
            dfqtn.index = pd.to_datetime(dfqtn.index)
    
        # Filter the dataframe between t0 and t1
        df_between = dfqtn.loc[t0:t1]
    
        if len(df_between) == 0:
            # No data in the requested interval
            print(f"No data found in Orlando's QTN DataFrame for {t0} - {t1}. "
                  "Using alternative download method...")
            dfqtn = download_QTN_PSP(t0, t1, credentials, varnames_QTN, settings)
        else:
            dfqtn = df_between
            #print('Orlando QTN', dfqtn.dropna())
            print("Successfully loaded Orlando's QTN data for the requested interval.")
    
    except Exception as e:
        # If reading from pickle failed for any reason (path, file missing, etc.)
        print(f"Failed to load Orlando's QTN data: {e}. "
              "Using alternative method...")
  
        dfqtn = download_QTN_PSP(t0, t1, credentials, varnames_QTN, settings)
    
    try:
        # Process the downloaded data
        dfqtn           = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfqtn)
        big_gaps        = func.find_big_gaps(dfqtn, settings['Big_Gaps']['QTN_big_gaps'], str(ind1), str(ind2))
        
        diagnostics_QTN = func.resample_timeseries_estimate_gaps(dfqtn, settings['part_resol'], large_gaps=10)
        dfqtn_flag      = 'QTN'

    except Exception:
        # Handle any exceptions during processing by resetting values
        dfqtn, dfqtn_flag,  big_gaps = None, 'No QTN', None
        diagnostics_QTN   = {
                            "Init_dt"         : np.nan,
                            "resampled_df"    : None,
                            "Frac_miss"       : None,
                            "Large_gaps"      : None,
                            "Tot_gaps"        : None,
                            "resol"           : None
                        }

    return dfqtn, diagnostics_QTN, dfqtn_flag, big_gaps

def download_ephemeris_PSP(t0, t1, credentials, varnames, settings=None):
    try:
        username = credentials['psp']['fields']['username']
        password = credentials['psp']['fields']['password']
        
        ephemdata = pyspedas.psp.fields(trange=[t0, t1], datatype='ephem_spp_rtn', level='l1', 
                                         varnames=varnames, time_clip=True, username=username, password=password)#, no_update=np.invert(settings['use_local_data']))
        
        if len(ephemdata)==0:
            print("No data available for this interval.")
            return None

        col_names = map_col_names_PSP('EPHEMERIS', varnames)
        dfs       = [pd.DataFrame(index=get_data(data).times, 
                                data=get_data(data).y, 
                                columns=col_names[i]) for i, data in enumerate(ephemdata)]
        dfephem  = dfs[0].join(dfs[1:])


        dfephem.index = time_string.time_datetime(time=dfephem.index)
        dfephem.index = dfephem.index.tz_localize(None)


        dfephem['Dist_au'] = np.sqrt(np.sum(dfephem[['sc_pos_r','sc_pos_t','sc_pos_n']]**2, axis=1)) / au_to_km
        
        return dfephem
    
    except Exception as e:
        logging.exception("Ephemeris could not be loaded: %s", e)

        
def process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1, ind2, settings):
    try:
        # Attempt to download Ephemeris data
        dfephem = download_ephemeris_PSP(t0, t1, credentials, varnames_EPHEM,  settings)
        # Process the downloaded data if successful
        dfephem = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfephem)
        return dfephem
    except Exception:
        # Return None if any error occurs during download or processing
        return None
        



def create_particle_dataframe(end_time,
                              diagnostics_spc,
                              diagnostics_span,
                              dfqtn, 
                              dfqtn_flag,
                              big_gaps_span,
                              big_gaps_spc,
                              settings):

    def integrate_qtn_data(source_df, dfqtn):
        """
        Integrate QTN data into the source DataFrame.
        """

        try:

            source_df, dfqtn       = func.synchronize_dfs(source_df, dfqtn,  True)
            source_df['np']        = dfqtn['np_qtn'].values

            return source_df, 'QTN'

        except Exception as e:
            logging.exception("Failed to integrate QTN data: %s", e)
            source_df['np_sweap']   = source_df.pop('np')
            return source_df, 'No_QTN'
    
    # Default processing for '9th_perih_cut' mode
    if settings.get('particle_mode', '9th_perih_cut') == '9th_perih_cut':
        
        use_spc     = pd.Timestamp(end_time) < pd.Timestamp('2021-07-15')
        df_selected = diagnostics_spc['resampled_df'] if use_spc else diagnostics_span['resampled_df']
        big_gaps    = big_gaps_spc if use_spc else big_gaps_span
        
    elif settings['particle_mode'] == 'spc':
        df_selected = diagnostics_spc['resampled_df']                                             
        big_gaps    = big_gaps_spc
        
    elif settings['particle_mode'] == 'span':
        df_selected = diagnostics_span['resampled_df']
        big_gaps    = big_gaps_span
    else:
        raise ValueError(f"Unsupported particle mode: {settings['particle_mode']}")

    # Replace negative values with NaN
    try:
        df_selected = func.replace_negative_with_nan(df_selected)
    except:
        print('Bad!')

    # Integrate QTN data if flagged
    df_selected, dfqtn_flag = integrate_qtn_data(df_selected, dfqtn)

    return df_selected.interpolate().dropna(), settings['particle_mode'], dfqtn_flag, big_gaps




def download_efield(t0, t1, credentials, varnames, settings):
    
    print('E_field Variables', varnames)
    
    try:
        fields_vars = pyspedas.psp.fields(trange=[t0, t1], datatype='dfb_wf_dvdc', varnames = varnames, level='l2', time_clip=True)

        col_names = map_col_names_PSP('FIELDS-MAG', varnames)
        
        dfs       = [pd.DataFrame(index  = get_data(data).times, 
                                  data   = get_data(data).y, 
                                 columns = col_names[i]) for i, data in enumerate(fields_vars)]

        df_efield = dfs[0].join(dfs[1:])

        # Fix datetime index
        df_efield.index = time_string.time_datetime(time=df_efield.index)
        df_efield.index = df_efield.index.tz_localize(None)
        df_efield.index.name = 'datetime'
        
        return df_efield
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        return None, None
                    

def process_e_field_data(t0, t1,settings, credentials, varnames,  ind1, ind2):
    try:
        # Download e_field data
        df_efield = download_efield(t0, t1, credentials, varnames, settings)
        
        
        # Trim data to the originally requested interval
        df_efield = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, df_efield)

        # Identify big gaps in timeseries
        big_gaps_e_field = func.find_big_gaps(df_efield, settings['Big_Gaps']['E_big_gaps'], str(ind1), str(ind2))
        
        # Calculate diagnostics
        diagnostics_e_field = func.resample_timeseries_estimate_gaps(df_efield, 1, large_gaps=10)
        e_field_flag = 'e_field'
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Return None for df_efield and default values for diagnostics_e_field on error
        df_efield, diagnostics_e_field, e_field_flag, big_gaps_e_field = None, {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}, 'No e_field', None

    return df_efield, big_gaps_e_field, diagnostics_e_field



def sc_potential_derived_density(t0, t1, credentials, varnames, settings):
    """
    Downloads and processes spacecraft potential-derived density data from PSP FIELDS.
    
    Parameters:
        t0 (str): Start time in the format 'YYYY-MM-DD'.
        t1 (str): End time in the format 'YYYY-MM-DD'.
        credentials (dict): Credentials for data access.
        varnames (list): List of variable names to fetch.
        settings (dict): Configuration settings.
    
    Returns:
        pd.DataFrame: Processed time-series data with appropriate column names.
    """
    print('SC_pot Variables:', varnames)
    
    try:
        # Map variable names
        col_names = map_col_names_PSP('FIELDS-MAG', varnames)


        print(col_names)
        
        # Fetch the data with the specified datatype
        fields_vars = pyspedas.psp.fields(trange=[t0, t1], datatype='dfb_wf_vdc', level='l2', time_clip=True)
        
        
        dfs = [
            pd.DataFrame(
                index=get_data(data).times, 
                data=get_data(data).y, 
                columns=[fields_vars[i]]  # Ensure this is a list
            ) 
            for i, data in enumerate(fields_vars[:-1])
        ]
        
        df_density = dfs[0].join(dfs[1:])
    
        # Fix datetime index
        df_density.index = time_string.time_datetime(time=df_density.index)
        df_density.index = df_density.index.tz_localize(None)
        df_density.index.name = 'datetime'


        
        return df_density

    except Exception as e:
        logging.exception("An error occurred: %s", e)
        return None

def process_sc_potential_data(t0, t1, settings, credentials, varnames, ind1, ind2):
    """
    Processes spacecraft potential-derived density data, identifying gaps and resampling.

    Parameters:
        t0 (str): Start time in the format 'YYYY-MM-DD'.
        t1 (str): End time in the format 'YYYY-MM-DD'.
        settings (dict): Configuration settings.
        credentials (dict): Credentials for data access.
        varnames (list): List of variable names to fetch.
        ind1 (str): Start datetime for data trimming.
        ind2 (str): End datetime for data trimming.

    Returns:
        tuple: (Processed DataFrame, Identified large gaps, Diagnostics dictionary)
    """
    try:
        # Download spacecraft potential-derived density data
        df_density = sc_potential_derived_density(t0, t1, credentials, varnames, settings)
        
        if df_density is not None:
            # Trim data to the requested interval
            df_density = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, df_density)

            # Identify big gaps in time series
            big_gaps_density = func.find_big_gaps(df_density, settings['Big_Gaps']['SC_pot_big_gaps'], str(ind1), str(ind2))


            print('Dens gaps', big_gaps_density  )
            # Calculate diagnostics
            diagnostics_density = func.resample_timeseries_estimate_gaps(df_density, 1, large_gaps=10)
            density_flag = 'sc_pot'
        else:
            raise ValueError("Downloaded data is None")
    
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Return None for df_density and default diagnostic values on error
        df_density, diagnostics_density, density_flag, big_gaps_density = None, {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}, 'No sc_potential', None

    return df_density, big_gaps_density, diagnostics_density



def LoadTimeSeriesPSP(start_time, 
                      end_time, 
                      settings, 
                      vars_2_downnload,
                      cdf_lib_path,
                      credentials = None,
                      time_amount = 2,
                      time_unit   ='h'
                     ):
    """" 
    Load Time Serie From SPEDAS, PSP 
    settings if not None, should be a dictionary, necessary settings:

    spc: boolean
    span: boolean
    mix_spc_span: dict:
        {'priority': 'spc' or 'span'}
    keep_spc_and_span: boolean

    keep_keys: ['np','Vth','Vx','Vy','Vz','Vr','Vt','Vn']
    
    Note that the priority is using SPAN
    """

    # default settings
    default_settings = {
        'particle_mode'   : '9th_perih_cut',
        'apply_hampel'    : True,
        'hampel_params'   : {'w':100, 'std':3},
        'part_resol'      : 900,
        'MAG_resol'       : 1,
        'Mag_SCAM_PSP'    : {'flag':False, 'noise_flag':False}

    }
 

    os.chdir(settings['Data_path'])
    # check local directory
    if os.path.exists("./psp_data"):
        pass
    else:
        working_dir = os.getcwd()
        os.makedirs(str(Path(working_dir).joinpath("psp_data")), exist_ok=True)

    
    settings = {**default_settings, **settings}
    

    
    # Ensure the dates have appropriate format
    t0i, t1i = func.ensure_time_format(start_time, end_time)
    
    # Since pyspedas does not always return what tou ask for we have to enforce it
    t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
    t1 = func.add_time_to_datetime_string(t1i,  time_amount, time_unit)

    # Add something extra to lower-cadence timeseries
    t0i_e  = func.add_time_to_datetime_string(t0i, -2, 'm')
    t1i_e = func.add_time_to_datetime_string(t1i,  2, 'm')
    
    
    # In order to return the originaly requested interval
    ind1  = func.string_to_datetime_index(t0i)
    ind2  = func.string_to_datetime_index(t1i)
    
    ind1_e  = func.string_to_datetime_index(t0i_e)
    ind2_e  = func.string_to_datetime_index(t1i_e)
    
    # Specify variables to download
    varnames_MAG, varnames_QTN, varnames_SPAN, varnames_SPC,  varnames_SPAN_alpha, varnames_EPHEM, varnames_E_field, varnames_SC_pot = default_variables_to_download_PSP(vars_2_downnload)
    
    print( varnames_QTN)
    # Download QTN data
    try:
        dfqtn, diagnostics_QTN, dfqtn_flag, dfqtn_big_gaps = process_qtn_data(t0, t1, credentials, varnames_QTN, ind1_e, ind2_e, settings)
    except:
        traceback.print_exc()

    
    print('FLAG QTN', dfqtn_flag)
    #Download ephemeris

    dfephem                            = process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1_e, ind2_e, settings)
    

    # Add some thresholds
    mean_dist        = round(np.nanmean(dfephem['Dist_au'].values),2)
    dist_threshold   = (settings['max_PSP_dist'] > mean_dist) | (settings['max_PSP_dist']   == None)
    qtn_threshold    = (dfqtn_flag  == 'QTN')                 | (settings['must_have_qtn']  == False)


    if (dist_threshold) & (qtn_threshold):
        
        # Download sc_potential data
        df_SC_pot, big_gaps_SC_pot, diagnostics_SC_pot = (
                                                    process_sc_potential_data(t0, t1, settings, credentials, varnames_SC_pot, ind1, ind2)
                                                    if vars_2_downnload.get('sc_pot', False)
                                                    else (None, None,  {
                                                                        "Init_dt"         : np.nan,
                                                                        "resampled_df"    : None,
                                                                        "Frac_miss"       : None,
                                                                        "Large_gaps"      : None,
                                                                        "Tot_gaps"        : None,
                                                                        "resol"           : None
                                                                    }))
            
        # Download Electric field data
        df_e_field, big_gaps_e_field, diagnostics_e_field = (
                                                    process_e_field_data(t0, t1, settings, credentials, varnames_E_field, ind1, ind2)
                                                    if vars_2_downnload.get('E_field', False)
                                                    else (None, None,  {
                                                                        "Init_dt"         : np.nan,
                                                                        "resampled_df"    : None,
                                                                        "Frac_miss"       : None,
                                                                        "Large_gaps"      : None,
                                                                        "Tot_gaps"        : None,
                                                                        "resol"           : None
                                                                    }))
                                                        

        
        # Download magnetic field data
        dfmag, big_gaps, diagnostics_MAG                   = process_mag_field_data(t0, t1, settings,
                                                                  credentials, varnames_MAG, ind1, ind2)
       
        # Download SPAN data
        dfspan, diagnostics_SPAN, span_flag, big_gaps_span = process_span_data(t0, t1, credentials,
                                                                varnames_SPAN, varnames_SPAN_alpha, settings, ind1_e, ind2_e)
  

        # Download SPC data 
        dfspc, diagnostics_SPC, spc_flag, big_gaps_spc     = process_spc_data(t0, t1, credentials, varnames_SPC, settings, ind1_e, ind2_e)


        try:

            dfpar, part_flag, dfqtn_flag, big_gaps_par    = create_particle_dataframe(end_time,
                                                                                       diagnostics_SPC,
                                                                                       diagnostics_SPAN,
                                                                                       pd.DataFrame(diagnostics_QTN["resampled_df"]),
                                                                                       dfqtn_flag,
                                                                                       big_gaps_span,
                                                                                       big_gaps_spc,
                                                                                       settings)

            diagnostics_PAR        = func.resample_timeseries_estimate_gaps(dfpar, 
                                                                            settings['part_resol'],
                                                                            large_gaps=10)
   

            keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
            misc = {
                'SPC'              : func.filter_dict(diagnostics_SPC,  keys_to_keep),
                'SPAN'             : func.filter_dict(diagnostics_SPAN, keys_to_keep),
                'QTN'              : func.filter_dict(diagnostics_QTN, keys_to_keep),
                'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
                'E'                : func.filter_dict(diagnostics_e_field,  keys_to_keep),
                'SC_pot'           : func.filter_dict(diagnostics_SC_pot,  keys_to_keep),
                'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep),
                'part_flag'        : part_flag,
                'qtn_flag'         : dfqtn_flag
            }
            
            # Lazy way to do this. Fix that
            if dfqtn_flag =='No_QTN':
                diagnostics_PAR["resampled_df"]['np']   = diagnostics_PAR["resampled_df"].pop('np_sweap')

            return diagnostics_QTN["resampled_df"], diagnostics_MAG["resampled_df"], diagnostics_PAR["resampled_df"], diagnostics_e_field["resampled_df"], diagnostics_SC_pot["resampled_df"], dfephem.interpolate(), big_gaps,  dfqtn_big_gaps, big_gaps_par, big_gaps_SC_pot,  misc

        except Exception as e:
            logging.exception("An error occurred: %s", e)
            return None, None, None, None, None, None, None, None, None, None, None
    else:
        if (dist_threshold == False) and (qtn_threshold == False):
            logging.info(BG_BLUE+'Discarded, No qtn and d=%s' + RESET, mean_dist)
        elif dist_threshold == False:
            logging.info(BG_BLUE+'Discarded, d=%s' + RESET, mean_dist)
        elif qtn_threshold == False:
            logging.info(BG_BLUE+'Discarded, no qtn dat.' + RESET)


            

def LoadSCAMFromSPEDAS_PSP(t0,
                           t1,
                           credentials,
                           settings):
    """ 
    load scam data with pyspedas and return a dataframe
    Input:
        start_time, end_time                pd.Timestamp
        (optional) credentials              dictionary, {'username':..., 'password':...}
    Output:
        return None if no data is present, otherwise a dataframe containing all the scam data
    """ 


    username = credentials['psp']['fields']['username']
    password = credentials['psp']['fields']['password']

    # use credentials
    try:
        if settings['in_rtn']:
            
            print('Working on RTN frame, SCAM DATA')
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf',
                        varnames = ['psp_fld_l3_merged_scam_wf_RTN'], level='l3', time_clip=1, downloadonly = False,
                username = username,
                password = password )
        else:
            print('Working on SC frame, SCAM DATA')
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf',
                        varnames = ['psp_fld_l3_merged_scam_wf_SC'], level='l3', time_clip=1, downloadonly = False,
                username = username, 
                password = password )                
            
            
        if scam_vars == []:
            return None

        if settings['in_rtn']:
            data   = get_data(scam_vars[0])
            dfscam = pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['Br','Bt','Bn'])
        else:
            data   = get_data(scam_vars[0])
            dfscam = pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['Bx','By','Bz']) 
            
        dfscam.index = time_string.time_datetime(time=dfscam.index)
        dfscam.index = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'
        



    except:
        traceback.print_exc()
        dfscam                 = None
        big_gaps               = None
        diagnostics_MAG        = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
 
    return dfscam
