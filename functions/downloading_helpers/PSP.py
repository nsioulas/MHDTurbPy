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
    return varnames_MAG, varnames_QTN, varnames_SPAN, varnames_SPC,  varnames_SPAN_alpha,varnames_EPHEM


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
    
    

def download_MAG_FIELD_PSP(t0, t1, mag_resolution, credentials, varnames, settings):
    try:
        for j, varname in enumerate(varnames): 

            try:
                
                
                print('Using private mag data')
                if varname == 'B_RTN':
                    print('Using RTN frame mag data.')
                    if mag_resolution> 230:               # It's better to use lower resol if you want to resample to SPC, SPAN cadence. 
                        datatype = 'mag_RTN_4_Sa_per_Cyc'
                    else:
                        datatype = 'mag_RTN'
                else:
                    print('Using SC frame mag data.')
                    if mag_resolution> 230:
                        datatype = 'mag_SC_4_Sa_per_Cyc'
                    else:
                        datatype = 'mag_SC'

                username = credentials['psp']['fields']['username']
                password = credentials['psp']['fields']['password']
                MAGdata = pyspedas.psp.fields(trange=[t0, t1], datatype=datatype, level='l2', 
                                              time_clip=True, username=username, password=password, no_update=settings['use_local_data'])
                

               
            except:
                traceback.print_exc()
                print('Using public mag data')
                
                if varname == 'B_RTN':
                    print('Using RTN frame mag data.')
                    if mag_resolution> 230:
                        datatype = 'mag_rtn_4_per_cycle'
                    else:
                        datatype = 'mag_rtn'
                else:
                    print('Using SC frame mag data.')
                    if mag_resolution> 230:
                        datatype = 'mag_sc_4_per_cycle'
                    else:
                        datatype = 'mag_sc'
                MAGdata = pyspedas.psp.fields(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True, no_update=settings['use_local_data'])           
                


            if j == 0:
                col_names = map_col_names_PSP('FIELDS-MAG', [datatype])
                if mag_resolution< 230:
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
                if mag_resolution< 230:
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
        

def process_mag_field_data(t0, t1, settings, credentials, varnames_MAG, ind1, ind2):
    try:
        dfmag = download_MAG_FIELD_PSP(t0, t1, settings['MAG_resol'], credentials, varnames_MAG, settings)
        try:
            dfmag = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
        except Exception:
            # Adjust index format and retry processing
            dfmag.index = pd.to_datetime(dfmag.index, format='%Y-%m-%d %H:%M:%S.%f')
            dfmag = func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(ind1), pd.to_numeric(ind2), dfmag)
        
        # Identify big gaps in timeseries
        big_gaps = func.find_big_gaps(dfmag, settings['Big_Gaps']['Mag_big_gaps'])
        # Resample the input dataframes
        diagnostics_MAG = func.resample_timeseries_estimate_gaps(dfmag, settings['MAG_resol'], large_gaps=10)
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
                                        username=username, password=password, no_update=settings['use_local_data'])
            if len(spcdata)==0:
                print("No data available for this interval.")
                return None, None
        except:
            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")

            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='l3', 
                                        varnames=varnames, time_clip=True, no_update=settings['use_local_data'])

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

        # Identify big gaps in timeseries
        big_gaps_spc = func.find_big_gaps(dfspc, settings['Big_Gaps']['Par_big_gaps'])
        
        # Calculate diagnostics
        diagnostics_SPC = func.resample_timeseries_estimate_gaps(dfspc, settings['part_resol'], large_gaps=10)
        spc_flag = 'SPC'
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Return None for dfspc and default values for diagnostics_SPC on error
        dfspc, diagnostics_SPC, spc_flag, big_gaps_spc = None, {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}, 'No SPC', None

    return dfspc, diagnostics_SPC, spc_flag, big_gaps_spc


    
def download_SPAN_PSP(t0, t1, credentials, varnames, varnames_alpha, settings ):   
    
    print('Span Variables', varnames)
    
    try:
        try:
            username = credentials['psp']['sweap']['username']
            password = credentials['psp']['sweap']['password']
            
            

            spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00', level='L3', 
                varnames = varnames, time_clip=True, username=username, password=password, no_update=settings['use_local_data'])

            spandata_alpha = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf0a', level='L3', 
                varnames=varnames_alpha, time_clip=True, username=username, password=password, no_update=settings['use_local_data'])
            
            if len(spandata)==0:
                print("No data available for this interval.")
                return None, None
        except:
            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")
                
            spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00_l3_mom', level='l3', 
                varnames=varnames, time_clip=True, no_update=settings['use_local_data'])

            spandata_alpha = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf0a_l3_mom', level='l3', 
                varnames=varnames_alpha, time_clip=True, no_update=settings['use_local_data'])

        #SPAN protons
        col_names = map_col_names_PSP('SPAN', varnames)
        dfs = [pd.DataFrame(index=get_data(data).times, 
                            data=get_data(data).y, 
                            columns=col_names[i]) for i, data in enumerate(spandata)]
        dfspan = dfs[0].join(dfs[1:])
        
        
        # Estimate distance in [au]
        dfspan['Dist_au'] = dfspan['Dist_au']/au_to_km
        
        #  Estimate Vth
        # k T (eV) = 1/2 mp Vth^2 => Vth = 13.84112218*sqrt(TEMP)
        dfspan['Vth'] = 13.84112218 * np.sqrt(dfspan['TEMP'])

        # for span the thermal speed is defined as the trace, hence have a sqrt(3) different from spc
        dfspan['Vth'] = dfspan['Vth']/np.sqrt(3)
        
        # Fix datetime index
        dfspan.index       = time_string.time_datetime(time=dfspan.index)
        dfspan.index       = dfspan.index.tz_localize(None)
        dfspan.index.name  = 'datetime'
        
#         #SPAN Alphas
#         col_names_alpha = map_col_names_PSP('SPAN-alpha', varnames_alpha)
#         dfs_alpha = [pd.DataFrame(index=get_data(data).times, 
#                             data=get_data(data).y, 
#                             columns=col_names_alpha[i]) for i, data in enumerate(spandata_alpha)]
#         dfspan_alpha = dfs_alpha[0].join(dfs_alpha[1:])
        
#         # Fix datetime index
#         dfspan_alpha.index       = time_string.time_datetime(time=dfspan_alpha.index)
#         dfspan_alpha.index       = dfspan_alpha.index.tz_localize(None)
#         dfspan_alpha.index.name  = 'datetime'        
        return dfspan#, dfspan_alpha


    except Exception as e:
        logging.exception("An error occurred: %s", e)
        return None, None
    
    
import logging
import numpy as np

def process_span_data(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings, ind1, ind2):
    try:
        dfspan = download_SPAN_PSP(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha, settings)

        if settings['apply_hampel']:
            # Determine columns for Hampel filter application
            columns_for_hampel = ['Vr', 'Vt', 'Vn', 'np', 'Vth'] if 'Vr' in dfspan.keys() else ['Vx', 'Vy', 'Vz', 'np', 'Vth']
            ws_hampel, n_hampel = settings['hampel_params']['w'], settings['hampel_params']['std']

            # Apply Hampel filter to specified columns
            for column in columns_for_hampel:
                try:
                    outliers_indices = func.hampel(dfspan[column], window_size=ws_hampel, n=n_hampel)
                    dfspan.loc[dfspan.index[outliers_indices], column] = np.nan
                except Exception as e:
                    logging.exception("An error occurred while filtering %s: %s", column, e)

            print(f'Applied Hampel filter to SPAN columns: {columns_for_hampel}, Window size: {ws_hampel}')

        # Trim data to the originally requested interval
        dfspan = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfspan)
        
        # Identify big gaps in timeseries
        big_gaps_span = func.find_big_gaps(dfspan, settings['Big_Gaps']['Par_big_gaps'])

        # Calculate diagnostics
        diagnostics_SPAN = func.resample_timeseries_estimate_gaps(dfspan, settings['part_resol'], large_gaps=10)
        span_flag = 'SPAN'
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        dfspan, diagnostics_SPAN, span_flag, big_gaps_span = None, {'Frac_miss': 100, 'Large_gaps': 100, 'Tot_gaps': 100, 'resol': 100}, 'No SPAN', None

    return dfspan, diagnostics_SPAN, span_flag, big_gaps_span



def download_QTN_PSP(t0, t1, credentials, varnames, settings):
    
    print('QTN', varnames)
    try:
        try:
            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']
            
            qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_V1V2', level='l3',
                                        varnames=varnames,
                                        time_clip=True, username=username, password=password, no_update=settings['use_local_data'])
            
            if len(qtndata)==0:
                print("No data available for this interval.")
                return None
        except:
            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")
            

            qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_v1v2', level='l3',
                        varnames=varnames,
                        time_clip=True)
            
        col_names = map_col_names_PSP('QTN', varnames)
        dfs       = [pd.DataFrame(index=get_data(data).times, 
                                data=get_data(data).y, 
                                columns=col_names[i]) for i, data in enumerate(qtndata)]
        dfqtn = dfs[0].join(dfs[1:])

        dfqtn['np_qtn'] = dfqtn['ne_qtn'] / 1.08  # 4% of alpha particle
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
    
    #Temporary fix
    if ((pd.Timestamp(t0)>pd.Timestamp('2023-09-20')) &  (pd.Timestamp(t1)<pd.Timestamp('2023-10-01 23:59:39'))): 
        
        ll_path       = '/Users/nokni/work/turb_amplitudes/final_data/qtn/'
        dfqtn           = pd.read_pickle(str(Path(ll_path).joinpath('qtn.pkl')))

    else:
        dfqtn = download_QTN_PSP(t0, t1, credentials, varnames_QTN, settings)
    
    try:
        # Process the downloaded data
        dfqtn           = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfqtn)
        big_gaps        = func.find_big_gaps(dfqtn, settings['Big_Gaps']['QTN_big_gaps'])
        
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

def download_ephemeris_PSP(t0, t1, credentials, varnames, settings):
    try:
        username = credentials['psp']['fields']['username']
        password = credentials['psp']['fields']['password']
        
        ephemdata = pyspedas.psp.fields(trange=[t0, t1], datatype='ephem_spp_rtn', level='l1', 
                                         varnames=varnames, time_clip=True, username=username, password=password, no_update=settings['use_local_data'])
        
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
                              df_spc, 
                              df_span, 
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
            new_dfqtn               = func.newindex(dfqtn, source_df.index)
            df_particle             = source_df.join(new_dfqtn['np_qtn']).pipe(func.replace_negative_with_nan)
            df_particle['np_sweap'] = df_particle.pop('np')
            df_particle['np']       = df_particle.pop('np_qtn')
            return df_particle, 'QTN'

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

    return df_selected, settings['particle_mode'], dfqtn_flag, big_gaps



def LoadTimeSeriesPSP(start_time, 
                      end_time, 
                      settings, 
                      vars_2_downnload,
                      cdf_lib_path,
                      credentials = None,
                      time_amount =12,
                      time_unit ='h'
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
        'MAG_resol'       : 1

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

    # In order to return the originaly requested interval
    ind1  = func.string_to_datetime_index(t0i)
    ind2  = func.string_to_datetime_index(t1i)
    
    # Specify variables to download
    varnames_MAG, varnames_QTN, varnames_SPAN, varnames_SPC,  varnames_SPAN_alpha, varnames_EPHEM = default_variables_to_download_PSP(vars_2_downnload)
    
    
    # Download QTN data
    dfqtn, diagnostics_QTN, dfqtn_flag, dfqtn_big_gaps = process_qtn_data(t0, t1, credentials, varnames_QTN, ind1, ind2, settings)
        
    print('FLAG QTN', dfqtn_flag)
    #Download ephemeris
    dfephem                            = process_ephemeris(t0, t1, credentials, varnames_EPHEM, ind1, ind2, settings)
        

    # Add some thresholds
    mean_dist        = round(np.nanmean(dfephem['Dist_au'].values),2)
    dist_threshold   = (settings['max_PSP_dist'] > mean_dist) | (settings['max_PSP_dist']   == None)
    qtn_threshold    = (dfqtn_flag  == 'QTN')                 | (settings['must_have_qtn']  == False)
        
    if (dist_threshold) & (qtn_threshold):
        

        # Download magnetic field data
        dfmag, big_gaps, diagnostics_MAG                   = process_mag_field_data(t0, t1, settings,
                                                                  credentials, varnames_MAG, ind1, ind2)
       
        # Download SPAN data
        dfspan, diagnostics_SPAN, span_flag, big_gaps_span = process_span_data(t0, t1, credentials,
                                                                varnames_SPAN, varnames_SPAN_alpha, settings, ind1, ind2)
        # Download SPC data 
        dfspc, diagnostics_SPC, spc_flag, big_gaps_spc     = process_spc_data(t0, t1, credentials, varnames_SPC, settings, ind1, ind2)

      
        try:

            dfpar, part_flag, dfqtn_flag, big_gaps_par    = create_particle_dataframe(end_time,
                                                                       diagnostics_SPC,
                                                                       diagnostics_SPAN,
                                                                       dfspc,
                                                                       dfspan,
                                                                       dfqtn,
                                                                       dfqtn_flag,
                                                                       big_gaps_span,
                                                                       big_gaps_spc,
                                                                       settings)
            
            diagnostics_PAR                 = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'], large_gaps=10)
   

            keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
            misc = {
                'SPC'              : func.filter_dict(diagnostics_SPC,  keys_to_keep),
                'SPAN'             : func.filter_dict(diagnostics_SPAN, keys_to_keep),
                'QTN'              : func.filter_dict(diagnostics_QTN, keys_to_keep),
                'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
                'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep),
                'part_flag'        : part_flag,
                'qtn_flag'         : dfqtn_flag
            }
            
            # Lazy way to do this. Fix that
            if dfqtn_flag =='No_QTN':
                diagnostics_PAR["resampled_df"]['np']   = diagnostics_PAR["resampled_df"].pop('np_sweap')
                

            return diagnostics_MAG["resampled_df"], diagnostics_PAR["resampled_df"], dfephem.interpolate(), big_gaps,  dfqtn_big_gaps, big_gaps_par, misc
        
     
        except Exception as e:
            logging.exception("An error occurred: %s", e)
            return None, None, None, None, None, None, None
    else:
        if (dist_threshold == False) and (qtn_threshold == False):
            logging.info(BG_BLUE+'Discarded, No qtn and d=%s' + RESET, mean_dist)
        elif dist_threshold == False:
            logging.info(BG_BLUE+'Discarded, d=%s' + RESET, mean_dist)
        elif qtn_threshold == False:
            logging.info(BG_BLUE+'Discarded, no qtn dat.' + RESET)


def LoadSCAMFromSPEDAS_PSP(in_RTN, start_time, end_time, credentials = None):
    """ 
    load scam data with pyspedas and return a dataframe
    Input:
        start_time, end_time                pd.Timestamp
        (optional) credentials              dictionary, {'username':..., 'password':...}
    Output:
        return None if no data is present, otherwise a dataframe containing all the scam data
    """ 
    
    # Ensure the dates have appropriate format
    t0i, t1i = func.ensure_time_format(start_time, end_time)
    
    # Since pyspedas does not always return what tou ask for we have to enforce it
    t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
    t1 = func.add_time_to_datetime_string(t1i,  time_amount, time_unit)

    # In order to return the originaly requested interval
    ind1  = func.string_to_datetime_index(t0i)
    ind2  = func.string_to_datetime_index(t1i)

    # use credentials
    try:
        if in_RTN:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf',
                        varnames = ['psp_fld_l3_merged_scam_wf_RTN'], level='l3', time_clip=1, downloadonly = False,
                username = credentials['username'], password = credentials['password'], no_update=settings['use_local_data'])
        else:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf',
                        varnames = ['psp_fld_l3_merged_scam_wf_SC'], level='l3', time_clip=1, downloadonly = False,
                username = credentials['username'], password = credentials['password'], no_update=settings['use_local_data'])                
            
            
        if scam_vars == []:
            return None

        if  in_RTN:
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


        # Return the originaly requested interval
        try:
            dfscam                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfscam)
        except:
            dfscam.index = pd.to_datetime(dfscam.index, format='%Y-%m-%d %H:%M:%S.%f')
            dfscam                 = func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(ind1), pd.to_numeric(ind2), dfscam)



        # Identify big gaps in timeseries
        big_gaps              = func.find_big_gaps(dfscam , settings['gap_time_threshold'])        
        # Resample the input dataframes
        diagnostics_MAG       = func.resample_timeseries_estimate_gaps(dfscam , settings['MAG_resol']  , large_gaps=10)      


    except:
        traceback.print_exc()
        dfscam                 = None
        big_gaps               = None
        diagnostics_MAG        = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
 
    return dfscam
