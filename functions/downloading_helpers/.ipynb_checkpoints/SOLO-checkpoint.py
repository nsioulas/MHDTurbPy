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


#Important!! Make sure your current directory is the MHDTurbPy folder!
os.chdir("/Users/nokni/work/MHDTurbPy/")


# Make sure to use the local spedas
sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data


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



import pytz
# SPDF API
from cdasws import CdasWs
cdas = CdasWs()




def default_variables_to_download_SOLO(vars_2_downnload):
    
    if vars_2_downnload['mag'] is None:
        varnames_MAG          = ['B_RTN']
    else:
        varnames_MAG          = vars_2_downnload['mag']
        
    if vars_2_downnload['qtn'] is None:
        varnames_QTN          = ['electron_density','electron_core_temperature']
    else:
        varnames_QTN          = vars_2_downnload['qtn']
        
    if vars_2_downnload['swa'] is None:
        varnames_SWA          = ['N', 'V_RTN', 'T']
    else:
        varnames_SWA          = vars_2_downnload['swa']        
    


    if vars_2_downnload['ephem'] is None:
        varnames_EPHEM         = ['position','velocity']  
    else:
        varnames_EPHEM          = vars_2_downnload['ephem']
    return varnames_MAG,  varnames_SWA ,varnames_EPHEM


def map_col_names_SOLO(instrument, varnames):
    
    # Mapping between variable names and column names for FIELDS
    fields_MAG_cols = {
        'rtn-normal'                   : ['Br', 'Bt', 'Bn'],
        'srf-normal'                   : ['Bx', 'By', 'Bz'],
        'rtn-burst'                    : ['Br', 'Bt', 'Bn'],
        'srf-burst'                    : ['Bx', 'By', 'Bz'],

    }

    # Mapping between variable names and column names for SPAN
    swa_cols = {
        'N'               : ['np'] ,
        'T'               : ['T'] ,
        'V_RTN'           : ['Vr','Vt','Vn'],
        'V_SRF'           : ['Vx','Vy','Vz'],
        'V_SOLO_RTN'      : ['sc_vel_r','sc_vel_t','sc_vel_n']

    }

    # Mapping between variable names and column names for EPHEMERIS
    ephem_cols = {
        'position'            : ['sc_pos_r','sc_pos_t','sc_pos_n'],
        'velocity'            : ['sc_vel_r','sc_vel_t','sc_vel_n'],
    }    
  
    
    if instrument == 'SWA':
        return [swa_cols[var] for var in varnames if var in swa_cols]
    elif instrument == 'MAG':
        print('ok')
        return [fields_MAG_cols[var] for var in varnames if var in fields_MAG_cols]
    elif instrument =='EPHEMERIS':
         return [ephem_cols[var] for var in varnames if var in ephem_cols]
    else:
        return []
    
    


def download_MAG_SOLO(t0, t1, mag_resolution,  varnames, download_SCAM=False):
    # Just to make sure!
    if download_SCAM:
        mag_resolution = 1

    try:
        dfmag = pd.DataFrame()
        for varname in varnames: 
            if varname == 'B_RTN':
                datatype = 'rtn-normal' if mag_resolution > 230 else 'rtn-burst'
            else:
                datatype = 'srf-normal' if mag_resolution > 230 else 'srf-burst'
                
            MAGdata = pyspedas.solo.mag(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True)
            col_names = map_col_names_SOLO('MAG', [datatype])
            #print(col_names)
            

            dfs = [pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=col_names[i]) 
                   for i, data in enumerate([MAGdata[0]])]
            df = dfs[0].join(dfs[1:])
            dfmag = dfmag.join(df, how='outer')
        
        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)
        
        # In case there is too little burst data!
        from dateutil import parser
        int_dur    = (parser.parse(t1) - parser.parse(t0)).total_seconds() / 3600
        deviation  = abs((dfmag.index[-1] - parser.parse(t1)) / np.timedelta64(1, 'h')) + abs((dfmag.index[0] - parser.parse(t0)) / np.timedelta64(1, 'h'))
        if deviation >= 0.1 * int_dur:
            print('Too little burst data!')
            dfmag = pd.DataFrame()
            for varname in varnames:
                if varname == 'B_RTN':
                    datatype = 'rtn-normal'
                else:
                    datatype = 'srf-normal'
            
                MAGdata = pyspedas.solo.mag(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True)
                col_names = map_col_names_SOLO('MAG', [datatype])


                dfs = [pd.DataFrame(index=get_data(data).times, data=get_data(data).y, columns=col_names[i]) 
                   for i, data in enumerate([MAGdata[0]])]
                df = dfs[0].join(dfs[1:])
                dfmag = dfmag.join(df, how='outer')
        
            dfmag.index = time_string.time_datetime(time=dfmag.index)
            dfmag.index = dfmag.index.tz_localize(None)
        else:
            print('Ok, We have enough burst mag data')

        return dfmag
    except Exception as e:
        print(f'Error occurred while retrieving MAG data: {e}')
        return None

    
def download_ephem_SOLO(t0, t1, cdf_lib_path):
    
    # Set enivronemnt 
    os.environ["CDF_LIB"] = cdf_lib_path
    
    time = [(pd.Timestamp(t0)-pd.Timedelta('3d')).to_pydatetime( ).replace(tzinfo=pytz.UTC), (pd.Timestamp(t1)+pd.Timedelta('3d')).to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('SOLO_HELIO1DAY_POSITION', ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON'], time[0], time[1])

    dfdis = pd.DataFrame(
        index = data['Epoch'],
        data = data[['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']]
    )
    dfdis.index.name = 'datetime'

    dfdis['Dist_au'] = dfdis['RAD_AU']
    
    return dfdis

def download_SWA_SOLO(t0, t1, varnames):   
    try:
        swadata = pyspedas.solo.swa(trange=[t0, t1],varnames = varnames, datatype='pas-grnd-mom')

        #SWA protons
        col_names = map_col_names_SOLO('SWA', varnames)
        dfs = [pd.DataFrame(index=get_data(data).times, 
                            data=get_data(data).y, 
                            columns=col_names[i]) for i, data in enumerate(swadata)]
        dfswa = dfs[0].join(dfs[1:])

        #  Estimate Vth
        dfswa['Vth'] = 13.84112218 * np.sqrt(dfswa['T']) 

        # Fix datetime index
        dfswa.index       = time_string.time_datetime(time=dfswa.index)
        dfswa.index       = dfswa.index.tz_localize(None)
        dfswa.index.name  = 'datetime'
        
        dfswa['np_qtn']   = dfswa['np']
        dfswa['ne_qtn']   = dfswa['np']
      
        return dfswa
    except Exception as e:
        print(f'Error occurred while retrieving SWA data: {e}')
        return None, None
    
    

def LoadTimeSeriesSOLO(start_time, 
                      end_time, 
                      settings, 
                      vars_2_downnload,
                      cdf_lib_path,
                      credentials = None,
                      download_SCAM =False,
                      gap_time_threshold=10,
                      time_amount =12,
                      time_unit ='h'
                     ):

    # check local directory
    if os.path.exists("./solar_orbiter_data"):
        pass
    else:
        working_dir = os.getcwd()
        os.makedirs(str(Path(working_dir).joinpath("solar_orbiter_data")), exist_ok=True)

    # default settings
    default_settings = {
        'use_hampel'   : False,
        'part_resol'   : 900,
        'MAG_resol'    : 1

    }
 
    settings = {**default_settings, **settings}

    # Ensure the dates have appropriate format
    t0i, t1i = func.ensure_time_format(start_time, end_time)
    
    # Since pyspedas does not always return what tou ask for we have to enforce it
    t0 = func.add_time_to_datetime_string(t0i, -time_amount, time_unit)
    t1 = func.add_time_to_datetime_string(t1i,  time_amount, time_unit)

    # In order to return the originaly requested interval
    ind1  = func.string_to_datetime_index(t0i)
    ind2  = func.string_to_datetime_index(t1i)
    
    # Define variales to download
    varnames_MAG,  varnames_SWA ,varnames_EPHEM = default_variables_to_download_SOLO(vars_2_downnload)
    
    # Load Magnetic field data
    try:
        dfmag                 = download_MAG_SOLO(t0, t1, settings['MAG_resol'], varnames_MAG, download_SCAM=download_SCAM)
        
        # Return the originaly requested interval
        dfmag                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
        
        # Identify big gaps in timeseries
        big_gaps              = func.find_big_gaps(dfmag, gap_time_threshold)

        # Resample the input dataframes
        diagnostics_MAG       = func.resample_timeseries_estimate_gaps(dfmag , settings['MAG_resol']  , large_gaps=10)
    except:
        traceback.print_exc()
        dfmag                 = None
        big_gaps              = None
        diagnostics_MAG       = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
        
    # Load particle data
    try:
        dfpar = download_SWA_SOLO(t0, t1, varnames_SWA)
        
        # Return the originaly requested interval
        dfpar = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)
        
        # Resample the input dataframes
        diagnostics_PAR  = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'], large_gaps=10)
        
        # apply hampel filter
        if settings['use_hampel'] == True:
            list_quants = ['np', 'T', 'Vth', 'Vr', 'Vt', 'Vn']
            for k in list_quants:
                ns, _ = func.hampel_filter(dfpar[k].values, 100)
                dfpar[k] = ns
                
    except:
        dfpar                = None
        traceback.print_exc()
        
        diagnostics_PAR      = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
        pass
    
    # Load distance data
    try:
        dfdis                = download_ephem_SOLO(t0, t1, cdf_lib_path)

        # if dfpar is not None:
        #     dfdis            = func.newindex(dfdis, dfpar.index)
        #     dfpar['Dist_au'] = dfdis['Dist_au']
    except:
        dfdis                = None
        traceback.print_exc()
        pass


    keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
    misc = {
        'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
        'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep),

    }

    return dfmag, dfpar, dfdis, big_gaps, misc
    
