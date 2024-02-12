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
au_to_km        = 1.496e8  # Conversion factor
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
        
    if vars_2_downnload['rpw'] is None:
        varnames_RPW          = ['bia-density-10-seconds']
    else:
        varnames_RPW          = vars_2_downnload['rpw']
        
    if vars_2_downnload['swa'] is None:
        varnames_SWA          = ['N', 'V_RTN', 'T']
    else:
        varnames_SWA          = vars_2_downnload['swa']        
    
    if vars_2_downnload['ephem'] is None:
        varnames_EPHEM         = ['position','velocity']  
    else:
        varnames_EPHEM          = vars_2_downnload['ephem']
    return varnames_MAG,  varnames_SWA ,varnames_EPHEM, varnames_RPW


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
    
    
    rpw_cols = {
        'bia-density-10-seconds'      : ['ne_qtn'] ,
        'bia-density'                 : ['ne_qtn'] ,
    }

    # Mapping between variable names and column names for EPHEMERIS
    ephem_cols = {
        'position'            : ['sc_pos_r','sc_pos_t','sc_pos_n'],
        'velocity'            : ['sc_vel_r','sc_vel_t','sc_vel_n'],
    }    
  
    
    if instrument == 'SWA':
        return [swa_cols[var] for var in varnames if var in swa_cols]
    
    if instrument == 'RPW':
        return [rpw_cols[var] for var in varnames if var in rpw_cols]
    
    elif instrument == 'MAG':
        return [fields_MAG_cols[var] for var in varnames if var in fields_MAG_cols]
    elif instrument =='EPHEMERIS':
         return [ephem_cols[var] for var in varnames if var in ephem_cols]
    else:
        return []
    
    


from dateutil import parser

def download_MAG_SOLO(t0, t1, mag_resolution, varnames):
    def retrieve_mag_data(datatype):
        MAGdata = pyspedas.solo.mag(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True)
        col_names = map_col_names_SOLO('MAG', [datatype])
        df = pd.DataFrame(index=get_data(MAGdata[0]).times, data=get_data(MAGdata[0]).y, columns=col_names[0])
        return df

    try:
        dfmag = pd.DataFrame()
        mag_flag = None

        for varname in varnames: 
            if varname == 'B_RTN':
                if mag_resolution > 230:
                    datatype = 'rtn-normal' 
                    mag_flag ='Regular'
                    print('Using normal-resol data!')
                else:
                    datatype = 'rtn-burst'
            else:
                datatype = 'srf-normal' if mag_resolution > 230 else 'srf-burst'

            df = retrieve_mag_data(datatype)
            dfmag = dfmag.join(df, how='outer')

        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)

        int_dur = (parser.parse(t1) - parser.parse(t0)).total_seconds() / 3600
        deviation = abs((dfmag.index[-1] - parser.parse(t1)) / np.timedelta64(1, 'h')) + abs((dfmag.index[0] - parser.parse(t0)) / np.timedelta64(1, 'h'))

        if deviation >= 0.1 * int_dur:
            print('Too little burst data!')
            dfmag = pd.DataFrame()
            for varname in varnames:
                datatype = 'rtn-normal' if varname == 'B_RTN' else 'srf-normal'

                df = retrieve_mag_data(datatype)
                dfmag = dfmag.join(df, how='outer')

            dfmag.index = time_string.time_datetime(time=dfmag.index)
            dfmag.index = dfmag.index.tz_localize(None)
            mag_flag ='Regular'
        else:
            if mag_flag != 'Regular':
                print('Ok, We have enough burst mag data')
                mag_flag ='Burst'

        return dfmag, mag_flag
    except Exception as e:
        logging.exception("An error occurred: %s", e)
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
    
    
    
    
def download_RPW_SOLO(t0, 
                      t1, 
                      varnames):   
    
    for varname in varnames: 
        if varname == 'bia-density-10-seconds':
            datatype = 'bia-density-10-seconds'
            varname  = ['DENSITY']
        else:
            datatype = 'bia-density'
            varname  = ['DENSITY']

        MAGdata = pyspedas.solo.mag(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True)
        
        col_names = map_col_names_SOLO('RPW', [datatype])

    try:
        rpwdata = pyspedas.solo.rpw(trange=[t0, t1], level='l3', varnames = varname, datatype=datatype)


        dfs       = [pd.DataFrame(index=get_data(data).times, 
                                data=get_data(data).y, 
                                columns=col_names[i]) for i, data in enumerate(rpwdata)]
        dfrpw = dfs[0].join(dfs[1:])


        # Fix datetime index
        dfrpw.index       = time_string.time_datetime(time=dfrpw.index)
        dfrpw.index       = dfrpw.index.tz_localize(None)
        dfrpw.index.name  = 'datetime'
        
        dfrpw['np_qtn']   = dfrpw['ne_qtn']/ 1.08  # 4% of alpha particle

      
        return dfrpw
    
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        return None, None
    

def LoadTimeSeriesSOLO(start_time, 
                      end_time, 
                      settings, 
                      vars_2_downnload,
                      cdf_lib_path,
                      credentials     = None,
                      time_amount     = 12,
                      time_unit       = 'H'
                     ):

    os.chdir(settings['Data_path'])
    
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
 
    try:
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
        varnames_MAG,  varnames_SWA ,varnames_EPHEM, varnames_RPW = default_variables_to_download_SOLO(vars_2_downnload)


        # Load rpw data
        try:
            dfrpw = download_RPW_SOLO(t0, t1, varnames_RPW)

            # Return the originaly requested interval
            dfrpw = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfrpw)
            
            # Identify big gaps in timeseries
            big_gaps_qtn = func.find_big_gaps(dfrpw, settings['Big_Gaps']['QTN_big_gaps'])


            # Resample the input dataframes
            diagnostics_RPW = func.resample_timeseries_estimate_gaps(dfrpw, settings['part_resol'], large_gaps=10)

            # apply hampel filter
            if settings['use_hampel'] == True:
                list_quants = ['np_qtn']
                for k in list_quants:
                    ns, _ = func.hampel_filter(dfrpw[k].values, 100)
                    dfrpw[k] = ns

            dfqtn_flag = 'QTN'

        except:
            dfrpw                = None
            dfqtn_flag           =  'NO_QTN'
            traceback.print_exc()

            diagnostics_RPW      = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}


        if ((dfqtn_flag  == 'QTN') & (settings['must_have_qtn'])) | (settings['must_have_qtn']==False):

            # Load particle data
            try:
                dfpar = download_SWA_SOLO(t0, t1, varnames_SWA)

                # Return the originaly requested interval
                dfpar = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)
                
                # Identify big gaps in timeseries
                big_gaps_par = func.find_big_gaps(dfpar, settings['Big_Gaps']['Par_big_gaps'])

                # Resample the input dataframes
                diagnostics_PAR  = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'], large_gaps=10)

                # apply hampel filter
                if settings['use_hampel'] == True:
                    list_quants = ['np', 'T', 'Vth', 'Vr', 'Vt', 'Vn']
                    for k in list_quants:
                        ns, _ = func.hampel_filter(dfpar[k].values, 100)
                        dfpar[k] = ns

                part_flag    = 'SWA'

                # interpolate QTN to index of either SPC or SPAN and fill nan!
                try: 

                    dfrpw          = dfrpw[~dfrpw.index.duplicated(keep='first')]
                    dfpar          = dfpar[~dfpar.index.duplicated(keep='first')]
                    dfrpw          = func.newindex(dfrpw, dfpar.index)

                    dfpar['np']    = dfrpw['np_qtn']

                    qtn_flag    = 'QTN'
                except Exception as e:
                    logging.exception("No qtn data because: %s", e)
                    qtn_flag    = 'No_QTN'

            except Exception as e:
                logging.exception("No qtn data because: %s", e)
                dfpar                = None


                diagnostics_PAR      = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
                pass

            if dfpar is not  None:

                # Load Magnetic field data
                try:
                    dfmag, mag_flag       = download_MAG_SOLO(t0, t1, settings['MAG_resol'], varnames_MAG)

                    # Return the originaly requested interval
                    dfmag                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)

                    # Identify big gaps in timeseries
                    big_gaps              = func.find_big_gaps(dfmag, settings['Big_Gaps']['Mag_big_gaps'])

                    # Resample the input dataframes
                    diagnostics_MAG       = func.resample_timeseries_estimate_gaps(dfmag , settings['MAG_resol']  , large_gaps=10)
                except:
                    traceback.print_exc()
                    dfmag                 = None
                    big_gaps              = None
                    diagnostics_MAG       = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}


                # Load distance data
                try:
                        # Since pyspedas does not always return what tou ask for we have to enforce it
                        
                    #t0i, t1i = func.ensure_time_format(start_time, end_time)
                    #t0a = func.add_time_to_datetime_string(t0i, -30, 'H')
                    #t1a = func.add_time_to_datetime_string(t1i,  30, 'H')
                    
                    fname = '/Volumes/Zesen-4TB/solar_orbiter_data/distance/solo_dist.pkl'
                    dfdis = pd.read_pickle(fname)
                    dfdis = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfdis)


                except Exception as e:
                    dfdis                = None
                    logging.exception("No qtn data because: %s", e)
                    pass


                keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
                misc = {
                    'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
                    'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep),
                    'QTN'              : func.filter_dict(diagnostics_RPW,  keys_to_keep),
                    'part_flag'        : part_flag,
                    'qtn_flag'         : qtn_flag

                }
            else:
                dfmag, mag_flag, dfpar, dfdis, big_gaps, big_gaps_qtn,  big_gaps_par, misc = None, None, None, None, None, None, None, None
                

            return diagnostics_MAG["resampled_df"], mag_flag , diagnostics_PAR["resampled_df"], dfdis, big_gaps,big_gaps_qtn,  big_gaps_par, misc
        else: 
            print('No qtn data, and thus we wont consider the interval as specified in settings')

    except:
        traceback.print_exc()