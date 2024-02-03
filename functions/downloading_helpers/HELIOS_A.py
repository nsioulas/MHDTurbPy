from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import traceback
import numpy as np
from scipy.optimize import curve_fit
import gc
import time
from scipy.signal import savgol_filter
import pandas as pd
#from numba import jit,njit,prange,objmode 
import os
from pathlib import Path
from glob import glob
from gc import collect
import warnings

import datetime
import pytz
#from numba import jit,njit,prange

# SPDF API
from cdasws import CdasWs
cdas = CdasWs()

# SPEDAS API
# make sure to use the local spedas
import sys
sys.path.insert(0,"../pyspedas")
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func
import Figures as figs
from   SEA import SEA
import three_D_funcs as threeD


au_to_km = 1.496e8  # Conversion factor
rsun     = 696340   # Sun radius in units of  [km]

from scipy import constants
psp_ref_point   =  0.06176464216946593 # in [au]
mu0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu_0             =  constants.mu_0  # Vacuum magnetic permeability [N A^-2]
m_p             =  constants.m_p   # Proton mass [kg]
k          = constants.k                  # Boltzman's constant     [j/K]
au_to_km   = 1.496e8
au_to_rsun = 215.032
T_to_Gauss = 1e4



def LoadTimeSeriesHELIOS_A_particles(start_time,
                                 end_time):
    """ 
    Load HELIOS_A Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """

    from cdasws import CdasWs
    cdas = CdasWs()


    vars =['Vp_R','Vp_T','Vp_N','crot','Np','Vp','Tp', 'R_Helio','ESS_Ang','clong','clat']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('HELIOS1_40SEC_MAG-PLASMA', vars, time[0], time[1])


    dfpar = pd.DataFrame(
        index = data['Epoch'],
        data = {

            'Vr' : data['Vp_R'],
            'Vt' : data['Vp_T'],
            'Vn' : data['Vp_N'],
            'np' : data['Np'],
            'Tp' : data['Tp'],
        }
    )


    dfpar[dfpar['Vr'] < -1e30] = np.nan
    dfpar['Vth'] = 0.128487*np.sqrt(dfpar['Tp']) # vth[km/s] = 0.128487 * √Tp[K]

        
    dfdis = pd.DataFrame(
        index = data['Epoch'],
        data = {
                'Dist_au'  : data['R_Helio'],
                'lon'      : data['clong'],
                'lat'      : data['clat'],
                'RAD_AU'   : data['R_Helio'],
        })

    return dfpar, dfdis


def LoadHighResMagHELIOS_A(start_time,
                       end_time,
                       verbose = True):
    """ 
    Load HELIOS_A Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """
    
    print('Note this is NOT in RTN!!!')
    
    vars = ['BXSSE','BYSSE','BZSSE','B']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('HEL1_6SEC_NESSMAG', vars, time[0], time[1])

    dfmag = pd.DataFrame(
        index = data['Epoch'],
        data = {
            'Bx': data['BXSSE'],
            'By': data['BYSSE'],
            'Bz': data['BZSSE'],
            'Btot': data['B']
        }
    )

    dfmag[(np.abs(dfmag['Btot']) > 1e3)] = np.nan
    dfmag1 = dfmag.resample('%ds' %(6)).mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    infos = {
        'resolution': 1
    }

    return dfmag, dfmag1, infos


def LoadMagHELIOS_A(start_time,
                    end_time,
                    verbose = True):
    """ 
    Load HELIOS_A Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """
    
    vars = ['B_R','B_T','B_N']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('HELIOS1_40SEC_MAG-PLASMA', vars, time[0], time[1])

    if verbose:
        print("Done.")

    dfmag = pd.DataFrame(
        index = data['Epoch'],
        data = {
                'Br' : data['B_R'],
                'Bt' : data['B_T'],
                'Bn' : data['B_N']
        }
    )
    
    dfmag['Btot'] = np.sqrt(dfmag['Br']**2 + dfmag['Bt']**2 + dfmag['Bn']**2)
    dfmag[(np.abs(dfmag['Btot']) > 1e3)] = np.nan
    dfmag1 = dfmag.resample('%ds' %(40)).mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    infos = {
        'resolution': 1
    }

    return dfmag, dfmag1, infos


def LoadTimeSeriesHELIOS_A(start_time, 
                      end_time, 
                      settings, 
                      gap_time_threshold=10,
                      time_amount =4,
                      time_unit ='h'
                     ):
    """" 
    Load Time Series from HELIOS_A sc
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
        'apply_hampel'    : True,
        'hampel_params'   : {'w':100, 'std':3},
        'part_resol'   : 3000,
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
    
    
   

    try:
        # Download Magnetic field data

        dfmag, dfmag1, infos = LoadMagHELIOS_A(pd.Timestamp(t0),
                                                  pd.Timestamp(t1),
                                                  verbose = True)

        print('Not ok!!')
        # Return the originaly requested interval
        try:
            dfmag                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
        except:
            dfmag.index = pd.to_datetime(dfmag.index, format='%Y-%m-%d %H:%M:%S.%f')
            
            dfmag                 = func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(ind1), pd.to_numeric(ind2), dfmag)

         # Identify big gaps in timeseries
        big_gaps              = func.find_big_gaps(dfmag , gap_time_threshold)        
        # Resample the input dataframes
        diagnostics_MAG       = func.resample_timeseries_estimate_gaps(dfmag , settings['MAG_resol']  , large_gaps=10)      
        
        print('Mag fraction missing', diagnostics_MAG['Frac_miss'])
    except:
        
        traceback.print_exc()
        dfmag                 = None
        big_gaps              = None
        diagnostics_MAG       = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}

    try:        
        # Download particle data
        dfpar, dfdis     = LoadTimeSeriesHELIOS_A_particles(pd.Timestamp(t0),
                                                        pd.Timestamp(t1))
        
        # Return the originaly requested interval
        dfpar                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar)       
        # Resample the input dataframes
        diagnostics_PAR = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'] , large_gaps=10)
        
        print('Par fraction missing', diagnostics_PAR['Frac_miss'])
    except:
        traceback.print_exc()
        dfpar                 = None
        diagnostics_PAR       = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
    

    try:

        #Create final particle dataframe
        
                    
        if settings['apply_hampel']:
            print('Applying hampel filter to particle data!')
            if 'Vr' in dfpar.keys():
                list_2_hampel = ['Vr','Vt','Vn','np','Vth']
            else:
                list_2_hampel = ['Vx','Vy','Vz','np','Vth']
                
            ws_hampel  = settings['hampel_params']['w']
            n_hampel   = settings['hampel_params']['std']
                
            for k in list_2_hampel:
                try:
                    outliers_indices = func.hampel(dfpar[k], window_size = ws_hampel, n = n_hampel)
                    # print(outliers_indices)
                    dfpar.loc[dfpar.index[outliers_indices], k] = np.nan
                except:
                     traceback.print_exc()
            print('Applied hampel filter to SPAN columns :', list_2_hampel, 'Windows size', ws_hampel)

        keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
        misc = {
            'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
            'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep),

        }

        return diagnostics_MAG["resampled_df"].interpolate().dropna(), dfpar.interpolate().dropna(), dfdis, big_gaps, misc
    except:
        traceback.print_exc()
