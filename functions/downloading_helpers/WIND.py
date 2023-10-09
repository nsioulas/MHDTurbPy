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


def LoadTimeSeriesWind_particles(start_time,
                                 end_time, 
                                 three_sec_resol =True):
    """ 
    Load Wind Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """

    if three_sec_resol == True:
        from cdasws import CdasWs
        cdas = CdasWs()
        vars = ['P_DENS','P_VELS','P_TEMP','TIME']
        time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
        status, data = cdas.get_data('WI_PM_3DP', vars, time[0], time[1])


        dfpar = pd.DataFrame(
            index = data['Epoch'],
            data = {
                'Vr': data['P_VELS'][:,0],
                'Vt': data['P_VELS'][:,1],
                'Vn': data['P_VELS'][:,2],
                'np': data['P_DENS'],
                'Tp': data['P_TEMP']
            }
        )
        
        
        dfpar[dfpar['Vr'] < -1e30] = np.nan
        dfpar['Vth'] = 0.128487*np.sqrt(dfpar['Tp']) # vth[km/s] = 0.128487 * √Tp[K]
        
        
        length = len(data['P_VELS'][:,0][::2])

    elif three_sec_resol =='Very_low_res':
        print('Loading very low resolution particle data!!')
        
        from cdasws import CdasWs
        cdas = CdasWs()
        vars         = ['MOM.P.DENSITY','MOM.P.VELOCITY', 'MOM.P.VTHERMAL','TIME']
        time         = [start_time, end_time]
        status, data = cdas.get_data('WI_PLSP_3DP', vars, str(time[0]), str(time[1]))
        
        dfpar = pd.DataFrame(
            index = data['Epoch'],
            data = {
                'Vr': data['MOM$P$VELOCITY'].T[0],
                'Vt': data['MOM$P$VELOCITY'].T[1],
                'Vn': data['MOM$P$VELOCITY'].T[2],
                'np': data['MOM$P$DENSITY'],
                'Vth': data['MOM$P$VTHERMAL']
            }
        )
        
        dfpar[dfpar['Vr'] < -1e30] = np.nan
        

        length = len(data['MOM$P$VELOCITY'].T[0][::2])
        
        
        
    dfdis = pd.DataFrame(
        index = data['Epoch'][::2],
        data = {
            'Dist_au': np.ones(length), 
            'lon'    : np.ones(length), 
            'lat'    : np.ones(length), 
            'RAD_AU' : np.ones(length)
        })



    return dfpar, dfdis



def LoadHighResMagWind(start_time,
                       end_time,
                       three_sec_resol= False,
                       verbose = True):
    """ 
    Load Wind Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """

    if three_sec_resol== True:

        vars = ['B3GSE','B3F1']
        time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
        status, data = cdas.get_data('WI_H0_MFI', vars, time[0], time[1])

        if verbose:
            print("Done.")

        dfmag = pd.DataFrame({'Epoch':data['Epoch3'],
                                'Br': data['B3GSE'][:,0],
                                'Bt': data['B3GSE'][:,1],
                                'Bn': data['B3GSE'][:,2],
                                'Btot': data['B3F1']
            }
        ).set_index('Epoch')
        
        dfmag[(np.abs(dfmag['Btot']) > 1e3)] = np.nan

        dfmag1 = dfmag.resample('1s').mean()

        if verbose:
            print("Input tstart = %s, tend = %s" %(time[0], time[1]))
            print("Returned tstart = %s, tend = %s" %(data['Epoch3'][0], data['Epoch3'][-1]))

        infos = {
            'resolution': 1
        }

    elif three_sec_resol == False:
        
        
        vars = ['BGSE','BF1']
        time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
        status, data = cdas.get_data('WI_H2_MFI', vars, time[0], time[1])

        if verbose:
            print("Done.")

        dfmag = pd.DataFrame({'Epoch':data['Epoch'],
                'Br': data['BGSE'][:,0],
                'Bt': data['BGSE'][:,1],
                'Bn': data['BGSE'][:,2],
                'Btot': data['BF1']
            }
        ).set_index('Epoch')
        
        
        dfmag[(np.abs(dfmag['Btot']) > 1e3)] = np.nan

        dfmag1 = dfmag.resample('1s').mean()

        if verbose:
            print("Input tstart = %s, tend = %s" %(time[0], time[1]))
            print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

        infos = {
            'resolution': 1
        }
    
    elif three_sec_resol =='Very_low_res':
        print('Loading very low resolution magnetic field data!!')
        
        try:

            vars         = ['MOM.P.MAGF']
            time         = [start_time, end_time]
            status, data = cdas.get_data('WI_PLSP_3DP', vars, str(time[0]), str(time[1]))

            if verbose:
                print("Done.")

            # Create dataframes for B, V, and N
            dfmag = pd.DataFrame({'Epoch': data['Epoch'],
                                     'Br': data['MOM$P$MAGF'].T[0],
                                     'Bt': data['MOM$P$MAGF'].T[1],
                                     'Bn': data['MOM$P$MAGF'].T[2],
                                 }).set_index('Epoch').interpolate()


            dfmag[(np.abs(dfmag['Br']) > 1e3)] = np.nan

            dfmag1 = dfmag.resample('500s').mean()

            if verbose:
                print("Input tstart = %s, tend = %s" %(time[0], time[1]))
                print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

            infos = {
                'resolution': 500
            }
        except:
            print('Somthing wrong')
            traceback.print_exc()


    return dfmag, dfmag1, infos


def LoadTimeSeriesWIND(start_time, 
                      end_time, 
                      settings, 
                      gap_time_threshold=10,
                      time_amount =4,
                      time_unit ='h',
                      three_sec_resol= False
                     ):
    """" 
    Load Time Series from WIND sc
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

        dfmag, dfmag1, infos = LoadHighResMagWind(pd.Timestamp(t0),
                                                  pd.Timestamp(t1),
                                                  three_sec_resol= three_sec_resol,
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
        dfpar, dfdis     = LoadTimeSeriesWind_particles(pd.Timestamp(t0),
                                                        pd.Timestamp(t1),
                                                        three_sec_resol =three_sec_resol )
        
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
