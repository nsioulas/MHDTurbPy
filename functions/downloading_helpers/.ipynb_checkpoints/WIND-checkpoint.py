# Basic libraries
import pandas as pd
import numpy as np
import sys
import traceback
import time
import datetime
import pytz

# Locate files
import os
from pathlib import Path
from glob import glob


# SPDF API
from cdasws import CdasWs
cdas = CdasWs()

# SPEDAS API
# make sure to use the local spedas
sys.path.insert(0,"../pyspedas")
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



def LoadTimeSeriesWind_particles(start_time,
                                 end_time,
                                 settings):
    """ 
    Load Wind Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """

    if settings['part_resol'] <=3:
        from cdasws import CdasWs
        cdas = CdasWs()
        vars = ['P_DENS','P_VELS','P_TEMP','TIME']
        time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
        status, data         = cdas.get_data('WI_PM_3DP', vars, time[0], time[1])
        
        
        dfpar = pd.DataFrame(
            index = data['Epoch'],
            data = {
                'Vr'     : data['P_VELS'][:,0],
                'Vt'     : data['P_VELS'][:,1],
                'Vn'     : data['P_VELS'][:,2],
                #'np_3DP' : data['P_DENS'],
                'np' : data['P_DENS'],
                'Tp'     : data['P_TEMP']
            }
        )
        
#         # Also use qtn data to remove offset
#         try:
#             vars_qtn                 = ['Ne','Ne_peak','Ne_Quality']
#             status_qtn, data_qtn = cdas.get_data('WI_H0_WAV', vars_qtn, time[0], time[1])
#             dfqtn     = pd.DataFrame(
#                 index = data_qtn['Epoch'],
#                 data = {

#                     'np': data_qtn['NE$']*0.96,

#                 }
#             )
        
        
#             # Estimate offset between the timeseries
#             dnp         = np.nanmedian(dfqtn.values) - np.nanmedian(dfpar['np_3DP'].values)
            
            
#             dfqtn           = func.newindex(dfqtn, dfpar.index)
#             dfpar['np']     = dfpar['np_3DP'] + dnp
#             dfpar['np_qtn'] = dfqtn.values
            
            
#             qtn_flag = 'QTN'
        
#         except:
#             dfpar['np']     = dfpar['np_3DP'].values
            
#             del dfpar['np_3DP']
            
#             qtn_flag = 'No_QTN'
            
        
        qtn_flag                   = 'No_QTN'
        dfpar[dfpar['Vr'] < -1e30] = np.nan
        dfpar['Vth']               = 0.128487*np.sqrt(dfpar['Tp']) # vth[km/s] = 0.128487 * √Tp[K]
        
        
        length = len(data['P_VELS'][:,0][::2])

    elif settings['part_resol'] >3:
        
        qtn_flag = None
        
        print('Loading very low resolution particle data!!')
        
        from cdasws import CdasWs
        cdas = CdasWs()
        vars         = ['MOM.P.DENSITY','MOM.P.VELOCITY', 'MOM.P.VTHERMAL','TIME']
        time         = [start_time, end_time]
        status, data = cdas.get_data('WI_PLSP_3DP', vars, str(time[0]), str(time[1]))
        
        dfpar = pd.DataFrame(
            index = data['Epoch'],
            data = {
                'Vr' : data['MOM$P$VELOCITY'].T[0],
                'Vt' : data['MOM$P$VELOCITY'].T[1],
                'Vn' : data['MOM$P$VELOCITY'].T[2],
                'np' : data['MOM$P$DENSITY'],
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



    return dfpar, dfdis, qtn_flag



def LoadHighResMagWind(start_time,
                       end_time,
                       settings,
                       verbose         = True):
    """ 
    Load Wind Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """

    if settings['MAG_resol'] == 3:

        vars = ['B3GSE','B3F1']
        time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
        status, data = cdas.get_data('WI_H0_MFI', vars, time[0], time[1])

        if verbose:
            print("Done.")

        dfmag = pd.DataFrame({'Epoch'  : data['Epoch3'],
                                'Br'   : data['B3GSE'][:,0],
                                'Bt'   : data['B3GSE'][:,1],
                                'Bn'   : data['B3GSE'][:,2],
                                'Btot' : data['B3F1']
            }
        ).set_index('Epoch')
        
        dfmag[(np.abs(dfmag['Btot']) > 1e3)] = np.nan


        if verbose:
            print("Input tstart = %s, tend = %s" %(time[0], time[1]))
            print("Returned tstart = %s, tend = %s" %(data['Epoch3'][0], data['Epoch3'][-1]))


    elif settings['MAG_resol'] < 3:
        
        
        vars = ['BGSE','BF1']
        time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
        status, data = cdas.get_data('WI_H2_MFI', vars, time[0], time[1])

        if verbose:
            print("Done.")

        dfmag = pd.DataFrame({'Epoch'  : data['Epoch'],
                                 'Br'  : data['BGSE'][:,0],
                                 'Bt'  : data['BGSE'][:,1],
                                 'Bn'  : data['BGSE'][:,2],
                                 'Btot': data['BF1']
            }
        ).set_index('Epoch')
        
        
        dfmag[(np.abs(dfmag['Btot']) > 1e3)] = np.nan


        if verbose:
            print("Input tstart = %s, tend = %s" %(time[0], time[1]))
            print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    else:
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

            if verbose:
                print("Input tstart = %s, tend = %s" %(time[0], time[1]))
                print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))


        except:
            print('Something wrong')
            traceback.print_exc()


    return dfmag


def LoadTimeSeriesWIND(start_time, 
                      end_time, 
                      settings, 
                      gap_time_threshold  =  10,
                      time_amount         =  4,
                      time_unit           =  'h'
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

        dfmag                = LoadHighResMagWind(pd.Timestamp(t0),
                                                  pd.Timestamp(t1),
                                                  settings,
                                                  verbose          = True)

        
        # Return the originaly requested interval
        try:
            dfmag                 = func.use_dates_return_elements_of_df_inbetween(ind1,
                                                                                   ind2,
                                                                                   dfmag)
            
        except:
            dfmag.index           = pd.to_datetime(dfmag.index, format='%Y-%m-%d %H:%M:%S.%f')
            dfmag                 = func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(ind1),
                                                                                   pd.to_numeric(ind2),
                                                                                   dfmag)

        # Identify big gaps in timeseries
        big_gaps = func.find_big_gaps(dfmag, settings['Big_Gaps']['Mag_big_gaps']) 
        
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
        dfpar, dfdis,qtn_flag     = LoadTimeSeriesWind_particles(pd.Timestamp(t0),
                                                        pd.Timestamp(t1),
                                                        settings)
        
        # Return the originaly requested interval
        dfpar                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfpar) 
        
        
        # Identify big gaps in timeseries
        big_gaps_par = func.find_big_gaps(dfpar, settings['Big_Gaps']['Par_big_gaps'])
        
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

        return diagnostics_MAG["resampled_df"], None, dfpar, dfdis, big_gaps, big_gaps_par, None, misc, qtn_flag
    except:
        traceback.print_exc()
