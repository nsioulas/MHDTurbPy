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


# Change current directory to the MHDTurbPy folder
os.chdir("/Users/nokni/work/MHDTurbPy/")

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
    
    

def download_MAG_FIELD_PSP(t0, t1, mag_resolution, credentials, varnames):
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
                                              time_clip=True, username=username, password=password)
                

               
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
                MAGdata = pyspedas.psp.fields(trange=[t0, t1], datatype=datatype, level='l2', time_clip=True)           
                


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
        




def download_SPC_PSP(t0, t1, credentials, varnames):
    
    print('Spc Variables', varnames)
    
    try:
        try:
            username = credentials['psp']['sweap']['username']
            password = credentials['psp']['sweap']['password']

            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='L3', 
                                        varnames=varnames, time_clip=True, 
                                        username=username, password=password)
            if len(spcdata)==0:
                print("No data available for this interval.")
                return None, None
        except:
            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")

            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='l3', 
                                        varnames=varnames, time_clip=True)

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
        print(f'Error occurred while retrieving SPC data: {e}')
        return None, None

    
def download_SPAN_PSP(t0, t1, credentials, varnames, varnames_alpha ):   
    
    print('Span Variables', varnames)
    
    try:
        try:
            username = credentials['psp']['sweap']['username']
            password = credentials['psp']['sweap']['password']
            
            

            spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00', level='L3', 
                varnames = varnames, time_clip=True, username=username, password=password)

            spandata_alpha = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf0a', level='L3', 
                varnames=varnames_alpha, time_clip=True, username=username, password=password)
            
            if len(spandata)==0:
                print("No data available for this interval.")
                return None, None
        except:
            if credentials is None:
                print("No credentials were provided. Attempting to utilize publicly accessible data.")
                
            spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00_l3_mom', level='l3', 
                varnames=varnames, time_clip=True)

            spandata_alpha = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf0a_l3_mom', level='l3', 
                varnames=varnames_alpha, time_clip=True)

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
        print(f'Error occurred while retrieving SPAN data: {e}')
        return None, None

def download_QTN_PSP(t0, t1, credentials, varnames):
    
    print('QTN', varnames)
    try:
        try:
            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']
            
            qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_V1V2', level='l3',
                                        varnames=varnames,
                                        time_clip=True, username=username, password=password)
            
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
        
        return dfqtn
    except Exception as e:
        print(f'Error occurred while retrieving QTN data: {e}')
        return None

def download_ephemeris_PSP(t0, t1, credentials, varnames):
    try:
        username = credentials['psp']['fields']['username']
        password = credentials['psp']['fields']['password']
        
        ephemdata = pyspedas.psp.fields(trange=[t0, t1], datatype='ephem_spp_rtn', level='l1', 
                                         varnames=varnames, time_clip=True, username=username, password=password)
        
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
    
    except:
        traceback.print_exc()
        raise ValueError("Ephemeris could not be loaded!")
        

        
def create_particle_dataframe(diagnostics_SPC,
                              diagnostics_SPAN,
                              start_time, 
                              end_time, 
                              df_spc, 
                              df_span,
                              dfqtn,
                              settings):
    """
    Creates a DataFrame for particle data based on the specified settings and diagnostics.
    
    Args:
        diagnostics_SPC (dict): Diagnostic information for SPC.
        diagnostics_SPAN (dict): Diagnostic information for SPAN.
        start_time (str): Start time for data sampling.
        end_time (str): End time for data sampling.
        df_spc, df_span, dfqtn (DataFrame): DataFrames for SPC, SPAN, and QTN data.
        settings (dict): Settings for particle data creation.

    Returns:
        DataFrame: The particle data DataFrame.
        str: Particle flag.
        str or None: QTN flag.
    """

    # Default is '9th_perih_cut'!
    particle_mode = settings.get('particle_mode', '9th_perih_cut')

    try:
        if particle_mode == 'spc':
            return process_instrument_mode('spc', diagnostics_SPC, dfqtn)

        elif particle_mode == 'span':
            return process_instrument_mode('span', diagnostics_SPAN, dfqtn)

        elif particle_mode == '9th_perih_cut':
            return process_9th_perih_cut(start_time, end_time, df_spc, df_span, dfqtn)
        elif particle_mode == 'keep_both':

            return process_keep_both(df_spc, df_span, dfqtn)
        else:
            raise ValueError(f"Particle mode '{particle_mode}' not supported.")
            
        
    except:
        traceback.print_exc()
def process_instrument_mode(mode, 
                            diagnostics, 
                            dfqtn):
    df_particle                         = diagnostics['resampled_df']
    df_particle[df_particle < -1e5]     = np.nan
    df_particle, qtn_flag               = use_qtn_df(df_particle, dfqtn)
    return df_particle, mode, qtn_flag

def process_9th_perih_cut(start_time,
                          end_time, 
                          df_spc, 
                          df_span,
                          dfqtn):
    
    use_spc                             = pd.Timestamp(end_time) < pd.Timestamp('2021-07-15')
    source_df                           = df_spc if use_spc else df_span
    df_particle, qtn_flag               = use_qtn_df(source_df, dfqtn)
    
    
    return df_particle, 'empirical', qtn_flag

def use_qtn_df(source_df,
                    dfqtn):
    try:
        new_dfqtn                                    = func.newindex(dfqtn, source_df.index)
        df_particle                                  = source_df.join(new_dfqtn['np_qtn'])
        df_particle[df_particle < -1e5]              = np.nan
        df_particle['np']                            = df_particle['np_qtn']
        df_particle.drop(columns=['np_qtn'], inplace = True)
        return df_particle, 'QTN'
    except Exception:
        return source_df, 'No_QTN'

def process_keep_both(df_spc,
                      df_span,
                      dfqtn):
    freq        = '5s'
    df_particle = pd.DataFrame()
    df_particle['np_qtn'] = dfqtn['np_qtn'].resample(freq).mean().interpolate()
    keep_keys = set(df_spc.columns).union(df_span.columns).intersection(['Vx', 'Vy', 'Vz', 'Vr', 'Vt', 'Vn', 'Vth', 'Dist_au', 'np'])
    for key in keep_keys:
        df_particle = add_instrument_data(df_particle, df_spc, df_span, key, freq)
    df_particle['np'] = df_particle['np_qtn']
    return df_particle, 'both', 'QTN' if 'np_qtn' in df_particle.columns else 'No_QTN'

def add_instrument_data(df_particle, df_spc, df_span, key, freq):
    try:
        df_particle[f'{key}_spc']    = df_spc[key].resample(freq).mean().interpolate()
    except KeyError:
        warnings.warn(f"Key: {key} not present in df_spc!")
        df_particle[f'{key}_spc']    = np.nan
    try:
        df_particle[f'{key}_span']   = df_span[key].resample(freq).mean().interpolate()
    except KeyError:
        warnings.warn(f"Key: {key} not present in df_span!")
        df_particle[f'{key}_span']   = np.nan
    return df_particle

# def create_particle_dataframe(diagnostics_SPC, diagnostics_SPAN, start_time, end_time, dfspc, dfspan,  dfqtn, settings):

    
    
#     # default to method suggested by SWEAP team if particle_mode is not provided
#     if settings['particle_mode'] == None:
#         part_instrument = '9th_perih_cut'
#     else:
#         part_instrument = settings['particle_mode']

    
#     # Define particle resolution
#     part_resolution = settings['part_resol']

#     if part_instrument == 'spc':
#         freq      = f"{round(diagnostics_SPC['Init_dt']*1000)}ms"
#         dfpar     = diagnostics_SPC['resampled_df']
#         part_flag = 'spc'

#     elif part_instrument == 'span':

#         freq      = f"{round(diagnostics_SPAN['Init_dt']*1000)}ms"
#         dfpar     = diagnostics_SPAN['resampled_df']
#         part_flag = 'span'
        
        
        
        

#     # before encounter 9 (Perihelion: 2021-08-09/19:11) use SPC for solar wind speed
#     # at and after encounter 8, mix SPC and SPAN for solar wind speed
#     # prioritize QTN for density, and fill with SPC, and with SPAN
#     elif part_instrument == '9th_perih_cut':
        
#         source_df   = dfspc if pd.Timestamp(end_time) < pd.Timestamp('2021-07-15') else dfspan
#         diagnostics = diagnostics_SPC if pd.Timestamp(end_time) < pd.Timestamp('2021-07-15') else diagnostics_SPAN

#         # interpolate QTN to index of either SPC or SPAN and fill nan!
#         try: 
#     
#             new_dfqtn = func.newindex(dfqtn, source_df.index)
#             dfpar     = source_df.join(new_dfqtn['np_qtn'])
            
#             dfpar[dfpar < -1e5] = np.nan
#             dfpar['np']         = dfpar['np_qtn']
#             del dfpar['np_qtn'], dfqtn
            
#             qtn_flag    = 'QTN'
#         except:
#             qtn_flag    = 'No_QTN'
#             dfpar       = source_df
#             print('No qtn data!')
            

#         part_flag = 'empirical'
        

#         del source_df

#     elif part_instrument  =='keep_both':
#         # A regular cadence
#         freq ='5s'
#         part_flag = 'empirical'
#         # add qtn
#         try:
#             dfpar['np_qtn'] = dfqtn['np_qtn'].resample(freq).mean().interpolate()
#         except:
#             warnings.warn("No QTN data!")
#             dfpar['np_qtn'] = np.nan

#         # Keep only the keys that exist in either dfspc or dfspan
#         keep_keys = ['Vx','Vy','Vz','Vr','Vt','Vn','Vth','Dist_au']
#         keep_keys = [k for k in keep_keys if k in dfspc.columns or k in dfspan.columns]
        
#         # add SPC
#         for k in keep_keys:
#             try:
#                 dfpar[k+"_spc"] = dfspc[k].resample(freq).mean().interpolate()
#             except:
#                 warnings.warn("key: %s not present in dfspc!" %(k))
#                 dfpar[k+"_spc"] = np.nan

#         # add SPAN
#         for k in keep_keys:
#             try:
#                 if k == 'na':
#                     dfpar[k+"_span"] = dfspan_a[k].resample(freq).mean().interpolate()
#                 else:
#                     dfpar[k+"_span"] = dfspan[k].resample(freq).mean().interpolate()
#             except:
#                 warnings.warn("key: %s not present in dfspan!" %(k))
#                 dfpar[k+"_span"] = np.nan

#         # set default
#         # default density: QTN
#         dfpar['np'] = dfpar['np_qtn']

#         # before encounter 9, default set to SPC
#         if dfpar.index[-1] < pd.Timestamp('2021-07-01'):
#             for k in keep_keys:
#                 dfpar[k] = dfpar[k+'_spc']
#         # at and after encounter 9, default set to SPAN
#         else:
#             for k in keep_keys:
#                 dfpar[k] = dfpar[k+'_span']

#         # raise ValueError("particle mode: %s under construction!" %(part_instrument))
    
#     else:
#         raise ValueError("particle mode: %s not supported!" %(part_instrument))
#     try:
#         del dfpar['na']
#     except:
#         pass
#     dfpar[dfpar < -1e5] = np.nan
    
#     return dfpar, part_flag, qtn_flag





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
    
    
    try:              
        # Download QTN data
        dfqtn                 = download_QTN_PSP(t0, t1, credentials, varnames_QTN)
        
        # Return the originaly requested interval
        dfqtn                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfqtn)
        diagnostics_QTN       = func.resample_timeseries_estimate_gaps(dfqtn, settings['part_resol'] , large_gaps=10)
        dfqtn_flag            = 'QTN'
    except:
        traceback.print_exc()
        dfqtn                 = None
        diagnostics_QTN       = None
        dfqtn_flag            = 'No QTN'
        

        
        
    if ((dfqtn_flag  == 'QTN') & (settings['must_have_qtn'])) | (settings['must_have_qtn']==False):
        
        try:
            # Download Magnetic field data
            dfmag                 = download_MAG_FIELD_PSP(t0, t1, settings['MAG_resol'], credentials, varnames_MAG)

            # Return the originaly requested interval
            try:
                dfmag                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfmag)
            except:
                dfmag.index = pd.to_datetime(dfmag.index, format='%Y-%m-%d %H:%M:%S.%f')
                dfmag                 = func.use_dates_return_elements_of_df_inbetween(pd.to_numeric(ind1), pd.to_numeric(ind2), dfmag)



             # Identify big gaps in timeseries
            big_gaps              = func.find_big_gaps(dfmag , settings['gap_time_threshold'])        
            # Resample the input dataframes
            diagnostics_MAG       = func.resample_timeseries_estimate_gaps(dfmag , settings['MAG_resol']  , large_gaps=10)      


        except:
            traceback.print_exc()
            dfmag                 = None
            big_gaps              = None
            diagnostics_MAG       = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}

        try:        
            # Download SPAN data
            dfspan  = download_SPAN_PSP(t0, t1, credentials, varnames_SPAN, varnames_SPAN_alpha)


            if settings['apply_hampel']:
                if 'Vr' in dfspan.keys():
                    list_2_hampel = ['Vr','Vt','Vn','np','Vth']
                else:
                    list_2_hampel = ['Vx','Vy','Vz','np','Vth']

                ws_hampel  = settings['hampel_params']['w']
                n_hampel   = settings['hampel_params']['std']

                for k in list_2_hampel:
                    try:
                        outliers_indices = func.hampel(dfspan[k], window_size = ws_hampel, n = n_hampel)

                        dfspan.loc[dfspan.index[outliers_indices], k] = np.nan
                    except:
                         traceback.print_exc()
                print('Applied hampel filter to SPAN columns :', list_2_hampel, 'Windows size', ws_hampel)


            # Return the originaly requested interval
            dfspan                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfspan)


            diagnostics_SPAN       = func.resample_timeseries_estimate_gaps(dfspan, settings['part_resol'] , large_gaps=10)
            span_flag = 'SPAN'
        except:

            traceback.print_exc()
            dfspan                = None

            diagnostics_SPAN      = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}
            span_flag = 'No SPAN'
        try:      
            # Download SPC data
            dfspc                 = download_SPC_PSP(t0, t1, credentials, varnames_SPC)

            # Return the originaly requested interval

            dfspc                 = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfspc)
            

            if settings['apply_hampel']:
                if 'Vr' in dfspc.keys():
                    list_2_hampel = ['Vr','Vt','Vn','np','Vth']
                else:
                    list_2_hampel = ['Vx','Vy','Vz','np','Vth']

                ws_hampel  = settings['hampel_params']['w']
                n_hampel   = settings['hampel_params']['std']

                for k in list_2_hampel:
                    try:
                        outliers_indices = func.hampel(dfspc[k], window_size = ws_hampel, n = n_hampel)
                        dfspc.loc[dfspc.index[outliers_indices], k] = np.nan
                    except:
                         traceback.print_exc()
                print('Applied hampel filter to SPC columns :', list_2_hampel)


             # Resample the input dataframes
            diagnostics_SPC       = func.resample_timeseries_estimate_gaps(dfspc , settings['part_resol'] , large_gaps=10)

            spc_flag = 'SPC'
        except:
            traceback.print_exc()
            dfspc                 = None
            diagnostics_SPC       = {'Frac_miss':100, 'Large_gaps':100, 'Tot_gaps':100, 'resol':100}

            spc_flag = 'No SPC'

        try: 

            # Download Ephemeris data
            dfephem               = download_ephemeris_PSP(t0, t1, credentials, varnames_EPHEM)


            dfephem                = func.use_dates_return_elements_of_df_inbetween(ind1, ind2, dfephem)


        except:
            dfephem               = None

        try:

            #Create final particle dataframe
            
            # Maybe fix later!
            

            dfpar, part_flag, qtn_flag = create_particle_dataframe(diagnostics_SPC,diagnostics_SPAN, start_time, end_time, dfspc, dfspan, dfqtn, settings)
            diagnostics_PAR            = func.resample_timeseries_estimate_gaps(dfpar, settings['part_resol'], large_gaps=10)



            keys_to_keep           = ['Frac_miss', 'Large_gaps', 'Tot_gaps', 'resol']
            misc = {
                'SPC'              : func.filter_dict(diagnostics_SPC,  keys_to_keep),
                'SPAN'             : func.filter_dict(diagnostics_SPAN, keys_to_keep),
                'QTN'              : func.filter_dict(diagnostics_QTN, keys_to_keep),
                'Par'              : func.filter_dict(diagnostics_PAR,  keys_to_keep),
                'Mag'              : func.filter_dict(diagnostics_MAG,  keys_to_keep),
                'part_flag'        : part_flag,
                'qtn_flag'         : qtn_flag
            }

            return diagnostics_MAG["resampled_df"], diagnostics_PAR["resampled_df"], dfephem.interpolate(), big_gaps, misc
        except:
            traceback.print_exc()
    else:
        print('No qtn data, and thus we wont consider the interval as specified in settings')
        

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
                username = credentials['username'], password = credentials['password'])
        else:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf',
                        varnames = ['psp_fld_l3_merged_scam_wf_SC'], level='l3', time_clip=1, downloadonly = False,
                username = credentials['username'], password = credentials['password'])                
            
            
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
