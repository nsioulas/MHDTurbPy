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
import gc
from scipy.interpolate import interp1d

# Make sure to use the local spedas
sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data


""" Import manual functions """

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import general_functions as func


sys.path.insert(1, os.path.join(os.getcwd(), 'functions/downloading_helpers'))
from  PSP import  LoadTimeSeriesPSP, download_ephemeris_PSP
from  SOLO import LoadTimeSeriesSOLO
from  WIND import LoadTimeSeriesWIND
from  HELIOS_A import LoadTimeSeriesHELIOS_A
from  HELIOS_B import LoadTimeSeriesHELIOS_B



def download_files( ok,
                    df,
                    settings,
                    vars_2_downnload,
                    cdf_lib_path,
                    credentials,
                    save_path,
                    three_sec_resol    = False):
    
    try:
        t0 = df['Start'][ok]
        t1 = df['End'][ok]

        """Setup for main function"""
        path0  = Path(save_path)
        tfmt   = "%Y-%m-%d_%H-%M-%S"
                                                             
        

        start_time  = df['Start'][ok]
        end_time    = df['End'][ok]
        
        
        # Define folder name
        foldername  = "%s_%s_sc_%d" %(str(start_time.strftime(tfmt)), str(end_time.strftime(tfmt)), 0)

        #if not os.path.exists(path0.joinpath(foldername).joinpath('final_data.pkl')):
        if (not os.path.exists(path0.joinpath(foldername)))  | (settings['overwrite_files']):
            if (not os.path.exists(path0.joinpath(foldername))):
                print('Creating new folder', path0.joinpath(foldername))
            else:
                print('Overwriting folder', path0.joinpath(foldername))   
                
                
            # Running the main function
            big_gaps, flag_good, final, general, sig_c_sig_r_timeseries, dfdis =     main_function(
                                                                                                    start_time         , 
                                                                                                    end_time           , 
                                                                                                    settings           , 
                                                                                                    vars_2_downnload   ,
                                                                                                    cdf_lib_path       ,
                                                                                                    credentials        ,
                                                                                                    three_sec_resol    = three_sec_resol)

            try:
                final['Par']['V_resampled'] = final['Par']['V_resampled'].join(func.newindex(dfdis[['sc_vel_r', 'sc_vel_t', 'sc_vel_n']],
                                                                                             final['Par']['V_resampled'].index))

            except:
                pass


            if flag_good == 1:
                os.makedirs(path0.joinpath(foldername), exist_ok=True)
                
                if settings['save_all']== True:
                    func.savepickle(final, str(path0.joinpath(foldername)), "final.pkl" )
                    func.savepickle(general, str(path0.joinpath(foldername)), "general.pkl" )
                    func.savepickle(sig_c_sig_r_timeseries, str(path0.joinpath(foldername)), "sig_c_sig_r.pkl" )
                    func.savepickle(big_gaps, str(path0.joinpath(foldername)), "gaps.pkl" )
                
                else:
                    print('Deleting memory intensive timeseries per your commands. Only keeping derived quantities!')
                    
                    del final["Mag"]['B_resampled'], final["Par"]['V_resampled']
                    func.savepickle(final, str(path0.joinpath(foldername)), "final.pkl" )
                    func.savepickle(general, str(path0.joinpath(foldername)), "general.pkl" )

                gc.collect()

                print("%d out of %d finished" %(ok, len(df)))
            else:
                os.makedirs(path0.joinpath(foldername), exist_ok=True)
                print("%s - %s failed!" %(ok, len(df)))   
    except Exception as e:
        os.makedirs(path0.joinpath(foldername), exist_ok=True)
        print('failed at index', ok, 'with error:', str(e))
        traceback.print_exc()

        
        
        

def main_function( 
                start_time         , 
                end_time           , 
                settings           , 
                vars_2_downnload   ,
                cdf_lib_path       ,
                credentials        ,  
                three_sec_resol    = True
              ):

    
    # Parker Solar Probe
    if settings['sc']=='PSP':
        
        print('Working on PSP data')
        mag_flag                                     = None
        dfmag, dfpar, dfdis, big_gaps, misc          = LoadTimeSeriesPSP(
                                                                          start_time, 
                                                                          end_time, 
                                                                          settings, 
                                                                          vars_2_downnload,
                                                                          cdf_lib_path,
                                                                          credentials        = credentials,
                                                                          time_amount        = settings['addit_time_around'],
                                                                          time_unit          = 'h')
        
        #print(dfpar)
        # Delete na because it is causing issues!
        try:
            del dfpar['na']
        except:
            print('No na')
        
        
    # Solar Orbiter
    elif settings['sc']=='SOLO':
        
        
        print('Working on SOLO data')
        
        try:
            dfmag, mag_flag, dfpar, dfdis, big_gaps, misc           =  LoadTimeSeriesSOLO(
                                                                                          start_time, 
                                                                                          end_time, 
                                                                                          settings, 
                                                                                          vars_2_downnload,
                                                                                          cdf_lib_path,
                                                                                          credentials        = credentials,
                                                                                          time_amount        = settings['addit_time_around'],
                                                                                          time_unit          = 'h')

            if (settings['SOLO_use_burst']) and (mag_flag != 'Burst'):
                dfmag, dfapr =None, None
                print('We dont have Burst Data, thus we wont consider this interval')
                
        except:
            traceback.print_exc()
    elif settings['sc']=='HELIOS_A':
        
        print('Working on HELIOS_A data')
        try:
            mag_flag                            = None
            dfmag, dfpar, dfdis, big_gaps, misc =  LoadTimeSeriesHELIOS_A(
                                                                              start_time, 
                                                                              end_time, 
                                                                              settings) 
        except:
            traceback.print_exc()
            
            
    elif settings['sc']=='HELIOS_B':
        
        print('Working on HELIOS_B data')
        try:
            mag_flag                            = None
            dfmag, dfpar, dfdis, big_gaps, misc =  LoadTimeSeriesHELIOS_B(
                                                                          start_time, 
                                                                          end_time, 
                                                                          settings) 
        except:
            traceback.print_exc()
        
    elif settings['sc']=='WIND':

        print('Working on WIND data')
        try:
            mag_flag = None
            dfmag, dfpar, dfdis, big_gaps, misc =  LoadTimeSeriesWIND(
                                                                          start_time, 
                                                                          end_time, 
                                                                          settings, 
                                                                          three_sec_resol= three_sec_resol) 
        except:
            traceback.print_exc()

    if dfpar is not None:
        print('Missing', misc['Par']['Frac_miss'])
        if misc['Par']['Frac_miss'] < settings['Max_par_missing'] :
            try:
                
                """ Make sure both correspond to the same interval """ 
                dfpar                                             = func.use_dates_return_elements_of_df_inbetween(dfmag.index[0], dfmag.index[-1], dfpar) 


                try:
                    general_dict =  calc.general_dict_func(misc,
                                                          dfmag,
                                                          dfpar,
                                                          dfdis)

                    
                    if settings['estimate_derived_param']:
                        mag_dict, part_dict, sig_c_sig_r_timeseries                 = calc.calculate_diagnostics(
                                                                                                                    dfmag,
                                                                                                                    dfpar,
                                                                                                                    dfdis, 
                                                                                                                    settings,
                                                                                                                    misc
                                                                                                                )
                    else:
                        mag_dict, part_dict, sig_c_sig_r_timeseries = {}, {}, {}
                        mag_dict['B_resampled']                     = dfmag.dropna().interpolate()

                except:
                    traceback.print_exc()   

                
                # Also keep Velocity field data
                part_dict['V_resampled'] = dfpar.interpolate()
                
                
                # Now save everything in final_dict as a dictionary
                final_dict = { 
                               "Mag"          : mag_dict,
                               "Par"          : part_dict,
                               "Mag_flag"     : mag_flag
                                
                              }

                # also save a general dict with basic info (add what is missing from particletimeseries)
                general_dict["Fraction_missing_part"]  = misc['Par']['Frac_miss']
                general_dict["Resolution_part"]        = misc['Par']["resol"]
                
                if (settings['sc']=='PSP') | (settings['sc']=='SOLO'):
                    try:
                        general_dict["Fraction_missing_qtn"]  = misc['QTN']['Frac_miss']
                        general_dict["Resolution_qtn"]        = misc['QTN']["resol"]
                    except:
                        general_dict["Fraction_missing_qtn"]  = None
                        general_dict["Resolution_qtn"]        = None

                general_dict["sc"]                            = settings["sc"]

                flag_good = 1

            except:
                traceback.print_exc()

                flag_good = 0
                print('No MAG data!')
                big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
        else:
            traceback.print_exc()
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
    



        
        
def generate_intervals(start_time_str,
                       end_time_str,
                       settings,
                       data_path):
    
    import datetime
    
    # Parse the start and end times
    start_date = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
    end_date = datetime.datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')


    if settings.get('Only_1_interval', False):
        # Generate only one interval
        start_times = [start_date]
        end_times = [end_date]

        print("Generating only one interval based on the provided start and end times.")
    else:
        # Generate the start times
        start_times = pd.date_range(start_date, end_date, freq=settings['Step'])

        # Generate the end times
        try:
            end_times = [time + pd.Timedelta(hours=int(settings['duration'][:-1])) for time in start_times]
        except:
            dur  = round(int(settings['duration'][:-3])/60,2)
            end_times = [time + pd.Timedelta(hours=dur) for time in start_times]            

        # Trim the last end time if it exceeds the end_date
        if end_times[-1] > end_date:
            end_times = end_times[:-1]
            start_times = start_times[:-1]

    # Create a DataFrame
    df = pd.DataFrame({'Start': start_times, 'End': end_times})

    # Print details in a professional format
    print(f"Start Time: {start_date}")
    print(f"End Time: {end_date}")
    if not settings.get('Only_1_interval', False):
        print(f"Step: {settings['Step']}")
        print(f"Duration: {settings['duration']}")
        print(f"Number of Intervals: {len(start_times)}")
    else:
        print(f"Considering an interval spanning: {end_date, start_date}")

    return df