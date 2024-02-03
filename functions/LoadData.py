# %matplotlib tk
from distutils.log import warn
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


# -----------  Load Data ----------- #
# all data will be loaded with LoadTimeSeriesWrapper
# LoadTimeSeriesWrapper will return a dataframe containing all the quantities with standardized naming.


def LoadTimeSeriesWrapper(
    sc, start_time, end_time,
    settings = None, credentials = None):
    """ A wrapper for time series loading """

    # change to root dir, mainly for spedas
    if 'rootdir' in settings.keys():
        os.chdir(settings['rootdir'])
    else:
        pass

    # clean settings
    if settings is not None:
        pass
    else:
        settings = {}

    # set verbose
    if 'verbose' in settings.keys():
        verbose = settings['verbose']
    else:
        verbose = True
        settings['verbose'] = verbose
    
    # print settings
    if verbose:
        print("Current Settings...")
        for k, v in settings.items():
            print("{} : {}".format(k, v))

    if sc == 0:
        df, misc = LoadTimeSeriesPSP(
            start_time,end_time,
            settings = settings,
            credentials = credentials
            )
    elif sc == 1:
        df, misc = LoadTimeSeriesSOLO(
            start_time,end_time,
            settings = settings,
            credentials = credentials
            )
    elif sc == 2:
        df, misc = LoadTimeSeriesHelios1(
            start_time,end_time,
            settings = settings,
            credentials = credentials
            )
    elif sc == 3:
        df, misc = LoadTimeSeriesHelios2(
            start_time,end_time,
            settings = settings,
            credentials = credentials
            )
    elif sc == 4:
        df, misc = LoadTimeSeriesULYSSES(
            start_time,end_time,
            settings = settings,
            credentials = credentials
            )
    elif sc == 5:
        df, misc = LoadTimeSeriesWind(
            start_time,end_time,
            settings = settings,
            credentials = credentials
            )
    else:
        raise ValueError("sc=%d not supporteda!" %(sc))

    
    # print settings
    if verbose:
        print("Final Settings...")
        for k, v in settings.items():
            print("{} : {}".format(k, v))

    return df, misc


def LoadTimeSeriesHelios1(
    start_time, end_time, 
    settings = {}, credentials = None):
    """ 
    Load Helios 1 Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """
    
    verbose = settings['verbose']

    if verbose:
        print("Loading Helios-A data from CDAWEB...")

    vars = ['R_Helio','ESS_Ang','clong','clat','HGIlong','B_R','B_T','B_N','Vp_R','Vp_T','Vp_N','crot','Np','Vp','Tp','V_Azimuth','V_Elev','B_x','B_y','B_z','stdev_B_x','stdev_B_y','stdev_B_z','N_a','V_a','T_a','Np2','Vp2']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('HELIOS1_40SEC_MAG-PLASMA', vars, time[0], time[1])

    if verbose:
        print("Done.")

    df = pd.DataFrame(
        index = data['Epoch']
    ).join(
        pd.DataFrame(
            index = data['Epoch'],
            data = {
                'Dist_au': data['R_Helio'],
                'lon':data['clong'],
                'lat':data['clat'],
                'Br' : data['B_R'],
                'Bt' : data['B_T'],
                'Bn' : data['B_N'],
                'Vr' : data['Vp_R'],
                'Vt' : data['Vp_T'],
                'Vn' : data['Vp_N'],
                'np' : data['Np'],
                'Tp' : data['Tp'],
                'Vth': 0.128487*np.sqrt(data['Tp']), # vth[km/s] = 0.128487 * √Tp[K]
                'Bx' : data['B_x'],
                'By' : data['B_y'],
                'Bz' : data['B_z']
                }
        )
    ) 

    # set nan (nan = -1e31)
    df[df < -1e30] = np.nan

    misc = {'settings': settings}

    return df, misc


def LoadTimeSeriesHelios2(
    start_time, end_time, 
    settings = {}, credentials = None):
    """ 
    Load Helios 1 Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """
    
    verbose = settings['verbose']

    if verbose:
        print("Loading Helios-B data from CDAWEB...")

    vars = ['R_Helio','ESS_Ang','clong','clat','HGIlong','B_R','B_T','B_N','Vp_R','Vp_T','Vp_N','crot','Np','Vp','Tp','V_Azimuth','V_Elev','B_x','B_y','B_z','stdev_B_x','stdev_B_y','stdev_B_z','N_a','V_a','T_a','Np2','Vp2']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('HELIOS2_40SEC_MAG-PLASMA', vars, time[0], time[1])

    if verbose:
        print("Done.")

    df = pd.DataFrame(
        index = data['Epoch']
    ).join(
        pd.DataFrame(
            index = data['Epoch'],
            data = {
                'Dist_au': data['R_Helio'],
                'lon':data['clong'],
                'lat':data['clat'],
                'Br' : data['B_R'],
                'Bt' : data['B_T'],
                'Bn' : data['B_N'],
                'Vr' : data['Vp_R'],
                'Vt' : data['Vp_T'],
                'Vn' : data['Vp_N'],
                'np' : data['Np'],
                'Tp' : data['Tp'],
                'Vth': 0.128487*np.sqrt(data['Tp']), # vth[km/s] = 0.128487 * √Tp[K]
                'Bx' : data['B_x'],
                'By' : data['B_y'],
                'Bz' : data['B_z']
                }
        )
    ) 

    # set nan (nan = -1e31)
    df[df < -1e30] = np.nan

    misc = {'settings': settings}

    return df, misc


def LoadTimeSeriesSOLO(start_time, end_time, settings = {}, credentials = None):
    """ Load Time Series from SPEDAS with Solar Orbiter """


    # check local directory
    if os.path.exists("./solar_orbiter_data"):
        pass
    else:
        raise ValueError("No local data folder is present!")

    # resolution
    if 'resolution' in settings.keys():
        resolution = settings['resolution']
    else:
        resolution = 5
        settings['resolution'] = resolution

    # mag_rtn_only
    if 'mag_rtn_only' in settings.keys():
        pass
    else:
        settings['mag_rtn_only'] = True

    t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

    swadata = pyspedas.solo.swa(trange=[t0, t1], datatype='pas-grnd-mom')
    data = get_data(swadata[0])

    dfpar = pd.DataFrame(
        # index = time_string.time_datetime(time=data.times, tz=None)
        index = data.times
    )
    temp = get_data('N')
    dfpar = dfpar.join(
        pd.DataFrame(
            # index = time_string.time_datetime(time=np.times, tz=None),
            index = temp.times,
            data = temp.y,
            columns = ['np']
        )
    )
    temp = get_data('T')
    dfpar = dfpar.join(
        pd.DataFrame(
            index = temp.times,
            data = temp.y,
            columns = ['T']
        )
    )
    dfpar['Vth'] = 13.84112218 * np.sqrt(dfpar['T']) # 1eV = 13.84112218 km/s (kB*T = 1/2*mp*Vth^2)

    temp = get_data('V_RTN')
    dfpar = dfpar.join(
        pd.DataFrame(
            index = temp.times,
            data = temp.y,
            columns = ['Vr','Vt','Vn']
        )
    )

    temp = get_data('V_SRF')
    dfpar = dfpar.join(
        pd.DataFrame(
            index = temp.times,
            data = temp.y,
            columns = ['Vx','Vy','Vz']
        )
    )

    temp = get_data('V_SOLO_RTN')
    dfpar = dfpar.join(
        pd.DataFrame(
            index = temp.times,
            data = temp.y,
            columns = ['sc_vel_r','sc_vel_t','sc_vel_n']
        )
    )

    dfpar.index = time_string.time_datetime(time=dfpar.index)
    dfpar.index = dfpar.index.tz_localize(None)
    dfpar.index.name = 'datetime'


    time = [(pd.Timestamp(t0)-pd.Timedelta('3d')).to_pydatetime( ).replace(tzinfo=pytz.UTC), (pd.Timestamp(t1)+pd.Timedelta('3d')).to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('SOLO_HELIO1DAY_POSITION', ['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON'], time[0], time[1])

    dfdis = pd.DataFrame(
        index = data['Epoch'],
        data = data[['RAD_AU','SE_LAT','SE_LON','HG_LAT','HG_LON','HGI_LAT','HGI_LON']]
    ).resample("%ds" %(resolution)).interpolate()
    dfdis.index.name = 'datetime'

    # join dfpar and dfdis
    dfpar = dfpar.resample("%ds" %(resolution)).mean().join(dfdis)
    dfpar['Dist_au'] = dfpar['RAD_AU']

    names = pyspedas.solo.mag(trange=[t0,t1], datatype='rtn-normal', level='l2', time_clip=True)
    data = get_data(names[0])
    dfmag1 = pd.DataFrame(
        index = data[0],
        data = data[1]
    )
    dfmag1.columns = ['Br','Bt','Bn']

    # use only rtn mag
    if settings['mag_rtn_only']:
            dfmag = dfmag1
    else:
        names = pyspedas.solo.mag(trange=[t0,t1], datatype='srf-normal', level='l2', time_clip=True)
        data = get_data(names[0])
        dfmag2 = pd.DataFrame(
            index = data[0],
            data = data[1]
        )
        dfmag2.columns = ['Bx','By','Bz']

        dfmag = dfmag1.join(dfmag2)

    
    dfmag.index = time_string.time_datetime(time=dfmag.index)
    dfmag.index = dfmag.index.tz_localize(None)

    dfts = dfmag.resample('%ds' %(resolution)).mean().join(
        dfpar.resample('%ds' %(resolution)).mean())

    dfts['Dist_au'] = dfts['RAD_AU']

    misc = {'settings': settings}

    return dfts, misc


def LoadTimeSeriesPSP(
    start_time, end_time, 
    settings = {}, credentials = None
    ):
    """" 
    Load Time Serie From SPEDAS, PSP 
    settings if not None, should be a dictionary, necessary settings:

    spc_only: boolean
    span_only: boolean
    mix_spc_span: dict:
        {'priority': 'spc' or 'span'}
    keep_spc_and_span: boolean

    keep_keys: ['np','Vth','Vx','Vy','Vz','Vr','Vt','Vn']
    
    Note that the priority is using SPAN
    """

    # default settings
    default_settings = {
        'particle_mode': 'keep_all',
        'resolution': 5,
        'use_hampel': False,
        'interpolate_qtn': True,
        'interpolate_rolling': True,
        'verbose': True,
        'must_have_qtn': False,
        'rtn_only': True
    }

    for k in default_settings.keys():
        if k not in settings.keys():
            settings[k] = default_settings[k]

    # check local directory
    if os.path.exists("./psp_data"):
        pass
    else:
        raise ValueError("No local data folder ./psp_data is present!")

    t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

    # Quasi-Thermal-Noise
    try:
        # Quasi-Thermal Noise for electron density
        try:    
            qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_v1v2', level='l3', 
                        varnames = [
                            'electron_density',
                            'electron_core_temperature'
                        ], 
                        time_clip=True)
            temp = get_data(qtndata[0])
        except:
            print("No QTN data is presented in the public repository!")
            print("Trying unpublished data... please provide credentials...")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']

            qtndata = pyspedas.psp.fields(trange=[t0, t1], datatype='sqtn_rfs_V1V2', level='l3', 
            varnames = [
                'electron_density',
                'electron_core_temperature'
            ], 
            time_clip=True, username=username, password=password)
            temp = get_data(qtndata[0])


        dfqtn = pd.DataFrame(
            index = temp.times,
            data = temp.y,
            columns = ['ne_qtn']
        )

        dfqtn['np_qtn'] = dfqtn['ne_qtn']/1.08 # 4% of alpha particle
        dfqtn.index = time_string.time_datetime(time=dfqtn.index)
        dfqtn.index = dfqtn.index.tz_localize(None)
        dfqtn.index.name = 'datetime'
    except:
        print("No QTN Data!")
        dfqtn = None
        #if settings['must_have_qtn']:
            #raise ValueError("Must have QTN! No QTN!")

    # Magnetic field
    try:
        try:
            names = pyspedas.psp.fields(trange=[t0,t1], datatype='mag_rtn_4_per_cycle', level='l2', time_clip=True)
            data = get_data(names[0])
            dfmag1 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag1.columns = ['Br','Bt','Bn']

            if settings['rtn_only']:
                dfmag = dfmag1
            else:
                names = pyspedas.psp.fields(trange=[t0,t1], datatype='mag_sc_4_per_cycle', level='l2', time_clip=True)
                data = get_data(names[0])
                dfmag2 = pd.DataFrame(
                    index = data[0],
                    data = data[1]
                )
                dfmag2.columns = ['Bx','By','Bz']

                dfmag = dfmag1.join(dfmag2)


        except:
            print("No MAG data is presented in the public repository!")
            print("Trying unpublished data... please provide credentials...")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']

            names = pyspedas.psp.fields(trange=[t0,t1], 
                datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True,
                username=username, password=password
            )
            data = get_data(names[0])
            dfmag1 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag1.columns = ['Br','Bt','Bn']

            if settings['rtn_only']:
                dfmag = dfmag1

            else:
                names = pyspedas.psp.fields(trange=[t0,t1], 
                    datatype='mag_SC_4_Sa_per_Cyc', level='l2', time_clip=True,
                    username=username, password=password
                )
                data = get_data(names[0])
                dfmag2 = pd.DataFrame(
                    index = data[0],
                    data = data[1]
                )
                dfmag2.columns = ['Bx','By','Bz']

                dfmag = dfmag1.join(dfmag2)


        dfmag.index = time_string.time_datetime(time=dfmag.index)
        dfmag.index = dfmag.index.tz_localize(None)
    except:
        print("No MAG Data!")
        dfmag = None

    # SPC
    try:
        try:
            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='l3', 
                                    varnames = [
                                        'np_moment',
                                        'wp_moment',
                                        'vp_moment_RTN',
                                        'vp_moment_SC',
                                        'sc_pos_HCI',
                                        'sc_vel_HCI',
                                        'carr_latitude',
                                        'carr_longitude',
                                        'na_fit'
                                    ], 
                                    time_clip=True)

            data = get_data(spcdata[0])
        except:
            print("No SPC data is presented in the public repository!")
            print("Trying unpublished data... please provide credentials...")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            username = credentials['psp']['sweap']['username']
            password = credentials['psp']['sweap']['password']

            spcdata = pyspedas.psp.spc(trange=[t0, t1], datatype='l3i', level='L3', 
                                    varnames = [
                                        'np_moment',
                                        'wp_moment',
                                        'vp_moment_RTN',
                                        'vp_moment_SC',
                                        'sc_pos_HCI',
                                        'sc_vel_HCI',
                                        'carr_latitude',
                                        'carr_longitude',
                                        'na_fit'
                                    ], 
                                    time_clip=True, username=username, password=password)

            data = get_data(spcdata[0])

        dfspc = pd.DataFrame(
            # index = time_string.time_datetime(time=data.times, tz=None)
            index = data.times
        )

        temp = get_data(spcdata[0])
        dfspc = dfspc.join(
            pd.DataFrame(
                # index = time_string.time_datetime(time=np.times, tz=None),
                index = temp.times,
                data = temp.y,
                columns = ['np']
            )
        )

        temp = get_data(spcdata[1])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vth']
            )
        )

        temp = get_data(spcdata[2])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vr','Vt','Vn']
            )
        )

        temp = get_data(spcdata[3])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vx','Vy','Vz']
            )
        )

        temp = get_data(spcdata[4])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['sc_x','sc_y','sc_z']
            )
        )

        temp = get_data(spcdata[5])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['sc_vel_x','sc_vel_y','sc_vel_z']
            )
        )

        temp = get_data(spcdata[6])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['carr_lat']
            )
        )

        temp = get_data(spcdata[7])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['carr_lon']
            )
        )

        temp = get_data(spcdata[8])
        dfspc = dfspc.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['na']
            )
        )        

        # calculate Dist_au
        dfspc['Dist_au'] = (dfspc[['sc_x','sc_y','sc_z']]**2).sum(axis=1).apply(np.sqrt)/au_to_km

        dfspc.index = time_string.time_datetime(time=dfspc.index)
        dfspc.index = dfspc.index.tz_localize(None)
        dfspc.index.name = 'datetime'
    except:
        print("No SPC Data!")
        dfspc = None

    # SPAN
    try:
        try:
            spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00_l3_mom', level='l3', 
                    varnames = [
                        'DENS',
                        'VEL_SC',
                        'VEL_RTN_SUN',
                        'TEMP',
                        'SUN_DIST',
                        'SC_VEL_RTN_SUN'
                    ], 
                    time_clip=True)
            temp = get_data(spandata[0])

            spandata_a = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf0a_l3_mom', level='l3', 
                    varnames = [
                        'DENS'
                    ], 
                    time_clip=True)
            temp_a = get_data(spandata_a[0])
        except:
            print("No SPAN data is presented in the public repository!")
            print("Trying unpublished data... please provide credentials...")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            username = credentials['psp']['sweap']['username']
            password = credentials['psp']['sweap']['password']

            spandata = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf00', level='L3', 
                    varnames = [
                        'DENS',
                        'VEL_SC',
                        'VEL_RTN_SUN',
                        'TEMP',
                        'SUN_DIST',
                        'SC_VEL_RTN_SUN'
                    ], 
                    time_clip=True, username=username, password=password)
            temp = get_data(spandata[0])

            spandata_a = pyspedas.psp.spi(trange=[t0, t1], datatype='spi_sf0a', level='L3', 
                    varnames = [
                        'DENS'
                    ], 
                    time_clip=True, username=username, password=password)
            temp_a = get_data(spandata_a[0])


        dfspan = pd.DataFrame(
            index = temp.times,
            data = temp.y,
            columns = ['np']
        )

        temp = get_data(spandata[1])
        dfspan = dfspan.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vx', 'Vy', 'Vz']
            )
        )

        temp = get_data(spandata[2])
        dfspan = dfspan.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Vr', 'Vt', 'Vn']
            )
        )

        temp = get_data(spandata[3])
        dfspan = dfspan.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['TEMP']
            )
        )

        temp = get_data(spandata[4])
        dfspan = dfspan.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['Dist_au']
            )
        )
        dfspan['Dist_au'] = dfspan['Dist_au']/au_to_km

        temp = get_data(spandata[5])
        dfspan = dfspan.join(
            pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['sc_vel_r','sc_vel_t','sc_vel_n']
            )
        )

        # alpha particle
        temp = get_data(spandata_a[0])
        dfspan_a = pd.DataFrame(
                index = temp.times,
                data = temp.y,
                columns = ['na']
            )

        # calculate Vth from TEMP
        # k T (eV) = 1/2 mp Vth^2 => Vth = 13.84112218*sqrt(TEMP)
        dfspan['Vth'] = 13.84112218 * np.sqrt(dfspan['TEMP'])

        # for span the thermal speed is defined as the trace, hence have a sqrt(3) different from spc
        dfspan['Vth'] = dfspan['Vth']/np.sqrt(3)

        dfspan.index = time_string.time_datetime(time=dfspan.index)
        dfspan.index = dfspan.index.tz_localize(None)
        dfspan.index.name = 'datetime'

        dfspan_a.index = time_string.time_datetime(time=dfspan_a.index)
        dfspan_a.index = dfspan_a.index.tz_localize(None)
        dfspan_a.index.name = 'datetime'
    except:
        print("No SPAN!")
        dfspan = None
        dfspan_a = None

    # merge particle data

    if 'particle_mode' in settings.keys():
        parmode = settings['particle_mode']
    else:
        parmode = 'empirical'

    # create empty dfpar with index
    freq = "%ds" %(settings['resolution'])
    index = pd.date_range(
        start = start_time, 
        end = end_time, 
        freq = freq
    )
    dfpar = pd.DataFrame(
        index = index
    )

    print("Parmode: %s" %(parmode))
    # fill dfpar with values
    if parmode == 'spc_only':
        dfpar = dfpar.join(dfspc.resample(freq).mean().interpolate())
    elif parmode == 'span_only':
        dfpar = dfpar.join(dfspan.resample(freq).mean().interpolate()).join(dfspan_a.resample(freq).mean().interpolate())
    elif parmode == 'empirical':
        # empirical use of data
        # encounter date: https://sppgway.jhuapl.edu/index.php/encounters
        # before encounter 9 (Perihelion: 2021-08-09/19:11) use SPC for solar wind speed
        # at and after encounter 8, mix SPC and SPAN for solar wind speed
        # prioritize QTN for density, and fill with SPC, and with SPAN
        
        # proton density with QTN
        if dfqtn is None:
            dfpar['np'] = np.nan
        else:
            dfpar = dfpar.join(dfqtn.resample(freq).mean())

            # the density is not very fluctuating, hence interpolate
            try:
                if settings['interpolate_qtn']:
                    print("QTN is interpolated!")
                    dfpar['ne_qtn'] = dfpar['ne_qtn'].interpolate()
            except:
                pass

            dfpar['np'] = dfpar['ne_qtn']/1.08

        keep_keys = ['Vx','Vy','Vz','Vr','Vt','Vn','Vth','Dist_au']

        # if no SPC or no SPAN data:
        if dfspc is None:
            dfspc = pd.DataFrame(index = dfpar.index)
            dfspc[keep_keys] = np.nan

        if dfspan is None:
            dfspan = pd.DataFrame(index = dfpar.index)
            dfspan[keep_keys] = np.nan


        # Perihelion cut
        ind1 = dfpar.index < pd.Timestamp('2021-07-01')
        ind2 = dfpar.index >= pd.Timestamp('2021-07-01')

        ind11 = dfspc.index < pd.Timestamp('2021-07-01')
        ind12 = dfspc.index >= pd.Timestamp('2021-07-01')

        ind21 = dfspan.index < pd.Timestamp('2021-07-01')
        ind22 = dfspan.index >= pd.Timestamp('2021-07-01')

        # before encounter 9 use spc
        dfpar1 = dfspc.loc[ind11,keep_keys].resample(freq).mean()

        # use span after 2021-07-15
        dfpar2 = dfspan.loc[ind22,keep_keys].resample(freq).mean()
        # dfpar2 = dfpar21.copy()
        # ind21t = dfpar21['Vr'].apply(np.isnan)
        # ind22t = dfpar22['Vr'].apply(np.isnan)
        # dfpar2.loc[~ind21t,['MOMENT_FLAG','DENSITY_FLAG']] = 1 # SPC
        # dfpar2.loc[~ind22t,['MOMENT_FLAG','DENSITY_FLAG']] = 2 # SPAN
        # fill SPAN empty with SPC
        # dfpar2.loc[ind21t] = dfpar22.loc[ind21t]

        # # combine qtn
        # dfpar_qtn = pd.DataFrame(
        #     index = index
        # ).join(dfqtn.resample(freq).mean())
        
        # merge qtn and particle data
        dftemp = pd.concat([dfpar1,dfpar2])
        dfpar = dfpar.join(dftemp[keep_keys])

        # join carr longitude
        try:
            dfpar = dfpar.join(dfspc[['carr_lon','carr_lat']].resample(freq).mean())
        except:
            print("No carr lon and lat information from SPC!")
            dfpar[['carr_lon','carr_lat']] = np.nan

    elif parmode == 'keep_all':
        # keep all particle information, default np is qtn, moments are empirical
        # default all are interpolated!

        # add qtn
        try:
            dfpar['np_qtn'] = dfqtn['np_qtn'].resample(freq).mean().interpolate()
        except:
            warnings.warn("No QTN data!")
            dfpar['np_qtn'] = np.nan

        keep_keys = ['np','Vx','Vy','Vz','Vr','Vt','Vn','Vth','Dist_au','na']
        # add SPC
        for k in keep_keys:
            try:
                dfpar[k+"_spc"] = dfspc[k].resample(freq).mean().interpolate()
            except:
                warnings.warn("key: %s not present in dfspc!" %(k))
                dfpar[k+"_spc"] = np.nan

        # add SPAN
        for k in keep_keys:
            try:
                if k == 'na':
                    dfpar[k+"_span"] = dfspan_a[k].resample(freq).mean().interpolate()
                else:
                    dfpar[k+"_span"] = dfspan[k].resample(freq).mean().interpolate()
            except:
                warnings.warn("key: %s not present in dfspan!" %(k))
                dfpar[k+"_span"] = np.nan

        # join carr longitude
        try:
            dfpar = dfpar.join(dfspc[['carr_lon','carr_lat']].resample(freq).mean())
        except:
            print("No carr lon and lat information from SPC!")
            dfpar[['carr_lon','carr_lat']] = np.nan

        # set default
        # default density: QTN
        if len(dfpar['np_qtn'].dropna())>0:
            dfpar['np'] = dfpar['np_qtn']
        

        # before encounter 9, default set to SPC
        if dfpar.index[-1] < pd.Timestamp('2021-07-01'):
            for k in keep_keys:
                dfpar[k] = dfpar[k+'_spc']
        # at and after encounter 9, default set to SPAN
        else:
            for k in keep_keys:
                dfpar[k] = dfpar[k+'_span']

        # raise ValueError("particle mode: %s under construction!" %(parmode))
    
    else:
        raise ValueError("particle mode: %s not supported!" %(parmode))


    # if settings['use_hampel'] == True:
    #     for k in dfpar.columns:
    #         ns, _ = hampel_filter_forloop_numba(dfpar[k].values, 100)
    #         dfpar[k] = ns




    # create dfts
    dfts = dfmag.resample(freq).mean().join(
        dfpar.resample(freq).mean()
    )

    # load l1 ephemeris
    if credentials is not None:
        try:
            print("Loading ephemeris...")
            fields_vars = pyspedas.psp.fields(
                trange=[t0, t1], 
                datatype='ephem_spp_rtn', level='l1', 
                time_clip=True, 
                username=credentials['psp']['fields']['username'], 
                password=credentials['psp']['fields']['password'])

            data = get_data('position')
            dfe = pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['sc_pos_r','sc_pos_t','sc_pos_n']
            )

            data = get_data('velocity')
            dfe = dfe.join(
                pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['sc_vel_r','sc_vel_t','sc_vel_n']
                )
            )

            dfe.index = time_string.time_datetime(time=dfe.index)
            dfe.index = dfe.index.tz_localize(None)

            dfe = dfe.resample(freq).interpolate()

            try:
                # when particla_mode = span_only, there are overlapping columns
                dfts.drop(columns = ['sc_vel_r', 'sc_vel_t', 'sc_vel_n'], inplace = True)
            except:
                pass

            dfts = dfts.join(dfe)
            dfts['Dist_au'] = (dfe[['sc_pos_r','sc_pos_t','sc_pos_n']]**2).sum(axis=1).apply(np.sqrt)/au_to_km
        except:
            raise ValueError("Ephemeris could not be loaded!")

    
    if 'local_carr_lon' in settings.keys():
        print("Loading local carr lon from %s" %(Path(settings['local_carr_lon']).absolute()))
        df_carr = pd.read_pickle(settings['local_carr_lon'])
        ind = (df_carr.index > start_time - pd.Timedelta('1d')) & (df_carr.index < end_time + pd.Timedelta('1d'))
        df_carr = df_carr[ind]
        df_carr = df_carr.resample(freq).interpolate()

        dfts.drop(columns = ['carr_lon','carr_lat'], inplace=True)
        dfts = dfts.join(df_carr)
        

    else:
        df_carr = None


    misc = {
        'dfqtn': dfqtn,
        'dfspc': dfspc,
        'dfspan': dfspan,
        'dfspan_a': dfspan_a,
        'parmode': parmode,
        'settings': settings,
        'dfpar': dfpar,
        'dfmag': dfmag,
        'dfe': dfe,
        'df_carr': df_carr
    }

    return dfts, misc


def LoadTimeSeriesULYSSES(
    start_time, end_time, settings = {}, credentials = None
    ):
    """
    Ulysses
    """ 
    if os.path.exists("./ulysses_data"):
        pass
    else:
        raise ValueError("No local data folder ./ulysses_data present!")

    # resolution
    if 'resolution' in settings.keys():
        resolution = settings['resolution']
    else:
        resolution = 300
        settings['resolution'] = resolution

    t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

    # vhm_vars = pyspedas.ulysses.vhm(trange=[t0,t1], datatype='1sec')
    # data = get_data('B_RTN')

    # dfmag = pd.DataFrame(
    #     index = data[0],
    #     data = data[1]
    # )
    # dfmag.columns = ['Br','Bt','Bn']

    # dfmag.index = time_string.time_datetime(time=dfmag.index)
    # dfmag.index = dfmag.index.tz_localize(None)  

    vars = ['B_RTN','B_MAG']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)-pd.Timedelta('10H'), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)+pd.Timedelta('10H')]
    status, data = cdas.get_data('UY_1MIN_VHM', vars, time[0], time[1])

    dfmag = pd.DataFrame(
        index = data['Epoch'],
        data = {
            'Br': data['B_RTN'][:,0],
            'Bt': data['B_RTN'][:,1],
            'Bn': data['B_RTN'][:,2]
        }
    )

    swoops_vars = pyspedas.ulysses.swoops(trange=[t0,t1])

    data = get_data('Density')

    dfpar = pd.DataFrame(
        index = data.times
    )

    dfpar = dfpar.join(
        pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['np','na']
        )
    )

    data = get_data('Temperature')
    dfpar = dfpar.join(
        pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['Tlow','Thigh']
        )
    )

    dfpar['Tp'] = (dfpar['Tlow']+dfpar['Thigh'])/2
    dfpar['Vth'] = 0.128487*np.sqrt(dfpar['Tp']) # vth[km/s] = 0.128487 * √Tp[K]

    data = get_data("Velocity")
    dfpar = dfpar.join(
        pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['Vr','Vt','Vn']
        )
    )

    dfpar.index = time_string.time_datetime(time=dfpar.index)
    dfpar.index = dfpar.index.tz_localize(None)    

    # load ephemeris
    vars = ['RAD_AU','SE_LAT','SE_LON']
    time = [(pd.Timestamp(t0)-pd.Timedelta('3d')).to_pydatetime( ).replace(tzinfo=pytz.UTC), (pd.Timestamp(t1)+pd.Timedelta('3d')).to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('ULYSSES_HELIO1DAY_POSITION', vars, time[0], time[1])

    dfdis = pd.DataFrame(
        index = data['Epoch'],
        data = data[['RAD_AU','SE_LAT','SE_LON']]
    ).resample("%ds" %(resolution)).interpolate()

    dfdis['Dist_au'] = dfdis['RAD_AU']
    dfdis['lon'] = dfdis['SE_LON']
    dfdis['lat'] = dfdis['SE_LAT']

    # resample and join
    dfts = ((dfmag.resample('%ds' %(resolution)).mean())[['Br','Bt','Bn']]).join(
        (dfpar.resample('%ds' %(resolution)).mean())[['np','Tp','Vr','Vt','Vn','Vth']]
    ).join(
        (dfdis.resample('%ds' %(resolution)).interpolate())[['Dist_au','lon','lat']]
    )

    misc = {'settings': settings}

    # set nan (nan = -1e31)
    dfts[dfts < -1e30] = np.nan

    return dfts, misc


def LoadTimeSeriesWind(start_time, end_time, settings = {}, credentials = None 
):
    """ 
    Load Wind Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """
    
    verbose = settings['verbose']
    
    # resolution
    if 'resolution' in settings.keys():
        resolution = settings['resolution']
    else:
        resolution = 300
        settings['resolution'] = resolution


    if verbose:
        print("Loading low-res WIND data from CDAWEB...")

    vars = ['B3GSE','B3F1']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('WI_H0_MFI', vars, time[0], time[1])

    if verbose:
        print("Done.")

    dfmag = pd.DataFrame(
        index = data['Epoch3'],
        data = {
            'Br': data['B3GSE'][:,0],
            'Bt': data['B3GSE'][:,1],
            'Bn': data['B3GSE'][:,2],
            'Btot': data['B3F1']
        }
    )

    # fill val = -1e31
    dfmag[dfmag['Br'] < -1e30] = np.nan


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
    
    
    dfdis = pd.DataFrame(
        index = data['Epoch'][::2],
        data = {
            'Dist_au': np.ones(len(data['P_VELS'][:,0][::2])), 
            'lon'    : np.ones(len(data['P_VELS'][:,0][::2])), 
            'lat'    : np.ones(len(data['P_VELS'][:,0][::2])), 
            'RAD_AU' : np.ones(len(data['P_VELS'][:,0][::2]))
        }
    
    
    )
    
    # resample and join
    dfts = ((dfmag.resample('%ds' %(resolution)).mean())[['Br','Bt','Bn']]).join(
        (dfpar.resample('%ds' %(resolution)).mean())[['np','Tp','Vr','Vt','Vn','Vth']]
    ).join(
        (dfdis.resample('%ds' %(resolution)).interpolate())[['Dist_au','lon','lat']]
    )


    misc = {'settings': settings}

    # set nan (nan = -1e31)
    dfts[dfts < -1e30] = np.nan

    return dfts, misc


# load High-res MAG data
def LoadHighResMagWrapper(
    sc, start_time, end_time, verbose = True, credentials = None, resolution = None):
    """
    A Wrapper to load high resolution magnetic field data
    Works for Helios 1/2 (sc = 2,3) and Ulysses (sc = 4)
    Note that CDAWEB sometimes does not return the correct interval

    return:
        dfmag_raw: raw high res mag dataframe
        dfmag_resampled: resampled mag dataframe
        infos: resample information
    """

    if verbose:
        print("Load High Res Mag data for sc = %d" %(sc))
        print("Required tstart = %s, tend = %s" %(start_time, end_time))

    if sc == 0:
        dfmag, dfmag1, infos = LoadHighResMagPSP(
            start_time-pd.Timedelta('10H'), end_time+pd.Timedelta('10H'), verbose, 
            credentials = credentials, resolution = resolution)
    elif sc == 1:
        dfmag, dfmag1, infos = LoadHighResMagSOLO(
            start_time-pd.Timedelta('10H'), end_time+pd.Timedelta('10H'), verbose)
    elif sc == 2:
        dfmag, dfmag1, infos = LoadHighResMagHelios1(
            start_time-pd.Timedelta('10H'), end_time+pd.Timedelta('10H'), verbose)
    elif sc == 3:
        dfmag, dfmag1, infos = LoadHighResMagHelios2(
            start_time-pd.Timedelta('10H'), end_time+pd.Timedelta('10H'), verbose)
    elif sc == 4:
        dfmag, dfmag1, infos = LoadHighResMagUlysses(
            start_time-pd.Timedelta('10H'), end_time+pd.Timedelta('10H'), verbose)
    elif sc == 5:
        print('Loading Wind')
        dfmag, dfmag1, infos = LoadHighResMagWind(
            start_time-pd.Timedelta('10H'), end_time+pd.Timedelta('10H'), verbose)
  
    else:
        raise ValueError("sc = %d not supported!" %(sc))

    # filter the correct time range
    ind = (dfmag.index  >= start_time) & (dfmag.index < end_time)
    dfmag = dfmag[ind]

    ind = (dfmag1.index  >= start_time) & (dfmag1.index < end_time)
    dfmag1 = dfmag1[ind]

    if verbose:
        print("Final tstart = %s, tend = %s" %(dfmag1.index[0], dfmag1.index[-1]))

    dfmag_raw = dfmag
    dfmag_resampled = dfmag1

    return dfmag_raw, dfmag_resampled, infos


def LoadHighResMagSOLO(
    start_time, end_time, verbose = True
    ):

    vars = ['B_SRF']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('SOLO_L2_MAG-SRF-NORMAL', vars, time[0], time[1])

    dfmag = pd.DataFrame(
        index = data['EPOCH'],
        data = {
            'Bx': data['B_SRF'][:,0],
            'By': data['B_SRF'][:,1],
            'Bz': data['B_SRF'][:,2]
        }
    )
    dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

    dfmag1 = dfmag.resample('1s').mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(
            data['EPOCH'][0], 
            data['EPOCH'][-1]))

    infos = {
        'resolution': 1
    }

    return dfmag, dfmag1, infos

def LoadHighResMagPSP(
    start_time, end_time, verbose = True, credentials = None, resolution = None, load_4_per_cyc = True, use_spedas = True,
    ):
    """
    resolution in ms!
    """
    if load_4_per_cyc:
        try:
            if use_spedas:
                print("Loading mag_rtn_4_per_cycle from SPEDAS")
                t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
                t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

                names = pyspedas.psp.fields(trange=[t0,t1], 
                    datatype='mag_rtn_4_per_cycle', level='l2', time_clip=True
                )
                data = get_data(names[0])
                dfmag2 = pd.DataFrame(
                    index = data[0],
                    data = data[1]
                )
                dfmag2.columns = ['Bx','By','Bz']

                dfmag = dfmag2
                dfmag.index = time_string.time_datetime(time=dfmag.index)
                dfmag.index = dfmag.index.tz_localize(None)

                dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

            else:
                print("Loading psp_fld_l2_mag_RTN_4_Sa_per_Cyc from SPDF")
                vars = ['psp_fld_l2_mag_RTN_4_Sa_per_Cyc']
                time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
                status, data = cdas.get_data('PSP_FLD_L2_MAG_RTN_4_SA_PER_CYC', vars, time[0], time[1])

                dfmag = pd.DataFrame(
                    index = data['epoch_mag_RTN_4_Sa_per_Cyc'],
                    data = {
                        'Bx': data['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][:,0],
                        'By': data['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][:,1],
                        'Bz': data['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][:,2]
                    }
                )
                dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

                if verbose:
                    print("Input tstart = %s, tend = %s" %(time[0], time[1]))
                    print("Returned tstart = %s, tend = %s" %(
                        data['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][0], 
                        data['psp_fld_l2_mag_RTN_4_Sa_per_Cyc'][-1]))

        except:
            print("Loading unpublished PSP high-res mag data....")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
            t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']

            names = pyspedas.psp.fields(trange=[t0,t1], 
                datatype='mag_RTN_4_Sa_per_Cyc', level='l2', time_clip=True,
                username=username, password=password
            )
            data = get_data(names[0])
            dfmag2 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag2.columns = ['Bx','By','Bz']

            dfmag = dfmag2
            dfmag.index = time_string.time_datetime(time=dfmag.index)
            dfmag.index = dfmag.index.tz_localize(None)

            dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

    else:
        try:
            if use_spedas:
                print("Loading mag_rtn from SPEDAS")
                t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
                t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

                names = pyspedas.psp.fields(trange=[t0,t1], 
                    datatype='mag_rtn', level='l2', time_clip=True
                )
                data = get_data(names[0])
                dfmag2 = pd.DataFrame(
                    index = data[0],
                    data = data[1]
                )
                dfmag2.columns = ['Bx','By','Bz']

                dfmag = dfmag2
                dfmag.index = time_string.time_datetime(time=dfmag.index)
                dfmag.index = dfmag.index.tz_localize(None)

                dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

            else:
                print("Loading psp_fld_l2_mag_RTN from SPDF")
                vars = ['psp_fld_l2_mag_RTN']
                time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
                status, data = cdas.get_data('PSP_FLD_L2_MAG_RTN', vars, time[0], time[1])

                dfmag = pd.DataFrame(
                    index = data['epoch_mag_RTN'],
                    data = {
                        'Bx': data['psp_fld_l2_mag_RTN'][:,0],
                        'By': data['psp_fld_l2_mag_RTN'][:,1],
                        'Bz': data['psp_fld_l2_mag_RTN'][:,2]
                    }
                )
                dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

                if verbose:
                    print("Input tstart = %s, tend = %s" %(time[0], time[1]))
                    print("Returned tstart = %s, tend = %s" %(
                        data['psp_fld_l2_mag_RTN'][0], 
                        data['psp_fld_l2_mag_RTN'][-1]))

        except:
            print("Loading unpublished PSP high-res mag data....")
            if credentials is None:
                raise ValueError("No credentials are provided!")

            t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
            t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

            username = credentials['psp']['fields']['username']
            password = credentials['psp']['fields']['password']

            names = pyspedas.psp.fields(trange=[t0,t1], 
                datatype='mag_RTN', level='l2', time_clip=True,
                username=username, password=password
            )
            data = get_data(names[0])
            dfmag2 = pd.DataFrame(
                index = data[0],
                data = data[1]
            )
            dfmag2.columns = ['Bx','By','Bz']

            dfmag = dfmag2
            dfmag.index = time_string.time_datetime(time=dfmag.index)
            dfmag.index = dfmag.index.tz_localize(None)

            dfmag['Btot'] = np.sqrt(dfmag['Bx']**2+dfmag['By']**2+dfmag['Bz']**2)

    if resolution is None:
        dfmag1 = dfmag.resample('100ms').mean()
        resolution = 0.1
        print("PSP Resolution: %.2f, fraction missing %.2f" %(resolution, np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1)*100))
        
        if np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1) > 0.1:
            dfmag1 = dfmag.resample('500ms').mean()
            resolution = 0.5
            print("PSP Resolution: %.2f, fraction missing %.2f" %(resolution, np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1)*100))
        
        if np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1) > 0.1:
            dfmag1 = dfmag.resample('1000ms').mean()
            resolution = 1.0
            print("PSP Resolution: %.2f, fraction missing %.2f" %(resolution, np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1)*100))
        
        print("Final PSP Resolution: %.2f, fraction missing %.2f" %(resolution, np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1)*100))
        infos = {
            'resolution': resolution,
            'fraction_missing': np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1)
        }
    else:
        dfmag1 = dfmag.resample("%dms" %(resolution)).mean()
        infos = {
            'resolution': resolution/1000
        }
        print("Final PSP Resolution: %.2f, fraction missing %.2f" %(resolution, np.sum(np.isnan(dfmag1['Bx']))/len(dfmag1)*100))



    return dfmag, dfmag1, infos

def LoadHighResMagHelios1(
    start_time, end_time, verbose = True
    ):
    """
    Load High Res Helios-1 Mag Data:
    CDAWEB Key: HEL1_6SEC_NESSMAG
    Input: 
    start_time/end_time | pd.Timestamp
    Output: 
    DataFrame with Bx/By/Bz
    """

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

    dfmag1 = dfmag.resample('%ds' %(6)).mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    infos = {
        'resolution': 6
    }

    return dfmag, dfmag1, infos


def LoadHighResMagHelios2(
    start_time, end_time, verbose = True
    ):
    """
    Load High Res Helios-1 Mag Data:
    CDAWEB Key: HEL1_6SEC_NESSMAG
    Input: 
    start_time/end_time | pd.Timestamp
    Output: 
    DataFrame with Bx/By/Bz
    """

    vars = ['BXSSE','BYSSE','BZSSE','B']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('HEL2_6SEC_NESSMAG', vars, time[0], time[1])

    dfmag = pd.DataFrame(
        index = data['Epoch'],
        data = {
            'Bx': data['BXSSE'],
            'By': data['BYSSE'],
            'Bz': data['BZSSE'],
            'Btot': data['B']
        }
    )

    dfmag1 = dfmag.resample('%ds' %(6)).mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    infos = {
        'resolution': 6
    }

    return dfmag, dfmag1, infos


def LoadHighResMagUlysses(
    start_time, end_time, verbose = True
    ):
    """
    Load High Res Ulysses Mag Data:
    CDAWEB Key: UY_1SEC_VHM
    Output: 
    DataFrame with Bx/By/Bz
    (Note: The actual return is in RTN, but deemed as SC frame nevertheless)
    """

    vars = ['B_RTN','B_MAG']
    time = [start_time.to_pydatetime( ).replace(tzinfo=pytz.UTC), end_time.to_pydatetime( ).replace(tzinfo=pytz.UTC)]
    status, data = cdas.get_data('UY_1SEC_VHM', vars, time[0], time[1])

    dfmag = pd.DataFrame(
        index = data['Epoch'],
        data = {
            'Bx': data['B_RTN'][:,0],
            'By': data['B_RTN'][:,1],
            'Bz': data['B_RTN'][:,2],
            'Btot': data['B_MAG']
        }
    )

    dfmag1 = dfmag.resample('1s').mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    infos = {
        'resolution': 1
    }

    return dfmag, dfmag1, infos


def LoadHighResMagWind(start_time, end_time, verbose = True
):
    """ 
    Load Wind Plasma Data 
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    """

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


    dfmag1 = dfmag.resample('1s').mean()

    if verbose:
        print("Input tstart = %s, tend = %s" %(time[0], time[1]))
        print("Returned tstart = %s, tend = %s" %(data['Epoch'][0], data['Epoch'][-1]))

    infos = {
        'resolution': 1
    }

    return dfmag, dfmag1, infos

def LoadSCAMFromSPEDAS_PSP(start_time, end_time, credentials = None):
    """ 
    load scam data with pyspedas and return a dataframe
    Input:
        start_time, end_time                pd.Timestamp
        (optional) credentials              dictionary, {'username':..., 'password':...}
    Output:
        return None if no data is present, otherwise a dataframe containing all the scam data
    """

    # check pyspedas
    if os.path.exists(Path(".").absolute().parent.joinpath("pyspedas")):
        print("Using pyspedas at %s" %(str(Path(".").absolute().parent.joinpath("pyspedas"))))
    else:
        raise ValueError("Please clone pyspedas to %s" %(str(Path(".").absolute().parent.joinpath("pyspedas"))))

    t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
    t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")

    if credentials is None:
        scam_vars = pyspedas.psp.fields(
            trange=[t0, t1], datatype='merged_scam_wf', level='l3', time_clip=True, downloadonly = False
        )

        if scam_vars == []:
            return None

        data = get_data(scam_vars[0])
        dfscam = pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['Bu','Bv','Bw']
        )

        data = get_data(scam_vars[1])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Bx','By','Bz']
            )
        )

        data = get_data(scam_vars[2])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Br','Bt','Bn']
            )
        )

        dfscam.index = time_string.time_datetime(time=dfscam.index)
        dfscam.index = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'

        return dfscam

    else:
        # have credentials

        try:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf', level='l3', time_clip=True, downloadonly = False,
                username = credentials['username'], password = credentials['password']
            )
        except:
            raise ValueError('Wrong Username or Password!')

        if scam_vars == []:
            return None

        data = get_data(scam_vars[0])
        dfscam = pd.DataFrame(
            index = data.times,
            data = data.y,
            columns = ['Bu','Bv','Bw']
        )

        data = get_data(scam_vars[1])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Bx','By','Bz']
            )
        )

        data = get_data(scam_vars[2])
        dfscam = dfscam.join(
            pd.DataFrame(
                index = data.times,
                data = data.y,
                columns = ['Br','Bt','Bn']
            )
        )

        dfscam.index = time_string.time_datetime(time=dfscam.index)
        dfscam.index = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'

        return dfscam

