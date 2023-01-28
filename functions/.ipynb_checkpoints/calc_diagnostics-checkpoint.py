import numpy as np
import pandas as pd
import sys
import gc
import time
from numba import jit, njit, prange, objmode 
import os
from pathlib import Path
from glob import glob
from gc import collect
import traceback
import datetime


# SPEDAS API
sys.path.insert(0,"/Users/nokni/work/MHDTurbPy/pyspedas")
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data

# Import TurbPy
sys.path.insert(1,'/Users/nokni/work/MHDTurbPy/functions')
import TurbPy as turb
import general_functions as func
import LoadData


def LoadSCAMFromSPEDAS_PSP(in_RTN, start_time, end_time, credentials = None):
    """ 
    load scam data with pyspedas and return a dataframe
    Input:
        start_time, end_time                pd.Timestamp
        (optional) credentials              dictionary, {'username':..., 'password':...}
    Output:
        return None if no data is present, otherwise a dataframe containing all the scam data
    """

    # # check pyspedas
    # if os.path.exists(Path(".").absolute().parent.joinpath("pyspedas")):
    #     print("Using pyspedas at %s" %(str(Path(".").absolute().parent.joinpath("pyspedas"))))
    # else:
    #     raise ValueError("Please clone pyspedas to %s" %(str(Path(".").absolute().parent.joinpath("pyspedas"))))
    #print(type(start_time))
    if type(start_time) =='str':
        t0 = start_time
        t1 = end_time
    else:
        try:
            t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
            t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")
        except:
            t0 = start_time
            t1 = end_time
    
    if credentials is None:
        if in_RTN:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf', 
                            varnames = ['psp_fld_l3_merged_scam_wf_RTN'], level='l3', time_clip=0, downloadonly = False ) 
        else:
            scam_vars = pyspedas.psp.fields(
                trange=[t0, t1], datatype='merged_scam_wf', 
                            varnames = ['psp_fld_l3_merged_scam_wf_SC'], level='l3', time_clip=0, downloadonly = False )            

        if scam_vars == []:
            return None

        if  in_RTN:
            data = get_data(scam_vars[0])
            dfscam = pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['Br','Bt','Bn']
                )
        else:
            data = get_data(scam_vars[0])
            dfscam = pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['Bx','By','Bz']
                )         

        dfscam.index      = time_string.time_datetime(time=dfscam.index)
        dfscam.index      = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'
        #print("SCAM data", dfscam)
        return dfscam

    else:
        # use credentials

        try:
            if in_RTN:
                scam_vars = pyspedas.psp.fields(
                    trange=[t0, t1], datatype='merged_scam_wf',
                            varnames = ['psp_fld_l3_merged_scam_wf_RTN'], level='l3', time_clip=0, downloadonly = False,
                    username = credentials['username'], password = credentials['password'])
            else:
                scam_vars = pyspedas.psp.fields(
                    trange=[t0, t1], datatype='merged_scam_wf',
                            varnames = ['psp_fld_l3_merged_scam_wf_SC'], level='l3', time_clip=0, downloadonly = False,
                    username = credentials['username'], password = credentials['password'])                
        except:
            raise ValueError('Wrong Username or Password!')

        if scam_vars == []:
            return None

        if  in_RTN:
            data = get_data(scam_vars[0])
            dfscam = pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['Br','Bt','Bn']
                )
        else:
            data = get_data(scam_vars[0])
            dfscam = pd.DataFrame(
                    index = data.times,
                    data = data.y,
                    columns = ['Bx','By','Bz']
                )         

        dfscam.index = time_string.time_datetime(time=dfscam.index)
        dfscam.index = dfscam.index.tz_localize(None)
        dfscam.index.name = 'datetime'
        #print("SCAM data", dfscam)

        r8      = dfscam.index.unique().get_loc(start_time, method='nearest');
        r8a     = dfscam.index.unique().get_loc(end_time, method='nearest');
        dfscam   = dfscam[r8:r8a]


        return dfscam


def LoadTimeSeriesFromSPEDAS_PSP(
    sc, in_RTN, SCAM,  start_time, end_time, 
    settings, credentials,
    rolling_rate = '1H', resolution = '5s',
    rootdir = None,
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
    # Define constants
    au_to_km   = 1.496e8                      ## Conversion factor au to km

    # change to root dir
    if rootdir is None:
        pass
    else:
        os.chdir(rootdir)

    if settings['verbose']:
        print("Current Settings...")
        for k, v in settings.items():
            print("{} : {}".format(k, v))

        print("Rolling Rate: %s" %(rolling_rate))

    # Parker Solar Probe
    if sc == 0:
        # check local directory
        if os.path.exists("./psp_data"):
            pass
        else:
            working_dir = os.getcwd()
            os.makedirs(str(Path(working_dir).joinpath("psp_data")), exist_ok=True)

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
                #print("No QTN data is presented in the public repository!")
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

        # Magnetic field
        try:
            try:

                if SCAM:
                    
                    dfmag = LoadSCAMFromSPEDAS_PSP(in_RTN, t0, t1, credentials = None )

                else:
                    if in_RTN:
                        names = pyspedas.psp.fields(trange=[t0,t1], 
                            datatype='mag_rtn', level='l2', time_clip=True,

                        )
                        data = get_data(names[0])
                        dfmag = pd.DataFrame(
                            index = data[0],
                            data = data[1]
                        )
                        dfmag.columns = ['Br','Bt','Bn']
                    else:
                        names = pyspedas.psp.fields(trange=[t0,t1], 
                            datatype='mag_sc', level='l2', time_clip=True,
                            # username=username, password=password
                        )
                        data = get_data(names[0])
                        dfmag = pd.DataFrame(
                            index = data[0],
                            data = data[1]
                        )
                        dfmag.columns = ['Bx','By','Bz']

                    dfmag.index = time_string.time_datetime(time=dfmag.index)
                    dfmag.index = dfmag.index.tz_localize(None)                
                
            except:
                print('Loading Unpublished data')
                username = credentials['psp']['fields']['username']
                password = credentials['psp']['fields']['password']
                if SCAM:
                    credentials1 =  {'username': username, 'password': password}
                    dfmag = LoadSCAMFromSPEDAS_PSP(in_RTN, t0, t1, credentials = credentials1 )

                else:
                    if in_RTN:
                        names = pyspedas.psp.fields(trange=[t0,t1], 
                            datatype='mag_RTN', level='l2', time_clip=True,
                            username=username, password=password
                        )
                        data = get_data(names[0])
                        dfmag = pd.DataFrame(
                            index = data[0],
                            data = data[1]
                        )
                        dfmag.columns = ['Br','Bt','Bn']
                    else:
                        names = pyspedas.psp.fields(trange=[t0,t1], 
                            datatype='mag_SC', level='l2', time_clip=True,
                            username=username, password=password
                        )
                        data = get_data(names[0])
                        dfmag = pd.DataFrame(
                            index = data[0],
                            data = data[1]
                        )
                        dfmag.columns = ['Bx','By','Bz']

                    dfmag.index = time_string.time_datetime(time=dfmag.index)
                    dfmag.index = dfmag.index.tz_localize(None) 
        except:
            traceback.print_exc()
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
                                            'carr_longitude'
                                        ], 
                                        time_clip=True)

                data = get_data(spcdata[0])
            except:
                #print("No SPC data is presented in the public repository!")
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
                                            'carr_longitude'
                                        ], 
                                        time_clip=True, username=username, password=password)

                data = get_data(spcdata[0])

            dfspc = pd.DataFrame(
                # index = time_string.time_datetime(time=data.times, tz=None)
                index = data.times
            )

            
            cols_array = [['np'],
                          ['Vth'],
                          ['Vr','Vt','Vn'],
                          ['Vx','Vy','Vz'],
                          ['sc_x','sc_y','sc_z'],
                          ['sc_vel_x','sc_vel_y','sc_vel_z'],
                          ['carr_lat'],
                          ['carr_lon']]
            
            for ind in range(len(cols_array)):
                temp = get_data(spcdata[ind])
                dfspc = dfspc.join(
                    pd.DataFrame(
                        # index = time_string.time_datetime(time=np.times, tz=None),
                        index = temp.times,
                        data = temp.y,
                        columns = cols_array[ind]
                    )
                )

            # calculate Dist_au
            dfspc['Dist_au'] = (dfspc[['sc_x','sc_y','sc_z']]**2).sum(axis=1).apply(np.sqrt)/au_to_km

            dfspc.index      = time_string.time_datetime(time=dfspc.index)
            dfspc.index      = dfspc.index.tz_localize(None)
            dfspc.index.name = 'datetime'
            dfspc            = dfspc.sort_index()
        except:
            traceback.print_exc()
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
            except:
                #print("No SPAN data is presented in the public repository!")
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

            # calculate Vth from TEMP
            # k T (eV) = 1/2 mp Vth^2 => Vth = 13.84112218*sqrt(TEMP)
            dfspan['Vth'] = 13.84112218 * np.sqrt(dfspan['TEMP'])

            # for span the thermal speed is defined as the trace, hence have a sqrt(3) different from spc
            dfspan['Vth'] = dfspan['Vth']/np.sqrt(3)

            dfspan.index = time_string.time_datetime(time=dfspan.index)
            dfspan.index = dfspan.index.tz_localize(None)
            dfspan.index.name = 'datetime'
        except:
            print("No SPAN!")
            dfspan = None

        # merge particle data

        if 'particle_mode' in settings.keys():
            parmode = settings['particle_mode']
        else:
            parmode = 'empirical'

        # # create empty dfpar_a1 with index
        freq = resolution
        index = pd.date_range(
            start = start_time, 
            end = end_time, 
            freq = freq
        )
        dfpar_a1 = pd.DataFrame(
            index = index
        )
        
        dfpar_a = pd.DataFrame()

        #print("Parmode: %s" %(parmode))
        # fill dfpar with values
        keep_keys = ['np', 'Vx','Vy','Vz','Vr','Vt','Vn','Vth','Dist_au']
        if parmode == 'spc_only':
            dftemp = dfspc
        elif parmode == 'span_only':
            dftemp = dfspan#.resample(freq).mean())
        elif parmode == 'empirical':
            # empirical use of data
            # encounter date: https://sppgway.jhuapl.edu/index.php/encounters
            # before encounter 9 (Perihelion: 2021-08-09/19:11) use SPC for solar wind speed
            # at and after encounter 8, mix SPC and SPAN for solar wind speed
            # prioritize QTN for density, and fill with SPC, and with SPAN
        
            

            # if no SPC or no SPAN data:
            if dfspc is None:
                dfspc = pd.DataFrame(index = dfpar_a1.index)
                dfspc[keep_keys] = np.nan

            if dfspan is None:
                dfspan = pd.DataFrame(index = dfpar_a1.index)
                dfspan[keep_keys] = np.nan


            # Perihelion cut
            ind1 = dfpar_a1.index < pd.Timestamp('2021-07-15')
            ind2 = dfpar_a1.index >= pd.Timestamp('2021-07-15')

            ind11 = dfspc.index < pd.Timestamp('2021-07-15')
            ind12 = dfspc.index >= pd.Timestamp('2021-07-15')

            ind21 = dfspan.index < pd.Timestamp('2021-07-15')
            ind22 = dfspan.index >= pd.Timestamp('2021-07-15')

            # before encounter 9 use spc
            dfpar1 = dfspc.loc[ind11,keep_keys]#.resample(freq).mean()

            # use span after 2021-07-15
            dfpar2 = dfspan.loc[ind22,keep_keys]#.resample(freq).mean()

            dftemp = pd.concat([dfpar1,dfpar2])
            #dfpar  = dfpar.join(dftemp[keep_keys])
            

        # proton density with QTN
        if dfqtn is None:
            dfpar_a['np_qtn'] = np.nan
        else:
            dfpar_a = dfqtn#.resample(freq).mean())

            # the density is not very fluctuating, hence interpolate
            try:
                if settings['interpolate_qtn']:
                    #print("QTN is interpolated!")
                    dfpar_a['ne_qtn'] = dfpar_a['ne_qtn'].dropna().interpolate(method='linear')
            except:
                pass

            dfpar_a['np_qtn'] = dfpar_a['ne_qtn']/1.08



        nindex  = dftemp.index
        dfpar_a = dfpar_a.reindex(dfpar_a.index.union(nindex)).interpolate(method='linear').reindex(nindex)
        dfpar   = dfpar_a.join(dftemp[keep_keys])
        if len(dfpar['np_qtn'].dropna())==0:
            dfpar['np_qtn'] = dfpar['np']
            dfpar['ne_qtn'] = dfpar['np']


        
       # else:
            #raise ValueError("particle mode: %s not supported!" %(parmode))

       # print(dfmag)
        if settings['use_hampel'] == True:
            for k in dfpar.columns:
                ns, _ = turb.hampel_filter(dfpar[k].values, 100)
                dfpar[k] = ns

        misc = {
            'dfqtn': dfqtn,
            'dfspc': dfspc,
            'dfspan': dfspan,
            'parmode': parmode,
            'settings': settings
        }

        return dfmag, dfpar, misc
    else:
        raise ValueError("sc = %d, wrong function!!" %(sc))

def LoadTimeSeriesFromSPEDAS_SOLO(dist_df, sc, in_RTN, SCAM, start_time, end_time, rootdir = None, rolling_rate = '1H', settings = None):
    import pyspedas
    from pyspedas.utilities import time_string
    from pytplot import get_data
    
    """ Load Time Series from SPEDAS with Solar Orbiter """
    # change to root dir
    if rootdir is None:
        pass
    else:
        os.chdir(rootdir)

    # Solar Orbiter
    if sc == 1:

        # check local directory
        if os.path.exists("./solar_orbiter_data"):
            pass
        else:
            working_dir = os.getcwd()
            os.makedirs(str(Path(working_dir).joinpath("solar_orbiter_data")), exist_ok=True)

        t0 = start_time.strftime("%Y-%m-%d/%H:%M:%S")
        t1 = end_time.strftime("%Y-%m-%d/%H:%M:%S")
        try:
            swadata = pyspedas.solo.swa(trange=[t0, t1], datatype='pas-grnd-mom')
            data = get_data(swadata[0])

            dfpar = pd.DataFrame(

                index = data.times
            )
            temp = get_data('N')
            dfpar = dfpar.join(
                pd.DataFrame(

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
           # print('Before', dfpar)

            """ Load ephimeris data """                  
            r8      = dist_df.index.unique().get_loc(dfpar.index[0], method='nearest');
            r8a     = dist_df.index.unique().get_loc(dfpar.index[-1], method='nearest');
            dfdis   = dist_df[r8:r8a]
            print('All right!')

            #print('After', dfpar)

            # Reindex distance data to particle data index
            nindex     = dfpar.index
            dfdis      = dfdis.reindex(dfdis.index.union(nindex)).interpolate(method='linear').reindex(nindex)

            # join dfpar and dfdis
            dfpar            = dfpar.join(dfdis)
            dfpar['np_qtn']  = dfpar['np']
            dfpar['ne_qtn']  = dfpar['np']


            if settings['use_hampel'] == True:
                list_quants = ['np', 'T', 'Vth', 'Vr', 'Vt']
                for k in list_quants:
                    ns, _ = turb.hampel_filter(dfpar[k].values, 100)
                    dfpar[k] = ns

        except:
            traceback.print_exc()
            print("No particle data!")
            dfpar = None
            dfdis = None
            
        try:

            if SCAM:
                if in_RTN:
                    try:
                        names     = pyspedas.solo.mag(trange=[t0,t1], datatype='rtn-burst', level='l2', time_clip=True)
                        data      = get_data(names[0])
                        dfmag     = pd.DataFrame(
                                                index = data[0],
                                                data  = data[1]
                                             )

                        dfmag.index = time_string.time_datetime(time=dfmag.index)
                        dfmag.index = dfmag.index.tz_localize(None)

                        # In case there is to few burst data!
                        from dateutil.parser import parse
                        int_dur    = ( parse(t1) - parse(t0)).total_seconds()/3600
                        deviation  = abs((dfmag.index[-1]- parse(t1))/np.timedelta64(1, 'h')) +abs((dfmag.index[0]-parse(t0))/np.timedelta64(1, 'h'))#/3600
                        if deviation>=0.1*int_dur:
                            print('Too little burst data!')
                            names     = pyspedas.solo.mag(trange=[t0,t1], datatype='rtn-normal', level='l2', time_clip=True)
                            data      = get_data(names[0])
                            dfmag     = pd.DataFrame(
                                                index = data[0],
                                                data  = data[1]
                                             )
                            dfmag.index = time_string.time_datetime(time=dfmag.index)
                            dfmag.index = dfmag.index.tz_localize(None)
                        else:
                            print('ok, enough burst mag data')
                    except:
                        print('Tried low resolution!')
                        names     = pyspedas.solo.mag(trange=[t0,t1], datatype='rtn-normal', level='l2', time_clip=True)
                        data      = get_data(names[0])
                        dfmag     = pd.DataFrame(
                                                index = data[0],
                                                data  = data[1]
                                             )
                        dfmag.index = time_string.time_datetime(time=dfmag.index)
                        dfmag.index = dfmag.index.tz_localize(None)

                    dfmag.columns = ['Br','Bt','Bn']            
                else:
                    names     = pyspedas.solo.mag(trange=[t0,t1], datatype='srf-burst', level='l2', time_clip=True)
                    data      = get_data(names[0])
                    dfmag     = pd.DataFrame(
                                                index = data[0],
                                                data  = data[1]
                                             )
                    dfmag.columns = ['Bx','By','Bz']    
                    dfmag.index = time_string.time_datetime(time=dfmag.index)
                    dfmag.index = dfmag.index.tz_localize(None)
            else:

                if in_RTN:

                    names     = pyspedas.solo.mag(trange=[t0,t1], datatype='rtn-normal', level='l2', time_clip=True)
                    data      = get_data(names[0])
                    dfmag     = pd.DataFrame(
                                                index = data[0],
                                                data  = data[1]
                                             )
                    dfmag.columns = ['Br','Bt','Bn']
                else:

                    names     = pyspedas.solo.mag(trange=[t0,t1], datatype='srf-normal', level='l2', time_clip=True)
                    data      = get_data(names[0])
                    dfmag     = pd.DataFrame(
                                                index = data[0],
                                                data  = data[1]
                                             )
                    dfmag.columns = ['Bx','By','Bz']


                dfmag.index = time_string.time_datetime(time=dfmag.index)
                dfmag.index = dfmag.index.tz_localize(None)

            #print(dfmag)
        except:
            traceback.print_exc()
            print("No MAG Data!")
            dfmag = None


        # nothing to be stored for solar orbiter
        misc = {'dfdis': dfdis}

        return dfmag, dfpar, misc


    else:
        raise ValueError("sc=%d not supported!" %(sc))



def resample_timeseries_estimate_gaps(df, resolution, large_gaps = 10):
    """
    Resample timeseries and gaps, default setting is for FIELDS data
    Resample to 10Hz and return the resampled timeseries and gaps infos
    Input: 
        df                  : input time series
        resolution          : resolution to resample [ms]
    Keywords:
        large_gaps  =   10 [s]      ## large gaps in timeseries [s]
    Outputs: 
        init_dt             :       initial resolution of df
        df_resampled        :       resampled dataframe
        fraction_missing    :       fraction of missing values in the interval
        total_large_gaps    :       fraction of large gaps in the interval
        total_gaps          :       total fraction of gaps in the interval  
        resolution          :       resolution of resmapled dataframe
        
        
    """
    # find keys
    keys = df.keys()
   

    # Find first col of df
    init_dt = (df[keys[1]].dropna().index.to_series().diff()/np.timedelta64(1, 's')).median()
    # print(init_dt)
    
    # Make sure that you resample to a resolution that is lower than initial df's resolution 
    while init_dt>resolution*1e-3:
        resolution = 1.005*resolution
        #print("New_resolution",resolution )
    

    
    # estimate duration of interval selected in seconds #
    interval_dur =  (df.index[-1] - df.index[0])/np.timedelta64(1, 's')

    # Resample time-series to desired resolution # 
    df_resampled = df.resample(str(int(resolution))+"ms").mean()

    # Estimate fraction of missing values within interval #
    fraction_missing = 100 *(df_resampled[keys[1]].isna().sum()/ len(df_resampled))

    # Estimate sum of gaps greater than large_gaps seconds
    res          = (df_resampled.dropna().index.to_series().diff()/np.timedelta64(1, 's'))
    
     # Gives you the fraction of  large gaps in timeseries 
    total_large_gaps   = 100*( res[res.values>large_gaps].sum()/ interval_dur  ) 
    
    # Gives you the total fraction  gaps in timeseries
    total_gaps         =  100*(res[res.values>int(resolution)*1e-3].sum()/ interval_dur  )
    # return a Dictionary
    final_dict = {"Init_dt"      : init_dt,
				  "resampled_df" : df_resampled,
				  "Frac_miss"    : fraction_missing,
				  "Large_gaps"   : total_large_gaps,
				  "Tot_gaps"     : total_gaps,
				  "resol"        :resolution
                  }
    
    return final_dict




def estimate_quants_particle_data(estimate_PSDv, part_resolution, f_min_spec, f_max_spec,  in_rtn, df, mag_resampled, subtract_rol_mean, smoothed = True):
   
    from scipy import constants
    mu_0            = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
    mu0             = constants.mu_0   #
    m_p             = constants.m_p    # Proton mass [kg]
    kb              = constants.k      # Boltzman's constant     [j/K]
    au_to_km        = 1.496e8
    T_to_Gauss      = 1e4
    
    # Particle data remove nan values
    df_part = df.dropna()
    
    # Reindex magnetic field data to particle data index
    dtv                 = (df_part.dropna().index.to_series().diff()/np.timedelta64(1, 's'))[1]
    dtb                 = (mag_resampled.dropna().index.to_series().diff()/np.timedelta64(1, 's'))[1]

    freq_final          = str(int(dtv*1e3))+'ms'
    print('final resol', freq_final)
   # freq_low, freq_high = 1e-10, 1/(np.sqrt(3)*dtv)
   # res                 = func.time_domain_filter(mag_resampled.values, dtb , freq_low, freq_high)

   # df1_var_names  = ['Br', 'Bt', 'Bn']
   # df2_var_names  = ['Vr']

   # mag_resampled1  = pd.DataFrame()
   # for i in range(len(df1_var_names)):
   #     mag_resampled1[df1_var_names[i]] = res.T[i]
    
   # mag_resampled1['DateTime']      = mag_resampled.index.values
   # mag_resampled1                  = mag_resampled1.set_index('DateTime')

   # nindex                          = df_part.index
   # BB                              = func.newindex(mag_resampled1, nindex , interp_method='linear')
    
    f_df = mag_resampled.resample(freq_final).mean().join(
         df_part.resample(freq_final).mean()
    )

    """Define magentic field components"""
    Bx     = f_df.values.T[0];  By     = f_df.values.T[1];  Bz     = f_df.values.T[2]; Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    #f_df      = mag_resampled.reindex(mag_resampled.index.union(nindex)).interpolate(method='linear').reindex(nindex)
    #print(f_df)
    if subtract_rol_mean:
        f_df[['Br_mean','Bt_mean','Bn_mean']]                   = f_df[['Br','Bt','Bn']].rolling('2H', center=True).mean().interpolate()
        f_df[['Vr_mean', 'Vt_mean','Vn_mean', 'np_mean']]  = f_df[['Vr','Vt','Vn', 'np']].rolling('2H', center=True).mean().interpolate()
       
    

    
        
    #Estimate median solar wind speed   
    Vth       = f_df.Vth.values;   Vth[Vth < 0] = np.nan; Vth_mean =np.nanmedian(Vth); Vth_std =np.nanstd(Vth);
    
    #Estimate median solar wind speed  
    if in_rtn:
        try:
            Vsw       = np.sqrt(f_df.Vr.values**2 + f_df.Vt.values**2 + f_df.Vn.values**2); Vsw_mean =np.nanmedian(Vsw); Vsw_std =np.nanstd(Vsw);
        except:
            Vsw       = np.sqrt(f_df.Vx.values**2 + f_df.Vy.values**2 + f_df.Vz.values**2); Vsw_mean =np.nanmedian(Vsw); Vsw_std =np.nanstd(Vsw);

    else:
        Vsw       = np.sqrt(f_df.Vx.values**2 + f_df.Vy.values**2 + f_df.Vz.values**2); Vsw_mean =np.nanmedian(Vsw); Vsw_std =np.nanstd(Vsw);
    Vsw[Vsw < 0] = np.nan
    Vsw[np.abs(Vsw) > 1e5] = np.nan

    # estimate mean number density
    Np        = np.nanmean([f_df['np_qtn'].values,f_df['np'].values], axis=0)  
    Np_mean =np.nanmedian(Np); Np_std =np.nanstd(Np);
        
    # Estimate Ion inertial length di in [Km]
    di        = 228/np.sqrt(Np); di[np.log10(di) < -3] = np.nan;  di_mean =np.nanmedian(di); di_std =np.nanstd(di);
    
    # Estimate plasma Beta
    km2m        = 1e3
    nT2T        = 1e-9
    cm2m        = 1e-2
    B_mag       = Bmag * nT2T                              # |B| units:      [T]
    temp        = 1./2 * m_p * (Vth*km2m)**2              # in [J] = [kg] * [m]^2 * [s]^-2
    dens        = Np/(cm2m**3)                            # number density: [m^-3] 
    beta        = (dens*temp)/((B_mag**2)/(2*mu_0))       # plasma beta 
    beta[beta < 0] = np.nan
    beta[np.abs(np.log10(beta))>4] = np.nan # delete some weird data
    beta_mean   = np.nanmedian(beta); beta_std   = np.nanstd(beta);
    
    
    # ion gyro radius
    rho_ci = 10.43968491 * Vth/B_mag #in [km]
    rho_ci[rho_ci < 0] = np.nan
    rho_ci[np.log10(rho_ci) < -3] = np.nan
    rho_ci_mean =np.nanmedian(rho_ci); rho_ci_std =np.nanstd(rho_ci);
 
    ### Define b and v ###
    if in_rtn:
        try:
            vr, vt, vn   = f_df.Vr.values, f_df.Vt.values, f_df.Vn.values
            br, bt, bn   = f_df.Br.values, f_df.Bt.values, f_df.Bn.values
        except:
            vr, vt, vn   = f_df.Vx.values, f_df.Vy.values, f_df.Vz.values
            br, bt, bn   = f_df.Bx.values, f_df.By.values, f_df.Bz.values
    else:
        vr, vt, vn       = f_df.Vx.values, f_df.Vy.values, f_df.Vz.values
        br, bt, bn       = f_df.Bx.values, f_df.By.values, f_df.Bz.values
      
    #VBangle_mean, dVB, VBangle_std = BVangle(br, bt, bn, vr, vt, vn , smoothed)  

    Va_r = 1e-15* br/np.sqrt(mu0*f_df['np'].values*m_p)   
    Va_t = 1e-15* bt/np.sqrt(mu0*f_df['np'].values*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
    Va_n = 1e-15* bn/np.sqrt(mu0*f_df['np'].values*m_p)   
    
    # Estimate VB angle
    vbang = np.arccos((Va_r * vr + Va_t * vt + Va_n * vn)/np.sqrt((Va_r**2+Va_t**2+Va_n**2)*(vr**2+vt**2+vn**2)))
    vbang = vbang/np.pi*180#
    VBangle_mean, VBangle_std = np.nanmean(vbang), np.nanstd(vbang)
    
    # Also save the components of Vsw an Valfven
    alfv_speed = [np.nanmean(Va_r), np.nanmean(Va_t), np.nanmean(Va_n)]
    sw_speed   = [np.nanmean(vr), np.nanmean(vt), np.nanmean(vn)]

    # sign of Br within the window
    signB = - np.sign(np.nanmean(Va_r))
    
    # # Estimate fluctuations of fields #
    if subtract_rol_mean:
        va_r = Va_r - 1e-15*f_df['Br_mean'].values/np.sqrt(mu0*f_df['np_mean'].values*m_p);    v_r = vr - f_df['Vr_mean'].values
        va_t = Va_t - 1e-15*f_df['Bt_mean'].values/np.sqrt(mu0*f_df['np_mean'].values*m_p);    v_t = vt - f_df['Vt_mean'].values
        va_n = Va_n - 1e-15*f_df['Bn_mean'].values/np.sqrt(mu0*f_df['np_mean'].values*m_p);    v_n = vn - f_df['Vn_mean'].values

    else:
        va_r = Va_r - np.nanmean(Va_r);   v_r = vr - np.nanmean(vr)
        va_t = Va_t - np.nanmean(Va_t);   v_t = vt - np.nanmean(vt)
        va_n = Va_n - np.nanmean(Va_n);   v_n = vn - np.nanmean(vn)
    


    # Estimate Zp, Zm components
    Zpr = v_r +  signB *va_r; Zmr = v_r - signB *va_r
    Zpt = v_t +  signB *va_t; Zmt = v_t - signB *va_t
    Zpn = v_n +  signB *va_n; Zmn = v_n - signB *va_n


    # Estimate energy in Zp, Zm
    Z_plus_squared  = Zpr**2 +  Zpt**2 + Zpn**2
    Z_minus_squared = Zmr**2 +  Zmt**2 + Zmn**2
    
    # Estimate amplitude of fluctuations
    Z_amplitude                = np.sqrt( (Z_plus_squared + Z_minus_squared)/2 ) ; Z_amplitude_mean    = np.nanmedian(Z_amplitude); Z_amplitude_std = np.nanstd(Z_amplitude);

    # Kin, mag energy
    Ek = v_r**2 + v_t**2 + v_n**2
    Eb = va_r**2 + va_t**2 + va_n**2
    
    #Estimate normalized residual energy
    sigma_r      = (Ek-Eb)/(Ek+Eb);                                                         sigma_r[np.abs(sigma_r) > 1e5] = np.nan;
    sigma_c      = (Z_plus_squared - Z_minus_squared)/( Z_plus_squared + Z_minus_squared);  sigma_c[np.abs(sigma_c) > 1e5] = np.nan
    
    #Save in DF format to estimate spectraly
    nn_df       = pd.DataFrame({'DateTime': f_df.index.values,
                                'Zpr'     : Zpr,    'Zpt': Zpt, 'Zpn' : Zpn,
                                'Zmr'     : Zmr,    'Zmt': Zmt, 'Zmn' : Zmn, 
                                'va_r'    : va_r,  'va_t': va_t,'va_n': va_n,
                                'v_r'     : v_r,   'v_t' : v_t, 'v_n' : v_n,
                                'beta'    : beta,  'np'  : Np,  'Tp'  : temp,
                                'sigma_c' : sigma_c,         'sigma_r': sigma_r}).set_index('DateTime')
    nn_df       = nn_df.mask(np.isinf(nn_df)).dropna().interpolate(method='linear')
    
    # Estimate mean, median,... of  normalized residual energy
    sigma_r_mean = np.nanmean(sigma_r); sigma_r_median =np.nanmedian(sigma_r); sigma_r_std =np.nanstd(sigma_r);
    
    # Estimate mean, median,... of  normalized cross helicity
    sigma_c_mean = np.nanmean(np.abs(sigma_c)); sigma_c_median = np.nanmedian(np.abs(sigma_c)); sigma_c_std = np.nanstd(np.abs(sigma_c));     sigma_c_median_no_abs  = np.nanmedian(sigma_c);     sigma_c_mean_no_abs    = np.nanmean(sigma_c);   

    #Estimate  z+, z- PSD
    if estimate_PSDv:
        f_Zplus, psd_Zplus    = turb.TracePSD(nn_df['Zpr'].values, nn_df['Zpt'].values, nn_df['Zpn'].values, 0,  part_resolution)
        f_Zminus, psd_Zminus  = turb.TracePSD(nn_df['Zmr'].values, nn_df['Zmt'].values, nn_df['Zmn'].values, 0,  part_resolution)
    else:
        f_Zplus, psd_Zplus    = None, None
        f_Zminus, psd_Zminus  = None, None 

    #Estimate  v,b PSD
    if estimate_PSDv==0:
        f_vv, psd_vv         = None, None
        f_bb, psd_bb         = None, None
    else:
        f_vv, psd_vv         = turb.TracePSD(nn_df['v_r'].values, nn_df['v_t'].values, nn_df['v_n'].values, 0,  part_resolution)
        f_bb, psd_bb         = turb.TracePSD(nn_df['va_r'].values, nn_df['va_t'].values, nn_df['va_n'].values, 0,  part_resolution)      
    
    # Only keep indices within the range of frequencies specified
    if estimate_PSDv:
        inda1           = (f_Zplus>f_min_spec) &  (f_Zplus<f_max_spec) & (np.abs(psd_Zplus)<1e10)  & (np.abs(psd_Zminus)<1e10)
        sigma_c_spec    =  (np.nansum(psd_Zplus[inda1])-np.nansum(psd_Zminus[inda1])) /  (np.nansum(psd_Zplus[inda1]) + np.nansum(psd_Zminus[inda1])) 
        sigma_r_spec    = (np.nansum(psd_vv[inda1])-np.nansum(psd_bb[inda1])) /  (np.nansum(psd_vv[inda1]) + np.nansum(psd_bb[inda1])) 
    else:
    #  Estimate normalized cross helicity and normalized residual energy spectraly
        sigma_c_spec    = None
        sigma_r_spec    = None
    # Also keep a dict containing psd_vv, psd_bb, psd_Zplus, psd_Zminus
    if estimate_PSDv:
        dict_psd = {"f_vb": f_vv, 'psd_v': psd_vv, "psd_b": psd_bb, "f_zpm":f_Zplus, "psd_zp":psd_Zplus , "psd_zm":psd_Zminus}
    else:
        dict_psd ={}
    #del nn_df['Zpr'], nn_df['Zpt'], nn_df['Zpn'], nn_df['Zmr'], nn_df['Zmt'], nn_df['Zmn']
    
  
    
    return  dict_psd, nn_df, sw_speed, alfv_speed,  f_Zplus, psd_Zplus, f_Zminus, psd_Zminus, sigma_r_spec, sigma_c_spec, beta_mean, beta_std, Z_amplitude_mean, Z_amplitude_std, sigma_r_mean, sigma_r_median, sigma_r_std, sigma_c_median, sigma_c_mean, sigma_c_median_no_abs, sigma_c_mean_no_abs, sigma_c_std, Vth_mean, Vth_std , Vsw_mean, Vsw_std, Np_mean, Np_std, di_mean, di_std, rho_ci_mean, rho_ci_std, VBangle_mean, VBangle_std

# Probably correct
def calc_mag_diagnostics(gap_time_threshold, dist1, estimate_SPD, mag_data, mag_resolution, sc = 0):
    

    if sc == 0:
        try:
            dist = pd.read_pickle(Path(r"C:\\Users\\zhuang\\work\\giannis\\sc_distance\\psp_distance_all.dat"))
        except:
            dist = dist1
    elif sc == 1:

        try:
            dist = pd.read_pickle(Path(r'C:\\Users\\zhuang\\work\\giannis\\sc_distance\\solo_distance_all.dat'))
        except:
            try:
                dist = dist1
            except:
                raise ValueError("ERROR")
        # file_to_read.close()
    else:
        raise ValueError("ERROR")
    

#     mag_data = df['spdf_data']['spdf_infos']['mag_sc']['dataframe']


    """ Make distance dataframe to begin at the same time as magnetic field timeseries """ 
    try:
        r8      = dist.index.unique().get_loc(mag_data.index[0], method='nearest');
        r8a     = dist.index.unique().get_loc(mag_data.index[-1], method='nearest');
        r_psp   = np.nanmean(dist[r8:r8a])
    except:
        print("error here, 1")

        
    """ Identify big gaps in timeseries"""
    try:
        ### Identify  big gaps in our timeseries ###
        f2          = mag_data[mag_data.Br>-1e3]
        time        = (f2.index.to_series().diff()/np.timedelta64(1, 's'))
        big_gaps    = time[time>gap_time_threshold]
    except:
        print("error here, 2")

    """ Resample magnetic field data, estimate fraction of gaps"""
    mag_results    = resample_timeseries_estimate_gaps(mag_data, resolution = mag_resolution, large_gaps = 10)
    # mag_init_dt, mag_resampled, mag_fraction_missing, mag_total_large_gaps, mag_total_gaps, mag_resolution = mag_results
    mag_init_dt, mag_resampled, mag_fraction_missing, mag_total_large_gaps, mag_total_gaps, mag_resolution = mag_results["Init_dt"],  mag_results["resampled_df"], mag_results["Frac_miss"], mag_results["Large_gaps"], mag_results["Tot_gaps"], mag_results["resol"]
    try:
        mag_interpolate = mag_resampled.interpolate(method = 'linear').dropna()
    except:
        print("error here, 3")

    """Define magentic field components"""
    # Bx     = mag_resampled.values.T[0];  By     = mag_resampled.values.T[1];  Bz     = mag_resampled.values.T[2]; Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    Bx = mag_interpolate.values.T[0]; By = mag_interpolate.values.T[1]; Bz = mag_interpolate.values.T[2]; Bmag = np.sqrt(Bx**2 + By**2 + Bz**2);
    try:
        if estimate_SPD:
            """Estimate PSD of  magnetic field"""
            remove_mean, dt_MAG       = 1 ,mag_resolution*1e-3
            f_B, psd_B                = turb.TracePSD(Bx,By,Bz, remove_mean,dt_MAG)

            """Smooth PSD of  magnetic field"""    
            f_B_mid, f_B_mean, psd_B_smooth   =  func.smoothing_function(f_B, psd_B, window=2, pad = 1)
        else:
            """Estimate PSD of  magnetic field"""
           # remove_mean, dt_MAG       = 1 ,mag_resolution*1e-3
            f_B, psd_B                 = None, None #TracePSD(Bx,By,Bz, remove_mean,dt_MAG)

            """Smooth PSD of  magnetic field"""    
            f_B_mid, f_B_mean, psd_B_smooth   = None, None, None #smoothing_function(f_B, psd_B, window=2, pad = 1)


    except:
        print('error here')

    try:
        general_dict = {
                "Start_Time"           : mag_data.index[0],
                "End_Time"             : mag_data.index[-1],  
                "d"                    : r_psp,
                "Fraction_missing_MAG" : mag_fraction_missing,
                "Resolution_MAG"       : mag_resolution,
                       }
    except:
        print('Error edw!')

    mag_dict = {
                "B_resampled"      : mag_interpolate,
                "PSD_f_orig"       : psd_B,
                "f_B_orig"         : f_B,
                "PSD_f_smoothed"   : psd_B_smooth,
                "f_B_mid"          : f_B_mid,
                "f_B_mean"         : f_B_mean,
                "Fraction_missing" : mag_fraction_missing,
                "resolution"       : mag_resolution
                }
    
    return big_gaps, general_dict, mag_dict

def calc_particle_diagnostics(estimate_PSD_V, subtract_rol_mean, f_min_spec, f_max_spec, in_rtn, mag_data, spc_data, mag_resolution, spc_resolution = 200, sc = 0, smoothed = True):

    spc_results = resample_timeseries_estimate_gaps(spc_data, resolution = spc_resolution, large_gaps = 10)

    spc_init_dt, spc_resampled, spc_fraction_missing, spc_total_large_gaps, spc_total_gaps, spc_resolution = spc_results["Init_dt"],  spc_results["resampled_df"], spc_results["Frac_miss"], spc_results["Large_gaps"], spc_results["Tot_gaps"], spc_results["resol"]

    """ Resample magnetic field data, estimate fraction of gaps"""
    mag_results    = resample_timeseries_estimate_gaps(mag_data, resolution = mag_resolution, large_gaps = 10)
    mag_init_dt, mag_resampled, mag_fraction_missing, mag_total_large_gaps, mag_total_gaps, mag_resolution = mag_results["Init_dt"],  mag_results["resampled_df"], mag_results["Frac_miss"], mag_results["Large_gaps"], mag_results["Tot_gaps"], mag_results["resol"]

    """Define Span velocity field components"""
    spc_interpolated = spc_resampled.interpolate(method = 'linear').dropna()
    if in_rtn:

        Vx_spc = spc_interpolated['Vr'].values;  Vy_spc = spc_interpolated['Vt'].values;  Vz_spc = spc_interpolated['Vn'].values 
    else:
        Vx_spc = spc_interpolated['Vx'].values;  Vy_spc = spc_interpolated['Vy'].values;  Vz_spc = spc_interpolated['Vz'].values

    """Estimate PSD of  Velocity field: SPAN data"""
    remove_mean, dt_SPC = 1 ,spc_resolution*1e-3
    #print(Vx_spc)
    f_V_SPC, psd_V_SPC                     = turb.TracePSD(Vx_spc, Vy_spc, Vz_spc, remove_mean,dt_SPC)

    """Estimate derived SPAN quantities"""
    part_quants_spc = estimate_quants_particle_data(estimate_PSD_V, dt_SPC, f_min_spec, f_max_spec,  in_rtn,  spc_interpolated, mag_resampled, subtract_rol_mean, smoothed)
    dict_psd, nn_df, sw_speed, alfv_speed, f_Zplus, psd_Zplus, f_Zminus, psd_Zminus, sigma_r_spec_spc, sigma_c_spec_spc, beta_mean_spc, beta_std_spc, Z_amplitude_mean_spc, Z_amplitude_std_spc, sigma_r_mean_spc, sigma_r_median_spc, sigma_r_std_spc, sigma_c_median_spc, sigma_c_mean_spc, sigma_c_median_no_abs_spc, sigma_c_mean_no_abs_spc, sigma_c_std_spc, Vth_mean_spc, Vth_std_spc , Vsw_mean_spc, Vsw_std_spc, Np_mean_spc, Np_std_spc, di_mean_spc, di_std_spc, rho_ci_mean_spc, rho_ci_std_spc, VBangle_mean_spc, VBangle_std_spc = part_quants_spc  
    

    spc_dict = { 
                 "V_resampled"           : spc_interpolated, #dataframe containing all particle data
                 "dict_psd"              : dict_psd,# dictionary containing psd's of Zp, Zm, V, B
                 "f_V"                   : f_V_SPC,  #array containing spacecraft frequency for power spectrum
                 "psd_V"                 : psd_V_SPC, #power spectrum 
                 "Va"                    : alfv_speed, # alfven speed 
                 "Sw_speed"              : sw_speed, #solar wind speed
                 "f_Zplus"               : f_Zplus,  # array containing spacecraft frequency for z+ power spectrum
                 "PSD_Zplus"             : psd_Zplus, # power spectrum of z+
                 "f_Zminus"              : f_Zminus, ## array containing spacecraft frequency for z- power spectrum
                 "PSD_Zminus"            : psd_Zminus, # power spectrum of z-
                 'di_mean'               : di_mean_spc, #mean value of ion inertial lenght for the interval
                 'di_std'                : di_std_spc, #standard deviation of -//-
                 'rho_ci_mean'           : rho_ci_mean_spc, # mean ion gyroradius
                 'rho_ci_std'            : rho_ci_std_spc,  #std ion gyroradius
                 'sigma_c_mean'          : sigma_c_mean_spc, #normalized cross-helicity mean 
                 'sigma_c_std'           : sigma_c_std_spc, #std cross helicity
                 'sigma_c_median'        : sigma_c_median_spc, 
                 'sigma_c_mean_no_abs'   : sigma_c_mean_no_abs_spc,
                 'sigma_c_median_no_abs' : sigma_c_median_no_abs_spc, 
                 'sigma_r_median'        : sigma_r_median_spc, #normalized residual energy
                 'sigma_c_spec'          : sigma_c_spec_spc,
                 'sigma_r_spec'          : sigma_r_spec_spc,
                 'sigma_r_mean'          : sigma_r_mean_spc,
                 'sigma_r_std'           : sigma_r_std_spc,
                 'Vsw_mean'              : Vsw_mean_spc, #bulk solar wind speed
                 'Vsw_std'               : Vsw_std_spc,
                 'VBangle_mean'          : VBangle_mean_spc, #angle between backround magnetic field and solar wind flow
                 'VBangle_std'           : VBangle_std_spc,
                 'beta_mean'             : beta_mean_spc, #plasma b parameter
                 'beta_std'              : beta_std_spc,
                 'Vth_mean'              : Vth_mean_spc, #thermal velocity of ions
                 'Vth_std'               : Vth_std_spc,
                 'Np_mean'               : Np_mean_spc, #number density of ions
                 'Np_std'                : Np_std_spc,
                 'Z_mean'                : Z_amplitude_mean_spc, #amplitude of fluctuations
                 'Z_std'                 : Z_amplitude_std_spc,
                 "Fraction_missing"      : spc_fraction_missing, #fraction of timeseries Nan
                 "resolution"            : spc_resolution #resolution of timeseries
              }
                                                          #means are calculated for each interval (1H)
    return spc_dict, nn_df


def use_SPDF_data(spc_resolution,mag_resolution, df ):
    spdf_data = df['spdf_data']
    mag_data  = df['spdf_data']['spdf_infos']['mag_rtn']['dataframe']
    spc_data = pd.DataFrame(index = spdf_data['spdf_infos']['spc']['data']['Epoch'])
    spc_data = spc_data.join(
        pd.DataFrame(
            index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
            data = spdf_data['spdf_infos']['spc']['data']['vp_moment_SC_gd'],
            columns = ['Vx','Vy','Vz']
        )
    ).join(
        pd.DataFrame(
            index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
            data = spdf_data['spdf_infos']['spc']['data']['vp_moment_RTN_gd'],
            columns = ['Vr','Vt','Vn']
        )
    ).join(
        pd.DataFrame(
            index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
            data = spdf_data['spdf_infos']['spc']['data']['sc_vel_HCI'],
            columns = ['Vsc_x','Vsc_y','Vsc_z']
        )
    ).join(
        pd.DataFrame(
            index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
            data = spdf_data['spdf_infos']['spc']['data']['np_moment_gd'],
            columns = ['np']
        )
    ).join(
        pd.DataFrame(
            index = spdf_data['spdf_infos']['spc']['data']['Epoch'],
            data = spdf_data['spdf_infos']['spc']['data']['wp_moment_gd'],
            columns = ['Vth']
        )
    )

    # pre-clean the data
    indnan = spdf_data['spdf_infos']['spc']['data']['general_flag']!=0
    spc_data.loc[indnan,:] = np.nan
    # more cleaning
    spc_data[spc_data.abs() > 1e20] = np.nan
    spc_data.index.name = 'datetime'


    # Only keep data from the desired interval
    r3      = spc_data.index.unique().get_loc(mag_data.index[0], method='nearest');
    r3a     = spc_data.index.unique().get_loc(mag_data.index[-1], method='nearest');

    spc_data = spc_data[r3:r3a]


    spc_results = resample_timeseries_estimate_gaps(spc_data, resolution = spc_resolution, large_gaps = 10)
    
    #mag_data = df['spdf_data']['spdf_infos']['mag_sc']['dataframe']

    """ Resample magnetic field data, estimate fraction of gaps"""
    mag_results    = resample_timeseries_estimate_gaps(mag_data, resolution = mag_resolution, large_gaps = 10)
   
    return spc_results, mag_results




def set_up_main_loop(final_path, settings, only_one_interval, t0, t1, step, duration, sc):
    # Dont change the next two variables


    """ end of user defined parameters """

    # Define lists with dates
    tstarts = []
    tends   = []


    # i1 = 1
    if only_one_interval:
        tstarts.append(t0)
        tends.append(t1)
        
    else:
        i1 = 0
        while True:
            tstart = t0+i1*pd.Timedelta(step)
            tend = tstart + pd.Timedelta(duration)
            if tend > t1:
                break
            tstarts.append(tstart)
            tends.append(tend)
            i1 += 1

    if sc==0:
        path0 = Path(final_path)
    else:
        path0 =Path(final_path)
    tfmt = "%Y-%m-%d_%H-%M-%S"
    return tstarts, tends, tfmt, path0, settings 




def set_up_main_loop_nikos(t0, t1, step, duration, sc, save_path):
    # Dont change the next two variables
    settings            = {
                            'particle_mode': 'empirical',
                            'final_freq': '5s',
                            'use_hampel': True,
                            'interpolate_qtn': True,
                            'interpolate_rolling': True,
                            'verbose': False,
                            'must_have_qtn': False
                           } 



    """ end of user defined parameters """

    # Define lists with dates
    tstarts = []
    tends   = []


    # i1 = 1
    i1 = 0
    while True:
        tstart = t0+i1*pd.Timedelta(step)
        tend = tstart + pd.Timedelta(duration)
        if tend > t1:
            break
        tstarts.append(tstart)
        tends.append(tend)
        i1 += 1



    if sc==0:
        path0 = Path(save_path).joinpath("PSP")#.joinpath()
    else:
        path0 = Path(save_path).joinpath("SolO")   
    tfmt = "%Y-%m-%d_%H-%M-%S"
    return tstarts, tends, tfmt, path0, settings

def create_dfs(estimate_PSD_V, subtract_rol_mean,dist_df, dfpar, dfmag, f_min_spec, f_max_spec, 
                start_time, end_time,
                estimate_PSD   , 
                sc             , 
                SCAM           ,
                in_RTN         ,
                mag_resolution,
                gap_time_threshold):
    
    settings    = {
                'particle_mode': 'empirical',
                'final_freq': '5s',
                'use_hampel': True,
                'interpolate_qtn': True,
                'interpolate_rolling': True,
                'verbose': False,
                'must_have_qtn': True
                } 


    """Now calculates quantities related to magnetic field timeseries"""
    # You have to fix the calc mag diagnostics function to be compatible with the current code ( you can find the function in CalcDiagnostics.Py)

    if dfpar is not None:
        if len(dfpar.dropna()) > 0 :
            try:
                #if sc==1:
                    # Fixes an error related to how Speedas downloads particle data
                r8      = dfpar.index.unique().get_loc(dfmag.index[0], method='nearest');
                r8a     = dfpar.index.unique().get_loc(dfmag.index[-1], method='nearest');
                dfpar   = dfpar[r8:r8a]
                try:       
                    big_gaps, general_dict, mag_dict = calc_mag_diagnostics(gap_time_threshold, dfpar['Dist_au'], estimate_PSD, dfmag, mag_resolution , sc = 0)
                except:
                    traceback.print_exc()          

                """Now calculates quantities related to particle timeseries"""  



                res_particles, sig_c_sig_r_timeseries = calc_particle_diagnostics(estimate_PSD_V, subtract_rol_mean, f_min_spec, f_max_spec, in_RTN, dfmag, dfpar, mag_resolution , spc_resolution = 200, sc = 0, smoothed = True)
                                                               #f_min_spec, f_max_spec, in_rtn, mag_data, spc_data, mag_resolution, spc_resolution = 200, sc = 0, smoothed = True
                # Now save everything in final_dict as a dictionary
                final_dict = { 
                               "Mag"          : mag_dict,
                               "Par"          : res_particles
                              }

                # also save a general dict with basic info (add what is missing from particletimeseries)
                general_dict["Fraction_missing_part"] = res_particles["Fraction_missing"]
                general_dict["Resolution_part"]       = res_particles["resolution"]

                flag_good = 1


            except:
                traceback.print_exc()

                flag_good = 0
                print('No MAG data.')
                big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
        else:
            final_dict = None
            print('No particle data.')

            flag_good = 0
            big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
    else:
        final_dict = None
        print('No particle data.')

        flag_good = 0
        big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None


        
    return big_gaps, flag_good, final_dict, general_dict, sig_c_sig_r_timeseries



def final_func(estimate_PSD_V, subtract_rol_mean, settings, dist_df, start_time, end_time, f_min_spec, f_max_spec,  
                estimate_PSD   , 
                sc             , 
                credentials    ,
                SCAM           ,
                in_RTN         ,
                mag_resolution,
                gap_time_threshold):

    if sc==0:

        dfmag, dfpar, misc = LoadTimeSeriesFromSPEDAS_PSP(sc, in_RTN, SCAM,  start_time, end_time, 
                                                          settings, credentials,
                                                          rolling_rate = '1H', resolution = '5s')
    elif sc ==1:
       # print('SOLO')
        dfmag, dfpar, misc = LoadTimeSeriesFromSPEDAS_SOLO(dist_df, sc, in_RTN, SCAM, start_time, end_time,
                                                           rootdir = None, rolling_rate = '1H', settings = None)
    elif sc ==3:
        # print('SOLO')
        final_dataframe = LoadData.LoadTimeSeriesWrapper(
        sc, start_time, end_time,
        settings = {}, credentials = None)
       
        final_dataframe = final_dataframe[0]
        dfmag    =  final_df[['Br','Bt','Bn','Bx','By','Bz']]
        dfpar    =  final_df[['Vr','Vt','Vn','np','Tp','Vth']]
        dist_df  =  final_df[['Dist_au','lon','lat']]
        misc     =  final_dataframe[1]


    """Now calculates quantities related to magnetic field timeseries"""
    # You have to fix the calc mag diagnostics function to be compatible with the current code ( you can find the function in CalcDiagnostics.Py)

    if dfpar is not None:
        if len(dfpar.dropna()) > 0 :
            try:
                if sc==1:
                    # Fixes an error related to how Speedas downloads particle data
                    r8      = dfpar.index.unique().get_loc(dfmag.index[0], method='nearest');
                    r8a     = dfpar.index.unique().get_loc(dfmag.index[-1], method='nearest');
                    dfpar   = dfpar[r8:r8a]
                try:       
                    big_gaps, general_dict, mag_dict = calc_mag_diagnostics(gap_time_threshold, dfpar['Dist_au'], estimate_PSD, dfmag, mag_resolution , sc = 0)
                except:
                    traceback.print_exc()          

                """Now calculates quantities related to particle timeseries"""


                res_particles, sig_c_sig_r_timeseries = calc_particle_diagnostics(estimate_PSD_V, subtract_rol_mean, f_min_spec, f_max_spec, in_RTN, dfmag, dfpar, mag_resolution , spc_resolution = 200, sc = 0, smoothed = True)

                # Now save everything in final_dict as a dictionary
                final_dict = { 
                               "Mag"          : mag_dict,
                               "Par"          : res_particles
                              }

                # also save a general dict with basic info (add what is missing from particletimeseries)
                general_dict["Fraction_missing_part"] = res_particles["Fraction_missing"]
                general_dict["Resolution_part"]       = res_particles["resolution"]

                flag_good = 1


            except:
                traceback.print_exc()

                flag_good = 0
                print('No MAG data.')
                big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
        else:
            final_dict = None
            print('No particle data.')

            flag_good = 0
            big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None
    else:
        final_dict = None
        print('No particle data.')

        flag_good = 0
        big_gaps, final_dict, general_dict, sig_c_sig_r_timeseries = None, None, None, None


        
    return big_gaps, flag_good, final_dict, general_dict, sig_c_sig_r_timeseries
    





# Particles df dunction
""" Takes final['Par']['V_resampled'] as argument,
rearranges the columns in a new df and saves the df as a csv"""
def save_par_df_func(func_dict, path_par):
    par_df = pd.DataFrame({
                'Vr'  : func_dict['Vr'],
                'Vt'  : func_dict['Vt'],
                'Vn'  : func_dict['Vn'],
                'np'  : func_dict['np'],
                'Vth' : func_dict['Vth']
                })
    
    # NOTE: The csv HAD the first column as "datetime" instead of "DateTime"
    par_df.index.names = ['DateTime']
    
    # Missing joinpath("Folder_name")
    #path_par = path0.joinpath("Particle_data.csv")
    par_df.to_csv(path_par)

    
# Magnetic field df function
""" Takes final['Mag']['B_resampled'] as argument,
rearranges the columns in a new df and saves the df as a csv"""
def save_mag_df_func(func_dict, path_mag):
    mag_df = pd.DataFrame({
                'Br'  : func_dict['Br'],
                'Bt'  : func_dict['Bt'],
                'Bn'  : func_dict['Bn']
                })
    
    # NOTE: The csv DID NOT have a name in the first column
    mag_df.index.names = ['DateTime']
    
    #Missing joinpath("Folder_name")
    #path_mag = path0.joinpath("Magnetic_data.csv")
    mag_df.to_csv(path_mag)