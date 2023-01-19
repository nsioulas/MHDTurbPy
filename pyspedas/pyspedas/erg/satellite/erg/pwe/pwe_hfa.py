import cdflib
from pytplot import clip, options, ylim, zlim, get_data

from ..load import load


def pwe_hfa(trange=['2017-04-01', '2017-04-02'],
            datatype='spec',
            mode='low',
            level='l2',
            suffix='',
            get_support_data=False,
            varformat=None,
            varnames=[],
            downloadonly=False,
            notplot=False,
            no_update=False,
            uname=None,
            passwd=None,
            time_clip=False,
            ror=True):
    """
    This function loads data from the PWE experiment from the Arase mission

    Parameters:
        trange : list of str
            time range of interest [starttime, endtime] with the format
            'YYYY-MM-DD','YYYY-MM-DD'] or to specify more or less than a day
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']

        datatype: str
            Data type; Valid options:

        level: str
            Data level; Valid options:

        suffix: str
            The tplot variable names will be given this suffix.  By default,
            no suffix is added.

        get_support_data: bool
            Data with an attribute "VAR_TYPE" with a value of "support_data"
            will be loaded into tplot.  By default, only loads in data with a
            "VAR_TYPE" attribute of "data".

        varformat: str
            The file variable formats to load into tplot.  Wildcard character
            "*" is accepted.  By default, all variables are loaded in.

        varnames: list of str
            List of variable names to load (if not specified,
            all data variables are loaded)

        downloadonly: bool
            Set this flag to download the CDF files, but not load them into
            tplot variables

        notplot: bool
            Return the data in hash tables instead of creating tplot variables

        no_update: bool
            If set, only load data from your local cache

        time_clip: bool
            Time clip the variables to exactly the range specified in the trange keyword

        ror: bool
            If set, print PI info and rules of the road

    Returns:
        List of tplot variables created.

    """

    initial_notplot_flag = False
    if notplot:
        initial_notplot_flag = True

    file_res = 3600. * 24

    if level == 'l2':
        prefix = 'erg_pwe_hfa_'+level+'_' + mode + '_'

    if level == 'l2':
        pathformat = 'satellite/erg/pwe/hfa/'+level+'/'+datatype+'/'+mode + \
            '/%Y/%m/erg_pwe_hfa_'+level+'_'+datatype+'_'+mode+'_%Y%m%d_v??_??.cdf'
    elif level == 'l3':
        prefix = 'erg_pwe_hfa_'+level+'_1min_'
        pathformat = 'satellite/erg/pwe/hfa/'+level + \
            '/%Y/%m/erg_pwe_hfa_'+level+'_1min_%Y%m%d_v??_??.cdf'

    loaded_data = load(pathformat=pathformat, trange=trange, level=level, datatype=datatype, file_res=file_res, prefix=prefix, suffix=suffix, get_support_data=get_support_data,
                       varformat=varformat, varnames=varnames, downloadonly=downloadonly, notplot=notplot, time_clip=time_clip, no_update=no_update, uname=uname, passwd=passwd)

    if (len(loaded_data) > 0) and ror:

        try:
            if isinstance(loaded_data, list):
                if downloadonly:
                    cdf_file = cdflib.CDF(loaded_data[-1])
                    gatt = cdf_file.globalattsget()
                else:
                    gatt = get_data(loaded_data[-1], metadata=True)['CDF']['GATT']
            elif isinstance(loaded_data, dict):
                gatt = loaded_data[list(loaded_data.keys())[-1]]['CDF']['GATT']

            # --- print PI info and rules of the road

            print(' ')
            print(' ')
            print(
                '**************************************************************************')
            print(gatt["LOGICAL_SOURCE_DESCRIPTION"])
            print('')
            print('Information about ERG PWE HFA')
            print('')
            print('PI: ', gatt['PI_NAME'])
            print("Affiliation: "+gatt["PI_AFFILIATION"])
            print('')
            print('RoR of ERG project common: https://ergsc.isee.nagoya-u.ac.jp/data_info/rules_of_the_road.shtml.en')
            print(
                'RoR of PWE/HFA: https://ergsc.isee.nagoya-u.ac.jp/mw/index.php/ErgSat/Pwe/Hfa')
            print('')
            print('Contact: erg_pwe_info at isee.nagoya-u.ac.jp')
            print(
                '**************************************************************************')
        except:
            print('printing PI info and rules of the road was failed')

    if initial_notplot_flag or downloadonly:
        return loaded_data

    if (level == 'l2') and (mode == 'low') and (not notplot):

        # set spectrogram plot option
        options(prefix + 'spectra_eu' + suffix, 'Spec', 1)
        options(prefix + 'spectra_ev' + suffix, 'Spec', 1)
        options(prefix + 'spectra_bgamma' + suffix, 'Spec', 1)
        options(prefix + 'spectra_esum' + suffix, 'Spec', 1)
        options(prefix + 'spectra_er' + suffix, 'Spec', 1)
        options(prefix + 'spectra_el' + suffix, 'Spec', 1)
        options(prefix + 'spectra_e_mix' + suffix, 'Spec', 1)
        options(prefix + 'spectra_e_ar' + suffix, 'Spec', 1)

        if prefix + 'spectra_er' + suffix in loaded_data:
            # remove minus values in y array
            clip(prefix + 'spectra_er' + suffix, 0., 5000.)
        if prefix + 'spectra_el' + suffix in loaded_data:
            # remove minus values in y array
            clip(prefix + 'spectra_el' + suffix, 0., 5000.)

        if prefix + 'spectra_eu' + suffix in loaded_data:
            # remove minus values in y array
            clip(prefix + 'spectra_eu' + suffix, 0., 5000.)
            # set ylim
            ylim(prefix + 'spectra_eu' + suffix,  2.0, 10000.0)
            # set zlim
            zlim(prefix + 'spectra_eu' + suffix,  1e-10, 1e-3)

        if prefix + 'spectra_ev' + suffix in loaded_data:
            # remove minus values in y array
            clip(prefix + 'spectra_ev' + suffix, 0., 5000.)
            # set ylim
            ylim(prefix + 'spectra_ev' + suffix,  2.0, 10000.0)
            # set zlim
            zlim(prefix + 'spectra_ev' + suffix,  1e-10, 1e-3)

        if prefix + 'spectra_bgamma' + suffix in loaded_data:
            # set ylim
            ylim(prefix + 'spectra_bgamma' + suffix, 2.0, 200.0)
            # set zlim
            zlim(prefix + 'spectra_bgamma' + suffix, 1e-4, 1e+2)

        if prefix + 'spectra_esum' + suffix in loaded_data:
            # set ylim
            ylim(prefix + 'spectra_esum' + suffix,  2.0, 10000.0)
            # set zlim
            zlim(prefix + 'spectra_esum' + suffix,  1e-10, 1e-3)

        if prefix + 'spectra_e_ar' + suffix in loaded_data:
            # set ylim
            ylim(prefix + 'spectra_e_ar' + suffix,  2.0, 10000.0)
            # set zlim
            zlim(prefix + 'spectra_e_ar' + suffix, -1, 1)

        # set y axis to logscale
        options(prefix + 'spectra_eu' + suffix, 'ylog', 1)
        options(prefix + 'spectra_ev' + suffix, 'ylog', 1)
        options(prefix + 'spectra_bgamma' + suffix, 'ylog', 1)
        options(prefix + 'spectra_esum' + suffix, 'ylog', 1)
        options(prefix + 'spectra_er' + suffix, 'ylog', 1)
        options(prefix + 'spectra_el' + suffix, 'ylog', 1)
        options(prefix + 'spectra_e_mix' + suffix, 'ylog', 1)
        options(prefix + 'spectra_e_ar' + suffix, 'ylog', 1)

        # set z axis to logscale
        options(prefix + 'spectra_eu' + suffix, 'zlog', 1)
        options(prefix + 'spectra_ev' + suffix, 'zlog', 1)
        options(prefix + 'spectra_bgamma' + suffix, 'zlog', 1)
        options(prefix + 'spectra_esum' + suffix, 'zlog', 1)
        options(prefix + 'spectra_er' + suffix, 'zlog', 1)
        options(prefix + 'spectra_el' + suffix, 'zlog', 1)
        options(prefix + 'spectra_e_mix' + suffix, 'zlog', 1)

        # set ytitle
        options(prefix + 'spectra_eu' + suffix, 'ytitle', 'ERG PWE/HFA (EU)')
        options(prefix + 'spectra_ev' + suffix, 'ytitle', 'ERG PWE/HFA (EV)')
        options(prefix + 'spectra_esum' + suffix,
                'ytitle', 'ERG PWE/HFA (ESUM)')
        options(prefix + 'spectra_e_ar' + suffix,
                'ytitle', 'ERG PWE/HFA (E_AR)')
        options(prefix + 'spectra_bgamma' + suffix,
                'ytitle', 'ERG PWE/HFA (BGAMMA)')

        # set ysubtitle
        options(prefix + 'spectra_eu' + suffix, 'ysubtitle', 'frequency [Hz]')
        options(prefix + 'spectra_ev' + suffix, 'ysubtitle', 'frequency [Hz]')
        options(prefix + 'spectra_esum' + suffix,
                'ysubtitle', 'frequency [Hz]')
        options(prefix + 'spectra_e_ar' + suffix,
                'ysubtitle', 'frequency [Hz]')
        options(prefix + 'spectra_bgamma' + suffix,
                'ysubtitle', 'frequency [Hz]')

        # set ztitle
        options(prefix + 'spectra_eu' + suffix, 'ztitle', 'mV^2/m^2/Hz')
        options(prefix + 'spectra_ev' + suffix, 'ztitle', 'mV^2/m^2/Hz')
        options(prefix + 'spectra_esum' + suffix, 'ztitle', 'mV^2/m^2/Hz')
        options(prefix + 'spectra_e_ar' + suffix, 'ztitle', 'LH:-1/RH:+1')
        options(prefix + 'spectra_bgamma' + suffix, 'ztitle', 'pT^2/Hz')

        # change colormap option
        options(prefix + 'spectra_eu' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_ev' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_bgamma' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_esum' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_er' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_el' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_e_mix' + suffix, 'Colormap', 'jet')
        options(prefix + 'spectra_e_ar' + suffix, 'Colormap', 'jet')

    elif level == 'l3':

        # set ytitle
        options(prefix + 'Fuhr' + suffix, 'ytitle', 'UHR frequency [Mhz]')
        options(prefix + 'ne_mgf' + suffix,
                'ytitle', 'eletctorn density [/cc]')

        # set y axis to logscale
        options(prefix + 'Fuhr' + suffix, 'ylog', 1)
        options(prefix + 'ne_mgf' + suffix, 'ylog', 1)

    return loaded_data
