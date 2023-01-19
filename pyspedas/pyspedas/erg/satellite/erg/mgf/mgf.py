import cdflib
import numpy as np
from pytplot import clip, get_data, options, ylim

from ..load import load


def mgf(trange=['2017-03-27', '2017-03-28'],
        datatype='8sec',
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
        ror=True,
        coord='dsi',
        version=None):
    """
    This function loads data from the MGF experiment from the Arase mission

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

        coord: str
            "sm", "dsi", "gse", "gsm", "sgi"

        version: str
            Set this value to specify the version of cdf files (such as "v03.03", "v03.04", ...)

    Returns:
        List of tplot variables created.

    """
    initial_notplot_flag = False
    if notplot:
        initial_notplot_flag = True

    if datatype == '8s' or datatype == '8':
        datatype = '8sec'
    elif datatype == '64':
        datatype = '64hz'
    elif datatype == '128':
        datatype = '128hz'
    elif datatype == '256':
        datatype = '256hz'

    prefix = 'erg_mgf_'+level+'_'
    if datatype == '8sec':
        file_res = 3600. * 24
        pathformat = 'satellite/erg/mgf/'+level+'/'+datatype + \
            '/%Y/%m/erg_mgf_'+level+'_'+datatype+'_%Y%m%d_'
    else:
        file_res = 3600.
        pathformat = 'satellite/erg/mgf/'+level+'/'+datatype + \
            '/%Y/%m/erg_mgf_'+level+'_'+datatype+'_' + coord + '_%Y%m%d%H_'

    if version is None:
        pathformat += 'v??.??.cdf'
    else:
        pathformat += version + '.cdf'

    loaded_data = load(pathformat=pathformat, file_res=file_res, trange=trange, level=level, datatype=datatype, prefix=prefix, suffix=suffix, get_support_data=get_support_data,
                       varformat=varformat, downloadonly=downloadonly, notplot=notplot, time_clip=time_clip, no_update=no_update, uname=uname, passwd=passwd)

    if (loaded_data is None) or (loaded_data == []):
        return loaded_data

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
            print(
                '**************************************************************************')
            print(gatt["LOGICAL_SOURCE_DESCRIPTION"])
            print('')
            print('Information about ERG MGF')
            print('')
            print('PI: ', gatt['PI_NAME'])
            print("Affiliation: "+gatt["PI_AFFILIATION"])
            print('')
            print('RoR of ERG project common: https://ergsc.isee.nagoya-u.ac.jp/data_info/rules_of_the_road.shtml.en')
            print(
                'RoR of MGF L2: https://ergsc.isee.nagoya-u.ac.jp/mw/index.php/ErgSat/Mgf')
            print('Contact: erg_mgf_info at isee.nagoya-u.ac.jp')
            print(
                '**************************************************************************')
        except:
            print('printing PI info and rules of the road was failed')

    if initial_notplot_flag or downloadonly:
        return loaded_data

    if datatype == '8sec':

        # remove -1.0e+30

        clip(prefix + 'mag_'+datatype+'_dsi'+suffix, -1e+6, 1e6)
        clip(prefix + 'mag_'+datatype+'_gse'+suffix, -1e+6, 1e6)
        clip(prefix + 'mag_'+datatype+'_gsm'+suffix, -1e+6, 1e6)
        clip(prefix + 'mag_'+datatype+'_sm'+suffix, -1e+6, 1e6)

        clip(prefix + 'magt_'+datatype+suffix, -1e+6, 1e6)

        clip(prefix + 'rmsd_'+datatype+'_dsi'+suffix, -1e+6, +1e+6)
        clip(prefix + 'rmsd_'+datatype+'_gse'+suffix, -1e+6, +1e+6)
        clip(prefix + 'rmsd_'+datatype+'_gsm'+suffix, -1e+6, +1e+6)
        clip(prefix + 'rmsd_'+datatype+'_sm'+suffix, -1e+6, +1e+6)

        clip(prefix + 'rmsd_'+datatype+suffix, 0., 80.)

        clip(prefix + 'dyn_rng_'+datatype+suffix, -120., +1e+6)

        clip(prefix + 'igrf_'+datatype+'_dsi'+suffix, -1e+6, +1e+6)
        clip(prefix + 'igrf_'+datatype+'_gse'+suffix, -1e+6, +1e+6)
        clip(prefix + 'igrf_'+datatype+'_gsm'+suffix, -1e+6, +1e+6)
        clip(prefix + 'igrf_'+datatype+'_sm'+suffix, -1e+6, +1e+6)

        # set yrange
        _, bdata = get_data(prefix + 'mag_'+datatype+'_dsi'+suffix)
        ylim(prefix + 'mag_'+datatype+'_dsi'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'mag_'+datatype+'_gse'+suffix)
        ylim(prefix + 'mag_'+datatype+'_gse'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'mag_'+datatype+'_gsm'+suffix)
        ylim(prefix + 'mag_'+datatype+'_gsm'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'mag_'+datatype+'_sm'+suffix)
        ylim(prefix + 'mag_'+datatype+'_sm'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))

        _, bdata = get_data(prefix + 'magt_'+datatype+suffix)
        ylim(prefix + 'magt_'+datatype+suffix,
             np.nanmin(bdata), np.nanmax(bdata))

        _, bdata = get_data(prefix + 'rmsd_'+datatype+suffix,)
        ylim(prefix + 'rmsd_'+datatype+suffix,
             np.nanmin(bdata), np.nanmax(bdata))

        _, bdata = get_data(prefix + 'rmsd_'+datatype+'_dsi'+suffix)
        ylim(prefix + 'rmsd_'+datatype+'_dsi'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'rmsd_'+datatype+'_gse'+suffix)
        ylim(prefix + 'rmsd_'+datatype+'_gse'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'rmsd_'+datatype+'_gsm'+suffix)
        ylim(prefix + 'rmsd_'+datatype+'_gsm'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'rmsd_'+datatype+'_sm'+suffix)
        ylim(prefix + 'rmsd_'+datatype+'_sm'+suffix,
             np.nanmin(bdata), np.nanmax(bdata))

        _, bdata = get_data(prefix + 'rmsd_'+datatype+suffix)
        ylim(prefix + 'rmsd_'+datatype+suffix,
             np.nanmin(bdata), np.nanmax(bdata))

        _, bdata = get_data(prefix + 'quality_'+datatype+suffix)
        ylim(prefix + 'quality_'+datatype+suffix,
             np.nanmin(bdata), np.nanmax(bdata))
        _, bdata = get_data(prefix + 'quality_'+datatype+'_gc'+suffix)
        ylim(prefix + 'quality_'+datatype+'_gc' +
             suffix, np.nanmin(bdata), np.nanmax(bdata))

        # set labels
        options(prefix + 'mag_'+datatype+'_dsi'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'mag_'+datatype+'_gse'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'mag_'+datatype+'_gsm'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'mag_'+datatype+'_sm'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])

        options(prefix + 'rmsd_'+datatype+'_dsi'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'rmsd_'+datatype+'_gse'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'rmsd_'+datatype+'_gsm'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'rmsd_'+datatype+'_sm'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])

        options(prefix + 'igrf_'+datatype+'_dsi'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'igrf_'+datatype+'_gse'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'igrf_'+datatype+'_gsm'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])
        options(prefix + 'igrf_'+datatype+'_sm'+suffix,
                'legend_names', ['Bx', 'By', 'Bz'])

        # set color of the labels
        options(prefix + 'mag_'+datatype+'_dsi' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'mag_'+datatype+'_gse' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'mag_'+datatype+'_gsm' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'mag_'+datatype+'_sm' +
                suffix, 'Color', ['b', 'g', 'r'])

        options(prefix + 'rmsd_'+datatype+'_dsi' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'rmsd_'+datatype+'_gse' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'rmsd_'+datatype+'_gsm' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'rmsd_'+datatype+'_sm' +
                suffix, 'Color', ['b', 'g', 'r'])

        options(prefix + 'quality_'+datatype+suffix, 'Color', ['r', 'g', 'b'])

        options(prefix + 'igrf_'+datatype+'_dsi' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'igrf_'+datatype+'_gse' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'igrf_'+datatype+'_gsm' +
                suffix, 'Color', ['b', 'g', 'r'])
        options(prefix + 'igrf_'+datatype+'_sm' +
                suffix, 'Color', ['b', 'g', 'r'])
    else:
        # remove -1.0e+30
        clip(prefix + 'mag_'+datatype+'_' + coord + suffix, -1e+6, 1e6)
        # set yrange
        _, bdata = get_data(prefix + 'mag_'+datatype+'_' + coord + suffix)
        ylim(prefix + 'mag_'+datatype+'_' + coord +
             suffix, np.nanmin(bdata), np.nanmax(bdata))
        # set labels
        options(prefix + 'mag_'+datatype+'_' + coord +
                suffix, 'legend_names', ['Bx', 'By', 'Bz'])
        # set color of the labels
        options(prefix + 'mag_'+datatype+'_' + coord +
                suffix, 'Color', ['b', 'g', 'r'])
    return loaded_data
