from .load import load
from pyspedas.utilities.datasets import find_datasets

# This routine was moved out of __init__.py. Please see that file for previous revision history.


def mfi(trange=['2018-11-5', '2018-11-6'],
        datatype='h0',
        prefix='',
        suffix='',  
        get_support_data=False, 
        varformat=None,
        varnames=[],
        downloadonly=False,
        force_download=False,
        notplot=False,
        no_update=False,
        time_clip=False,
        addmaster=True):
    """
    Load data from the WIND Fluxgate Magnetometer
    
    Parameters
    ----------
        trange : list of str
            time range of interest [starttime, endtime] with the format 
            'YYYY-MM-DD','YYYY-MM-DD'] or to specify more or less than a day 
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
            Default: ['2018-11-5', '2018-11-6']

        datatype: str
            Data type; Valid options: 'h0', 'h1', 'h2', 'h3-rtn', 'h4-rtn', 'k0'
            Default: 'h0'

        prefix: str
            The tplot variable names will be given this prefix.
            Default: ''

        suffix: str
            The tplot variable names will be given this suffix.
            Default: ''

        get_support_data: bool
            Data with an attribute "VAR_TYPE" with a value of "support_data"
            will be loaded into tplot.
            Default: False

        varformat: str
            The file variable formats to load into tplot.  Wildcard character
            "*" is accepted.
            Default: None (all variables will be loaded)

        varnames: list of str
            List of variable names to load (if empty list or not specified,
            all data variables are loaded)
            Default: [] (all variables will be loaded)

        downloadonly: bool
            Set this flag to download the CDF files, but not load them into
            tplot variables
            Default: False

        force_download: bool
            Set this flag to download the CDF files, even if the local copy is newer.
            Default: False

        notplot: bool
            Return the data in hash tables instead of creating tplot variables
            Default: False

        no_update: bool
            If set, only load data from your local cache
            Default: False

        time_clip: bool
            Time clip the variables to exactly the range specified in the trange keyword
            Default: False

        addmaster: bool
            If True, use the metadata from a master CDF at SPDF rather than the metadata in the data file
            Default: True


    Returns
    ----------
        List of tplot variables created.

    Examples
    --------

    >>> import pyspedas
    >>> from pytplot import tplot
    >>> vars = pyspedas.wind.mfi(trange=['2018-11-5', '2018-11-6'],datatype='h0')
    >>> tplot(vars)

    """
    return load(instrument='fgm', trange=trange, datatype=datatype, prefix=prefix, suffix=suffix, get_support_data=get_support_data, varformat=varformat, varnames=varnames, downloadonly=downloadonly, force_download=force_download, notplot=notplot, time_clip=time_clip, no_update=no_update, addmaster=addmaster)

