from .load import load
from typing import List, Union

# This routine was moved out of __init__.py.  Please see that file for previous revision history.

def hiscale(trange:List[str]=['2003-01-01', '2003-01-02'],
        datatype:str='lmde_m1',
        prefix:str='',
        suffix:str='',
        get_support_data:bool=False,
        varformat:str=None,
        varnames:List[str]=[],
        downloadonly:bool=False,
        force_download:bool=False,
        notplot:bool=False,
        no_update:bool=False,
        time_clip:bool=True) -> List[str]:
    """
    This function loads data from the HI-SCALE experiment from the Ulysses mission
    
    Parameters
    ----------
        trange : list of str
            time range of interest [starttime, endtime] with the format 
            ['YYYY-MM-DD','YYYY-MM-DD'] or to specify more or less than a day
            ['YYYY-MM-DD/hh:mm:ss','YYYY-MM-DD/hh:mm:ss']
            Default: ['2003-01-01', '2003-01-02']

        datatype: str
            Data type; Valid options: 'lf15_m1', 'lf60_m1', 'lm12_m1', 'lm30_m1', 'lmde_m1', 'wart_m1', 'wrtd_m1'
            Default: 'lmde_m1'

        prefix: str
            The tplot variable names will be given this prefix.
            Default: ''

        suffix: str
            The tplot variable names will be given this suffix.
            Default: ''

        get_support_data: bool
            If True, data with an attribute "VAR_TYPE" with a value of "support_data"
            will be loaded into tplot.
            Default: False

        varformat: str
            The file variable formats to load into tplot.  Wildcard character
            "*" is accepted.
            Default: '' (all variables loaded)

        varnames: list of str
            List of variable names to load (If empty list or not specified,
            all data variables are loaded)
            Default: []

        downloadonly: bool
            Set this flag to download the CDF files, but not load them into
            tplot variables
            Default: False

        force_download: bool
            Set this flag to download the CDF files, even if local file is newer
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

    Returns
    -------
        list of str
            List of tplot variables created.

    Examples
    --------
    >>> import pyspedas
    >>> from pytplot import tplot
    >>> hiscale_vars = pyspedas.ulysses.hiscale()
    >>> tplot('Electrons')
    """
    return load(instrument='hiscale', trange=trange, datatype=datatype, prefix=prefix, suffix=suffix, get_support_data=get_support_data, varformat=varformat, varnames=varnames, downloadonly=downloadonly, force_download=force_download, notplot=notplot, time_clip=time_clip, no_update=no_update)

