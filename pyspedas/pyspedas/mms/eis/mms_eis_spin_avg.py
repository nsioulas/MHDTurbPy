
import logging
import warnings
import numpy as np
from pytplot import get_data, store_data, options
from pyspedas import tnames

# use nanmean from bottleneck if it's installed, otherwise use the numpy one
# bottleneck nanmean is ~2.5x faster
try:
    import bottleneck as bn
    nanmean = bn.nanmean
except ImportError:
    nanmean = np.nanmean

logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

def mms_eis_spin_avg(probe='1', species='proton', data_units='flux', datatype='extof', data_rate='srvy', level='l2', suffix=''):
    """
    This function will spin-average the EIS spectrograms, and is automatically called from mms_load_eis
    
    Parameters
    ----------
        probe: str
            probe #, e.g., '4' for MMS4

        species: str
            species for calculation (default: 'proton')
            
        data_units: str
            'flux' or 'cps' (default: 'flux')

        datatype: str
            'extof' or 'phxtof' (default: 'extof')

        data_rate: str
            instrument data rate, e.g., 'srvy' or 'brst' (default: 'srvy')

        level: str
            data level ['l1a','l1b','l2pre','l2' (default)]

        suffix: str
            suffix of the loaded data

    Returns:
        List of tplot variables created.
    """
    prefix = 'mms' + probe + '_epd_eis_' + data_rate + '_' + level + '_'

    if data_units == 'flux':
        units_label = '1/(cm^2-sr-s-keV)'
    elif data_units == 'cps':
        units_label = '1/s'
    elif data_units == 'counts':
        units_label = 'counts'

    spin_data = get_data(prefix + datatype + '_spin' + suffix)
    if spin_data is None:
        logging.error('Error, problem finding EIS spin variable to calculate spin-averages')
        return

    spin_times, spin_nums = spin_data

    if spin_nums is not None:
        spin_starts = [spin_start for spin_start in np.where(spin_nums[1:] > spin_nums[:-1])[0]]

        telescopes = tnames(prefix + datatype + '_' + species + '_*' + data_units + '_t?' + suffix)

        if len(telescopes) != 6:
            logging.error('Problem calculating the spin-average for species: ' + species + ' (' + datatype + ')')
            return None

        out_vars = []

        for scope in range(0, 6):
            this_scope = telescopes[scope]
            scope_data = get_data(this_scope)
            
            if len(scope_data) <= 2:
                logging.error("Error, couldn't find energy table for the variable: " + this_scope)
                continue

            flux_times, flux_data, energies = scope_data

            spin_avg_flux = np.zeros([len(spin_starts), len(energies)])

            current_start = 0

            for spin_idx in range(0, len(spin_starts)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    spin_avg_flux[spin_idx, :] = nanmean(flux_data[current_start:spin_starts[spin_idx]+1, :], axis=0)
                current_start = spin_starts[spin_idx] + 1

            store_data(this_scope + '_spin', data={'x': flux_times[spin_starts], 'y': spin_avg_flux, 'v': energies})
            options(this_scope + '_spin', 'ztitle', units_label)
            options(this_scope + '_spin', 'ytitle', 'MMS' + probe + ' ' + datatype + ' ' + species + ' (spin)')
            options(this_scope + '_spin', 'ysubtitle', 'Energy [keV]')
            options(this_scope + '_spin', 'spec', True)
            options(this_scope + '_spin', 'ylog', True)
            options(this_scope + '_spin', 'zlog', True)
            out_vars.append(this_scope + '_spin')
        return out_vars
    else:
        logging.error('Error, problem finding EIS spin variable to calculate spin-averages')
        return None