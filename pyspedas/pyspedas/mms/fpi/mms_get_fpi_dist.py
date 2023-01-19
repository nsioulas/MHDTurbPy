
import logging
from copy import deepcopy
import numpy as np
from pyspedas import time_double
from pytplot import get_data

logging.captureWarnings(True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def mms_get_fpi_dist(tname, index=None, probe=None, data_rate=None, species=None, level='l2', single_time=None):
    """
    Returns 3D particle data structures containing MMS FPI
    data for use with SPEDAS particle routines. 
    
    Input
    ----------
        tname: str
            tplot variable name containing the FPI distribution data

    Parameters
    ----------
        probe: str
            Spacecraft probe #

        data_rate: str
            Instrument data rates for FPI include 'brst' and 'fast'

        species: str
            Species of the data specified by the 'tname' input
            (valid options: 'i' for ions and 'e' for electrons)

        index: int
            Index of time sample to return

    Returns
    ----------
        3D particle data structure(s) containing MMS FPI distribution functions
    """

    data_in = get_data(tname)

    if data_in is None:
        logging.error('Problem extracting the FPI distribution data.')
        return

    data = [0, 0, 0, 0, 0]

    if single_time is not None:
        index = np.searchsorted(data_in.times, time_double(single_time), side='left')

    if index is not None:
        data[0] = data_in[0][index]
        data[1] = data_in[1][index]
        if data_in[2].ndim == 1:
            data[2] = data_in[2]
        else:
            data[2] = data_in[2][index]
        if data_in[3].ndim == 1:
            data[3] = data_in[3]
        else:
            data[3] = data_in[3][index]
        if data_in[4].ndim == 1:
            data[4] = data_in[4]
        else:
            data[4] = data_in[4][index]
    else:
        data = data_in

    if species is None:
        species = tname.split('_')[1][1]

    if data_rate is None:
        data_rate = tname.split('_')[3]

    if species.lower() == 'i':
        mass = 1.04535e-2
        charge = 1.
        data_name = 'FPI Ion'
        if data_rate is not None and level is not None:
            if data_rate == 'brst' and level == 'l2':
                integ_time = 0.15
            if data_rate == 'brst' and level == 'acr':
                integ_time = 0.0375
        if data_rate == 'fast':
            integ_time = 4.5
    elif species.lower() == 'e':
        mass = 5.68566e-06
        charge = -1.
        data_name = 'FPI Electron'
        if data_rate is not None and level is not None:
            if data_rate == 'brst' and level == 'l2':
                integ_time = 0.03
            if data_rate == 'brst' and level == 'acr':
                integ_time = 0.0075
        if data_rate == 'fast':
            integ_time = 4.5
    else:
        logging.error('Invalid species: ' + species + '; valid options: "i" for ions and "e" for electrons')
        return

    out = {'project_name': 'MMS', 
           'spacecraft': probe, 
           'data_name': data_name, 
           'units_name': 'df_cm', 
           'units_procedure': 'mms_part_conv_units',
           'species': species,
           'valid': 1, # ???
           'charge': charge,
           'mass': mass}

    # the first dimension gets lost on single times
    if isinstance(data[0], np.ndarray) == False:
        data[1] = np.expand_dims(data[1], axis=0)
        data[0] = np.array([data[0]])

    # data[1] shape is: (32, 16, 32) (fast survey data)
    # which is: (phi, theta, energy)

    # we shuffle the output to be [time, energy, phi, theta]
    out_data = data[1].transpose([0, 3, 1, 2])
    out_energy = np.empty(data[1].shape).transpose([0, 3, 1, 2])
    out_theta = np.empty(data[1].shape).transpose([0, 3, 1, 2])
    out_phi = np.empty(data[1].shape).transpose([0, 3, 1, 2])
    out_bins = np.zeros(out_data.shape) + 1
    out_dphi = np.zeros(out_data.shape) + 11.25
    out_dtheta = np.zeros(out_data.shape) + 11.25
    out_denergy = np.zeros(out_data.shape)

    # elevations are constant across time
    # convert colat -> lat
    theta_reform = 90. - np.reshape(data[3], [1, 1, len(data[3])])

    # in the IDL code, we use reform to repeat the vector above
    # here, we'll do the same thing with np.repeat
    theta_rebin1 = np.repeat(theta_reform, data[1].shape[1], axis=0) # repeated across phi
    theta_rebinned = np.repeat(theta_rebin1, data[1].shape[3], axis=1) # repeated across energy

    # energies
    if data[4].ndim == 1:
        energy_reform = np.reshape(data[4], [len(data[4]), 1, 1])
        energy_rebin1 = np.repeat(energy_reform, len(data[3]), axis=2) # repeated across theta
        energy_table = np.repeat(energy_rebin1, len(data[2]), axis=1)
        energy_len = len(data[4])
    elif data[4].ndim == 2:
        energy_len = len(data[4][0])

    # phi
    if data[2].ndim == 1:
        phi_reform = np.reshape(data[2], [1, len(data[2]), 1])
        phi_rebin1 = np.repeat(phi_reform, energy_len, axis=0) # repeated across energy
        phi_rebinned = np.repeat(phi_rebin1, len(data[3]), axis=2)
        phi_len = len(data[2])
    elif data[2].ndim == 2:
        phi_len = len(data[2][0])

    for idx in range(len(data[0])):
        out_theta[idx] = theta_rebinned
        if data[4].ndim == 2: # time varying energy table
            energy_reform = np.reshape(data[4][idx], [energy_len, 1, 1])
            energy_rebin1 = np.repeat(energy_reform, len(data[3]), axis=2) # repeated across theta
            energy_table = np.repeat(energy_rebin1, phi_len, axis=1)
        out_energy[idx] = energy_table
        if data[2].ndim == 2:
            phi_reform = np.reshape(data[2][idx], [1, phi_len, 1])
            phi_rebin1 = np.repeat(phi_reform, energy_len, axis=0) # repeated across energy
            phi_rebinned = np.repeat(phi_rebin1, len(data[3]), axis=2)
        out_phi[idx] = phi_rebinned


    out_phi = (out_phi+180.) % 360
    out_theta = -out_theta

    out_list = []

    for time_idx, time in enumerate(data[0]):
        out_table = {**out}
        out_table['data'] = out_data[time_idx, :]
        out_table['bins'] = out_bins[time_idx, :]
        out_table['theta'] = out_theta[time_idx, :]
        out_table['phi'] = out_phi[time_idx, :]
        out_table['energy'] = out_energy[time_idx, :]
        out_table['dtheta'] = out_dtheta[time_idx, :]
        out_table['dphi'] = out_dphi[time_idx, :]
        out_table['denergy'] = out_denergy[time_idx, :]
        out_table['n_energy'] = energy_len
        out_table['n_theta'] = len(data[3])
        out_table['n_phi'] = phi_len
        out_table['start_time'] = time # note: assumes the FPI data weren't centered!
        out_table['end_time'] = time + integ_time
        out_list.append(out_table)

    return out_list
