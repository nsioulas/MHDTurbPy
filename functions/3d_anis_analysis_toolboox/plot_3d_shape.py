from joblib import Parallel, delayed
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
from scipy.optimize import fsolve





# Make sure to use the local spedas
sys.path.insert(0, os.path.join(os.getcwd(), 'pyspedas'))
import pyspedas
from pyspedas.utilities import time_string
from pytplot import get_data


""" Import manual functions """

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import calc_diagnostics as calc
import TurbPy as turb
import general_functions as func
import three_D_funcs as threeD

sys.path.insert(1, os.path.join(os.getcwd(), 'functions','3d_anis_analysis_toolboox'))
import collect_wave_coeffs 











# Bin intervals!
from joblib import Parallel, delayed, parallel_backend
from warnings import filterwarnings

def process_files(fnames,
                   sf_names,
                   normalize_with_B2=False,
                   save_path=None,
                   n_jobs=-1):
    """
    Processes the given files and returns a pandas DataFrame containing the s-functions.

    Args:
        fnames (list of str)     : A list of file paths to the input data files.
        sf_names (list of str)   : A list of file paths to the precomputed structure functions.
        normalize_with_B2 (bool) : If True, normalizes the s-functions with B2. Defaults to False.
        save_path (str)          : The file path to save the resulting DataFrame. If None, the DataFrame is not saved.
        n_jobs (int)             : The number of parallel jobs to run. Defaults to -1 (using all available CPUs).

    Returns:
        A pandas DataFrame containing the s-functions.
    """
    def process_file(fname, sf_name):
        # Load file containing sfunctions
        res = pd.read_pickle(sf_name)

        # Load file containing corresponing plasma, mag field parameters
        if normalize_with_B2:
            fin = pd.read_pickle(fname)
            B_mean = np.nanmean(np.linalg.norm(fin["Mag"]["B_resampled"].values, axis=1))**2

        # Define keys
        keys = list(res["sfuncs"].keys())
        temp_df = pd.DataFrame({"l_di": res["l_di"]}) 

        for key in keys:
            if normalize_with_B2:
                temp_df[key] = res["sfuncs"][key][:, 0] / B_mean
            else:
                temp_df[key] = res["sfuncs"][key][:, 0]

        return temp_df

    with parallel_backend('threading'):
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(process_file)(fname, sf_name) for fname, sf_name in zip(fnames, sf_names)
        )

    total_df = pd.concat(dfs, ignore_index=True)

    if save_path is not None:
        if normalize_with_B2:
            func.savepickle(total_df, Path(save_path), 'normalized_initial_SFs.pkl')
        else:
            func.savepickle(total_df, Path(save_path), 'initial_SFs.pkl')

    return total_df



from joblib import Parallel, delayed, parallel_backend
from warnings import filterwarnings

def bin_data(total_df,
                      keys,
                      what,
                      normalize_with_B2,
                      std_or_error_of_mean,
                      mov_aver_window,
                      loglog,
                      save_path,
                      n_jobs=-1):
    """
    Processes the given DataFrame and returns a new DataFrame containing binned data in parallel.

    Args:
        total_df (pd.DataFrame): The input DataFrame containing the s-functions.
        keys (list of str): A list of column names in the input DataFrame to process.
        what (str): The method to use for binning. Can be 'median' (default), 'mean', or 'sum'.
        std_or_error_of_mean (int): The number of standard deviations or standard errors of the mean to include in the binning. Defaults to 1.
        mov_aver_window (int): The size of the moving average window to use for smoothing. Defaults to 50.
        loglog (bool): If True (default), uses logarithmic binning. Otherwise, uses linear binning.
        save_path (str): The path where to save the output DataFrame.
        n_jobs (int): The number of parallel jobs to run. Defaults to -1, which means using all available CPUs.

    Returns:
        A pandas DataFrame containing the binned data.
    """
    # Define a helper function to bin a single key
    def bin_key(key):
        y = total_df[key].values

        x_b, y_b, z_b = func.binned_quantity(x,
                                             y,
                                             what,
                                             std_or_error_of_mean,
                                             mov_aver_window,
                                             loglog,
                                             return_counts=False)

        return x_b, y_b, z_b

    # First load the data:
    if normalize_with_B2:
        total_df = pd.read_pickle('/Users/nokni/work/3d_anisotropy/structure_functions_E1/data/3d_shape/normalized_initial_SFs.pkl')
    else:
        total_df = pd.read_pickle('/Users/nokni/work/3d_anisotropy/structure_functions_E1/data/3d_shape/initial_SFs.pkl')

    keys = list(total_df.keys())[1:]
    df = pd.DataFrame()
    x = total_df['l_di'].values

    with parallel_backend('threading'):
        # Run the binning in parallel
        results = Parallel(n_jobs=n_jobs)(delayed(bin_key)(key) for key in keys)

    # Collect the results and store them in the output DataFrame
    for key, (x_b, y_b, z_b) in zip(keys, results):
        df[f'x{key}'] = x_b
        df[f'y{key}'] = y_b
        df[f'z{key}'] = z_b

        
    # Save everything!
    df = {
          'df'  : df, 
          'keys': list(total_df.keys())
    }
    
    if normalize_with_B2:
        func.savepickle(df, Path(save_path), 'nbins_'+str(mov_aver_window)+'_normalized_final_SFs.pkl')
    else:
        func.savepickle(df, Path(save_path),'nbins_'+str(mov_aver_window)+'_final_SFs.pkl' )

    return df



def process_y0(df,
               keys,
               keep_points,
               y0):
    
    
    points = find_3D_shape(df, keys, y0)
    keep_points.append(points)
    #x, y, z = points.T[0], points.T[1], points.T[2]
    
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)

    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min
    return keep_points, x_size, y_size, z_size


def find_nears(arr: np.ndarray, value: float) -> tuple:
    """
    Returns the indices of the two elements in the input array that are closest to a given value.

    Args:
        arr: The input array.
        value: The value to search for.

    Returns:
        A tuple containing the indices of the two nearest elements.
    """
    arr = np.asarray(arr)
    idx_k = np.argpartition(np.abs(arr - value), 2)[:2]
    return tuple(idx_k)


def lin_coefs(x1: float, x2: float, y1: float, y2: float) -> tuple:
    """
    Returns the coefficients of a linear equation given two points.

    Args:
        x1: The x-coordinate of the first point.
        x2: The x-coordinate of the second point.
        y1: The y-coordinate of the first point.
        y2: The y-coordinate of the second point.

    Returns:
        A tuple containing the slope and y-intercept of the linear equation.
    """
    coeffs = np.polyfit([x1, x2], [y1, y2], 1)
    return tuple(coeffs)


def find_ell(xb: np.ndarray, yb: np.ndarray, y0: float) -> float:
    """
    Returns the x-coordinate of the point on an ellipse with the given y-coordinate.

    Args:
        xb: An array of x-coordinates on the ellipse.
        yb: An array of y-coordinates on the ellipse.
        y0: The desired y-coordinate.

    Returns:
        The x-coordinate of the point on the ellipse with the given y-coordinate.
    """
    idx_1, idx_2 = find_nears(yb, y0)
    x1, x2       = xb[idx_1], xb[idx_2]
    y1, y2       = yb[idx_1], yb[idx_2]
    s, b         = lin_coefs(x1, x2, y1, y2)
    x0 = np.interp(y0, [y1, y2], [x1, x2])
    return x0


def polar2cart(r: float, theta: float, phi: float) -> np.ndarray:
    """
    Converts polar coordinates to Cartesian coordinates.

    Args:
        r: The radial distance from the origin.
        theta: The polar angle in degrees (measured from the positive z-axis).
        phi: The azimuthal angle in degrees (measured from the positive x-axis).

    Returns:
        A numpy array containing the Cartesian coordinates [x, y, z].
    """
    x = r * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y = r * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = r * np.cos(np.radians(theta))

    return np.array([x, y, z])


def symmetry(coords):
    """
    Generate a DataFrame containing all possible symmetry combinations
    of the input coordinates.

    :param coords: A DataFrame containing 'xs', 'ys', and 'zs' columns
                   representing the x, y, and z coordinates, respectively.
    :type coords: pd.DataFrame
    :return: A DataFrame containing all possible symmetry combinations
             of the input coordinates.
    :rtype: pd.DataFrame
    """
    total = pd.DataFrame()

    # Generate all possible sign combinations
    signs = [(sx, sy, sz) for sx in [+1, -1] for sy in [+1, -1] for sz in [+1, -1]]

    for sx, sy, sz in signs:
        temp = pd.DataFrame({
            'xs': sx * coords['xs'].values,
            'ys': sy * coords['ys'].values,
            'zs': sz * coords['zs'].values
        })
        total = pd.concat([total, temp], ignore_index=True)

    return total


def find_3D_shape(df, keys, y0):
    # Calculate l_di for each key
    l_di = [find_ell(df[f'x{key}'].values, df[f'y{key}'].values, y0) for key in keys]

    # Extract thetas and phis from keys
    thetas = [(int(key.split("_")[1]) + int(key.split("_")[2])) / 2 for key in keys]
    phis = [(int(key.split("_")[4]) + int(key.split("_")[5])) / 2 for key in keys]

    # Convert polar coordinates to cartesian coordinates
    coords = pd.DataFrame({
        'xs': [polar2cart(l_di[i], thetas[i], phis[i])[0] for i in range(len(keys))],
        'ys': [polar2cart(l_di[i], thetas[i], phis[i])[1] for i in range(len(keys))],
        'zs': [polar2cart(l_di[i], thetas[i], phis[i])[2] for i in range(len(keys))]
    })

    # Apply symmetry transformation to coordinates
    total = symmetry(coords)

    # Convert coordinates to a numpy array
    points = np.array([total['xs'].values, total['ys'].values, total['zs'].values]).T

    return points


def find_closest_values_in_arrays(arr_list,
                                  L_list,
                                  limited_window=False,
                                  xlims=None):
    """
    Find the closest target to a specified value in each of the arrays in `arr_list` by interpolating the values
    in `L_list`.

    Parameters
    ----------
    arr_list : list of arrays
        List of arrays of target values.
    L_list : list of arrays or array
        List of arrays or a single array of the independent variable values.
    limited_window : bool, optional
        If True, the results will only be returned if the ell value falls within the `xlims` range.
        Default is False.
    xlims : tuple, optional
        Tuple of lower and upper bounds for ell values. Only used if `limited_window` is True.
        Default is None.

    Returns
    -------
    result_df : pandas DataFrame
        DataFrame with columns for each array in `arr_list` and the corresponding ell value and target value.
        If a corresponding value is not found, the value will be set to NaN.

    """
 

    if type(L_list) != list:
        L_list = [L_list] * len(arr_list)
        
    indices = [np.where(arr > -1e10)[0].astype(int) for arr in arr_list]
    arr_list = [arr[idx] for arr, idx in zip(arr_list, indices)]
    L_list = [L[idx] for L, idx in zip(L_list, indices)]

    
        
    max_index = np.argmax([arr[0] for arr in arr_list])
    max_arr = arr_list[max_index]
    result_dict  = []
    for i in range(len(max_arr)):
        closest_indices = [np.argmin(np.abs(max_arr[i] - arr)) for j, arr in enumerate(arr_list)]
        closest_vals    = [arr[idx] for arr, idx in zip([arr for j, arr in enumerate(arr_list)], closest_indices)]
        ells = []
        targets = []
        for val, arr, L in zip(closest_vals, [arr for j, arr in enumerate(arr_list) ], L_list):
            idx = closest_indices[closest_vals.index(val)]
            if idx > 0 and idx < len(arr) - 1:
                ell = np.interp(val, [arr[idx - 1], arr[idx + 1]], [L[idx - 1], L[idx + 1]])
                if limited_window and (ell < xlims[0] or ell > xlims[1]):
                    continue
                ells.append(ell)
                target = np.interp(ell, L, arr)
                targets.append(target)
                
        final_dict = {}
        for jj in range(len(arr_list)):
            final_dict["ell_"+str(jj)]      = ells[jj] if jj < len(ells) else np.nan
            final_dict["target"+str(jj)]    = targets[jj] if jj < len(targets) else np.nan
        result_dict.append(final_dict)

    return pd.DataFrame(result_dict)


from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def find_closest_values_in_arrays_new(arr_list, L_list, identifiers, limited_window=False, xlims=None, num_points=10):
    if type(L_list) != list:
        L_list = [L_list] * len(arr_list)

    result_dict = []
    for i, ya in enumerate(arr_list[0]):
        xa = L_list[0][i]
        
        # Interpolate for the second array
        interp_func = interp1d(arr_list[1], L_list[1], kind='linear', fill_value='extrapolate')
        xb_exact = interp_func(ya)
        
        final_dict = {}
        final_dict["x_"+str(identifiers[0])] = xa
        final_dict["x_"+str(identifiers[1])] = xb_exact

        result_dict.append(final_dict)

    return pd.DataFrame(result_dict)

# def find_closest_values_in_arrays_new(arr_list, L_list, identifiers, limited_window=False, xlims=None, num_points=10):
#     """
#     Find the x values in `L_list` at which all arrays in `arr_list` have the same target value.

#     Parameters
#     ----------
#     arr_list : list of arrays
#         List of arrays of target values.
#     L_list : list of arrays or array
#         List of arrays or a single array of the independent variable values.
#     identifiers : list of str
#         List of names for the variables.
#     limited_window : bool, optional
#         If True, the results will only be returned if the ell value falls within the `xlims` range.
#         Default is False.
#     xlims : tuple, optional
#         Tuple of lower and upper bounds for ell values. Only used if `limited_window` is True.
#         Default is None.
#     num_points : int, optional
#         Number of intermediate points to add between existing data points.
#         Default is 10.

#     Returns
#     -------
#     result_df : pandas DataFrame
#         DataFrame with columns for each array in `arr_list` and the corresponding x value and target value.
#         If a corresponding value is not found, the value will be set to NaN.
#     """

#     if type(L_list) != list:
#         L_list = [L_list] * len(arr_list)

#     result_dict = []
#     for i, ya in enumerate(arr_list[0]):
#         xa = L_list[0][i]
#         xb = None
#         yb = None
#         ratio = None
        
#         # Search for closest y value in yb
#         min_distance = np.inf
#         for j, y in enumerate(arr_list[1]):
#             if np.abs(ya - y) < min_distance:
#                 min_distance = np.abs(ya - y)
#                 yb = y
#                 xb = L_list[1][j]
        
#         # Interpolate for exact x value
#         if xb is not None and yb is not None:
#             x_vals = np.linspace(ya, yb, num_points+2)[1:-1]
#             y_vals = np.linspace(xa, xb, num_points+2)[1:-1]
#             interp_func = interp1d([ya, *x_vals, yb], [xa, *y_vals, xb], kind='linear', fill_value='extrapolate')
#             xb_exact = interp_func(yb)
#             ratio = yb/ya
        
#             final_dict = {}
#             final_dict["x_"+str(identifiers[0])] = xa
#             final_dict["x_"+str(identifiers[1])] = xb_exact
#             final_dict["ratio"] = ratio

#             result_dict.append(final_dict)

#     return pd.DataFrame(result_dict)


def estimate_wave_anisotropy(identif,
                             keep_all,
                             what,
                             std_or_error_of_mean,
                             mov_aver_window,
                             loglog,
                             new_method=False):

    w_aniso = {}
    for N in range(len(keep_all[identif[0]]['anis_anal']['xvals'])):
        #print(N)
        try:

            x_2, y_2          = keep_all[identif[1]]['anis_anal']['xvals'][N], keep_all[identif[1]]['anis_anal']['yvals'][N]
            x_1, y_1          =  keep_all[identif[0]]['anis_anal']['xvals'][N], keep_all[identif[0]]['anis_anal']['yvals'][N]
            sig_c             = keep_all['ell_perp']['anis_anal']['sig_c'][N]

            
            if new_method:
                res = {}
                keep_x1 = []
                keep_x2 = []
                for  (x_vval2, y_vval2) in zip(x_2, y_2):
                    
                    keep_x1.append(find_ell(x_1, y_1, y_vval2))
                    keep_x2.append(x_vval2)
                    
                res['x_'+str(identif[0])] = keep_x1
                res['x_'+str(identif[1])] = keep_x2
                res =pd.DataFrame(res)


                
            else:
                res               = find_closest_values_in_arrays_new([y_1, y_2],
                                                                     [x_1, x_2],
                                                                     identif,
                                                                     limited_window = False,
                                                                     xlims          = None)

            w_aniso[str(N)] = {
                               str('x_'+str(identif[0])): res['x_'+str(identif[0])].values,
                               str('x_'+str(identif[1])): res['x_'+str(identif[1])].values, 
                               'sig_c'                  : sig_c
                              }

        except:
            traceback.print_exc()

    # Create the dataframe
    w_aniso = pd.DataFrame(w_aniso).T.sort_values('sig_c')

    # Stack  individual lists
    xvals   = np.hstack(w_aniso['x_'+str(identif[0])].values)
    yvals   = np.hstack(np.hstack(w_aniso['x_'+str(identif[1])].values))
    
   # print(xvals)
    #print(yvals)
    
    # Estimate binned 
    keep_indices                                  = (xvals>0) & (yvals>0)
    binned_quant                                  = func.binned_quantity(xvals[keep_indices],
                                                                         yvals[keep_indices],
                                                                         what,
                                                                         std_or_error_of_mean,
                                                                         mov_aver_window,
                                                                         loglog)

    return w_aniso, binned_quant