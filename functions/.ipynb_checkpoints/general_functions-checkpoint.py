import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from datetime import datetime
from pathlib import Path
import pickle
from scipy import stats
import numba
from numba import jit, njit, prange, objmode
from scipy.optimize import curve_fit
import joblib
from joblib import Parallel, delayed
import statistics
from statistics import mode
import orderedstructs
import sys
import pytplot

import warnings
warnings.filterwarnings('ignore')


# Import urbPy
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))


from plasma_params import*
import signal_processing 



def create_folder_if_not_exists(path0, overwrite_files=False):
    folder_path = Path(path0)
    
    # Check if folder exists or overwrite_files is True
    if not folder_path.exists() or overwrite_files:
        folder_path.mkdir( exist_ok=True)
        print(f"Folder created or overwritten: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")

from collections.abc import Iterable

def ensure_iterable(obj):
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        return obj
    else:
        return [obj]


import orderedstructs

def savepickle_dill(df_2_save, save_path, filename):
    file_path = Path(save_path).joinpath(filename)
    with open(file_path, 'wb') as file:
        pickle.dump(df_2_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
import dill

def load_and_construct_lambdas(save_path, fname):
    """
    Load the coefficients and construct lambda functions.
    
    Parameters:
    save_path (str): Directory path where the file is saved.
    fname (str): Filename of the saved file.
    
    Returns:
    dict: Dictionary containing the reconstructed lambda functions.
    """
    # Full path to the file
    full_path = os.path.join(save_path, fname)

    # Load the coefficients
    with open(full_path, 'rb') as file:
        coefficients_dict = dill.load(file)

    # Reconstruct the lambda functions
    f_dict = {}
    for key, coeffs in coefficients_dict.items():
        f_dict[key] = lambda x, c=coeffs: sum(c_i * (x ** i) for i, c_i in enumerate(c[::-1]))

    return f_dict


def format_datetime_to_string(numpy_datetime):
    """
    Converts a numpy.datetime64 object to a string in 'YYYY-MM-DD HH:MM' format.
    
    Parameters:
    numpy_datetime (numpy.datetime64): The input datetime object.
    
    Returns:
    str: Formatted datetime string.
    """
    # Convert to datetime object
    datetime_obj = numpy_datetime.astype('datetime64[s]').tolist()
    
    # Convert to the desired string format: 'YYYY-MM-DD HH:MM'
    formatted_time = datetime_obj.strftime('%Y-%m-%d %H:%M')
    
    return formatted_time


def estimate_derivatives(x, y):
    """
    Estimate the first and second derivatives of a function using central differences.
    
    :param x: An array of x values.
    :param y: An array of y values.
    :return: Arrays of the first and second derivatives of y.
    """
    # First derivative (dy/dx)
    dy = np.gradient(y, x, edge_order=2)
    
    # Second derivative (d^2y/dx^2)
    d2y = np.gradient(dy, x, edge_order=2)
    
    return dy, d2y

def compute_curvature(x, y):
    """
    Compute the curvature of a function.
    
    :param x: An array of x values.
    :param y: An array of y values.
    :return: An array of curvature values.
    """
    dy, d2y = estimate_derivatives(x, y)
    
    # Curvature formula
    curvature = np.abs(d2y) / (1 + dy**2)**1.5
    
    return curvature


def tplot_to_dataframe(file_path, 
                       var_name=None, 
                       convert_time_to_datetime=True,
                       time_unit='s'):
    """
    Restore a TPlot file (IDL .tplot or .sav with TPlot variables) using pytplot
    and convert a specified TPlot variable into a pandas DataFrame with a DateTimeIndex.
    
    Parameters
    ----------
    file_path : str
        Full path to the TPlot save file (e.g. '.tplot' or '.sav').
    var_name : str, optional
        Name of the TPlot variable inside the file. If None, and the file
        contains exactly one variable, we use that one. Otherwise, this must be specified.
    convert_time_to_datetime : bool, optional
        If True, converts numeric time (often seconds since 1970) to a DateTimeIndex.
        If False, leaves time as numeric values.
    time_unit : str, optional
        If converting time to DateTimeIndex, the unit of the time array. Typically 's' 
        for seconds. Options include 'ms', 'ns', etc., depending on your data.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame indexed by time (either numeric or a DateTimeIndex),
        with one or more columns for the TPlot variable data.

    Raises
    ------
    FileNotFoundError
        If the file_path does not exist on disk.
    ValueError
        If the file contains no TPlot variables, or if var_name is specified but not found,
        or if multiple variables exist and var_name is not specified.
    """

    # 1. Check if the file exists on disk
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 2. Attempt to restore the TPlot file
    #    If this fails due to the file not being a valid TPlot file,
    #    pytplot might print a warning or raise an error, and no variables will load.
    pytplot.tplot_restore(file_path)

    # 3. See which TPlot variables were loaded
    all_vars = pytplot.tplot_names()

    if len(all_vars) == 0:
        # Means no TPlot variables recognized from this file
        raise ValueError(
            f"No TPlot variables found in file: {file_path}\n"
            f"Check if it's really a TPlot save file created with IDL tplot_save()."
        )

    # If user didn't specify var_name, auto-pick if there's exactly one
    if var_name is None:
        if len(all_vars) == 1:
            var_name = all_vars[0]
            print(f"Auto-selected single TPlot variable: {var_name}")
        else:
            raise ValueError(
                f"Multiple TPlot variables found. Please specify var_name.\n"
                f"Available variables: {all_vars}"
            )
    else:
        if var_name not in all_vars:
            raise ValueError(
                f"Requested var_name='{var_name}' not in loaded TPlot variables: {all_vars}"
            )

    # 4. Retrieve data for the chosen variable
    result = pytplot.get_data(var_name)
    if result is None:
        raise ValueError(
            f"Could not retrieve data for TPlot variable '{var_name}'. "
            "Possibly no valid data in the file."
        )

    # get_data can return either (time, data) or (time, data, metadata)
    if len(result) == 2:
        time_vals, data_vals = result
    elif len(result) == 3:
        time_vals, data_vals, _ = result
    else:
        raise ValueError("Unexpected format from pytplot.get_data().")

    # 5. Convert time array if requested
    if convert_time_to_datetime:
        # By default, TPlot times are often in seconds since 1970.
        time_index = pd.to_datetime(time_vals, unit=time_unit, origin='unix')
    else:
        # Leave time as numeric
        time_index = pd.Index(time_vals, name='time')

    # 6. Wrap in a DataFrame
    if data_vals.ndim == 1:
        # 1D data => single column named after var_name
        df = pd.DataFrame(data_vals, index=time_index, columns=[var_name])
    else:
        # 2D or more => multiple columns
        # Name each column var_name_0, var_name_1, etc.
        col_count = data_vals.shape[1]
        col_names = [f"{var_name}_{i}" for i in range(col_count)]
        df = pd.DataFrame(data_vals, index=time_index, columns=col_names)

    return df



# def synchronize_dfs(df_higher_freq, df_lower_freq, upsample=True, 
#                    order_up=3, order_down=5, percentage=1.15, interp_method='linear'):
#     """
#     Align two DataFrames based on their frequency by either upsampling the lower frequency DataFrame 
#     or downsampling the higher frequency DataFrame.

#     Args:
#         df_higher_freq: [pandas DataFrame] DataFrame with higher frequency data.
#         df_lower_freq: [pandas DataFrame] DataFrame with lower frequency data.
#         upsample: [bool] If True, upsample the lower frequency DataFrame; otherwise, downsample the higher frequency one.
#         order_up: [int] Order of the Butterworth filter for upsampling.
#         order_down: [int] Order of the Butterworth filter for downsampling.
#         percentage: [float] Multiplier for the cutoff frequency during downsampling.
#         interp_method: [str] Interpolation method for reindexing.

#     Returns:
#         Tuple[pandas DataFrame, pandas DataFrame]: Aligned DataFrames (high_freq_df, low_freq_df).
#     """
    

#     if overlapping_start >= overlapping_end:
#         raise ValueError("No overlapping time range between high_df and low_df.")

#     # Trim both DataFrames to the overlapping range before any processing

#     if upsample:
#         # Upsample the lower frequency DataFrame to match the higher frequency one
#         aligned_lower_freq = signal_processing.upsample_and_filter(
#             low_df         = df_lower_freq, 
#             high_df        = df_higher_freq, 
#             order          = order_up, 
#             interp_method  = interp_method
#         )
#         # Ensure both DataFrames cover the exact same time range after synchronization
#         aligned_high_freq = df_higher_freq.loc[aligned_lower_freq.index]
#         return aligned_high_freq, aligned_lower_freq

#     else:
#         # Downsample the higher frequency DataFrame to match the lower frequency one
#         aligned_higher_freq  = signal_processing.downsample_and_filter(
#             high_df          = df_higher_freq, 
#             low_df           = df_lower_freq, 
#             order            = order_down, 
#             percentage       = percentage
#         )
        
#         # Ensure both DataFrames cover the exact same time range after synchronization
#         aligned_low_freq = df_lower_freq.loc[aligned_higher_freq.index]
#         return aligned_higher_freq, aligned_low_freq



def synchronize_dfs(df_higher_freq, df_lower_freq, upsample):
    """
    Align two dataframes based on their frequency, upsample lower frequency 
    dataframe if specified, otherwise downsample the higher frequency one.

    In the end, we ensure that we return two DataFrames that:
      1. Are strictly in the overlapping time range,
      2. Have no NaNs at the beginning or end (trimmed away),
      3. Are time-interpolated (so small internal gaps are filled).
    """
    if upsample:
        # 1) Upsample lower frequency to match higher frequency
        aligned_lower_freq = signal_processing.upsample_dataframe(df_lower_freq, df_higher_freq)

        # 2) Overlap clamp
        overlapping_start = max(df_higher_freq.index.min(), aligned_lower_freq.index.min())
        overlapping_end   = min(df_higher_freq.index.max(), aligned_lower_freq.index.max())

        df_higher_freq     = df_higher_freq.loc[overlapping_start:overlapping_end]
        aligned_lower_freq = aligned_lower_freq.loc[overlapping_start:overlapping_end]

        # 3) Interpolate both to fill small gaps
        df_higher_freq     = df_higher_freq.interpolate(method='time')
        aligned_lower_freq = aligned_lower_freq.interpolate(method='time')

        # 4) Remove leading/trailing NaNs from both

        # (a) Find first/last valid index in each
        fvi_high = df_higher_freq.first_valid_index()
        lvi_high = df_higher_freq.last_valid_index()

        fvi_low  = aligned_lower_freq.first_valid_index()
        lvi_low  = aligned_lower_freq.last_valid_index()

        # If either is entirely NaN, just return empty slices
        if fvi_high is None or fvi_low is None:
            return (df_higher_freq.iloc[0:0], aligned_lower_freq.iloc[0:0])

        # (b) Take the maximum of first_valid_indices => start
        new_start = max(fvi_high, fvi_low)
        # (c) Take the minimum of last_valid_indices  => end
        new_end   = min(lvi_high, lvi_low)

        # (d) Trim both DataFrames
        df_higher_freq     = df_higher_freq.loc[new_start:new_end]
        aligned_lower_freq = aligned_lower_freq.loc[new_start:new_end]

        return df_higher_freq, aligned_lower_freq

    else:
        # 1) Downsample higher frequency to match lower frequency
        try:
            # Interpolate + dropna inside, if needed (unchanged)
            aligned_higher_freq = signal_processing.downsample_and_filter(
                df_higher_freq.interpolate().dropna(),
                df_lower_freq.interpolate().dropna()
            )
            
            # 2) Overlap clamp
            overlapping_start = max(aligned_higher_freq.index.min(), df_lower_freq.index.min())
            overlapping_end   = min(aligned_higher_freq.index.max(), df_lower_freq.index.max())

            aligned_higher_freq = aligned_higher_freq.loc[overlapping_start:overlapping_end]
            df_lower_freq       = df_lower_freq.loc[overlapping_start:overlapping_end]

            # 3) Interpolate both (fills small internal gaps)
            aligned_higher_freq = aligned_higher_freq.interpolate(method='time')
            df_lower_freq       = df_lower_freq.interpolate(method='time')

            # 4) Remove leading/trailing NaNs from both
            fvi_high = aligned_higher_freq.first_valid_index()
            lvi_high = aligned_higher_freq.last_valid_index()

            fvi_low  = df_lower_freq.first_valid_index()
            lvi_low  = df_lower_freq.last_valid_index()

            if fvi_high is None or fvi_low is None:
                return (aligned_higher_freq.iloc[0:0], df_lower_freq.iloc[0:0])

            new_start = max(fvi_high, fvi_low)
            new_end   = min(lvi_high, lvi_low)

            aligned_higher_freq = aligned_higher_freq.loc[new_start:new_end]
            df_lower_freq       = df_lower_freq.loc[new_start:new_end]

            return aligned_higher_freq, df_lower_freq

        except Exception as e:
            print(f'Error aligning dataframes: {e}')
            # Optionally handle error or return the original dataframes
            return df_higher_freq, df_lower_freq


def clean_data(x, y):
    """Remove non-finite values from the data."""
    finite_mask = np.isfinite(x) & np.isfinite(y)
    return x[finite_mask], y[finite_mask]



def symlogspace(start, end, num=50, linthresh=1):
    """
    Generate a symmetric logarithmic scale array.

    Parameters:
    - start, end: The starting and ending values of the sequence.
    - num: Number of samples to generate.
    - linthresh: The range within which the plot is linear (to avoid having a zero value).

    Returns:
    - ndarray
    """

    if start * end > 0:
        raise ValueError("Start and end values must have different signs for symlogspace to be meaningful.")

    # Divide the number of bins to account for both negative and positive values.
    num_half = num // 2

    # Create the logarithmic spaces for negative and positive values.
    log_neg = np.logspace(np.log10(linthresh), np.log10(abs(start)), num_half)
    log_pos = np.logspace(np.log10(linthresh), np.log10(end), num_half)

    # Combine the negative and positive logarithmic spaces.
    return np.concatenate((-log_neg[::-1], log_pos))

def most_common(List):
    return(mode(List))

def load_files(load_path, filenames, conect_2= '', sort= True):
    import glob
    
    pattern = Path(load_path, '*', conect_2, filenames)
    print(pattern)
    if sort:
        fnames = np.sort(glob.glob(str(pattern)))
    else:
        
        fnames = glob.glob(str(pattern))      
    
    return fnames


@njit(parallel=True)
def custom_nansum_product(xvec, yvec, axis):
    result = np.zeros(xvec.shape[1-axis], dtype=xvec.dtype)
    # Parallelizing the outer loop
    for j in prange(xvec.shape[1-axis]):
        for i in range(xvec.shape[axis]):
            if axis == 0:
                if not np.isnan(xvec[i, j]) and not np.isnan(yvec[i, j]):
                    result[j] += xvec[i, j] * yvec[i, j]
            else:
                if not np.isnan(xvec[j, i]) and not np.isnan(yvec[j, i]):
                    result[j] += xvec[j, i] * yvec[j, i]
    return result



def find_matching_files_with_common_parent(f_names,
                                           f_file_name,
                                           gen_names,
                                           gen_file_name,
                                           num_parents_f=1,
                                           num_parents_g=1):
    
    gen_parent = [Path(gen_name).parents[num_parents_g-1] for gen_name in gen_names]
    f_parents  = [Path(f_name).parents[num_parents_f-1] for f_name in f_names]
    
    
    parents    = list(set(gen_parent).intersection(f_parents))
    
    f_names    = [Path(parent).joinpath( f_file_name) for parent in parents]
    gen_names  = [Path(parent).joinpath( gen_file_name) for parent in parents]

    return list(np.sort(np.array(f_names).astype(str))),  list(np.sort(np.array(gen_names).astype(str)))



def delete_files_and_folders(file_and_folder_list):
    import shutil
    from pathlib import Path
    for fname in file_and_folder_list:
        try:
            if Path(fname).is_file():
                Path(fname).unlink()
                print(f"Deleted file: {fname}")
            elif Path(fname).is_dir():
                shutil.rmtree(fname)
                print(f"Deleted directory: {fname}")
        except Exception as e:
            print(f"Error deleting {fname}: {e}")


def generate_date_range_df(Start_date, 
                           End_date, 
                           step,
                           step2):
    """
    Generate a DataFrame with a date range.

    Args:
        Start_date (str): The starting date in the format 'YYYY-MM-DD'.
        End_date (str): The ending date in the format 'YYYY-MM-DD'.
        step (int): The number of days in each interval.
        step2 (int): The number of days to subtract from the previous 'End_date'.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'Starting_date' and 'Ending_date'.
                      Both columns contain Timestamp objects representing date intervals.

    Example:
        >>> start_date = '2023-08-01'
        >>> end_date = '2023-08-15'
        >>> step = 3
        >>> step2 = 1
        >>> result_df = generate_date_range_df(start_date, end_date, step, step2)
        >>> print(result_df)
    """
    from datetime import datetime, timedelta
    start_datetime  = datetime.strptime(Start_date, '%Y-%m-%d')
    end_datetime    = datetime.strptime(End_date, '%Y-%m-%d')
    step_timedelta  = timedelta(days=step)
    step2_timedelta = timedelta(days=step2)

    dates = []
    while start_datetime < end_datetime:
        end_of_range = start_datetime + step_timedelta
        dates.append((start_datetime, end_of_range))
        start_datetime = end_of_range - step2_timedelta  # Subtract step2
        
    df = pd.DataFrame(dates, columns=['Start', 'End'])
    df['Start'] = pd.to_datetime(df['Start'])  # Convert to Timestamp
    df['End'] = pd.to_datetime(df['End'])      # Convert to Timestamp
    return df



import numpy as np
from numba import njit, prange





# Original function
def angle_between_vectors(V,
                          B,
                          return_denom  = False,
                          restrict_2_90 = False):
                    
    """
    Calculate the angle between two vectors.

    Args:
        V (np.ndarray)                : A 2D numpy array representing the first vector.
        B (np.ndarray)                : A 2D numpy array representing the second vector.
        return_denom (bool, optional) : Whether to return the denominator components.
        restrict_2_90(bool, optional) : Restrict angles to 0-90
            Defaults to False.
            
    Returns:
        np.ndarray                    : A 1D numpy array representing the angles in degrees between the two input vectors.
        tuple                         : A tuple containing the angle, dot product, V_norm, and B_norm (if denom is True).
    """
    
    V_norm      = estimate_vec_magnitude(V)
    B_norm      = estimate_vec_magnitude(B)
    dot_prod    = dot_product(V, B)

    if restrict_2_90:
        angle       = np.arccos(np.abs(dot_prod) / (V_norm * B_norm)) / np.pi * 180
    else:
        angle       = np.arccos(dot_prod / (V_norm * B_norm)) / np.pi * 180       
        
    if return_denom:
        return angle, dot_prod, V_norm, B_norm
    else:
        return angle
    
    
def dot_product(xvec, yvec ):
    # Determine which axis is shorter
    axis_to_sum = 1 if xvec.shape[0] > xvec.shape[1] else 0

    # Calculate the product of the two arrays, handling NaNs effectively

    # Sum along the determined shorter axis, skipping NaNs
    result = np.nansum(xvec * yvec, axis=axis_to_sum)
    
    return result

def estimate_vec_magnitude(a):
    """
    Estimate the magnitude of each vector in the input array `a`.

    Parameters:
    ----------
    a : numpy.ndarray
        The input array containing vectors. The shape of `a` should be (N, M) or (M, N), where N is the number of vectors,
        and M is the dimensionality of each vector.

    Returns:
    -------
    numpy.ndarray
        An array containing the magnitude of each vector in `a`. The output array will have shape (N,) if the input
        array `a` has shape (N, M), or shape (M,) if the input array `a` has shape (M, N).
    """

    shortest_axis = 0 if a.shape[0] <= a.shape[1] else 1

    return  np.sqrt(np.nansum(a**2, axis=shortest_axis))


def perp_vector(a, b, return_paral_comp = False):
    """
    This function calculates the component of a vector perpendicular to another vector.

    Parameters:
    a (ndarray) : A 2D numpy array representing the first vector.
    b (ndarray) : A 2D numpy array representing the second vector.

    Returns:
    ndarray     : A 2D numpy array representing the component of the first input vector that is perpendicular to the second input vector.
    """
    b_unit = b / estimate_vec_magnitude(b)[:, np.newaxis]
    proj   = dot_product(a, b_unit)[:, np.newaxis]* b_unit
    perp   = a - proj
    if return_paral_comp:
        
        return perp, proj
    else:
        return perp
        

def update_dates_strings(t0, t1, addit_time):
    """
    Update the given datetime strings by adding or subtracting a specific amount of time.

    This function takes two datetime strings `t0` and `t1`, and an `addit_time` (time duration in seconds) and
    returns updated datetime strings by subtracting `addit_time` seconds from the first datetime (`t0`) and adding
    `addit_time` seconds to the second datetime (`t1`).

    Parameters:
    ----------
    t0 : str
        The first datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
    t1 : str
        The second datetime string in the format 'YYYY-MM-DD HH:MM:SS'.
    addit_time : int or float
        The time duration in seconds to be added to `t1` and subtracted from `t0`.

    Returns:
    -------
    tuple of str
        A tuple containing two updated datetime strings. The first element of the tuple is the updated `t0` datetime
        string, and the second element is the updated `t1` datetime string.

    Example:
    --------
    >>> t0 = '2023-08-05 10:30:00'
    >>> t1 = '2023-08-06 12:45:00'
    >>> addit_time = 20

    >>> updated_t0, updated_t1 = update_dates_strings(t0, t1, addit_time)
    >>> print(updated_t0)
    '2023-08-05 10:29:40'
    >>> print(updated_t1)
    '2023-08-06 12:45:20'
    """
    from datetime import datetime, timedelta

    # Convert strings to datetime objects
    format_str = '%Y-%m-%d %H:%M:%S'
    dt0 = datetime.strptime(t0, format_str)
    dt1 = datetime.strptime(t1, format_str)

    # Subtract `addit_time` seconds from the first date
    new_dt0 = dt0 - timedelta(seconds=addit_time)

    # Add `addit_time` seconds to the second date
    new_dt1 = dt1 + timedelta(seconds=addit_time)

    # Convert datetime objects back to strings
    new_t0 = new_dt0.strftime(format_str)
    new_t1 = new_dt1.strftime(format_str)

    return new_t0, new_t1


def filter_dict(d, keys_to_keep):
    return dict(filter(lambda item: item[0] in keys_to_keep, d.items()))


from dateutil import parser

def format_date_to_str(date_input):
    """
    Takes a date input (string or object that can be converted to a string) and attempts to parse and format it
    into a 'YYYY-MM-DD HH:MM' format.

    Parameters:
    - date_input: The date input to be formatted. Can be a string or an object that can be converted to a string.

    Returns:
    - A string representing the formatted date in 'YYYY-MM-DD HH:MM' format if successful.
    - None if parsing fails.
    """
    try:
        # Convert input to string in case it's not already a string
        date_str = str(date_input)
        # Try to parse the date string
        date_obj = parser.parse(date_str)
        # Format the datetime object to the desired format
        formatted_date_str = date_obj.strftime('%Y-%m-%d %H:%M')
        return formatted_date_str
    except ValueError as e:
        print(f"Error parsing date: {e}")
        # Return None or consider a default value or re-raise the exception based on your use case
        return None
    
def replace_negative_with_nan(df):
    """
    Replace negative values with NaN in a DataFrame.
    """
    return df.where(df >= -1e5, np.nan)


def string_to_datetime_index(datetime_string, datetime_format='%Y-%m-%d %H:%M:%S'):
    return pd.to_datetime(datetime_string, format=datetime_format)

def string_to_timestamp(datetime_string, datetime_format='%Y-%m-%d %H:%M:%S'):
    """
    Converts a string representation of a date and time to a timestamp in the format 'Timestamp('YYYY-MM-DD HH:MM:SS.ssssss')'.

    Parameters:
    datetime_string (str): The string representation of the date and time.
    datetime_format (str, optional): The format of the input string. Defaults to '%Y-%m-%d %H:%M:%S'.

    Returns:
    str: The timestamp representation of the date and time in the format 'Timestamp('YYYY-MM-DD HH:MM:SS.ssssss')'.

    Raises:
    ValueError: If the input string does not match the specified format.
    """
    datetime_object = datetime.datetime.strptime(datetime_string, datetime_format)
    timestamp = datetime_object.timestamp()
    return f"Timestamp('{datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')}')"

def add_time_to_datetime_string(start_time, time_amount, time_unit):
    """
    Adds a specified amount of time to a datetime string and returns the result as a string.

    Parameters:
    start_time (str): The original datetime string, which may or may not include fractional seconds.
    time_amount (int): The amount of time to add.
    time_unit (str): The unit of the added time, either 's' (seconds), 'm' (minutes), 'h' (hours), or 'd' (days).

    Returns:
    str: The datetime string after the specified time has been added, in the format '%Y-%m-%d %H:%M:%S'.

    Raises:
    ValueError: If an invalid time unit is specified.
    """
    import datetime

    # Define the mapping of time units to their corresponding attributes in timedelta
    units = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days'}
    unit = units.get(time_unit)

    # Raise an error if the time unit is invalid
    if unit is None:
        raise ValueError("Invalid time unit")

    # Create a timedelta object with the specified time amount and unit
    delta = datetime.timedelta(**{unit: time_amount})

    # Try parsing the datetime string with and without fractional seconds
    formats = ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S']
    for fmt in formats:
        try:
            # Attempt to parse the datetime string
            start_datetime = datetime.datetime.strptime(start_time, fmt)
            break
        except ValueError:
            continue
    else:
        # If both parsing attempts fail, raise an exception
        raise ValueError("start_time does not match expected formats")

    # Add the time delta to the parsed datetime
    end_datetime = start_datetime + delta

    # Return the resulting datetime string in the specified format, without fractional seconds
    return end_datetime.strftime('%Y-%m-%d %H:%M:%S')
    

from datetime import timedelta
import re

def parse_time_duration(duration_str):
    # Define regex to capture value and unit
    time_regex = re.compile(r'(\d+)([a-z]+)')
    
    # Define supported time units and their equivalent in timedelta
    unit_mapping = {
        'ms': 'milliseconds',
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days'
    }
    
    matches = time_regex.findall(duration_str)
    if not matches:
        raise ValueError(f"Invalid time duration format: {duration_str}")
    
    kwargs = {}
    for value, unit in matches:
        if unit in unit_mapping:
            kwargs[unit_mapping[unit]] = int(value)
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
    
    return timedelta(**kwargs)


def count_fits_in_duration(df, wind_size):
    """
    This function takes a pandas dataframe with a datetime index and an integer wind_size as input.
    It calculates the total duration of the dataframe in hours and then finds how many times wind_size can fit in the duration.
    """
    
    # Calculate the total duration of the dataframe in hours
    total_duration = (df.index[-1] - df.index[0]).total_seconds() / 3600
    
    #print(total_duration)
    
    # Find how many times wind_size can fit into the duration
    fits_in_duration = int(total_duration // wind_size)
    
    return fits_in_duration


def smooth_filter(xv, arr, window):
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    # Convolve with sobel filter
    xv, arr = clean_data(xv, arr)
    grad    = signal.convolve(arr, [1,-1,0])[:-1]
    # Smooth gradient
    smooth_grad = smooth_grad = gaussian_filter1d(grad, window)
    
    return smooth_grad


@njit
def custom_median(array):
    sorted_array = np.sort(array)
    n = len(sorted_array)
    middle = n // 2
    if n % 2 == 0:
        return 0.5 * (sorted_array[middle - 1] + sorted_array[middle])
    else:
        return sorted_array[middle]

@njit
def calc_medians(window_size, arr, medians):
    half_window = window_size // 2
    n = len(arr)
    for i in range(half_window, n - half_window):
        window = arr[i - half_window:i + half_window + 1]
        medians[i] = custom_median(window)
    return medians

@njit
def calc_medians_std(window_size, arr, medians_diff, medians):
    half_window = window_size // 2
    k = 1.4826  # Scale factor for Gaussian distribution
    n = len(arr)
    for i in range(half_window, n - half_window):
        window = arr[i - half_window:i + half_window + 1]
        median = medians[i]
        abs_deviation = np.abs(window - median)
        mad = custom_median(abs_deviation)
        medians_diff[i] = k * mad
    return medians_diff

@njit(parallel=True)
def calc_medians_parallel(window_size, arr, medians):
    half_window = window_size // 2
    n = len(arr)
    for i in prange(half_window, n - half_window):
        window     = arr[i - half_window:i + half_window + 1]
        medians[i] = custom_median(window)
    return medians

@njit(parallel=True)
def calc_medians_std_parallel(window_size, arr, medians_diff, medians):
    half_window = window_size // 2
    k = 1.4826
    n = len(arr)
    for i in prange(half_window, n - half_window):
        window = arr[i - half_window:i + half_window + 1]
        median = medians[i]
        abs_deviation = np.abs(window - median)
        mad = custom_median(abs_deviation)
        medians_diff[i] = k * mad
    return medians_diff

def hampel(arr, window_size=5, n=3, parallel=True):
    """
    Apply Hampel filter to despike a time series by removing spurious data points.

    Parameters:
    ----------
    arr : numpy.ndarray, pandas.Series, or pandas.DataFrame
        The input time series as a 1-dimensional array.
    window_size : int, optional
        The size of the sliding window used to compute the median and MAD of neighboring values.
        It should be an odd integer to have a symmetric window around each point. The default value is 5.
    n : int or float, optional
        The number of MADs away from the median used to define outliers. Data points that deviate
        from the median by more than `n` times the MAD are considered outliers. The default value is 3.
    parallel : bool, optional
        Whether to use parallel computation for performance. The default is True.

    Returns:
    -------
    filtered_arr : numpy.ndarray
        A new filtered time series with outliers replaced by the median of neighboring values.
    outlier_indices : numpy.ndarray
        An array of indices corresponding to the positions of the identified outliers in the original `arr`.
    """
    # Convert input to numpy array if it's a pandas Series or DataFrame
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.values.flatten()
    elif not isinstance(arr, np.ndarray):
        raise ValueError("arr must be a numpy array or pandas Series or DataFrame!")
    
    if arr.ndim != 1:
        raise ValueError("Input array must be one-dimensional!")

    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1  # Make it odd
        print(f"window_size adjusted to {window_size} to make it odd.")

    medians      = np.full_like(arr, np.nan, dtype=np.float64)
    medians_diff = np.full_like(arr, np.nan, dtype=np.float64)

    if parallel:
        medians = calc_medians_parallel(window_size, arr, medians)
        medians_diff = calc_medians_std_parallel(window_size, arr, medians_diff, medians)
    else:
        medians = calc_medians(window_size, arr, medians)
        medians_diff = calc_medians_std(window_size, arr, medians_diff, medians)

    # Identify outliers
    threshold = n * medians_diff
    difference = np.abs(arr - medians)
    outlier_indices = np.where(difference > threshold)[0]

    # Create a copy of the original array to avoid modifying it
    filtered_arr = arr.copy()
    filtered_arr[outlier_indices] = medians[outlier_indices]

    return filtered_arr, outlier_indices

# solve for a and b
def best_fit(X, Y):
    """
    Function to calculate the best linear fit for a given set of data.

    Parameters
    ----------
    X: list or numpy array
        The x-values of the data set
    Y: list or numpy array
        The y-values of the data set

    Returns
    -------
    a: float
        The y-intercept of the best fit line
    b: float
        The slope of the best fit line
    """

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum(xi*yi for xi,yi in zip(X, Y)) - n * xbar * ybar
    denum = sum(xi**2 for xi in X) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

def powlaw(x, a, b) : 
    return a * np.power(x, b)
def expo(x, a, b) :
    return a*np.exp(-b*x)
def linlaw(x, a, b) : 
    return a + x * b

def curve_fit_log(xdata, ydata) : 
    """
    Function to fit data to a power law with weights according to a log scale.

    Parameters
    ----------
    xdata: numpy array
        The x-values of the data set to fit
    ydata: numpy array
        The y-values of the data set to fit

    Returns
    -------
    popt_log: tuple
        The parameters of the best fit line
    pcov_log: numpy array
        The covariance of the parameters
    ydatafit_log: numpy array
        The y-values of the best fit line

    """
    # Weights according to a log scale
    # Apply fscalex
    xdata_log = np.log10(xdata)
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log, ydatafit_log)



def find_fit(x, y, x0, xf):  
    x    = np.array(x)
    y    = np.array(y)
    ind1 = np.argsort(x)
     
    x   = x[ind1]
    y   = y[ind1]
    ind = y>-1e15

    x   = x[ind]
    y   = y[ind]
    # Apply fit on specified range #
   # print(len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0]))
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>-0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
        
        if (len(y[s:e])>1):
            fit = curve_fit_log(x[s:e],y[s:e])
            return fit, s, e, x, y
        else:
            return [0],0,0,0,[0]

def curve_fit_log_wrap(x, y, x0, xf):  
    
    from scipy.optimize import curve_fit
    
    def linlaw(x, a, b) : 
        return a + x * b


    def curve_fit_log(xdata, ydata) : 

        """Fit data to a power law with weights according to a log scale"""
        # Weights according to a log scale
        # Apply fscalex
        xdata_log = np.log10(xdata)
        # Apply fscaley
        ydata_log = np.log10(ydata)
        # Fit linear
        popt_log, pcov_log = curve_fit(linlaw, xdata_log, ydata_log)
        #print(popt_log, pcov_log)
        # Apply fscaley^-1 to fitted data
        ydatafit_log = np.power(10, linlaw(xdata_log, *popt_log))
        # There is no need to apply fscalex^-1 as original data is already available
        return (popt_log, pcov_log, ydatafit_log)

            
   # Apply fit on specified range #
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
        # s = np.min([s,e])
        # e = np.max([s,e])
        if s>e:
            s,e = e,s
        else:
            pass

        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = curve_fit_log(x[s:e],y[s:e])
            #print(fit)

            return fit, s, e, False
        else:
            return [np.nan, np.nan, np.nan],np.nan,np.nan, True




def find_fit_expo(x, y, x0, xf):  
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>-0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]
        
        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = curve_fit_log_expo(x[s:e],y[s:e])
            #print(fit)
            return fit, s, e, x, y
        else:
            return [0],0,0,0,[0]

def curve_fit_log_expo(xdata, ydata) : 
    """Fit data to an exponential law with weights according to a log scale"""
    # Weights according to a log scale
    # Apply fscaley
    ydata_log = np.log10(ydata)
    # Fit linear
    popt_log, pcov_log = curve_fit(linlaw, xdata, ydata_log)
    #print(popt_log, pcov_log)
    # Apply fscaley^-1 to fitted data
    #ydatafit_log = np.power(10, linlaw(xdata, *popt_log)
    # There is no need to apply fscalex^-1 as original data is already available
    return (popt_log, pcov_log)



def histogram(quant, bins2, logx):

    """
    Function to create a histogram of a given data set.

    Parameters
    ----------
    quant: numpy array
        The data set to create the histogram from
    bins2: int
        The number of bins to use in the histogram
    logx: boolean
        Indicates whether the x-axis should be in log scale

    Returns
    -------
    nout: list
        The frequency counts of the data in each bin
    bout: list
        The center of each bin
    errout: list
        The error on the frequency count in each bin

    """
    nout = []
    bout = []
    errout=[]
    if logx == True:
        binsa = np.logspace(np.log10(min(quant)),np.log10(max(quant)),bins2)
    else:
        binsa = np.linspace((min(quant)),(max(quant)),bins2)

    histoout,binsout = np.histogram(quant,binsa,density=True)
    erroutt = histoout/np.float64(np.size(quant))
    erroutt = np.sqrt(erroutt*(1.0-erroutt)/np.float64(np.size(quant)))
    erroutt[: np.size(erroutt)] = erroutt[: np.size(erroutt)] / (
        binsout[1 : np.size(binsout)] - binsout[: np.size(binsout) - 1]
    )

    bin_centersout   = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])

    for k in range(len(bin_centersout)):
        if (histoout[k]!=0.):
            nout.append(histoout[k])
            bout.append(bin_centersout[k])
            errout.append(erroutt[k])
    return nout, bout,errout


import numpy as np

def scotts_rule_PDF(x):
    """
    Estimate bin edges using Scott's rule for histogram binning.

    Scott’s rule minimizes the integrated mean squared error in the bin approximation
    under the assumption that the data is approximately Gaussian, increasing the number
    of bins for smaller scales.

    Parameters
    ----------
    x : array_like
        Input data array.

    Returns
    -------
    array
        An array containing the estimated bin edges.

    Notes
    -----
    This function estimates the bin edges for histogram binning using Scott's rule, which
    is based on the standard deviation of the data. The number of bins is adjusted for
    smaller scales, assuming the data follows a Gaussian distribution.
    """
    x = np.real(x)
    N = len(x)
    sigma = np.nanstd(x)

    # Scott's rule for bin width
    dui = 3.5 * sigma / N ** (1 / 3)

    # create bins
    return np.arange(np.nanmin(x), np.nanmax(x), dui)




def pdf(val,
        bins, 
        loglog      = False,
        density     = False, 
        scott_rule  = False):
    """
    Calculate the Probability Density Function (PDF) from a given dataset.

    Parameters
    ----------
    val : array_like
        Input data array.
    bins : int or array_like
        Number of bins or bin edges for the histogram.
    loglog : bool, optional
        If True, use logarithmic bins. Default is False.
    density : bool, optional
        If True, compute a probability density function. Default is False.
    scott_rule : bool, optional
        If True, use Scott's rule for estimating the number of bins. Default is False.

    Returns
    -------
    tuple
        A tuple containing the bin centers, PDF values, PDF errors, and raw counts.

    Notes
    -----
    This function calculates the Probability Density Function (PDF) from a given dataset using a histogram.
    The bins for the histogram can be specified as an integer (number of bins) or an array (bin edges).
    The function supports logarithmic bins if `loglog` is set to True.
    If `density` is True, the function computes a normalized probability density function.
    If `scott_rule` is True, Scott's rule is used for estimating the number of bins.

    """
    nout = []
    bout = []
    errout = []
    countsout = []

    val = np.array(val)
    val = val[np.abs(val) < 1e15]

    if loglog:
        binsa = np.logspace(np.log10(min(val)), np.log10(max(val)), bins)
    else:
        if scott_rule:
            binsa = scotts_rule_PDF(val)
        else:
            binsa = np.linspace(min(val), max(val), bins)

    if density:
        numout, binsout, patchesout = plt.hist(val, density=True, bins=binsa, alpha=0)
    else:
        numout, binsout, patchesout = plt.hist(val, density=False, bins=binsa, alpha=0)

    counts, _, _ = plt.hist(val, density=False, bins=binsa, alpha=0)

    if loglog:
        bin_centers = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])
    else:
        bin_centers = binsout[:-1] + 0.5 * (binsout[1:] - binsout[:-1])

    if density:
        histoout, edgeout = np.histogram(val, binsa, density=True)
    else:
        histoout, edgeout = np.histogram(val, binsa, density=False)

    erroutt = histoout / np.float64(np.size(val))
    erroutt = np.sqrt(erroutt * (1.0 - erroutt) / np.float64(np.size(val)))
    erroutt[: np.size(erroutt)] = erroutt[: np.size(erroutt)] / (
        edgeout[1 : np.size(edgeout)] - edgeout[: np.size(edgeout) - 1]
    )

    for i in range(len(numout)):
        if numout[i] != 0.0:
            nout.append(numout[i])
            bout.append(bin_centers[i])
            errout.append(erroutt[i])
            countsout.append(counts[i])

    return np.array(bout), np.array(nout), np.array(errout), np.array(countsout)


def moving_average(xvals, yvals, window_size):
    """
    Calculate the moving average of the data.

    Parameters
    ----------
    xvals : array_like
        Input array representing the independent variable (x).
    yvals : array_like
        Input array representing the dependent variable (y).
    window_size : int
        Size of the moving average window.

    Returns
    -------
    tuple
        A tuple containing the smoothed `xvals` and the corresponding smoothed `yvals`.

    Notes
    -----
    This function calculates the moving average of the data using a specified window size.
    The `xvals` and `yvals` inputs are sorted based on `xvals` before calculating the moving average.

    """
    # Turn input into np.arrays
    xvals, yvals = np.array(xvals), np.array(yvals)

    # Now sort them
    index = np.argsort(xvals).astype(int)
    xvals = xvals[index]
    yvals = yvals[index]

    window = np.ones(int(window_size)) / float(window_size)
    y_new = np.convolve(yvals, window, 'same')
    return xvals, y_new





def plot_plaw(start, end, exponent, c):
    """
    Calculate points on a power-law line within a specified range.

    Parameters
    ----------
    start : float
        Starting value for the x range.
    end : float
        Ending value for the x range.
    exponent : float
        Exponent of the power-law function.
    c : float
        Scaling constant for the power-law function.

    Returns
    -------
    tuple
        A tuple containing the `x` values and the corresponding `y` values representing the points on the power-law line.

    Notes
    -----
    This function calculates the points on a power-law line given by the equation f(x) = c * x ** exponent.
    The points are calculated within the specified range from `start` to `end`.
    The function returns the `x` values and the corresponding `y` values representing the points on the power-law line.

    """
    # Calculating the points on the line
    x = np.logspace(np.log10(start), np.log10(end), 10000)
    
    # Power-law function f(x) = c * x ** exponent
    f = lambda x: c * x ** exponent
    
    return x, f(x)


import numpy as np
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D


class LineAnnotation(Annotation):
    def __init__(
        self, text, line, x, xytext=(0, 5), textcoords="offset points", font_size=None, **kwargs
    ):
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])

        self.neighbours = n1, n2 = neighbours(x, xs, ys)

        # Calculate y by interpolating neighboring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            "fontsize": font_size,  # Set the font size using the font_size parameter
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


# def line_annotate(text, line, x, font_size=None, *args, **kwargs):
#     ax = line.axes
#     a = LineAnnotation(text, line, x, font_size=font_size, *args, **kwargs)
#     if "clip_on" in kwargs:
#         a.set_clip_path(ax.patch)
#     ax.add_artist(a)
#     return a
def line_annotate(ax, text, line, x, font_size=None, *args, **kwargs):
    a = LineAnnotation(text, line, x, font_size=font_size, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a




@jit(nopython=True, parallel=True)
def smoothing_function(x, y, mean=True, window=2):
    """
    Optimized smoothing function for time series data.
    [Description same as before...]
    """
    
    def optimized_bisection(array, value):
        """
        Optimized bisection search function.
        [Description same as before...]
        """
        n = len(array)
        if value < array[0]:
            return -1
        elif value > array[n-1]:
            return n
        jl, ju = 0, n-1
        while ju - jl > 1:
            jm = (ju + jl) >> 1
            if value >= array[jm]:
                jl = jm
            else:
                ju = jm
        return jl if value != array[n-1] else n-1

    len_x = len(x)
    max_x = np.max(x)
    xoutmid,  yout = np.full(len_x, np.nan), np.full(len_x, np.nan)

    for i in prange(len_x):
        x0 = x[i]
        xf = window * x0

        if xf < max_x:
            e = optimized_bisection(x, xf)
            if e < len_x:
                x_range = x[i:e]
                y_range = y[i:e]
                if mean:
                    yout[i] = np.nanmean(y_range)
                else:
                    yout[i] = np.nanmedian(y_range)
                xoutmid[i] = x0 + np.log10(0.4) * (x0 - x[e])
                #xoutmid[i] = np.nanmedian(x_range)
               

    return xoutmid, yout


def calculate_parker_spiral(B):
    """
    This function estimates φ_{rB} = arctan(Bt / Br) for arrays of Br and Bt.
    It also returns the rolling mean of the computed angles over a 24-hour window.
    
    Parameters:
    B : pd.DataFrame
        A DataFrame where:
        - B.iloc[:, 0] corresponds to Br (Radial component of the magnetic field).
        - B.iloc[:, 1] corresponds to Bt (Tangential component of the magnetic field).
        - The index is the datetime for each measurement.
    
    Returns:
    pd.DataFrame:
        A DataFrame with the rolling mean of φ_{rB} (in degrees) over 24-hour windows.
    """
    # Extract Br and Bt using positional indexing (0 for Br, 1 for Bt)
    Br = B.iloc[:, 0].to_numpy()
    Bt = B.iloc[:, 1].to_numpy()
    
    # Calculate the Parker Spiral angle (phi_rB) in degrees using arctan2
    phi_rB = np.degrees(np.arctan2(Bt, Br))
    
    # Create a DataFrame for phi_rB with datetime index
    df_phi = pd.DataFrame({'phi_rB': phi_rB}, index=B.index)
    
    # Apply the rolling mean over a 24-hour window
    df_phi_rolling = df_phi.rolling('24H', center=True).mean()
    
    return df_phi_rolling


def interp(df, new_index):
    """
    Interpolate a DataFrame's columns values to new index values and return a new DataFrame.

    This function takes a DataFrame `df` and a new index `new_index`. It performs linear interpolation on each column
    of the DataFrame to calculate the corresponding values at the new index points. The resulting interpolated values
    are returned as a new DataFrame.

    Parameters:
    ----------
    df : pandas DataFrame
        The input DataFrame to be interpolated. The DataFrame should have a valid index.
    new_index : array-like
        The new index values to which the columns of `df` will be interpolated.

    Returns:
    -------
    pandas DataFrame
        A new DataFrame containing the interpolated values of the columns from `df` at the new index points.

    Notes:
    -----
    The function uses NumPy's `np.interp` function to perform linear interpolation for each column of the DataFrame.
    If the new index values lie outside the range of the original DataFrame's index, the function will extrapolate
    based on the closest available values.
    
    """
    
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out




def simple_python_rolling_median(vector: np.ndarray,
                                 window_length: int) -> np.ndarray:
    """Computes a rolling median of a numpy vector returning a new numpy
    vector of the same length.
    NaNs in the input are not handled but a ValueError will be raised."""
    if vector.ndim != 1:
        raise ValueError(
            f'vector must be one dimensional not shape {vector.shape}'
        )
    skip_list = orderedstructs.SkipList(float)
    ret = np.empty_like(vector)
    for i in range(len(vector)):
        value = vector[i]
        skip_list.insert(value)
        if i >= window_length - 1:
            # // 4 for lower quartile
            # * 3 // 4 for upper quartile etc.
            median = skip_list.at(window_length // 2)
            skip_list.remove(vector[i - window_length + 1])
        else:
            median = np.nan
        ret[i] = median
    return ret




def  use_dates_return_elements_of_df_inbetween(t0, t1, df):
    """
    Return the rows of df between the nearest indices to t0 and t1 using iloc.

    Parameters:
    -----------
    t0 : datetime-like or str
        Start date (if str, converted to datetime).
    t1 : datetime-like or str
        End date (if str, converted to datetime).
    df : pd.DataFrame
        DataFrame with a sorted datetime-like index.

    Returns:
    --------
    pd.DataFrame
        A DataFrame slice from the nearest index to t0 up to the nearest index to t1.
    """
    df = df.sort_index()

    # Convert to datetime if necessary
    if isinstance(t0, str):
        t0 = pd.to_datetime(t0)
    if isinstance(t1, str):
        t1 = pd.to_datetime(t1)

    # Find nearest indices
    unique_idx = df.index.unique()
    start_idx = unique_idx.get_indexer([t0], method="nearest")[0]
    end_idx = unique_idx.get_indexer([t1], method="nearest")[0]

    # Slice using iloc
    return df.iloc[start_idx:end_idx]


# def find_big_gaps(df, gap_time_threshold):
#     """
#     Filter a data set by the values of its first column and identify gaps in time that are greater than a specified threshold.

#     Parameters:
#     df (pandas DataFrame): The data set to be filtered and analyzed.
#     gap_time_threshold (float): The threshold for identifying gaps in time, in seconds.

#     Returns:
#     big_gaps (pandas Series): The time differences between consecutive records in df that are greater than gap_time_threshold.
#     """
#     keys = df.keys()

#     filtered_data = df[df[keys[1]] > -1e10]
#     time_diff     = (filtered_data.index.to_series().diff() / np.timedelta64(1, 's'))
#     big_gaps      = time_diff[time_diff > gap_time_threshold]

#     return big_gaps


def find_big_gaps(
    df, 
    gap_time_threshold  = 10.0, 
    expected_start      = None, 
    expected_end        = None
):
    """
    Identifies "big gaps" where:
      1) The gap between consecutive filtered data points exceeds `gap_time_threshold`.
      2) The gap from an `expected_start` (if given) to the first filtered data point 
         exceeds `gap_time_threshold`.
      3) The gap from the last filtered data point to an `expected_end` (if given) 
         exceeds `gap_time_threshold`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DateTimeIndex and at least one column.
        Rows whose first column <= -1e10 will be excluded from gap checks.
    gap_time_threshold : float, optional
        The gap size threshold in seconds. Default=10.0 seconds.
    expected_start : None or str or pd.Timestamp, optional
        If provided (e.g. "2025-01-01 09:00:00"), we also check if there's a large
        gap between this time and the first valid row. If that difference in seconds 
        is > gap_time_threshold, it's listed as a gap.
    expected_end : None or str or pd.Timestamp, optional
        If provided, we also check if there's a large gap between the last valid row 
        and this time. If that difference is > gap_time_threshold, it's listed as a gap.

    Returns
    -------
    gaps_df : pandas.DataFrame
        A DataFrame with columns ["Start", "End"] listing the start and end of each
        detected gap. Gaps are returned if they exceed `gap_time_threshold` in seconds.

    Notes
    -----
    - We filter out rows in which the **first column** is <= -1e10 (same as your
      original logic).
    - The function sorts the DataFrame by its DateTimeIndex if not already sorted.
    - If there is no valid data, or if only 1 valid row remains, only the "start"
      or "end" gap (if any) can be reported (no consecutive row gaps).
    """

    def _check_full_empty_gap_only(expected_start, expected_end, gap_time_threshold):
        """
        Helper for the case where we have no valid data in the DF after filtering.
        If both expected_start and expected_end are given and the time difference
        is > threshold, we'll return one big gap from start->end. Otherwise, empty.
    
        Returns a DataFrame with columns ["Start", "End"].
        """
        if (expected_start is not None) and (expected_end is not None):
            # measure difference from expected_start to expected_end
            diff_sec = (expected_end - expected_start).total_seconds()
            if diff_sec > gap_time_threshold:
                return pd.DataFrame([{"Start": expected_start, "End": expected_end}],
                                    columns=["Start", "End"])
        # No gap or incomplete info, return empty
        return pd.DataFrame(columns=["Start", "End"])

    # ------------------------------------------------------------------------
    # 1) Ensure the DF is sorted by DateTimeIndex
    # ------------------------------------------------------------------------
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    # ------------------------------------------------------------------------
    # 2) Filter out rows where the first column <= -1e10
    # ------------------------------------------------------------------------
    filtered_df = df[df.iloc[:, 0] > -1e10].dropna()
    if filtered_df.empty:
        # No valid data at all => any "start" or "end" gap is measured purely
        # from expected_start to expected_end (if both are given).
        # Typically, we interpret: if expected_start & expected_end are present
        # and the gap is bigger than threshold => one big gap.
        # Otherwise, we just return empty. 
        return None

    # ------------------------------------------------------------------------
    # 3) Convert expected_start / expected_end to Timestamps if not None
    # ------------------------------------------------------------------------
    if expected_start is not None and not isinstance(expected_start, pd.Timestamp):
        expected_start = pd.Timestamp(expected_start)
    if expected_end is not None and not isinstance(expected_end, pd.Timestamp):
        expected_end = pd.Timestamp(expected_end)

      # Prepare a list to accumulate gap rows
    gap_rows = []

    # ------------------------------------------------------------------------
    # 4) Check gap from expected_start to first valid row
    # ------------------------------------------------------------------------
    if expected_start is not None:
        first_time = filtered_df.index[0]
        start_diff_sec = (first_time - expected_start).total_seconds()
        if start_diff_sec > gap_time_threshold:
            gap_rows.append({"Start": expected_start, "End": first_time})

    # ------------------------------------------------------------------------
    # If only 1 valid row, we won't find any consecutive row gaps, 
    # but we might still check from that row to expected_end.
    # ------------------------------------------------------------------------
    if len(filtered_df) == 1:
        # Only one valid row => skip consecutive checks, but check end gap
        if expected_end is not None:
            last_time = filtered_df.index[0]
            end_diff_sec = (expected_end - last_time).total_seconds()
            if end_diff_sec > gap_time_threshold:
                gap_rows.append({"Start": last_time, "End": expected_end})
        return pd.DataFrame(gap_rows, columns=["Start", "End"])

    # ------------------------------------------------------------------------
    # 5) Compute consecutive time differences (in seconds) in filtered data
    #    time_diffs[i] is gap from row (i-1) to row i.
    # ------------------------------------------------------------------------
    time_diffs = filtered_df.index.to_series().diff().dt.total_seconds()
    # Boolean mask: True at index i => gap between i-1 and i > threshold
    gap_mask = time_diffs > gap_time_threshold
    gap_mask_values = gap_mask.values  # convert to numpy boolean array

    # ------------------------------------------------------------------------
    # 6) Vectorized detection: we want all i where gap_mask[i] is True
    #    The "start" is index[i-1], "end" is index[i]
    # ------------------------------------------------------------------------
    gap_indices = np.where(gap_mask_values)[0]  # array of int positions
    f_idx = filtered_df.index  # DatetimeIndex

    # For each i in gap_indices, gap is (f_idx[i-1], f_idx[i])
    for i in gap_indices:
        # i-1 must be >= 0, which it should be for i>0
        start_time = f_idx[i - 1]
        end_time = f_idx[i]
        gap_rows.append({"Start": start_time, "End": end_time})

    # ------------------------------------------------------------------------
    # 7) Check gap from last valid row to expected_end
    # ------------------------------------------------------------------------
    if expected_end is not None:
        last_time = filtered_df.index[-1]
        end_diff_sec = (expected_end - last_time).total_seconds()
        if end_diff_sec > gap_time_threshold:
            gap_rows.append({"Start": last_time, "End": expected_end})

    # ------------------------------------------------------------------------
    # 8) Build a DataFrame of all gaps found
    # ------------------------------------------------------------------------
    gaps_df = pd.DataFrame(gap_rows, columns=["Start", "End"])
    return gaps_df

# def find_big_gaps(df, gap_time_threshold=10):
#     """
#     Identifies gaps where the time difference between consecutive filtered entries
#     exceeds the gap_time_threshold, in a vectorized manner.
    
#     Parameters:
#     - df: pandas DataFrame with a datetime index and at least one column.
#     - gap_time_threshold: float, the gap size threshold in seconds.
    
#     Returns:
#     - A DataFrame with the start and end times of the gaps.
#     """
#     # Filter rows based on the condition for the first column
#     filtered_df = df[df.iloc[:, 0] > -1e10]
    
#     # Calculate time differences in seconds between consecutive rows
#     time_diffs = filtered_df.index.to_series().diff().dt.total_seconds()
    
#     # Identify indices where time differences exceed the threshold
#     gap_mask = time_diffs > gap_time_threshold
    
#     # Using the mask, find the end times of the gaps
#     gap_ends = filtered_df.index[gap_mask]
    
#     # The start times are just before the ends, adjust indices accordingly
#     gap_starts = filtered_df.index[gap_mask.shift(-1, fill_value=False)]
    
#     # Remove the last element from starts and the first element from ends to align
#     if len(gap_starts) > 0 and len(gap_ends) > 0:  # Ensure there are gaps
#         gap_starts = gap_starts[:-1]
#         gap_ends = gap_ends[1:]
    
#     # Create a DataFrame to return the start and end times of gaps
#     gaps_df = pd.DataFrame({'Start': gap_starts, 'End': gap_ends})
    
#     return gaps_df

def percentile(y,percentile):
    return(np.percentile(y,percentile))


import numpy as np


def binned_statistics_exclude(x, values, bins, statistic='mean', N=2, log_binning=True, n_jobs=1):
    """
    Compute binned statistics with exclusion of outliers greater than N standard deviations from the bin-specific mean.

    Parameters:
    - x : (N,) array_like
        Input values to be binned.
    - values : (N,) array_like
        Data values to compute the statistics on.
    - bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins. If bins is a sequence, it defines the bin edges.
    - statistic : string in ['mean', 'sum', 'std', 'count'] or callable
        The statistic to compute (default is 'mean').
    - N : float
        Number of standard deviations to exclude values from the mean within each bin.
    - log_binning : bool
        If True, use logarithmic bins.
    - n_jobs : int, default=1
        Number of CPU cores to use when parallelizing. Use -1 for all cores.
    
    Returns:
    - result : (nbins,) array
        The computed statistic for each bin.
    """
    
    
    from joblib import Parallel, delayed

    def compute_bin_statistic(bin_values, statistic, N):
        # Function to compute statistics for a single bin
        mean_val = np.mean(bin_values)
        std_val  = np.std(bin_values)

        # Exclude values more than N std dev from the bin-specific mean
        mask_std = np.abs(bin_values - mean_val) <= N * std_val
        bin_values = bin_values[mask_std]

        if statistic == 'mean':
            return np.nanmean(bin_values)
        elif statistic == 'median':
            return np.nanmedian(bin_values)
        elif statistic == 'sum':
            return np.nansum(bin_values)
        elif statistic == 'std':
            return np.nanstd(bin_values)
        elif statistic == 'count':
            return len(bin_values)
        elif callable(statistic):
            return statistic(bin_values)
        else:
            return np.nan

    # Remove nan and inf values
    mask_valid = np.isfinite(x) & np.isfinite(values)
    x          = x[mask_valid]
    values     = values[mask_valid]
    
    # Determine bins 
    if log_binning:
        if isinstance(bins, int):
            bin_edges = np.logspace(np.log10(min(x)), np.log10(max(x)), bins+1)
        else:
            bin_edges = np.logspace(np.log10(min(bins)), np.log10(max(bins)), len(bins))
    else:
        if isinstance(bins, int):
            bin_edges = np.linspace(min(x), max(x), bins+1)
        else:
            bin_edges = bins
        
    bin_indices = np.digitize(x, bin_edges)

    # Compute statistic for each bin using parallel processing
    results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], statistic, N) 
                                      for i in range(1, len(bin_edges)))
    
    std_results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], 'std', N) 
                                      for i in range(1, len(bin_edges)))
    count_results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], 'count', N) 
                                      for i in range(1, len(bin_edges)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return bin_centers, np.array(results), np.array(std_results)/np.sqrt(count_results)

# Example usage


def binned_statistics_high_percentile(x, values, bins, statistic='mean', percentile=50, log_binning=True, n_jobs=1):
    """
    Compute binned statistics considering only values above the given percentile for each bin.

    Parameters:
    - x : (N,) array_like
        Input values to be binned.
    - values : (N,) array_like
        Data values to compute the statistics on.
    - bins : int or sequence of scalars
        If bins is an int, it defines the number of equal-width bins. If bins is a sequence, it defines the bin edges.
    - statistic : string in ['mean', 'sum', 'std', 'count'] or callable
        The statistic to compute (default is 'mean').
    - percentile : float, default=50
        Percentile below which data will be excluded from each bin. 
        Values should be between 0 and 100.
    - log_binning : bool
        If True, use logarithmic bins.
    - n_jobs : int, default=1
        Number of CPU cores to use when parallelizing. Use -1 for all cores.
    
    Returns:
    - bin_centers : (nbins,) array
        The center of each bin.
    - result : (nbins,) array
        The computed statistic for each bin.
    """
    
    from joblib import Parallel, delayed

    def compute_bin_statistic(bin_values, statistic, percentile):
        # Check if bin_values is empty
        if len(bin_values) == 0:
            return np.nan

        # Filter values that are below the provided percentile
        threshold = np.percentile(bin_values, percentile)
        bin_values = bin_values[bin_values >= threshold]

        if statistic == 'mean':
            return np.mean(bin_values)
        elif statistic == 'sum':
            return np.sum(bin_values)
        elif statistic == 'std':
            return np.std(bin_values)
        elif statistic == 'count':
            return len(bin_values)
        elif callable(statistic):
            return statistic(bin_values)
        else:
            return np.nan


    # Remove nan and inf values
    mask_valid = np.isfinite(x) & np.isfinite(values)
    x          = x[mask_valid]
    values     = values[mask_valid]
    
    # Determine bins 
    if log_binning:
        if isinstance(bins, int):
            bin_edges = np.logspace(np.log10(min(x)), np.log10(max(x)), bins+1)
        else:
            bin_edges = np.logspace(np.log10(min(bins)), np.log10(max(bins)), len(bins))
    else:
        if isinstance(bins, int):
            bin_edges = np.linspace(min(x), max(x), bins+1)
        else:
            bin_edges = bins
        
    bin_indices = np.digitize(x, bin_edges)

    # Compute statistic for each bin using parallel processing
    results = Parallel(n_jobs=n_jobs)(delayed(compute_bin_statistic)(values[bin_indices == i], statistic, percentile) 
                                      for i in range(1, len(bin_edges)))
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return bin_centers, np.array(results)



import numpy as np
from scipy import stats

def binned_quantity_percentile(x, y, what, std_or_error_of_mean, bins, loglog, percentile):
    # Ensure x and y are float type arrays
    x = x.astype(float)
    y = y.astype(float)
    
    # Filter out invalid y values
    ind = y > -1e15
    x = x[ind]
    y = y[ind]
    
    # Define bins in logarithmic scale if specified
    if loglog:
        bins = np.logspace(np.log10(min(x)), np.log10(max(x)), bins)
    
    # Calculate the binned percentiles
    percentiles, x_b, binnumber = stats.binned_statistic(x, y, lambda y: np.percentile(y, percentile), bins=bins)
    
    # Initialize arrays to store binned quantities for values above the percentile
    y_b = np.zeros(len(bins) - 1)
    z_b = np.zeros(len(bins) - 1)
    points = np.zeros(len(bins) - 1)
    
    # Iterate through each bin and calculate the statistics for values above the percentile
    for i in range(len(bins) - 1):
        bin_indices = np.where((x >= bins[i]) & (x < bins[i+1]))[0]
        if len(bin_indices) > 0:
            bin_y = y[bin_indices]
            bin_percentile_value = np.percentile(bin_y, percentile)
            bin_y_above_percentile = bin_y[bin_y >= bin_percentile_value]
            
            if len(bin_y_above_percentile) > 0:
                if what == 'mean':
                    y_b[i] = np.nanmean(bin_y_above_percentile)
                elif what == 'median':
                    y_b[i] = np.nanmedian(bin_y_above_percentile)
                elif what == 'sum':
                    y_b[i] = np.nansum(bin_y_above_percentile)
                elif what == 'std':
                    y_b[i] = np.nanstd(bin_y_above_percentile)
                elif what == 'var':
                    y_b[i] = np.nanvar(bin_y_above_percentile)
                
                z_b[i] = np.nanstd(bin_y_above_percentile)
                points[i] = len(bin_y_above_percentile)
    
    # Calculate the standard error of the mean if specified
    if std_or_error_of_mean == 0:
        z_b = z_b / np.sqrt(points)
    
    # Calculate the bin centers
    x_b = x_b[:-1] + 0.5 * (x_b[1:] - x_b[:-1])
    
    return x_b, y_b, z_b, percentiles


def ensure_time_format(start_time, end_time):
    
    """
    Ensure that the input start and end times are in the desired format and return them as formatted strings.

    This function takes `start_time` and `end_time` as inputs. It ensures that both `start_time` and `end_time`
    are in the desired format "%Y-%m-%d %H:%M:%S" and returns them as formatted strings.

    Parameters:
    ----------
    start_time : str or datetime-like object
        The start time of the desired time period. If provided as a datetime-like object, it will be converted
        to a string in the format "%Y-%m-%d %H:%M:%S".
    end_time : str or datetime-like object
        The end time of the desired time period. If provided as a datetime-like object, it will be converted
        to a string in the format "%Y-%m-%d %H:%M:%S".

    Returns:
    -------
    tuple of str
        A tuple containing two elements:
        1. The formatted start time in the format "%Y-%m-%d %H:%M:%S".
        2. The formatted end time in the format "%Y-%m-%d %H:%M:%S".

    Notes:
    -----
    The function uses the `datetime` module to handle datetime-like objects. If the input times are not provided
    as strings, the function converts them to the desired format. If the time is provided without a specific time
    (only date), the function appends "00:00:00" to the time before converting it to the desired format.

    """

    desired_format = "%Y-%m-%d %H:%M:%S"
    if not isinstance(start_time, str):
        start_time = datetime.strftime(start_time, desired_format)
    if not isinstance(end_time, str):
        end_time = datetime.strftime(end_time, desired_format)
    
    try:
        t0 = datetime.strptime(start_time, desired_format)
    except ValueError:
        t0 = datetime.strptime(start_time + " 00:00:00", desired_format)
    
    try:
        t1 = datetime.strptime(end_time, desired_format)
    except ValueError:
        t1 = datetime.strptime(end_time + " 00:00:00", desired_format)
        
    return t0.strftime(desired_format), t1.strftime(desired_format)


import numpy as np
from scipy import stats

def binned_quantity(x, y, what='mean', std_or_error_of_mean=True, bins=100, loglog=True, return_counts=False, return_percentiles=False, lower_percentile =25, higher_percentile = 75):
    """
    Calculate binned statistics of one variable (y) with respect to another variable (x).

    Parameters
    ----------
    x : array_like
        Input array. This represents the independent variable.
    y : array_like
        Input array. This represents the dependent variable.
    what : str or callable, optional
        The type of binned statistic to compute. This can be any of the options supported by `scipy.stats.binned_statistic()`.
        The default is 'mean'.
    std_or_error_of_mean : bool, optional
        Indicates whether to return the standard deviation (True) or the error of the mean (False) of the binned statistic.
        The default is True.
    bins : int or array_like, optional
        The number of bins to use for the histogram. If `loglog` is True, this value is used to generate logarithmic bins.
        The default is 100.
    loglog : bool, optional
        If True, logarithmic bins are used instead of linear bins. The default is True.
    return_counts : bool, optional
        If True, also return the number of points in each bin. The default is False.
    return_percentiles : bool, optional
        If True, also return the 25th and 75th percentiles for each bin. The default is False.

    Returns
    -------
    x_b : ndarray
        The centers of the bins.
    y_b : ndarray
        The value of the binned statistic.
    z_b : ndarray
        The standard deviation or error of the mean of the binned statistic.
    points : ndarray, optional
        The number of points in each bin. This is only returned if `return_counts` is True.
    percentiles : tuple of ndarrays, optional
        The 25th and 75th percentiles for each bin. This is only returned if `return_percentiles` is True.
    """
    
    if loglog:
        mask = np.where((y > -1e10) & (x > 0) )[0]        
    else:
        mask = np.where((y > -1e10) & (x > -1e10) )[0]
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)

    if loglog:
        bins = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), bins)

    # Binned statistic calculation
    y_b, x_b, _ = stats.binned_statistic(x, y, statistic=what, bins=bins)
    z_b, _, _ = stats.binned_statistic(x, y, statistic='std', bins=bins)
    points, _, _ = stats.binned_statistic(x, y, statistic='count', bins=bins)

    if std_or_error_of_mean == 0:
        z_b /= np.sqrt(points)

    x_b = x_b[:-1] + 0.5 * (x_b[1:] - x_b[:-1])

    result = (x_b, y_b, z_b, points) if return_counts else (x_b, y_b, z_b)

    if return_percentiles:
        percentile_25, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, lower_percentile), bins=bins)
        percentile_75, _, _ = stats.binned_statistic(x, y, statistic=lambda y: np.percentile(y, higher_percentile), bins=bins)
        percentiles = (percentile_25, percentile_75)
        result += (percentiles,)

    return result


# # --- Numba Helper Functions ---
# @njit(inline='always')
# def find_bin(x, bin_edges):
#     n = len(bin_edges)
#     if x < bin_edges[0]:
#         return 0
#     if x >= bin_edges[n - 1]:
#         return n - 2  # Last valid index for n-1 bins.
#     lo = 0
#     hi = n - 1
#     while hi - lo > 1:
#         mid = (lo + hi) // 2
#         if x < bin_edges[mid]:
#             hi = mid
#         else:
#             lo = mid
#     return lo

# @njit(parallel=True)
# def compute_bins_numba_no_atomic(valid_x, valid_y, bin_edges, nbins, counts, sums, sumsqs, nchunks):
#     n = valid_x.shape[0]
#     chunk_size = (n + nchunks - 1) // nchunks  # Ceiling division.
#     # Temporary arrays for each chunk.
#     temp_counts = np.zeros((nchunks, nbins), dtype=np.int64)
#     temp_sums   = np.zeros((nchunks, nbins), dtype=np.float64)
#     temp_sumsqs = np.zeros((nchunks, nbins), dtype=np.float64)
    
#     for i in prange(nchunks):
#         start = i * chunk_size
#         end = start + chunk_size
#         if end > n:
#             end = n
#         for j in range(start, end):
#             bin_index = find_bin(valid_x[j], bin_edges)
#             temp_counts[i, bin_index] += 1
#             temp_sums[i, bin_index]   += valid_y[j]
#             temp_sumsqs[i, bin_index] += valid_y[j] * valid_y[j]
    
#     # Reduce temporary arrays into final arrays.
#     for i in range(nchunks):
#         for b in range(nbins):
#             counts[b] += temp_counts[i, b]
#             sums[b]   += temp_sums[i, b]
#             sumsqs[b] += temp_sumsqs[i, b]

# # --- Numba Version ---
# def binned_quantity_numba(x, y, what='mean', std_or_error_of_mean=True,
#                           bins=100, loglog=True, return_counts=False,
#                           return_percentiles=False, lower_percentile=25,
#                           higher_percentile=75, nchunks=16):
#     """
#     Numba-accelerated binned statistic computation.
    
#     Parameters
#     ----------
#     x, y : array_like
#         Input arrays.
#     what : {'mean', 'sum', 'std', 'median'}, optional
#         The binned statistic to compute. For 'mean', 'sum', and 'std' the aggregation is done
#         using a Numba-accelerated summation routine. For 'median', the binning is computed in pure NumPy.
#     std_or_error_of_mean : bool, optional
#         For what=='mean', if True the third output is the standard deviation,
#         otherwise it is the error-of-mean. (Ignored for other statistics.)
#     bins : int or array_like, optional
#         Number of bins (if int) or the bin edges.
#     loglog : bool, optional
#         If True, logarithmic bins are used (requires x > 0).
#     return_counts : bool, optional
#         If True, also return the counts per bin.
#     return_percentiles : bool, optional
#         If True, also return the (lower, upper) percentiles per bin.
#     nchunks : int, optional
#         Number of chunks to use for parallelization.
    
#     Returns
#     -------
#     bin_centers : ndarray
#         The centers of the bins.
#     stat : ndarray
#         The binned statistic (mean, sum, std, or median).
#     z : ndarray
#         For what=='mean': the standard deviation (or error-of-mean if std_or_error_of_mean is False);
#         for what=='sum' or 'std': the standard deviation of the bin values;
#         for what=='median': the standard deviation of the values in each bin.
#     [counts] : ndarray, optional
#         The counts per bin (if return_counts is True).
#     [percentiles] : tuple of ndarrays, optional
#         The (lower, upper) percentiles per bin (if return_percentiles is True).
#     """
#     # Convert inputs to float arrays.
#     x = np.asarray(x, dtype=float)
#     y = np.asarray(y, dtype=float)
    
#     # Filter data.
#     if loglog:
#         mask = (y > -1e10) & (x > 0)
#     else:
#         mask = (y > -1e10) & (x > -1e10)
#     x = x[mask]
#     y = y[mask]
    
#     # Determine bin edges.
#     if loglog:
#         if np.isscalar(bins):
#             bin_edges = np.logspace(np.log10(np.nanmin(x)),
#                                     np.log10(np.nanmax(x)),
#                                     int(bins))
#         else:
#             bin_edges = np.asarray(bins, dtype=np.float64)
#     else:
#         if np.isscalar(bins):
#             bin_edges = np.linspace(np.nanmin(x),
#                                     np.nanmax(x),
#                                     int(bins) + 1)
#         else:
#             bin_edges = np.asarray(bins, dtype=np.float64)
    
#     nbins = len(bin_edges) - 1
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
#     # --- Compute the requested statistic ---
#     if what == 'median':
#         # For median, compute bin indices and then loop over bins.
#         bin_indices = np.searchsorted(bin_edges, x, side='right') - 1
#         bin_indices = np.clip(bin_indices, 0, nbins - 1)
#         stat = np.empty(nbins, dtype=np.float64)
#         z = np.empty(nbins, dtype=np.float64)
#         if return_counts:
#             counts = np.empty(nbins, dtype=np.int64)
#         for i in range(nbins):
#             y_in_bin = y[bin_indices == i]
#             if y_in_bin.size > 0:
#                 stat[i] = np.median(y_in_bin)
#                 z[i] = np.std(y_in_bin)
#                 if return_counts:
#                     counts[i] = y_in_bin.size
#             else:
#                 stat[i] = np.nan
#                 z[i] = np.nan
#                 if return_counts:
#                     counts[i] = 0
#     elif what in ('mean', 'sum', 'std'):
#         counts = np.zeros(nbins, dtype=np.int64)
#         sums   = np.zeros(nbins, dtype=np.float64)
#         sumsqs = np.zeros(nbins, dtype=np.float64)
#         compute_bins_numba_no_atomic(x, y, bin_edges, nbins, counts, sums, sumsqs, nchunks)
        
#         if what == 'mean':
#             stat = sums / np.where(counts == 0, 1, counts)
#         elif what == 'sum':
#             stat = sums
#         elif what == 'std':
#             means = sums / np.where(counts == 0, 1, counts)
#             variance = sumsqs / np.where(counts == 0, 1, counts) - means**2
#             stat = np.sqrt(np.maximum(variance, 0))
        
#         # Also compute z as the standard deviation (or error) per bin.
#         means_for_std = sums / np.where(counts == 0, 1, counts)
#         variance = sumsqs / np.where(counts == 0, 1, counts) - means_for_std**2
#         z = np.sqrt(np.maximum(variance, 0))
#         if what == 'mean' and not std_or_error_of_mean:
#             z = z / np.where(counts == 0, 1, np.sqrt(counts))
#     else:
#         raise NotImplementedError("Unsupported statistic. Use 'mean', 'sum', 'std', or 'median'.")
    
#     extra = ()
#     if return_percentiles:
#         bin_indices = np.searchsorted(bin_edges, x, side='right') - 1
#         bin_indices = np.clip(bin_indices, 0, nbins - 1)
#         perc_lower = np.empty(nbins, dtype=np.float64)
#         perc_upper = np.empty(nbins, dtype=np.float64)
#         for i in range(nbins):
#             y_in_bin = y[bin_indices == i]
#             if y_in_bin.size > 0:
#                 perc_lower[i] = np.percentile(y_in_bin, lower_percentile)
#                 perc_upper[i] = np.percentile(y_in_bin, higher_percentile)
#             else:
#                 perc_lower[i] = np.nan
#                 perc_upper[i] = np.nan
#         extra += ((perc_lower, perc_upper),)
    
#     if return_counts and what != 'median':
#         extra = (counts,) + extra
#     elif return_counts and what == 'median':
#         extra = (counts,) + extra
    
#     return (bin_centers, stat, z) + extra

# # --- Numba Wrapper for Warmup ---
# def binned_quantity(x, y, *args, **kwargs):
#     """
#     Wrapper that first calls binned_quantity_numba on a tiny subset (using only the first element)
#     to trigger the JIT compilation and then calls binned_quantity_numba on the full data.
#     Returns only the full-data results.
#     """
#     x = np.asarray(x)
#     if x.size > 0:
#         _ = binned_quantity_numba(x[:1], y[:1], *args, **kwargs)
#     return binned_quantity_numba(x, y, *args, **kwargs)




@jit(nopython=True, parallel=True)
def mean_manual(xpar,
                ypar,
                what='mean',
                std_or_std_mean=True,
                nbins=100,
                loglog=True,
                upper_percentile=95,
                remove_upper_percentile=False):
    """
    Calculate manually binned statistics of one variable (ypar) with respect to another variable (xpar).

    Parameters
    ----------
    xpar : array_like
        Input array. This represents the independent variable.
    ypar : array_like
        Input array. This represents the dependent variable.
    what : str, optional
        The type of binned statistic to compute. It can be 'mean' or 'median'. The default is 'mean'.
    std_or_std_mean : bool, optional
        Indicates whether to return the standard deviation (True) or the standard error of the mean (False) of the binned statistic.
        The default is True.
    nbins : int, optional
        The number of bins to use for the histogram. The default is 100.
    loglog : bool, optional
        If True, logarithmic bins are used instead of linear bins. The default is True.
    upper_percentile : int, optional
        The upper percentile value for removing outliers. The default is 95.
    remove_upper_percentile : bool, optional
        If True, remove upper percentiles to eliminate outliers. The default is False.

    Returns
    -------
    ndarray
        The centers of the bins for the independent variable.
    ndarray
        The value of the binned statistic (mean or median) for the dependent variable.
    ndarray
        The standard deviation or standard error of the mean of the binned statistic.

    Notes
    -----
    This function manually calculates binned statistics by dividing the data into specified bins and computing the mean or median
    for the dependent variable (ypar) in each bin. It also calculates the standard deviation or standard error of the mean
    depending on the `std_or_std_mean` parameter.

    Example
    --------
    >>> import numpy as np
    >>> from numba import jit, prange

    >>> @jit(nopython=True, parallel=True)
    ... def mean_manual(xpar, ypar, what='mean', std_or_std_mean=True, nbins=100, loglog=True, upper_percentile=95, remove_upper_percentile=False):
    ...     # (Your function implementation here)

    >>> x_values = np.random.rand(1000)
    >>> y_values = np.random.randn(1000)

    >>> x_mean, y_mean, y_std = mean_manual(x_values, y_values)
    >>> print(x_mean)
    >>> print(y_mean)
    >>> print(y_std)
    """
    
    xpar = np.array(xpar)
    ypar = np.array(ypar)
    ind = (ypar > -1e10) & (xpar > -1e10) & (~np.isinf(xpar)) & (~np.isinf(ypar))

    xpar = xpar[ind]
    ypar = ypar[ind]

    if loglog:
        bins = np.logspace(np.log10(np.nanmin(xpar)), np.log10(np.nanmax(xpar)), nbins)
    else:
        bins = np.linspace(np.nanmin(xpar), np.nanmax(xpar), nbins)

    res1 = np.digitize(xpar, bins)

    bin_counts = np.bincount(res1)

    ypar_mean  = np.zeros(nbins)
    ypar_std   = np.zeros(nbins)
    xpar_mean  = np.zeros(nbins)
    ypar_count = np.zeros(nbins)

    for i in prange(len(bin_counts)):
        if bin_counts[i] == 0:
            continue

        xvalues1 = xpar[res1 == i]
        yvalues1 = ypar[res1 == i]

        if remove_upper_percentile:
            percentile = np.percentile(yvalues1, upper_percentile)
            xvalues1 = xvalues1[yvalues1 < percentile]
            yvalues1 = yvalues1[yvalues1 < percentile]

        if what == 'mean':
            ypar_mean[i] = np.nanmean(yvalues1)
            xpar_mean[i] = np.nanmean(xvalues1)
        else:
            ypar_mean[i] = np.nanmedian(yvalues1)
            xpar_mean[i] = np.nanmedian(xvalues1)

        ypar_std[i] = np.nanstd(yvalues1)
        ypar_count[i] = bin_counts[i]

    if std_or_std_mean == 0:
        z_b = ypar_std / np.sqrt(ypar_count)
    else:
        z_b = ypar_std

    return xpar_mean, ypar_mean, z_b



def find_fit_semilogy(x, y, x0, xf): 
    def line(x, a, b):
        return a*x+b
    # Apply fit on specified range #
    if  len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0])>0:
        s = np.where(x == x.flat[np.abs(x - x0).argmin()])[0][0]
        e = np.where(x  == x.flat[np.abs(x - xf).argmin()])[0][0]

        if (len(y[s:e])>1): #& (np.median(y[s:e])>1e-1):  
            fit = fun.curve_fit(line, x[s:e],np.log10(y[s:e]))
            y = 10**line(x[s:e], fit[0][0], fit[0][1]) 
            return fit, s, e, x[s:e], y
        else:
            return [0],0,0,0,[0]
        
        
        
# import numpy as np
# from scipy.optimize import minimize

# def three_plaw_fit(x: np.ndarray, y: np.ndarray, num_segments: int = 3, max_iter: int = 10000,
#                    middle_weight: float = 1.0, initial_breakpoints: np.ndarray = None,
#                    breakpoint_bounds: list = None):
#     """
#     Optimize the breakpoints for a piecewise power-law fit with continuity constraints,
#     and weight the middle segment more heavily in the optimization.

#     Parameters:
#     -----------
#     x : np.ndarray
#         Independent variable data.
#     y : np.ndarray
#         Dependent variable data.
#     num_segments : int, optional
#         Number of segments for the piecewise power-law fit. Default is 3.
#     max_iter : int, optional
#         Maximum number of iterations for the optimizer. Default is 10000.
#     middle_weight : float, optional
#         Weight applied to the residuals of the middle segment. Default is 1.0.
#     initial_breakpoints : np.ndarray, optional
#         Initial guesses for the breakpoint x-values. Should be of length num_segments - 1.
#     breakpoint_bounds : list of tuples, optional
#         Bounds for the breakpoint x-values. Should be a list of tuples with length num_segments - 1.

#     Returns:
#     --------
#     fits_dict : dict
#         Dictionary containing the fit results for each segment.
#     """
#     # Ensure x and y are sorted by x
#     sort_idx = np.argsort(x)
#     x = x[sort_idx]
#     y = y[sort_idx]

#     # Remove any NaN or infinite values
#     finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
#     x = x[finite_mask]
#     y = y[finite_mask]

#     n = len(x)
#     logx = np.log(x)
#     logy = np.log(y)

#     # Initial guess for breakpoints (x-values)
#     if initial_breakpoints is None:
#         initial_breakpoints = np.linspace(
#             x[0],
#             x[-1],
#             num_segments + 1
#         )[1:-1]  # Exclude the first and last point

#     # Ensure initial breakpoints satisfy constraints
#     initial_breakpoints = np.array(initial_breakpoints)
#     initial_breakpoints += np.arange(num_segments - 1) * 1e-5

#     # Initial guesses for slopes and intercepts
#     initial_a = np.full(num_segments, -1.0)  # Initial slope guesses
#     initial_logc = np.full(num_segments, np.mean(logy))  # Initial intercept guesses

#     # Combine all variables into a single array
#     x0 = np.concatenate([initial_breakpoints, initial_a, initial_logc])

#     # Define bounds for variables
#     if breakpoint_bounds is None:
#         breakpoint_bounds = [(x[1], x[-2])] * (num_segments - 1)
#     else:
#         breakpoint_bounds = [(max(b[0], x[1]), min(b[1], x[-2])) for b in breakpoint_bounds]

#     bounds = breakpoint_bounds  # Bounds for breakpoints (x-values)
#     bounds += [(-np.inf, np.inf)] * (2 * num_segments)  # Bounds for slopes and intercepts

#     # Constraints: Ordering of breakpoints
#     constraints = []
#     for i in range(num_segments - 2):
#         def breakpoint_order_constraint(x_vars, i=i):
#             return x_vars[i + 1] - x_vars[i] - 1e-5
#         constraints.append({
#             'type': 'ineq',
#             'fun': breakpoint_order_constraint
#         })

#     # Continuity constraints at breakpoints
#     for i in range(num_segments - 1):
#         def continuity_constraint(x_vars, i=i):
#             # Breakpoint x-value
#             x_b = x_vars[i]
#             if x_b <= x[0] or x_b >= x[-1]:
#                 return 0  # Return zero to avoid errors

#             # Slopes and intercepts
#             a_i = x_vars[num_segments - 1 + i]
#             logc_i = x_vars[2 * num_segments - 1 + i]
#             a_next = x_vars[num_segments - 1 + i + 1]
#             logc_next = x_vars[2 * num_segments - 1 + i + 1]

#             # Continuity equation
#             y_i = logc_i + a_i * np.log(x_b)
#             y_next = logc_next + a_next * np.log(x_b)
#             return y_i - y_next  # Should be zero for continuity

#         constraints.append({
#             'type': 'eq',
#             'fun': continuity_constraint
#         })

#     # Objective function with weighted middle segment
#     def objective(x_vars):
#         # Extract variables
#         breakpoints = x_vars[:num_segments - 1]
#         a_i = x_vars[num_segments - 1:2 * num_segments - 1]
#         logc_i = x_vars[2 * num_segments - 1:]

#         # Clip and sort breakpoints
#         breakpoints = np.clip(breakpoints, x[1], x[-2])
#         breakpoints = np.sort(breakpoints)

#         # Determine the indices where the breakpoints occur
#         indices = np.searchsorted(x, breakpoints)
#         start_idx = np.concatenate(([0], indices))
#         end_idx = np.concatenate((indices, [n]))

#         residuals = []
#         for i in range(num_segments):
#             idx = slice(start_idx[i], end_idx[i])
#             x_seg = logx[idx]
#             y_seg = logy[idx]

#             y_fit = logc_i[i] + a_i[i] * x_seg
#             res = y_seg - y_fit

#             # Apply weighting to the middle segment
#             if num_segments == 3 and i == 1:
#                 weight = middle_weight
#             else:
#                 weight = 1.0

#             residuals.extend(weight * res)

#         residuals = np.array(residuals)
#         return np.sum(residuals ** 2)

#     # Minimize the total residuals with tighter tolerances
#     res = minimize(
#         objective,
#         x0,
#         method='SLSQP',
#         bounds=bounds,
#         constraints=constraints,
#         options={
#             'maxiter': int(max_iter),
#             'disp': True,
#             'ftol': 1e-12,   # Decrease function tolerance
#             'eps': 1e-12     # Decrease step size for numerical gradient
#         }
#     )

#     if not res.success:
#         print("Optimizer did not converge:", res.message)

#     # Extract optimized variables
#     x_vars = res.x
#     breakpoints = x_vars[:num_segments - 1]
#     a_i = x_vars[num_segments - 1:2 * num_segments - 1]
#     logc_i = x_vars[2 * num_segments - 1:]

#     # Clip and sort breakpoints
#     breakpoints = np.clip(breakpoints, x[1], x[-2])
#     breakpoints = np.sort(breakpoints)

#     # Determine the indices where the breakpoints occur
#     indices = np.searchsorted(x, breakpoints)
#     start_idx = np.concatenate(([0], indices))
#     end_idx = np.concatenate((indices, [n]))

#     # Get the x-values of the breakpoints
#     breakpoints_values = breakpoints  # These are already x-values

#     # Build fits_dict
#     fits_dict = {}
#     segment_labels = ['p{}'.format(i + 1) for i in range(num_segments)]
#     for i, label in enumerate(segment_labels):
#         idx_range = slice(start_idx[i], end_idx[i])
#         x_seg = x[idx_range]
#         y_seg = y[idx_range]

#         # Compute predicted y values
#         y_fit = np.exp(logc_i[i] + a_i[i] * np.log(x_seg))

#         # Compute residuals
#         residuals = np.log(y_seg) - (logc_i[i] + a_i[i] * np.log(x_seg))
#         residual_sum = np.sum(residuals ** 2)

#         # Compute standard error of the slope
#         n_seg = len(x_seg)
#         if n_seg > 2:
#             s_squared = residual_sum / (n_seg - 2)
#             Sxx = np.sum((np.log(x_seg) - np.mean(np.log(x_seg))) ** 2)
#             if Sxx > 0:
#                 s_a = np.sqrt(s_squared / Sxx)
#             else:
#                 s_a = np.nan
#         else:
#             s_a = np.nan

#         fits_dict[label] = {
#             'plaw-index': a_i[i],
#             'plaw-index-err': s_a,
#             'err': residual_sum,
#             'xv': x_seg,
#             'yv': y_fit,
#             'x_break': breakpoints_values[i] if i < num_segments - 1 else np.nan,
#             'n_iter': res.nit
#         }

#     return fits_dict



import numpy as np
from scipy.ndimage import gaussian_filter1d

def local_slope(x, y, bin_size=1, smoothing_sigma=3, return_max_diff_points=False):
    """
    Compute the local slope of y with respect to x in log-log space, and estimate y at midpoints.

    Parameters:
    -----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    bin_size : int, optional
        The number of data points to include in each bin. Default is 1 (no binning).
    smoothing_sigma : float, optional
        The standard deviation for Gaussian kernel used in smoothing. Default is 3.
    return_max_diff_points : bool, optional
        If True, the function returns the x-values where the absolute differences
        of the slopes are maximum.

    Returns:
    --------
    midpoints : np.ndarray
        The x-values at the middle of the bins where the slopes are estimated.
    slopes_smooth : np.ndarray
        The smoothed local slopes computed in log-log space.
    y_midpoints : np.ndarray
        The y-values estimated at the midpoints.
    max_diff_points : np.ndarray (optional)
        The x-values where the absolute differences of the slopes are maximum.
        Only returned if `return_max_diff_points` is True.
    """
    # Ensure x and y are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Ensure x and y are sorted by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove any NaN or infinite values and non-positive values
    finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[finite_mask]
    y = y[finite_mask]

    # Log-transform x and y
    logx = np.log(x)
    logy = np.log(y)

    # Binning
    if bin_size > 1:
        num_complete_bins = len(logx) // bin_size
        logx_binned = np.array([
            np.mean(logx[i * bin_size:(i + 1) * bin_size]) for i in range(num_complete_bins)
        ])
        logy_binned = np.array([
            np.mean(logy[i * bin_size:(i + 1) * bin_size]) for i in range(num_complete_bins)
        ])
    else:
        logx_binned = logx
        logy_binned = logy

    # Compute the differences in log-log space
    dlogx = np.diff(logx_binned)
    dlogy = np.diff(logy_binned)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = np.true_divide(dlogy, dlogx)
        slopes[~np.isfinite(slopes)] = 0  # Replace infinities and NaNs with zero

    # Compute midpoints of x and y in log space
    logx_mid = (logx_binned[:-1] + logx_binned[1:]) / 2
    midpoints = np.exp(logx_mid)

    logy_mid = (logy_binned[:-1] + logy_binned[1:]) / 2
    y_midpoints = np.exp(logy_mid)

    # Smooth the slopes to reduce noise
    slopes_smooth = gaussian_filter1d(slopes, sigma=smoothing_sigma)

    if return_max_diff_points:
        # Compute differences of slopes
        slope_diffs = np.diff(slopes_smooth)
        # Find indices where the absolute differences are maximum
        max_diff_indices = np.where(np.abs(slope_diffs) == np.max(np.abs(slope_diffs)))[0]
        # Corresponding x-values (midpoints between midpoints)
        x_max_diff = (midpoints[max_diff_indices] + midpoints[max_diff_indices + 1]) / 2
        return midpoints, slopes_smooth, y_midpoints, x_max_diff
    else:
        return midpoints, slopes_smooth, y_midpoints




from scipy.optimize import differential_evolution
import numpy as np

def three_plaw_fit(x: np.ndarray, y: np.ndarray, num_breaks: int = 2):
    """
    Fit a piecewise power-law (3 segments) to the data (x, y) by optimizing the breakpoints.
    Parameters:
    -----------
    x : np.ndarray
        Independent variable data.
    y : np.ndarray
        Dependent variable data.
    num_breaks : int, optional
        Number of breakpoints (Default is 2, resulting in 3 segments)
    Returns:
    --------
    fits_dict : dict
        Dictionary containing the fit results for each segment, including standard errors.
    """

    # Ensure x and y are sorted by x
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]

    # Remove any NaN or infinite values
    finite_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[finite_mask]
    y = y[finite_mask]

    logx = np.log(x)
    logy = np.log(y)

    # Define the objective function
    def objective(breakpoints):
        # Ensure breakpoints are sorted and within x range
        breakpoints = np.sort(breakpoints)
        if np.any(breakpoints <= x[0]) or np.any(breakpoints >= x[-1]):
            return np.inf  # Penalty for invalid breakpoints

        # Split data into segments
        residuals = []
        previous_idx = 0

        for bp in breakpoints:
            idx = np.searchsorted(x, bp, side='right')
            x_seg = logx[previous_idx:idx]
            y_seg = logy[previous_idx:idx]

            # Linear regression in log-log space
            if len(x_seg) > 1:
                A = np.vstack([x_seg, np.ones(len(x_seg))]).T
                slope, intercept = np.linalg.lstsq(A, y_seg, rcond=None)[0]
                y_fit = intercept + slope * x_seg
                residuals.extend(y_seg - y_fit)
            else:
                return np.inf  # Penalty for too few points in segment

            previous_idx = idx

        # Last segment
        x_seg = logx[previous_idx:]
        y_seg = logy[previous_idx:]
        if len(x_seg) > 1:
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            slope, intercept = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            y_fit = intercept + slope * x_seg
            residuals.extend(y_seg - y_fit)
        else:
            return np.inf  # Penalty for too few points in segment

        residuals = np.array(residuals)
        return np.sum(residuals ** 2)

    # Define bounds for breakpoints
    bounds = [(x[1], x[-2])] * num_breaks  # Avoid the very first and last points

    # Use differential evolution for global optimization
    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=False
    )

    if not result.success:
        print("Optimization failed.")
        return None

    # Get the best breakpoints
    best_breakpoints = np.sort(result.x)

    # Now, compute the final fit parameters
    fits_dict = {}
    segment_labels = ['p{}'.format(i + 1) for i in range(num_breaks + 1)]
    previous_idx = 0
    residuals_total = []

    for i, bp in enumerate(np.append(best_breakpoints, x[-1])):
        idx = np.searchsorted(x, bp, side='right')
        x_seg = x[previous_idx:idx]
        y_seg = y[previous_idx:idx]
        logx_seg = logx[previous_idx:idx]
        logy_seg = logy[previous_idx:idx]

        if len(x_seg) > 1:
            A = np.vstack([logx_seg, np.ones(len(logx_seg))]).T
            # Solve for parameters
            beta = np.linalg.lstsq(A, logy_seg, rcond=None)[0]
            slope, intercept = beta
            y_fit = np.exp(intercept + slope * logx_seg)
            residuals = logy_seg - (intercept + slope * logx_seg)
            residual_sum = np.sum(residuals ** 2)

            # Calculate standard errors
            n = len(logx_seg)
            p = 2  # Number of parameters (slope and intercept)
            dof = n - p  # Degrees of freedom
            if dof > 0:
                sigma_squared = np.sum(residuals ** 2) / dof
                # Compute covariance matrix
                cov_beta = sigma_squared * np.linalg.inv(np.dot(A.T, A))
                # Standard errors are square roots of diagonal elements
                standard_errors = np.sqrt(np.diag(cov_beta))
                slope_error, intercept_error = standard_errors
            else:
                slope_error = np.nan
                intercept_error = np.nan
        else:
            slope = np.nan
            intercept = np.nan
            slope_error = np.nan
            intercept_error = np.nan
            y_fit = np.full_like(y_seg, np.nan)
            residual_sum = np.nan

        fits_dict[segment_labels[i]] = {
            'plaw-index': slope,
            'plaw-index-error': slope_error,
            'plaw-intercept': intercept,
            'plaw-intercept-error': intercept_error,
            'err': residual_sum,
            'xv': x_seg,
            'yv': y_fit,
            'x_break': bp if i < num_breaks else np.nan
        }

        residuals_total.extend(residuals)
        previous_idx = idx

    fits_dict['breakpoints'] = best_breakpoints
    fits_dict['total_error'] = np.sum(np.array(residuals_total) ** 2)

    return fits_dict

import numpy as np
from joblib import Parallel, delayed

def mov_fit_func_joblib(xx,
                        yy,
                        w_size,
                        xmin,
                        xmax,
                        keep_plot=0,
                        pad=1,
                        n_jobs=-1):
    """
    Perform moving fits on the data within a specified range.
    Optimized with Joblib for parallel processing.
    
    Parameters
    ----------
    xx : ndarray
        Input array representing the independent variable (x).
    yy : ndarray
        Input array representing the dependent variable (y).
    w_size : float
        Window size used to perform the fits.
    xmin : float
        Minimum value of x for the fitting range.
    xmax : float
        Maximum value of x for the fitting range.
    keep_plot : bool
        If True, additional data for plotting fits is returned.
    pad : int
        Step size to reduce the number of points for fitting.
    n_jobs : int
        Number of parallel jobs to run (-1 uses all available CPUs).
    
    Returns
    -------
    dict
        A dictionary containing information about the fits.
    """
    
    # Convert inputs to arrays and filter based on valid ranges
    xx = np.asarray(xx)
    yy = np.asarray(yy)
    
    mask = (xx > -1e10) & (yy > -1e10)
    xx, yy = xx[mask], yy[mask]

    # Find indices in the range of interest
    index1 = np.searchsorted(xx, xmin, side='left')
    index2 = np.searchsorted(xx, xmax, side='right') - 1
    where_fit = np.arange(index1, index2 + 1, step=int(pad))  # Skip with stride of `pad`

    # Function to perform fit (to be run in parallel)
    def perform_fit(i):
        x0 = xx[i]
        xf = x0 * w_size

        if xf < 0.98 * xmax:
            fit, s, e, x1, y1 = find_fit(xx, yy, x0, xf)
            if len(np.shape(x1)) > 0:
                err = np.sqrt(fit[1][1][1])
                ind = fit[0][1]
                x_val = x1[s]

                result = {
                    'err': err,
                    'ind': ind,
                    'x_val': x_val,
                }

                if keep_plot:
                    result['plot_x'] = x1[s:e]
                    result['plot_y'] = 2 * fit[2]

                return result
        return None

    # Run fits in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(
        delayed(perform_fit)(i) for i in where_fit
    )

    # Extract valid results
    keep_err, keep_ind, keep_x = [], [], []
    xvals, yvals = [], []

    for result in results:
        if result is not None:
            keep_err.append(result['err'])
            keep_ind.append(result['ind'])
            keep_x.append(result['x_val'])
            if keep_plot:
                xvals.append(result['plot_x'])
                yvals.append(result['plot_y'])

    # Prepare the result dictionary
    result_dict = {
        'xvals': np.array(keep_x),
        'plaw': np.array(keep_ind),
        'fit_err': np.array(keep_err),
    }

    if keep_plot:
        result_dict['plot_x'] = xvals
        result_dict['plot_y'] = yvals

    return result_dict


import numpy as np
from scipy.optimize import curve_fit



def moving_fit(x, 
               y,
               fwin,
               df, 
               make_df_adapt_2_scale = True,
               df_multiplier         = 0.005
              ):
    """
    Process data by fitting curves within specified window and step sizes.

    Parameters:
    x (np.ndarray): Array of x values.
    y (np.ndarray): Array of y values.
    fwin (float): Window size for the logarithmic fitting.
    df (float): Step size for shifting the window in logarithmic scale.

    Returns:
    list: List of fit values.
    list: List of x midpoints of each fitting window.
    """
    # Delete NaNs
    ind = np.isnan(y)
    x = x[np.invert(ind)]
    y = y[np.invert(ind)]

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)


    x1 = xmin
    x2 = fwin*xmin
    
    if make_df_adapt_2_scale:
        df = x1*df_multiplier
        print('Using adaptive df. Init Value:',df)
    
    
    fit_vals  = []
    xmids     = []

    while x2 < xmax:
        try:
            fit, _, _, flag = curve_fit_log_wrap(x, y, x1, x2)

            fit_vals.append(fit[0][1])   
            xmids.append(x1)
                         

        except:
            fit_vals.append(np.nan)
            xmids.append(x1)
                            
        x1 = x1 + df
        x2 = fwin*x1
        
        if make_df_adapt_2_scale:
            df = x1*df_multiplier
        

    return  xmids, fit_vals

def freq2wavenum(freq, P, Vtot, di):
    """ Takes the frequency, the PSD, the SW velocity and the di.
        Gives the k* and the E(k*), normalised with di"""
    
    # xvals          =  xvals/Vtotal*(2*np.pi*di)
    # yvals          =  yvals*(2*np.pi*di)/Vtotal

    
    k_star = freq/Vtot*(2*np.pi*di)
    
    eps_of_k_star = P*Vtot/(2*np.pi*di)
    
    return k_star, eps_of_k_star

def freq2wavenum_only_kdi(freq, Vtot, di):
    """ Takes the frequency, the PSD, the SW velocity and the di.
        Gives the k* and the E(k*), normalised with di"""
    
    # xvals          =  xvals/Vtotal*(2*np.pi*di)
    # yvals          =  yvals*(2*np.pi*di)/Vtotal

    
    k_star = freq/Vtot*(2*np.pi*di)
    
    
    return k_star

import numpy as np

def freq2wavenum_only_kdi_arrays(freq, Vtot, di):
    """
    Returns a 2D array k_star of shape (len(Vtot), len(freq)),
    i.e. (14401, 184).
    """
    freq = np.asarray(freq).reshape(1, -1)     # (1, 184)
    Vtot = np.asarray(Vtot).reshape(-1, 1)     # (14401, 1)
    di   = np.asarray(di).reshape(-1, 1)       # (14401, 1)

    # Elementwise: (1,184) / (14401,1) * 2*pi*(14401,1) 
    # => shape (14401, 184)
    k_star = freq / Vtot * (2 * np.pi * di)  

    return k_star.T



import numpy as np

def integrate_psd_in_k_range_2d(kdi_2d, psd_2d, kmin, kmax):
    """
    Integrate the PSD in each column of `psd_2d` over the k-range
    [kmin, kmax], using the corresponding k-values in `kdi_2d`.

    Parameters
    ----------
    kdi_2d : (M, N) array
        2D array of wavenumbers. Each column can have its own k-values.
    psd_2d : (M, N) array
        2D array of PSD values matching the shape of kdi_2d.
    kmin : float
        Lower limit of k-range to integrate over.
    kmax : float
        Upper limit of k-range to integrate over.

    Returns
    -------
    integrated : (N,) ndarray
        The integrated PSD for each of the N columns.
        If no valid data in a column, returns np.nan in that position.
    """
    kdi_2d = np.asarray(kdi_2d)
    psd_2d = np.asarray(psd_2d)
    M, N   = kdi_2d.shape

    result = np.full(N, np.nan)  # Default to NaN

    for i in range(N):
        kvals = kdi_2d[:, i]
        pvals = psd_2d[:, i]

        # 1) Remove NaNs
        valid_mask = ~np.isnan(kvals) & ~np.isnan(pvals)
        kvals = kvals[valid_mask]
        pvals = pvals[valid_mask]
        
        # 2) Restrict to k in [kmin, kmax]
        range_mask = (kvals >= kmin) & (kvals <= kmax)
        kvals = kvals[range_mask]
        pvals = pvals[range_mask]
        
        # If nothing remains, leave result[i] = np.nan
        if kvals.size == 0:
            continue

        # 3) Sort k in ascending order (so integration is positive if pvals >= 0)
        sort_inds = np.argsort(kvals)
        kvals = kvals[sort_inds]
        pvals = pvals[sort_inds]

        # 4) Integrate with trapezoidal rule
        result[i] = np.trapz(pvals, x=kvals)

    return result




import numpy as np
from scipy.interpolate import griddata

def smooth_2d_data(X, Y, Z, Ntimes):
    """
    Interpolate 2D data on a new grid.

    Parameters:
    X (2D array): X-coordinates of the data.
    Y (2D array): Y-coordinates of the data.
    Z (2D array): Values at each (X, Y) point.
    Ntimes (int): Factor to scale the new grid size.

    Returns:
    Xn, Yn (2D arrays): New meshgrid for X and Y.
    data1 (2D array): Interpolated data on the new grid.
    """

    # Calculate differences and midpoints
    X_diff = np.diff(X, axis=1)
    Y_diff = np.diff(Y.T, axis=1)
    X_mid = X[:, :-1] + X_diff / 2
    Y_mid = (Y.T)[:,:-1] + Y_diff / 2

    # Flatten the arrays
    x = X_mid[1:, :].flatten()
    y = (Y_mid[1:,:].T).flatten()
    z = Z.flatten()

    # Filter out NaN values
    mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]

    # Define the new grid for interpolation
    xnew = np.logspace(np.log10(np.nanmin(X)), np.log10(np.nanmax(X)), int(Ntimes*len(X[0])))
    ynew = np.logspace(np.log10(np.nanmin(Y)), np.log10(np.nanmax(Y)), int(Ntimes*len(Y[0])))

    # Create a meshgrid for the new grid
    Xn, Yn = np.meshgrid(xnew, ynew)

    # Perform the interpolation
    data1 = griddata((x_filtered, y_filtered), z_filtered, (Xn, Yn), method='linear')

    return Xn, Yn, data1

# Example usage
# Xn, Yn, interpolated_data = interpolate_2d_data(X, Y, Z, Ntimes)


def smooth(x, n=5):
    """
    Apply a running mean smoothing to the input signal.

    Parameters
    ----------
    x : ndarray
        The signal to be smoothed.
    n : int, optional
        Window width for the running mean. The default is 5.

    Returns
    -------
    ndarray
        The smoothed signal of the same length as *x*.

    Notes
    -----
    This function applies a running mean smoothing to the input signal using a convolution operation.
    The running mean is calculated using a window of width `n`, and the smoothed signal is returned.
    The convolution operation is performed in 'same' mode to ensure that the output has the same length as the input.

    """
    return np.convolve(x, np.ones(n) / n, mode='same')




def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


def resample_find_equal_elements(keep_unique, interpolate, xarr1, yarr1, xarr2, yarr2,choose_min_max, interp_method, npoints,parx_min, parx_max, perx_min, perx_max ):
    
    if interpolate:
        df_par = pd.DataFrame({'x': xarr1, 'y': yarr1}).set_index('x')
        df_per = pd.DataFrame({'x': xarr2, 'y':yarr2}).set_index('x')



        if choose_min_max:
            parx_min, parx_max = np.nanmin(df_par.index.values),np.nanmax(df_par.index.values)
            perx_min, perx_max = np.nanmin(df_per.index.values),np.nanmax(df_per.index.values)   

        new_index_par  = np.logspace(np.log10(parx_min), np.log10(parx_max), npoints)
        new_index_per  = np.logspace(np.log10(perx_min), np.log10(perx_max), npoints)

        # new_index_par  = np.linspace((parx_min), (parx_max), npoints)
        # new_index_per  = np.linspace((perx_min), (perx_max), npoints)

        df_par         =    newindex(df_par, new_index_par, interp_method)
        df_per         =    newindex(df_per, new_index_per, interp_method)
        # df_par         = func.interp(df_par, new_index_par)
        # df_per         = func.interp(df_per, new_index_per)


        x_para, y_para   = np.real(df_par.index.values),np.real( df_par.values.T[0])
        x_pera, y_pera = np.real(df_per.index.values),np.real( df_per.values.T[0])
    else:

        x_para, y_para   = xarr1, yarr1
        x_pera, y_pera   = xarr2, yarr2  
    
    
    res = closest_argmin(y_para, y_pera)


    xparnew, yparnew = x_para,y_para
    xpernew, ypernew = x_pera[res], y_pera[res]

    if keep_unique:
        unq, unq_inv, unq_cnt = np.unique(np.sort(res), return_inverse=True, return_counts=True)

        xparnew1, yparnew1  = xparnew[unq], yparnew[unq]
        xpernew1, ypernew1 = xpernew[unq], ypernew[unq]
    else:
        xparnew1, yparnew1  = xparnew, yparnew
        xpernew1, ypernew1 = xpernew, ypernew     
    
    index1 =  np.argsort(xpernew1)
    
    
    return  xparnew1[index1], yparnew1[index1], xpernew1[index1], ypernew1[index1]


def fit(x, y, deg=1, fullyes=False):
    """
    Fit function wrapper that calls `nupmpy.polyfit`. Returns the fit parameters as well as the standard deviation
    of the fit.

    Args:
        x: [ndarray] X-coordinates of the data points
        y: [ndarray] Y-coordinates of the data points (same shape as *x*)
        deg: [int] Degree of the polynomial
        fullyes: [boolean] Whether to return the full set of arguments from the fit function or not.
        Argument forwarded to `numpy.polyfit()`

    Returns:
        fitpars: [list] List of fit parameters containing at least the polynomial coefficients, and also residuals,
        rank, etc. See numpy.polyfit for full details.
        fitpars_std: [numpy.ndarray] The standard deviation of each fit parameter estimate.

    """

    try:
        if np.any(np.isnan(x + y)):
            raise ValueError('Input argument *x* or *y* contains NAN.')
    except ValueError as err:
        err_fitpars = env.ERRORVAL*np.ones(deg+1)
        err_cov = env.ERRORVAL*np.ones(deg+1)
        return err_fitpars, err_cov

    if fullyes:
        fitpars, cov = np.polyfit(x, y, deg=deg, cov=True)
        fitpars_std = np.sqrt(np.diag(cov))
    else:
        fitpars = np.polyfit(x, y, deg=deg)
        fitpars_std = env.ERRORVAL

    return fitpars, fitpars_std


def savepickle(df_2_save, save_path, filename):
    """
    Save a list of variables into a single pickle file.

    Parameters
    ----------
    df_2_save : object
        The data or variables to be saved in the pickle file.
    save_path : str
        The path to the folder where the file will be saved.
    filename : str
        The name of the file to save (including the extension).

    Returns
    -------
    None

    Notes
    -----
    This function creates the specified directory (`save_path`) if it doesn't exist and saves the data or variables (`df_2_save`)
    into a single pickle file with the provided filename.

    """
    
    # Ensure the directory exists
    os.makedirs(str(save_path), exist_ok=True)
    
    # Use the highest protocol available for more efficient serialization
    # Open the file using a context manager to ensure it's properly closed after writing
    file_path = Path(save_path).joinpath(filename)
    with open(file_path, 'wb') as file:
        pickle.dump(df_2_save, file, protocol=pickle.HIGHEST_PROTOCOL)
        
      

# def savefeather(df, path_to_save, filename, include_index= True):
#     """
#     Saves a DataFrame to a Feather file.

#     Parameters:
#     - df: pandas.DataFrame, the DataFrame to save.
#     - path_to_save: str, the directory path where the Feather file will be saved.
#     - filename: str, the name of the Feather file to be saved.

#     Returns:
#     - None, saves the file to the specified path.
#     """
#     if include_index:
#         df.reset_index(inplace=True)
        
#     # Construct the full file path
#     full_file_path = f"{path_to_save}/{filename}"
    
#     # Save the DataFrame to a Feather file
#     df.to_feather(full_file_path)
    
def saveparquet(df, path_to_save, filename, column_names=None):
    """
    Saves a DataFrame to a Parquet file, with an option to save only specified columns.
    Checks if the save path exists, and creates it if it doesn't.

    Parameters:
    - df            : pandas.DataFrame, the DataFrame to save.
    - path_to_save  : str, the directory path where the Parquet file will be saved.
    - filename      : str, the name of the Parquet file to be saved.
    - column_names  : list (optional), a list of column names to save from the DataFrame. If None, all columns are saved.

    Returns:
    - None, saves the file to the specified path.
    """
    # Check if the path exists, create it if it doesn't
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save, exist_ok=True)

    # Construct the full file path
    full_file_path = os.path.join(path_to_save, filename)
    
    # If column_names is specified, select only those columns
    if column_names is not None:
        df_to_save = df[column_names]
    else:
        df_to_save = df
    
    # Save the DataFrame (or the subset) to a Parquet file
    df_to_save.to_parquet(full_file_path)
    
import pandas as pd

def load_parquet(path_to_save, filename= None, column_names= None, engine='pyarrow'):
    """
    Reads specific columns from a Parquet file using the specified engine.

    Parameters:
    - path_to_save: str, the directory path where the Parquet file is saved.
    - filename: str, the name of the Parquet file.
    - column_names: list, a list of column names to read from the Parquet file.
    - engine: str, the engine to use for reading the Parquet file ('pyarrow' or 'fastparquet').

    Returns:
    - A pandas DataFrame containing only the specified columns.
    """
    # Construct the full file path
    if filename== None:
        full_file_path = f"{path_to_save}"    
    else:
        full_file_path = f"{path_to_save}/{filename}"

    # Read specific columns from the Parquet file using the specified engine
    df = pd.read_parquet(full_file_path, columns=column_names, engine=engine)
    
    return df


def replace_filename_extension(oldfilename, newextension, addon=False):
    """
    Replace the extension of the file name with *newextension*

    Args:
        oldfilename: [str] file name to be changed
        newextension: [str] the new extension
        addon: [boolean] whether or not to add on the new extension, or replace old extension with new

    Returns:
        newfilename: [str] filename with the new extension
    """

    # extension is the part after the last period in the filename
    dot_ix = oldfilename.rfind('.')

    # if oldfilename doesn't have extension, then just add the new extension
    if dot_ix == -1:
        addon = True

    # if desired, just add the new extension forming double extension file like `filename.old.new`
    if addon:
        dot_ix = len(oldfilename)

    return oldfilename[:dot_ix] + '.' + newextension.strip('.')




# def newindex(df, ix_new, interp_method='linear'):
#     """
#     Reindex a DataFrame according to the new index *ix_new* supplied, ensuring no duplicate labels in the index.

#     Args:
#         df: [pandas DataFrame] The dataframe to be reindexed.
#         ix_new: [np.array or pandas Index] The new index.
#         interp_method: [str] Interpolation method to be used; forwarded to `pandas.DataFrame.interpolate`.

#     Returns:
#         df_reindexed: [pandas DataFrame] DataFrame interpolated and reindexed to *ix_new*.
#     """
#     # Remove duplicate indices
#     df = df[~df.index.duplicated(keep='first')].sort_index()

#     # Remove duplicates in new index and sort
#     ix_new = np.unique(ix_new)

#     # Trim the new index to the overlapping range to ensure synchronization constraints
#     start, end = max(df.index.min(), ix_new.min()), min(df.index.max(), ix_new.max())
#     ix_new     = ix_new[(ix_new >= start) & (ix_new <= end)]

#     # Reindex and interpolate
#     df_reindexed = df.reindex(ix_new).interpolate(method=interp_method).dropna()

#     return df_reindexed

def newindex(df, ix_new, interp_method='linear'):
    """
    Reindex a DataFrame according to the new index *ix_new* supplied, ensuring no duplicate labels in the index.

    Args:
        df: [pandas DataFrame] The dataframe to be reindexed.
        ix_new: [np.array] The new index.
        interp_method: [str] Interpolation method to be used; forwarded to `pandas.DataFrame.reindex.interpolate`.

    Returns:
        df3: [pandas DataFrame] DataFrame interpolated and reindexed to *ix_new*.
    """
    # Ensure df.index and ix_new do not contain duplicates
    df     = df[~df.index.duplicated(keep='first')].interpolate().dropna()
    ix_new = np.unique(ix_new)

    # Verify that reindexing is necessary and feasible
    if not np.array_equal(df.index.sort_values(), ix_new.sort()):
        # Sort the DataFrame index in increasing order
        df = df.sort_index(ascending=True)

        # Create combined index from old and new index arrays, ensuring no duplicates
        ix_com = np.unique(np.concatenate([df.index.values, ix_new]))

        # Re-index and interpolate over the non-matching points
        df2 = df.reindex(ix_com).interpolate(method=interp_method)

        # Reindex to the new index, ix_new
        return df2.reindex(ix_new)
    else:
        # If the current index and new index are effectively the same, no reindexing is needed
        print("No reindexing necessary; DataFrame index matches the new index.")
        return df



def listsearch(search_string, input_list):
    """
    Return matching items from a list

    Args:
        search_string: [str] String to search for (starting only)
        input_list: [list] List to search in

    Returns:
        found_list: [list] List of matching items

    """

    return [si for si in input_list if si.startswith(search_string)]


def window_selector(N, win_name='Hanning'):
    """
    Simply a wrapper for *get_window* from *scipy.signal*. Return the window coefficients.

    Args:
        N: [int] Window length
        win_name: [str] Name of the window

    Returns:
        w: [ndarray] Window coefficients
    """
    import scipy.signal as signal

    return signal.windows.get_window(win_name, N)


def chunkify(ts_in, chunk_duration):
    """
    Divide a given timeseries in to chunks of *chunk_duration*

    Args:
        ts_in: [pd.Timeseries] Input timeseries
        chunk_duration: [float] Duration of the chunks in seconds

    Returns:

    """
    #print('converting to chunks of len %.2f sec' % chunk_duration)

    dchunk_str = f'{str(chunk_duration)}S'

    return pd.date_range(
        ts_in[0].ceil('1s'), ts_in[-1].floor('1s'), freq=dchunk_str
    )



def progress_bar(jj, length):
    """
    Display a progress bar showing the completion percentage.

    Parameters
    ----------
    jj : int
        The current progress value.
    length : int
        The total length or maximum value for the progress bar.

    Returns
    -------
    None

    Notes
    -----
    This function displays a simple progress bar indicating the percentage of completion for a task.

    """
    percentage = round(100 * (jj / length), 2)
    print('Completed', percentage)


# def find_ind_of_closest_dates(df, dates):
#     """
#     Find the indices of the closest dates in a DataFrame to a list of input dates.

#     Parameters
#     ----------
#     df : pandas DataFrame
#         Input DataFrame containing time series data with a unique index.
#     dates : list-like
#         List of input dates for which the closest indices need to be found.

#     Returns
#     -------
#     list
#         A list containing the indices of the closest dates in the DataFrame `df` to each element in the `dates` list.

#     Notes
#     -----
#     This function calculates the indices of the closest dates in the DataFrame `df` to each date in the input `dates`.
#     It uses the pandas DataFrame `index.unique().get_loc()` method with the 'nearest' method to find the indices.

#     """
#     return [df.index.unique().get_loc(date, method='nearest') for date in dates]





def find_ind_of_closest_dates(df, dates):
    """
    Find the indices of the closest dates in a DataFrame to a list of input dates in a vectorized manner.

    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame containing time series data with a DateTime index.
    dates : list-like
        List of dates (as pandas Timestamps or compatible types) for which the closest indices need to be found.

    Returns
    -------
    list
        A list containing the indices of the closest dates in the DataFrame `df` to each date in the `dates` list.
    """
    # Ensure the DataFrame index is in datetime64[ns] format
    df_timestamps = df.index.values.astype('datetime64[ns]')
    # Convert input dates to numpy array in datetime64[ns] format
    input_dates = np.array(pd.to_datetime(dates).values.astype('datetime64[ns]'))
    # Calculate the absolute differences between all dates
    abs_diff = np.abs(df_timestamps[:, np.newaxis] - input_dates)
    # Find the index of the minimum difference for each input date
    closest_indices = np.argmin(abs_diff, axis=0)
    return closest_indices.tolist()



def find_closest_values_of_2_arrays(a, b):
    """
    Find the closest values of two arrays and return their indices.

    Parameters
    ----------
    a : array_like
        The first input array.
    b : array_like
        The second input array.

    Returns
    -------
    ndarray
        An array containing pairs of indices where the values in arrays `a` and `b` are closest to each other.

    Notes
    -----
    This function finds the closest values of two arrays `a` and `b` and returns their corresponding indices.
    It searches for the closest values of `b` in `a`, and for each unique index in `a`, it finds the index in `b`
    where the values are closest.

    Example
    --------
    >>> import numpy as np

    >>> def find_closest_values_of_2_arrays(a, b):
    ...     # (Your function implementation here)

    >>> a = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    >>> b = np.array([2.0, 4.0, 6.0, 8.0])
    >>> closest_indices = find_closest_values_of_2_arrays(a, b)
    >>> print(closest_indices)
    """
    dup = np.searchsorted(a, b)
    uni = np.unique(dup)
    uni = uni[uni < a.shape[0]]
    ret_b = np.zeros(uni.shape[0], dtype=int)
    for idx, val in enumerate(uni):
        bw = np.argmin(np.abs(a[val] - b[dup == val]))
        tt = dup == val
        ret_b[idx] = np.where(tt)[0][bw]
    return np.column_stack((uni, ret_b))


def find_cadence(df, mean_or_median_cadence='median'):
    """
    Find the cadence (time interval) between successive timestamps in a DataFrame's index.

    Parameters
    ----------
    df : pandas DataFrame
        The input DataFrame.
    mean_or_median_cadence : str, optional
        The type of cadence to compute. It can be 'Mean' or 'Median'. The default is 'Mean'.

    Returns
    -------
    float
        The mean or median cadence in seconds between successive timestamps in the DataFrame's index.

    Notes
    -----
    This function calculates the cadence (time interval) between successive timestamps in the DataFrame's index.
    It drops any rows with missing values and computes either the mean or median cadence based on the `mean_or_median_cadence` parameter.
    """
    keys = list(df.keys())
    if mean_or_median_cadence == 'mean':
        return np.nanmean((df[keys[0]].dropna().index.to_series().diff() / np.timedelta64(1, 's')))
    else:
        return np.nanmedian((df[keys[0]].dropna().index.to_series().diff() / np.timedelta64(1, 's')))





#def resample_timeseries_estimate_gaps(df, resolution, large_gaps=5)



# def resample_timeseries_estimate_gaps(
#     df,
#     resolution_ms=1000,
#     large_gaps=10.0,
#     aggregator="mean",
#     do_interpolation=True,
#     interpolation_method="time",
#     handle_infs_as_nans=True,
#     enforce_res_not_smaller=True,
#     gap_mode="median"
# ):
#     """
#     Resample a time series and estimate gaps, returning a dictionary with 
#     the same keys as originally specified.
#     """
#     # Prepare the return dictionary with default (in case of error).
#     results = {
#         "Init_dt"      : None,
#         "resampled_df" : None,
#         "Frac_miss"    : 100.0,  # 100% if something fails
#         "Large_gaps"   : None,
#         "Tot_gaps"     : None,
#         "resol"        : np.nan
#     }

#     try:
#         # 1) Ensure DataFrame is sorted by its DateTimeIndex
#         if not df.index.is_monotonic_increasing:
#             df = df.sort_index()

#         # 2) Optionally replace inf/-inf with NaN
#         if handle_infs_as_nans:
#             df = df.replace([np.inf, -np.inf], np.nan)

#         # 3) Compute original cadence (Init_dt) in seconds from consecutive diffs
#         time_diffs = df.index.to_series().diff().dt.total_seconds().dropna()
#         if len(time_diffs) < 1:
#             # if we cannot measure dt
#             init_dt = 0.0
#         else:
#             if gap_mode == "mean":
#                 init_dt = time_diffs.mean()
#             else:
#                 init_dt = time_diffs.median()
#         results["Init_dt"] = init_dt

#         # 4) Compute the total interval duration
#         if len(df.index) < 2:
#             interval_dur_s = 0.0
#         else:
#             interval_dur_s = (df.index[-1] - df.index[0]).total_seconds()

#         # 5) If there's a positive total duration, measure large & total gaps
#         if interval_dur_s > 0:
#             # Large gaps fraction
#             large_gap_mask = time_diffs > large_gaps
#             sum_large_gaps = time_diffs[large_gap_mask].sum()
#             total_large_gaps = 100.0 * sum_large_gaps / interval_dur_s
#             results["Large_gaps"] = total_large_gaps

#             # Possibly enforce final resolution not < init_dt
#             desired_res_s = resolution_ms / 1000.0
#             if enforce_res_not_smaller and init_dt > 0 and (desired_res_s < init_dt):
#                 desired_res_s = init_dt

#             final_res_ms = desired_res_s * 1000.0
#             results["resol"] = final_res_ms

#             # measure total gaps fraction above that final interval
#             tot_gap_mask = time_diffs > desired_res_s
#             sum_tot_gaps = time_diffs[tot_gap_mask].sum()
#             total_gaps = 100.0 * sum_tot_gaps / interval_dur_s
#             results["Tot_gaps"] = total_gaps

#             # 6) Resample with aggregator to get uniform time steps
#             resample_rule = f"{int(round(final_res_ms))}ms"
#             df_resampled_raw = getattr(df.resample(resample_rule), aggregator)()

#             # 7) OPTIONAL: measure fraction missing before interpolation
#             #    (If you only want the fraction in final, skip or keep for debugging)
#             n_vals_raw = df_resampled_raw.size
#             if n_vals_raw > 0:
#                 n_missing_raw = df_resampled_raw.isna().sum().sum()
#                 fraction_missing_raw = 100.0 * n_missing_raw / n_vals_raw
#             else:
#                 fraction_missing_raw = 0.0
#             # print("Fraction missing (pre‐interpolation):", fraction_missing_raw, "%")

#             # 8) Interpolate if requested
#             if do_interpolation:
#                 df_resampled_filled = df_resampled_raw.interpolate(method=interpolation_method)
#             else:
#                 df_resampled_filled = df_resampled_raw

#             # 9) Now measure final fraction of missing
#             n_vals_res = df_resampled_filled.size
#             if n_vals_res > 0:
#                 n_missing_res = df_resampled_filled.isna().sum().sum()
#                 fraction_missing = 100.0 * n_missing_res / n_vals_res
#             else:
#                 fraction_missing = 0.0

#             # 10) If you want absolutely no missing data in final (no NaNs),
#             #     you could do a second fill method:
#             #       df_resampled_filled = df_resampled_filled.fillna(method="ffill").fillna(method="bfill")
#             #
#             #     Or if there's an unbounded region, you might choose a constant fill:
#             #       df_resampled_filled = df_resampled_filled.fillna(0)

#             # Crucially: DO NOT dropna() if you want to keep a continuous time axis
#             #   Because dropna() would remove entire timestamps (rows), reintroducing time gaps

#             results["Frac_miss"]    = fraction_missing
#             results["resampled_df"] = df_resampled_filled

#         else:
#             # If there's no real duration, we can't measure these
#             results["Large_gaps"] = 0.0
#             results["Tot_gaps"]   = 0.0
#             results["Frac_miss"]  = 100.0
#             # We'll leave the rest as is
#     except Exception as e:
#         # If something goes wrong, results dict stays with safe defaults
#         print(f"ERROR in resample_timeseries_estimate_gaps: {e}")

#     return results



    
def resample_timeseries_estimate_gaps(df, resolution, large_gaps=5):
    """
    Resample a time series and estimate gaps.

    Parameters
    ----------
    df : pandas DataFrame
        Input time series data as a pandas DataFrame.
    resolution : int
        Resolution in milliseconds to resample the time series.
    large_gaps : int, optional
        Large gaps threshold in seconds. Gaps greater than this threshold are considered large.
        The default is 10.

    Returns
    -------
    dict
        A dictionary containing the following information:
        - 'Init_dt': Initial resolution of the input time series.
        - 'resampled_df': Resampled DataFrame with interpolated missing values.
        - 'Frac_miss': Fraction of missing values in the resampled interval.
        - 'Large_gaps': Fraction of large gaps (greater than `large_gaps` seconds) in the resampled interval.
        - 'Tot_gaps': Total fraction of gaps (greater than the resampled resolution) in the resampled interval.
        - 'resol': The actual resolution used for resampling.

    Notes
    -----
    This function resamples the input time series `df` to the specified `resolution` using the mean of data points within each resampled interval.
    If the initial resolution of `df` is greater than the desired `resolution`, the function increases the `resolution` slightly until it is lower than the initial resolution.
    The function estimates the fraction of missing values and gaps in the resampled data and provides the information in the returned dictionary.


    """
    try:
        keys    = list(df.keys())
        init_dt = find_cadence(df)


        # Estimate fraction of missing values within interval
        fraction_missing = 100 * len(df[(np.abs(df[keys[0]])>1e10) | (np.isnan(df[keys[0]])) |  (np.isinf(df[keys[0]]))  ])/ len(df)
        
        # Make sure that you resample to a resolution that is lower than the initial df's resolution
        while init_dt > resolution * 1e-3:
            resolution = 1.005 * resolution

        # Estimate duration of interval selected in seconds
        interval_dur = (df.index[-1] - df.index[0]).total_seconds()
        
        # Estimate sum of gaps greater than large_gaps seconds
        res = (df.dropna().index.to_series().diff() / np.timedelta64(1, 's'))

        # Gives you the fraction of large gaps in the time series
        total_large_gaps = 100 * (res[res > large_gaps].sum() / interval_dur)
        
        # Gives you the total fraction of gaps in the time series
        total_gaps = 100 * (res[res > resolution * 1e-3].sum() / interval_dur)

        # Resample time-series to desired resolution
        df_resampled = df.resample(f"{int(resolution)}ms").median().interpolate()


    except:
        init_dt           = None
        df_resampled      = None
        fraction_missing  = 100
        total_gaps        = None
        total_large_gaps  = None
        resolution        = np.nan

    return {
        "Init_dt"         : init_dt,
        "resampled_df"    : df_resampled,
        "Frac_miss"       : fraction_missing,
        "Large_gaps"      : total_large_gaps,
        "Tot_gaps"        : total_gaps,
        "resol"           : resolution
    }




