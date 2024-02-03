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


import numpy as np

import orderedstructs



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

def dot_product(xvec, yvec):
    
    """
    Calculate the dot product between two arrays and return the result.

    Parameters:
        xvec (numpy.ndarray): The first input array for the dot product.
        yvec (numpy.ndarray): The second input array for the dot product.

    Returns:
        numpy.ndarray: The result of the dot product.

    Raises:
        ValueError: If the dimensions of xvec and yvec are not compatible for dot product.
    """
    if xvec.shape != yvec.shape:
        raise ValueError("Incompatible dimensions for dot product.")

    len_x_0, len_x_1 = xvec.shape

    if len_x_0 > len_x_1:
        return custom_nansum_product(xvec, yvec, 1)
    else:
        return custom_nansum_product(xvec, yvec, 0)

# def dot_product(xvec, yvec):
#     """
#     Calculate the dot product between two arrays and return the result.

#     Parameters:
#         xvec (numpy.ndarray): The first input array for the dot product.
#         yvec (numpy.ndarray): The second input array for the dot product.

#     Returns:
#         numpy.ndarray: The result of the dot product.

#     Raises:
#         ValueError: If the dimensions of xvec and yvec are not compatible for dot product.
#     """
#     # Get the shapes of the input arrays
#     len_x_0, len_x_1 = xvec.shape
#     len_y_0, len_y_1 = yvec.shape

#     # Check if the dimensions are compatible for dot product
#     if len_x_0 == len_y_0 and len_x_1 == len_y_1:
#         # Calculate the dot product along the appropriate axis
#         if len_x_0 > len_x_1:
#             return np.abs(np.nansum(xvec * yvec, axis=1))
#         else:
#             return np.abs(np.nansum(xvec * yvec, axis=0))
#     else:
#         raise ValueError("Incompatible dimensions for dot product.")


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
    
    

@njit(parallel=True)
def estimate_vec_magnitude(a):
    """
    Estimate the magnitude of each vector in the input array `a`.

    This function calculates the magnitude of each vector in the input array `a` using the parallelized Numba JIT compiler.
    It takes advantage of parallel processing to improve performance for large arrays.

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
    shape = a.shape[0]
    norms_squared = np.empty(shape, dtype=a.dtype)
    shortest_axis = 0 if a.shape[0] <= a.shape[1] else 1
    for i in prange(shape):
        squared_sum = 0.0
        for j in prange(a.shape[shortest_axis]):
            squared_sum += a[i, j] ** 2
        norms_squared[i] = squared_sum
    return np.sqrt(norms_squared)


def perp_vector(a, b, return_paral_comp = False):
    """
    This function calculates the component of a vector perpendicular to another vector.

    Parameters:
    a (ndarray) : A 2D numpy array representing the first vector.
    b (ndarray) : A 2D numpy array representing the second vector.

    Returns:
    ndarray     : A 2D numpy array representing the component of the first input vector that is perpendicular to the second input vector.
    """
    b_unit = (b.T / estimate_vec_magnitude(b)).T
    proj   = (np.sum((a * b_unit), axis=1, keepdims=True))* b_unit
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


# def string_to_datetime_index(datetime_string, datetime_format='%Y-%m-%d %H:%M:%S.%f'):
#     return pd.to_datetime(datetime_string, format=datetime_format)

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
    start_time (str): The original datetime string in the format '%Y-%m-%d %H:%M:%S'.
    time_amount (int): The amount of time to add.
    time_unit (str): The unit of the added time, either 's' (seconds), 'm' (minutes), 'h' (hours), or 'd' (days).

    Returns:
    str: The datetime string after the specified time has been added, in the format '%Y-%m-%d %H:%M:%S'.

    Raises:
    ValueError: If an invalid time unit is specified.
    """
    import datetime
    units = {'s': 'seconds', 'm': 'minutes', 'h': 'hours', 'd': 'days'}
    unit = units.get(time_unit, None)
    if unit is None:
        raise ValueError("Invalid time unit")
    delta = datetime.timedelta(**{unit: time_amount})
    end_datetime = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S') + delta
    return end_datetime.strftime('%Y-%m-%d %H:%M:%S')



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

@jit(nopython=True)
def calc_medians(window_size, arr, medians): 
    for i in range(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        median = np.median(arr[id0:id1])
        medians[i] = median

@jit(nopython=True)
def calc_medians_std(window_size, arr, medians, medians_diff): 
    k = 1.4826
    for i in range(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        x = arr[id0:id1]
        medians_diff[i] = k * np.median(np.abs(x - np.median(x)))
        
        
@njit(parallel=True) 
def calc_medians_parallel(window_size, arr, medians): 
    for i in prange(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        median = np.median(arr[id0:id1])
        medians[i] = median

@njit(parallel=True) 
def calc_medians_std_parallel(window_size, arr, medians, medians_diff): 
    k = 1.4826  # scale factor for Gaussian distribution
    for i in prange(window_size, len(arr)-window_size, 1):
        id0 = i - window_size
        id1 = i + window_size
        x = arr[id0:id1]
        medians_diff[i] = k * np.median(np.abs(x - np.median(x)))
        
        


    
    return xv, [arr[0] + sum(smooth_grad[:x]) for x in range(len(arr))]


def hampel(arr, window_size=200, n=3, parallel=False):
    """
    Apply Hampel filter to despike a time series by removing spurious data points.

    The Hampel filter is a robust method used for detecting and replacing outliers (spikes) in a time series.
    This function applies the Hampel filter to the input array `arr` and replaces the outliers with the median of
    neighboring values within a specified window.

    Parameters:
    ----------
    arr : numpy.ndarray
        The input time series as a 1-dimensional numpy array.
    window_size : int, optional
        The size of the sliding window used to compute the median and standard deviation of neighboring values.
        The default value is 5.
    n_sigmas : int or float, optional
        The number of standard deviations away from the median used to define outliers. Data points that deviate
        from the median by more than `n_sigmas` times the median absolute deviation are considered outliers.
        The default value is 3.

    Returns:
    -------
    numpy.ndarray, tuple
        A tuple containing two elements:
        1. A new filtered time series as a numpy array with outliers replaced by the median of neighboring values.
        2. A tuple of indices corresponding to the positions of the identified outliers in the original `arr`.

    Notes:
    -----
    The Hampel filter is an effective method for despiking a time series in the presence of outliers and noise.
    The function uses the Numpy library for efficient array operations and parallel processing to improve performance
    for large arrays.

    Example:
    --------
    >>> import numpy as np

    >>> def hampel_filter(arr, window_size=5, n_sigmas=3):
    ...     # (Your function implementation here)

    >>> time_series = np.array([1.0, 2.0, 100.0, 3.0, 4.0, 200.0, 5.0, 6.0])
    >>> filtered_series, outliers = hampel_filter(time_series, window_size=3, n_sigmas=2)
    >>> print(filtered_series)
    array([1. , 2. , 3. , 3. , 4. , 5. , 5. , 6. ])
    >>> print(outliers)
    (array([2, 5]),)
    """
    if isinstance(arr, np.ndarray):
        pass
    elif isinstance(arr, pd.Series):
        arr = arr.values
    elif isinstance(arr, pd.DataFrame):
        arr = arr.values
    else:
        raise ValueError("arr must be a numpy array or pandas Series or DataFrame!")
    
    medians = np.ones_like(arr, dtype=float)*np.nan
    medians_diff = np.ones_like(arr, dtype=float)*np.nan
    if parallel:
        calc_medians_parallel(window_size, arr, medians)
        calc_medians_std_parallel(window_size, arr, medians, medians_diff)
    else:
        calc_medians(window_size, arr, medians)
        calc_medians_std(window_size, arr, medians, medians_diff)
    
    outlier_indices = np.where(np.abs(arr - medians) > n*(medians_diff))
    
    return outlier_indices[0]


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
        loglog=False,
        density=False, 
        scott_rule=False):
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




import numpy as np

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


import numpy as np
from numba import jit, prange

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
                xoutmid[i] = x0 + np.log10(0.5) * (x0 - x[e])
               

    return xoutmid, yout



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


def use_dates_return_elements_of_df_inbetween(t0, t1, df):
    """
    Find the mean of values in a DataFrame `df` within the range of the nearest indices of `t0` and `t1`.

    This function locates the nearest indices of `t0` and `t1` in the DataFrame `df`, which has a datetime-like
    index. It then returns the mean of the values in `df` within the range defined by the nearest indices.

    Parameters:
    ----------
    t0 : datetime-like object or str
        The start date to find the nearest index for. If provided as a string, it will be converted to a
        datetime-like object using `pd.to_datetime`.
    t1 : datetime-like object or str
        The end date to find the nearest index for. If provided as a string, it will be converted to a
        datetime-like object using `pd.to_datetime`.
    df : pandas DataFrame
        The DataFrame to use. The DataFrame's index should be a datetime-like object.

    Returns:
    -------
    pandas DataFrame
        A DataFrame containing the values of `df` within the range defined by the nearest indices of `t0` and `t1`.

    Notes:
    -----
    The function sorts the index of the DataFrame `df` in increasing order before locating the nearest indices for `t0`
    and `t1`. It is assumed that the DataFrame's index is sorted and consists of unique datetime-like values.

    """
    
    # sort the index in increasing order
    df = df.sort_index(ascending=True)

    if type(t0)==str:
        t0 = pd.to_datetime(t0)
        t1 = pd.to_datetime(t1)
    
    r8  = df.index.unique().get_indexer([t0], method='nearest')[0]
    r8a = df.index.unique().get_indexer([t1], method='nearest')[0]
    
    f_df = df[r8:r8a]
    
    return f_df

def find_big_gaps(df, gap_time_threshold):
    """
    Filter a data set by the values of its first column and identify gaps in time that are greater than a specified threshold.

    Parameters:
    df (pandas DataFrame): The data set to be filtered and analyzed.
    gap_time_threshold (float): The threshold for identifying gaps in time, in seconds.

    Returns:
    big_gaps (pandas Series): The time differences between consecutive records in df that are greater than gap_time_threshold.
    """
    keys = df.keys()

    filtered_data = df[df[keys[1]] > -1e10]
    time_diff     = (filtered_data.index.to_series().diff() / np.timedelta64(1, 's'))
    big_gaps      = time_diff[time_diff > gap_time_threshold]

    return big_gaps

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




def binned_quantity_percentile(x, y, what, std_or_error_of_mean, bins,loglog, percentile):
    x = x.astype(float); y =y.astype(float)
    ind = y>-1e15
    x = x[ind]
    y = y[ind]
    if loglog:
        bins = np.logspace(np.log10(min(x)),np.log10(max(x)), bins)
    y_b, x_b, binnumber     = stats.binned_statistic(x, y, what, bins   = bins)
    z_b, x_b, binnumber     = stats.binned_statistic(x, y, 'std',   bins  = bins)
    points , x_b, binnumber = stats.binned_statistic(x, y, 'count', bins= bins)    
    percentiles , x_b, binnumber = stats.binned_statistic(x, y, lambda y: np.percentile(y, percentile), bins= bins) 


    if std_or_error_of_mean==0:
        z_b =z_b/np.sqrt(points)
    x_b = x_b[:-1] + 0.5*(x_b[1:]-x_b[:-1]) 
 

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


def binned_quantity(x,
                    y, 
                    what='mean',
                    std_or_error_of_mean=True,
                    bins=100, 
                    loglog=True,
                    return_counts=False):
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
    """
    
    if loglog:
        mask = np.where((y > -1e10) & (x > 0) & (~np.isinf(x)) & (~np.isinf(y)))[0]        
    else:
        mask = np.where((y > -1e10) & (x > -1e10) & (~np.isinf(x)) & (~np.isinf(y)))[0]
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)

    if loglog:
        bins = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), bins)

    y_b, x_b, _ = stats.binned_statistic(x, y, statistic=what, bins=bins)

    z_b, _, _ = stats.binned_statistic(x, y, statistic='std', bins=bins)

    #if return_counts:
    points, _, _ = stats.binned_statistic(x, y, statistic='count', bins=bins)

    if std_or_error_of_mean == 0:
        z_b /= np.sqrt(points)

    x_b = x_b[:-1] + 0.5 * (x_b[1:] - x_b[:-1])

    return (x_b, y_b, z_b, points) if return_counts else (x_b, y_b, z_b)





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



def mov_fit_func(xx, yy, w_size, xmin, xmax, keep_plot=0):
    """
    Perform moving fits on the data within a specified range.

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
    numb_fits : int
        Number of fits to perform.
    keep_plot : bool
        If True, additional data for plotting fits is returned.

    Returns
    -------
    dict
        A dictionary containing information about the fits:
        - 'xvals': x values corresponding to the fitted data.
        - 'plaw': fitted parameters of the power law.
        - 'fit_err': errors of the fitted parameters.

        If `keep_plot` is True, the dictionary also contains:
        - 'plot_x': x values for plotting the fitted data.
        - 'plot_y': y values for plotting the fitted data.

    Notes
    -----
    This function performs moving fits on the data within the specified range (`xmin` to `xmax`) with a window size of `w_size`.
    The number of fits to perform is given by `numb_fits`.
    If `keep_plot` is True, additional data for plotting the fits is included in the returned dictionary.

    Example
    --------
    >>> import numpy as np
    >>> # Create example data for xx and yy
    >>> xx = np.linspace(1, 10, 100)
    >>> yy = 2 * xx + np.random.normal(0, 1, 100)
    >>> # Perform moving fits with specified parameters
    >>> result = mov_fit_func(xx, yy, w_size=2, xmin=2, xmax=8, numb_fits=3, keep_plot=True)
    >>> print(result)
    """
    keep_err = []
    keep_ind = []
    keep_x = []
    xvals = []
    yvals = []

    
    xx    = np.array(xx)
    yy    = np.array(yy)
    
    mask  = (xx>-1e10) & (yy>-1e10)
    
    
    xx, yy = xx[mask], yy[mask]
    
    index1 = np.searchsorted(xx, xmin, side='left')
    index2 = np.searchsorted(xx, xmax, side='right') - 1

    where_fit = np.arange(index1, index2 + 1)

    for i in where_fit:
        x0 = xx[i]
        xf = x0*w_size

        if xf < 0.98 * xmax:
            fit, s, e, x1, y1 = find_fit(xx, yy, x0,  xf)
            if len(np.shape(x1)) > 0:
                keep_err.append(np.sqrt(fit[1][1][1]))
                keep_ind.append(fit[0][1])
                #keep_x.append(np.nanmean(x1[s:e]))
                keep_x.append(x1[s])
                if keep_plot:
                    xvals.append(x1[s:e])
                    yvals.append(2 * fit[2])

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


            xmids.append(x1)
            fit_vals.append(fit[0][1])                

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
    
    eps_of_k_star = P*(2*np.pi*di)/Vtot
    
    return k_star, eps_of_k_star


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

    Example
    --------
    >>> import pickle
    >>> from pathlib import Path

    >>> def savepickle(df_2_save, save_path, filename):
    ...     # (Your function implementation here)

    >>> data_to_save = [1, 2, 3, 4, 5]
    >>> save_path = './data_folder'
    >>> filename = 'saved_data.pkl'
    >>> savepickle(data_to_save, save_path, filename)
    """
    os.makedirs(str(save_path), exist_ok=True)
    pickle.dump(df_2_save, open(Path(save_path).joinpath(filename), 'wb'))
    
    
    

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



def newindex(df, ix_new, interp_method='linear'):
    """
    Reindex a DataFrame according to the new index *ix_new* supplied.

    Args:
        df: [pandas DataFrame] The dataframe to be reindexed.
        ix_new: [np.array] The new index.
        interp_method: [str] Interpolation method to be used; forwarded to `pandas.DataFrame.reindex.interpolate`.

    Returns:
        df3: [pandas DataFrame] DataFrame interpolated and reindexed to *ix_new*.
    """

    # Ensure df.index and ix_new do not contain duplicates
    df = df[~df.index.duplicated(keep='first')]
    ix_new = np.unique(ix_new)

    # Sort the DataFrame index in increasing order
    df = df.sort_index(ascending=True)

    # Create combined index from old and new index arrays
    ix_com = np.unique(np.append(df.index, ix_new))

    # Sort the combined index (ascending order)
    ix_com.sort()

    # Re-index and interpolate over the non-matching points
    df2 = df.reindex(ix_com).interpolate(method=interp_method)

    return df2.reindex(ix_new)




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


def find_ind_of_closest_dates(df, dates):
    """
    Find the indices of the closest dates in a DataFrame to a list of input dates.

    Parameters
    ----------
    df : pandas DataFrame
        Input DataFrame containing time series data with a unique index.
    dates : list-like
        List of input dates for which the closest indices need to be found.

    Returns
    -------
    list
        A list containing the indices of the closest dates in the DataFrame `df` to each element in the `dates` list.

    Notes
    -----
    This function calculates the indices of the closest dates in the DataFrame `df` to each date in the input `dates`.
    It uses the pandas DataFrame `index.unique().get_loc()` method with the 'nearest' method to find the indices.

    """
    return [df.index.unique().get_loc(date, method='nearest') for date in dates]


def find_ind_of_closest_dates_updated(df, dates):
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
    df_timestamps = df.index.values.astype('datetime64[ns]')
    input_dates = np.array(pd.to_datetime(dates).values.astype('datetime64[ns]'))
    abs_diff = np.abs(df_timestamps[:, np.newaxis] - input_dates)
    closest_indices = abs_diff.argmin(axis=0)
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


def find_cadence(df, mean_or_median_cadence='mean'):
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
    keys = list(df.keys())
    try:
        init_dt = find_cadence(df)
    except:
        init_dt = find_cadence(df)

    if init_dt > -1e10:
        
        
        
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
        df_resampled = df.resample(f"{int(resolution)}ms").mean().interpolate()


    else:
        init_dt = None
        df_resampled = None
        fraction_missing = 100
        total_gaps = None
        total_large_gaps = None
        resolution = np.nan

    return {
        "Init_dt": init_dt,
        "resampled_df": df_resampled,
        "Frac_miss": fraction_missing,
        "Large_gaps": total_large_gaps,
        "Tot_gaps": total_gaps,
        "resol": resolution
    }




