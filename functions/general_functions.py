import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas as pd 
import pickle
from scipy import stats
from pathlib import Path
import numba
from numba import jit,njit,prange,objmode
import scipy as sc
from scipy.optimize import curve_fit

import matplotlib.colors as pltcolors
import matplotlib as mpl
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec


def update_dates_strings(t0, t1, addit_time):
    from datetime import datetime, timedelta
    # Convert strings to datetime objects
    format_str = '%Y-%m-%d %H:%M:%S'
    dt0 = datetime.strptime(t0, format_str)
    dt1 = datetime.strptime(t1, format_str)

    # Subtract 20 seconds from the first date
    new_dt0 = dt0 - timedelta(seconds=addit_time)

    # Add 20 seconds to the second date
    new_dt1 = dt1 + timedelta(seconds=addit_time)

    # Convert datetime objects back to strings
    new_t0 = new_dt0.strftime(format_str)
    new_t1 = new_dt1.strftime(format_str)

    return new_t0, new_t1



def filter_dict(d, keys_to_keep):
    return dict(filter(lambda item: item[0] in keys_to_keep, d.items()))


def string_to_datetime_index(datetime_string, datetime_format='%Y-%m-%d %H:%M:%S.%f'):
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

#@jit(nopython=True, parallel=True)
def hampel_filter(input_series, window_size, n_sigmas=3):
    """
    Method to apply a Hampel filter to a given time series to remove spurious data points.
    Parameters
    ----------
    input_series: numpy array
        The time series to be filtered
    window_size: int
        The size of the window to use in the filtering
    n_sigmas: float
        The number of standard deviations used to determine if a data point is a spike

    Returns
    -------
    new_series: numpy array
        The filtered time series
    indices: list
        A list of the indices of the removed data points
    """
    
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826 # scale factor for Gaussian distribution
    indices = []
    
    for i in range((window_size),(n - window_size)):
        x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)
    
    return new_series, indices

# from joblib import Parallel, delayed

# def hampel_filter(input_series, window_size, n_sigmas=3, njobs=-1):
#     """
#     Method to apply a Hampel filter to a given time series to remove spurious data points.
#     Parameters
#     ----------
#     input_series: numpy array
#         The time series to be filtered
#     window_size: int
#         The size of the window to use in the filtering
#     n_sigmas: float
#         The number of standard deviations used to determine if a data point is a spike

#     Returns
#     -------
#     new_series: numpy array
#         The filtered time series
#     indices: list
#         A list of the indices of the removed data points
#     """
#     n = len(input_series)
#     new_series = input_series.copy()
#     k = 1.4826 # scale factor for Gaussian distribution
#     indices = []

#     def _process_window(i):
#         x0 = np.nanmedian(input_series[(i - window_size):(i + window_size)])
#         S0 = k * np.nanmedian(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
#         if (np.abs(input_series[i] - x0) > n_sigmas * S0):
#             new_series[i] = x0
#             indices.append(i)

#     Parallel(n_jobs=njobs)(delayed(_process_window)(i) for i in range(window_size, n - window_size))

#     return new_series, indices



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

def find_fit_expo(x, y, x0, xf):  

    # Apply fit on specified range #
   # print(len(np.where(x == x.flat[np.abs(x - x0).argmin()])[0]))
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


def scotts_rule_PDF(x):
    """

    Scott’s rule minimizes the integrated mean squared error in the bin approximation under
    the assumption that the data is approximately Gaussian, increasing the number of bins for
    smaller scales 
    
    Inputs:
         
         x : Variable
    """
    x     = np.real(x)
    
    N     = len(x)
    sigma = np.nanstd(x)
    
    # Scott's rule for bin width
    dui    = 3.5*sigma/N**(1/3)
    #print(dui)
    # create bins
    return  np.arange(np.nanmin(x), np.nanmax(x), dui)

def pdf(val, bins, loglog, density,scott_rule =False):

    nout  =[]
    bout  =[]
    errout=[]
    countsout=[]

    val = np.array(val)
    val = val[np.abs(val)<1e15]

    if loglog ==1:
        binsa = np.logspace(np.log10(min(val)),np.log10(max(val)),bins)
    else:
        if scott_rule:
            binsa = scotts_rule_PDF(val)
        else:
            binsa = np.linspace((min(val)),(max(val)),bins)

    if density ==1:
        numout, binsout, patchesout = plt.hist(val,density= True,bins=binsa, alpha = 0)

    else:
        numout, binsout, patchesout = plt.hist(val,density= False,bins=binsa, alpha = 0)
    counts, _,_ = plt.hist(val,density= False,bins=binsa, alpha = 0)

    if loglog ==1:
        bin_centers = binsout[:-1] + np.log10(0.5) * (binsout[1:] - binsout[:-1])
    else:
        bin_centers = binsout[:-1] +         (0.5) * (binsout[1:] - binsout[:-1])

    if density ==1:
        histoout,edgeout=np.histogram(val,binsa,density= True)
    else:
        histoout,edgeout=np.histogram(val,binsa,density= False)

    erroutt = histoout/np.float64(np.size(val))
    erroutt = np.sqrt(erroutt*(1.0-erroutt)/np.float64(np.size(val)))
    erroutt[: np.size(erroutt)] = erroutt[: np.size(erroutt)] / (
        edgeout[1 : np.size(edgeout)] - edgeout[: np.size(edgeout) - 1]
    )

    for i in range(len(numout)):
        if (numout[i]!=0.):
            nout.append(numout[i])
            bout.append(bin_centers[i]) 
            errout.append(erroutt[i])
            countsout.append(counts[i])

    return  np.array(bout), np.array(nout), np.array(errout), np.array(countsout)



# Estimate a moving average
def moving_average(xvals, yvals, window_size):

    # Turn input into np.arrays
    xvals, yvals = np.array(xvals), np.array(yvals)
    
    # Now sort them
    index = np.argsort(xvals).astype(int)
    xvals = xvals[index]
    yvals = yvals[index]
    
    window = np.ones(int(window_size))/float(window_size)
    y_new  = np.convolve(yvals, window, 'same')
    return xvals, y_new



def plot_plaw(start, end, exponent, c):
    #calculating the points on the line
    x = np.logspace(np.log10(start), np.log10(end), 10000)
    
    f = lambda x: c * x ** exponent
    return x, f(x)


@jit(nopython=True, parallel=True)      
def smoothing_function(x,y, mean=True,  window=2, pad = 1):
    def bisection(array,value):
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        if (value < array[0]):
            return -1
        elif (value > array[n-1]):
            return n
        jl = 0# Initialize lower
        ju = n-1# and upper limits.
        while (ju-jl > 1):# If we are not yet done,
            jm=(ju+jl) >> 1# compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl=jm# and replace either the lower limit
            else:
                ju=jm# or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):# edge cases at bottom
            return 0
        elif (value == array[n-1]):# and top
            return n-1
        else:
            return jl

    len_x    = len(x)
    max_x    = np.max(x)
    xoutmid  = np.full(len_x, np.nan)
    xoutmean = np.full(len_x, np.nan)
    yout     = np.full(len_x, np.nan)
    
    for i in prange(len_x):
        x0 = x[i]
        xf = window*x0
        
        if xf < max_x:
            #e = np.where(x  == x[np.abs(x - xf).argmin()])[0][0]
            e = bisection(x,xf)
            if e<len_x:
                if mean:
                    yout[i]     = np.nanmean(y[i:e])
                    xoutmid[i]  = x0 + np.log10(0.5) * (x0 - x[e])
                    xoutmean[i] = np.nanmean(x[i:e])
                else:
                    yout[i]     = np.nanmedian(y[i:e])
                    xoutmid[i]  = x0 + np.log10(0.5) * (x0 - x[e])
                    xoutmean[i] = np.nanmean(x[i:e])                   

    return xoutmid, xoutmean,  yout



def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new|_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out


def use_dates_return_elements_of_df_inbetween(t0, t1, df):
    """
    This function finds the nearest date index of t0 and t1 in the data frame df, and returns the mean of values
    of df within the range of the nearest indices.

    Parameters:
    t0 (datetime-like object): The start date to find the nearest index for.
    t1 (datetime-like object): The end date to find the nearest index for.
    df (pandas DataFrame): The data frame to use. The data frame's index should be a datetime-like object.

    Returns:
    float: The mean of values of df within the range of the nearest indices of t0 and t1.
    """
    
    # sort the index in increasing order
    df = df.sort_index(ascending=True)
    
    if type(t0)==str:
        t0 = pd.to_datetime(t0)
        t1 = pd.to_datetime(t1)
    
    r8   = df.index.unique().get_loc(t0, method='nearest');
    r8a  = df.index.unique().get_loc(t1, method='nearest');
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


def binned_quantity_percentile(x, y, what, std_or_error_of_mean, mov_aver_window,loglog, percentile):
    x = x.astype(float); y =y.astype(float)
    ind = y>-1e15
    x = x[ind]
    y = y[ind]
    if loglog:
        mov_aver_window = np.logspace(np.log10(min(x)),np.log10(max(x)), mov_aver_window)
    y_b, x_b, binnumber     = stats.binned_statistic(x, y, what, bins   = mov_aver_window)
    z_b, x_b, binnumber     = stats.binned_statistic(x, y, 'std',   bins  = mov_aver_window)
    points , x_b, binnumber = stats.binned_statistic(x, y, 'count', bins= mov_aver_window)    
    percentiles , x_b, binnumber = stats.binned_statistic(x, y, lambda y: np.percentile(y, percentile), bins= mov_aver_window) 


    if std_or_error_of_mean==0:
        z_b =z_b/np.sqrt(points)
    x_b = x_b[:-1] + 0.5*(x_b[1:]-x_b[:-1]) 
    #x_b = x_b[1:]  

    return x_b, y_b, z_b, percentiles


def ensure_time_format(start_time, end_time):
    from datetime import datetime
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

#@njit(nopython=True)
def binned_quantity(x, y, what, std_or_error_of_mean, mov_aver_window,loglog, return_counts=False):
    """
    Calculate binned statistics of one variable (y) with respect to another variable (x).

    Parameters
    ----------
    x : array_like
        Input array. This represents the independent variable.
    y : array_like
        Input array. This represents the dependent variable.
    what : str or callable
        The type of binned statistic to compute. This can be any of the options supported by `scipy.stats.binned_statistic()`.
    std_or_error_of_mean : int
        Indicates whether to return the standard deviation (std_or_error_of_mean=1) or the error of the mean (std_or_error_of_mean=0).
    mov_aver_window : int
        The number of bins to use for the histogram. If `loglog` is True, this value is used to generate logarithmic bins.
    loglog : bool
        If True, logarithmic bins are used instead of linear bins.
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

    mask = y > -1e10
    x    = np.asarray(x[mask], dtype=float)
    y    = np.asarray(y[mask], dtype=float)
    #print(len(y))


    if loglog:
        mov_aver_window     = np.logspace(np.log10(np.nanmin(x)),np.log10(np.nanmax(x)), mov_aver_window)

    y_b, x_b, _     = stats.binned_statistic(x, y, what,    bins   = mov_aver_window) 

    z_b, _, _       = stats.binned_statistic(x, y, 'std',   bins  = mov_aver_window)
    #if return_counts:
    points, x_b, _     = stats.binned_statistic(x, y, 'count',   bins  = mov_aver_window)


    if std_or_error_of_mean==0:
        z_b =z_b/np.sqrt(points)
    x_b = x_b[:-1] + 0.5*(x_b[1:]-x_b[:-1])

    return (x_b, y_b, z_b, points) if return_counts else (x_b, y_b, z_b)

def progress_bar(jj, length):
    print('Completed', round(100*(jj/length),2))



@jit(nopython=True, parallel=True)

def mean_manual(xpar, ypar, what, std_or_std_mean, nbins, loglog, upper_percentile=95, remove_upper_percentile=False):
    xpar = np.array(xpar)
    ypar = np.array(ypar)
    ind = (xpar > -1e9) & (ypar > -1e9)

    xpar = xpar[ind]
    ypar = ypar[ind]

    if loglog:
        bins = np.logspace(np.log10(np.nanmin(xpar)), np.log10(np.nanmax(xpar)), nbins)
    else:
        bins = np.linspace(np.nanmin(xpar), np.nanmax(xpar), nbins)

    res1 = np.digitize(xpar, bins)

    bin_counts = np.bincount(res1)

    ypar_mean  = []
    ypar_std   = []
    xpar_mean  = []
    ypar_count = []
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
            ypar_mean.append(np.nanmean(yvalues1))
            xpar_mean.append(np.nanmean(xvalues1))
        else:
            ypar_mean.append(np.nanmedian(yvalues1))
            xpar_mean.append(np.nanmedian(xvalues1))
            

        ypar_std.append(np.nanstd(yvalues1))
        ypar_count.append(bin_counts[i])

    if std_or_std_mean == 0:
        z_b = np.array(ypar_std) / np.sqrt(ypar_count)
    else:
        z_b = np.array(ypar_std)

    return np.array(xpar_mean), np.array(ypar_mean), z_b



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


from joblib import Parallel, delayed


# def worker(i, xx, yy, w_size, xmax):
#     x0 = xx[i]
#     xf = w_size * x0
#     if xf < 0.95 * xmax:
#         fit, s, e, x1, y1 = find_fit(xx, yy, x0, xf) 
#         if len(np.shape(x1)) > 0:
#             fit_err = np.sqrt(fit[1][1][1])
#             ind = fit[0][1]
#             x = x1[s]
#             return fit_err, ind, x
#     return None

# def mov_fit_func(xx, yy, w_size, xmin, xmax, nfit=0, keep_plot=0):
#     xvals =[]
#     yvals =[]

#     where_fit = np.where((xx > xmin) & (xx < xmax))[0]

#     if where_fit.size == 0:
#         return {np.nan}

#     results = Parallel(n_jobs=-1)(delayed(worker)(i, xx, yy, w_size, xmax) for i in where_fit)

#     # filter out None results
#     results = [res for res in results if res is not None]

#     keep_err, keep_ind, keep_x = zip(*results)

#     return_dict = {'xvals': keep_x, 'plaw': keep_ind, 'fit_err': keep_err}
#     if keep_plot:
#         # these lists will be empty in this version of code since they are not populated in worker function
#         return_dict.update({"plot_x": xvals, 'plot_y': yvals})

#     return return_dict

def mov_fit_func(xx, yy, w_size,  xmin, xmax, numb_fits, keep_plot):
    keep_err = []
    keep_ind = []
    keep_x =[]
    xvals =[]
    yvals =[]

    index1   = np.where(xx>xmin)[0].astype(int)#[0]

    if index1.size == 0:
        return {np.nan}


    index1     = index1[0]
    index2     =  np.where(xx<xmax)[0].astype(int)[-1]

    where_fit = np.arange(index1, index2+1, 1)

    for i in range(index1, len(where_fit)):
        x0 = xx[int(where_fit[i])]
        xf = w_size*x0
        if xf<0.95*xmax:
            fit, s, e, x1, y1       = find_fit(xx, yy, x0, xf) 
            if len(np.shape(x1))>0:

                keep_err.append(np.sqrt(fit[1][1][1]))
                keep_ind.append(fit[0][1])
                #keep_x.append(x1[s])# + np.log10(0.5) * (x1[s] - x1[e])) 
                keep_x.append(x1[s])# + np.log10(0.5) * (x1[s] - x1[e])) 
                if keep_plot:
                    xvals.append(x1[s:e])
                    yvals.append(2*fit[2])
    return (
        {
            'xvals': keep_x,
            'plaw': keep_ind,
            'fit_err': keep_err,
            "plot_x": xvals,
            'plot_y': yvals,
        }
        if keep_plot
        else {'xvals': keep_x, 'plaw': keep_ind, 'fit_err': keep_err}
    )

def angle_between_vectors(V, B):
    """
    This function calculates the angle between two vectors.

    Parameters:
    V (ndarray): A 2D numpy array representing the first vector.
    B (ndarray): A 2D numpy array representing the second vector.

    Returns:
    ndarray    : A 1D numpy array representing the angles in degrees between the two input vectors.
    """
    V_norm = np.linalg.norm(V, axis=1, keepdims=True).T[0]
    B_norm = np.linalg.norm(B, axis=1, keepdims=True).T[0]
    dot_product = (V * B).sum(axis=1)
    angle = np.arccos(dot_product / (V_norm * B_norm))/ np.pi * 180

    return angle 



def freq2wavenum(freq, P, Vtot, di):
    """ Takes the frequency, the PSD, the SW velocity and the di.
        Gives the k* and the E(k*), normalised with di"""
    
    # xvals          =  xvals/Vtotal*(2*np.pi*di)
    # yvals          =  yvals*(2*np.pi*di)/Vtotal

    
    k_star = freq/Vtot*(2*np.pi*di)
    
    eps_of_k_star = P*(2*np.pi*di)/Vtot
    
    return k_star, eps_of_k_star



def smooth(x, n=5):
    """
    Generic smoothing function. At this stage it is just running mean with
    window width set by n.

    Args:
        x: [ndarray] Signal to smooth
        n: [int] Window width

    Returns:
        xs: [ndarray] Smoothed signal of same length as *x*
    """
    return np.convolve(x, np.ones(n) / n, mode='same')


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




def savepickle(df_2_save,  save_path, filename):
    """
    Save a list of variables in to a single pickle file.

    Args:
        savepath: [str]path to folder within which file will be saved
        filename: the name of the file to save

    Returns:

    """
    import pickle
    
    os.makedirs(str(save_path), exist_ok=True)
    pickle.dump(df_2_save,open(Path(save_path).joinpath(filename),'wb'))

    return


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
        df: [pandas DataFrame] The dataframe to be reindexed
        ix_new: [np.array] The new index
        interp_method: [str] Interpolation method to be used; forwarded to `pandas.DataFrame.reindex.interpolate`

    Returns:
        df3: [pandas DataFrame] DataFrame interpolated and reindexed to *ixnew*

    """
    
    # sort the index in increasing order
    df = df.sort_index(ascending=True)

    # create combined index from old and new index arrays
    ix_com = np.unique(np.append(df.index, ix_new))

    # sort the combined index (ascending order)
    ix_com.sort()

    # re-index and interpolate over the non-matching points
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


from scipy import signal, special
def time_domain_filter(data, dt, freq_low, freq_high):
    """

    Purpose: band-pass filter of data, assuming constant dt between points
    Parameters:
        data: input nx3 array
        dt: in seconds
        freq_low: low coutoff frequency in Hz
        freq_high: high cutoff frequency in Hz
    Returns: nx3 array of band-pass filtered data
    """
    
    nyquist = 1./(2.*dt)
    flow = freq_low/nyquist
    fhigh = freq_high/nyquist
    A = 120. # from Ergun's fa_fields_filter
    f = fhigh if flow == 0.0 else flow
    nterms = int(5./f)
    nterms = min(nterms, 5000.)
    out = digital_filter(flow,fhigh,A,nterms)

    if len(np.shape(data)) <= 1:
        return signal.convolve(data,out,mode='same',method='direct')

    new_series_x = signal.convolve(data[:,0],out,mode='same',method='direct')
    new_series_y = signal.convolve(data[:,1],out,mode='same',method='direct')
    new_series_z = signal.convolve(data[:,2],out,mode='same',method='direct')
    return np.transpose(np.vstack((new_series_x,new_series_y,new_series_z)))

def digital_filter(flow,fhigh,aGibbs,nterms):
    
    fstop = float(1) if fhigh < flow else float(0)
    # Computes Kaiser weights W(N,K) for digital filters
    # W = coef = returned array of Kaiser weights
    # N = value of N in W(N,K), ie number of terms
    # A = Size of gibbs phenomenon wiggles in -DB

    if aGibbs <= 21 :
        alpha = 0.
    elif (aGibbs >= 50) :
        alpha = 0.1102*(aGibbs-8.7)
    else:
        alpha = 0.5842*(aGibbs-21)**(0.4) + 0.07886*(aGibbs-21)

    arg = (np.arange(nterms)+1)/nterms
    coef = special.iv(0,alpha*np.sqrt(1.-arg**2))/special.iv(0,alpha)
    t = (np.arange(nterms)+1)*np.pi
    coef = coef*(np.sin(t*fhigh)-np.sin(t*flow))/t
    coef = np.concatenate((np.flip(coef), [fhigh - flow + fstop], coef))
    return coef



def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


def find_ind_of_closest_dates(df, dates):
    return [df.index.unique().get_loc(date, method='nearest') for date in dates]

def find_closest_values_of_2_arrays(a, b):
    dup = np.searchsorted(a, b)
    uni = np.unique(dup)
    uni = uni[uni < a.shape[0]]
    ret_b = np.zeros(uni.shape[0])
    for idx, val in enumerate(uni):
        bw = np.argmin(np.abs(a[val]-b[dup == val]))
        tt = dup == val
        ret_b[idx] = np.where(tt)[0][bw]
    return np.column_stack((uni, ret_b))

def find_cadence(df):
    return np.nanmean((df.dropna().index.to_series().diff()/np.timedelta64(1, 's')))

def resample_timeseries_estimate_gaps(df, resolution, large_gaps=10):
    """
    Resample timeseries and estimate gaps, default setting is for FIELDS data
    Resample to 10Hz and return the resampled timeseries and gaps infos
    Input: 
        df                  :       input time series
        resolution          :       resolution to resample [ms]
    Keywords:
        large_gaps  =   10 [s]      large gaps in timeseries [s]
    Outputs: 
        init_dt             :       initial resolution of df
        df_resampled        :       resampled dataframe
        fraction_missing    :       fraction of missing values in the interval
        total_large_gaps    :       fraction of large gaps in the interval
        total_gaps          :       total fraction of gaps in the interval  
        resolution          :       resolution of resmapled dataframe
        
        
    """
    keys = df.keys()
    try:
        init_dt = find_cadence(df[keys[1]])
    except:
        init_dt = find_cadence(df[keys[0]])
    if init_dt > -1e10:
        
        # Make sure that you resample to a resolution that is lower than initial df's resolution
        while init_dt > resolution * 1e-3:
            resolution     = 1.005 * resolution
        
        # estimate duration of interval selected in seconds #
        interval_dur = (df.index[-1] - df.index[0]).total_seconds()
        
        # Resample time-series to desired resolution # 
        df_resampled       = df.resample(f"{int(resolution)}ms").mean()
        
        # Estimate fraction of missing values within interval #
        try:
            fraction_missing   = 100 * df_resampled[keys[1]].isna().mean()
        except:
            fraction_missing   = 100 * df_resampled[keys[0]].isna().mean()        
        # Estimate sum of gaps greater than large_gaps seconds
        res = (df_resampled.dropna().index.to_series().diff() / np.timedelta64(1, 's'))
        
        # Gives you the fraction of  large gaps in timeseries
        total_large_gaps  = 100 * (res[res > large_gaps].sum() / interval_dur)
        
        # Gives you the total fraction  gaps in timeseries
        total_gaps       = 100 * (res[res > resolution * 1e-3].sum() / interval_dur)
    else:
        init_dt            = None
        df_resampled       = None
        fraction_missing   = 100
        total_gaps         = None
        total_large_gaps   = None
        resolution = np.nan
    return {
        "Init_dt": init_dt,
        "resampled_df": df_resampled.interpolate(),
        "Frac_miss": fraction_missing,
        "Large_gaps": total_large_gaps,
        "Tot_gaps": total_gaps,
        "resol": resolution
    }



def str2bool(v):
    '''
    FROM:
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def stringChop(var_string, var_remove):
    ''' stringChop
    PURPOSE / OUTPUT:
        Remove the occurance of the string 'var_remove' at both the start and end of the string 'var_string'.
    '''
    if var_string.endswith(var_remove):
        var_string = var_string[:-len(var_remove)]
    if var_string.startswith(var_remove):
        var_string = var_string[len(var_remove):]
    return var_string

def createFolder(folder_name):
    ''' createFolder
    PURPOSE:
        Create the folder passed as a filepath to inside the folder.
    OUTPUT:
        Commandline output of the success/failure status of creating the folder.
    '''
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name)
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name)

    print(' ')

def setupInfo(filepath):
    ''' setupInfo
    PURPOSE:
        Collect filenames that will be processed and the number of these files
    '''
    global bool_debug_mode
    ## save the the filenames to process
    file_names = list(filter(meetsCondition, sorted(os.listdir(filepath))))
    ## check files
    if bool_debug_mode:
        print('The files in the filepath:')
        print('\t' + filepath)
        print('\tthat satisfied meetCondition are the files:')
        print('\t\t' + '\n\t\t'.join(file_names))
        print(' ')
    ## return data
    return [file_names, len(file_names) // 2]

def createFilePath(names):
    ''' creatFilePath
    PURPOSE / OUTPUT:
        Turn an ordered list of names and concatinate them into a filepath.
    '''
    return ('/'.join([x for x in names if x != '']))



