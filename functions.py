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



@jit(nopython=True, parallel=True)
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

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

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


# 9) PDF plot

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
    erroutt[0:np.size(erroutt)] = erroutt[0:np.size(erroutt)] /(binsout[1:np.size(binsout)]-binsout[0:np.size(binsout)-1])

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

    if loglog ==1:
        binsa = np.logspace(np.log10(min(val)),np.log10(max(val)),bins)
    else:
        if scott_rule:
            binsa = scotts_rule_PDF(val)
            #print(binsa)
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
    erroutt[0:np.size(erroutt)] = erroutt[0:np.size(erroutt)] /(edgeout[1:np.size(edgeout)]-edgeout[0:np.size(edgeout)-1])
 
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
    x = np.linspace(start, end, 1000)
    
    f = lambda x: c * x ** exponent
    return x, f(x)



def find_duration(df,what,theta1,theta2):
    small = (df.index[1] - df.index[0])/np.timedelta64(1, 's')
    a = pd.DataFrame(df[what].values, columns = [what])
    a['condition'] = a[what].between(theta1,theta2) #(a.PVI  >= 2) #and (a.PVI  <= 6)#(a.PVI > 4)
    a['crossing']  = (a.condition != a.condition.shift()).cumsum()
    a['count']     = a.groupby(['condition', 'crossing']).cumcount(ascending=False) + 1
    a.loc[a.condition == False, 'count'] = 0
    krata        = np.zeros(len(a['count'].values))
    krata1       = np.zeros(len(a['count'].values))
    duration     = np.zeros(len(a['count'].values))
    indexs       = a['count'].values
    values       = a['condition'].values
    PVIs         = a[what].values

    for i in range(1,len(values)-1):
        if values[i]==True:
            if values[i-1]==False:
                krata[i]     = i
                krata1[i]    = i+int(indexs[i])
                time         = df.index[i+int(indexs[i])] - df.index[i]
                duration[i]  = time/ np.timedelta64(1, 's')  -small
   
    return krata, krata1, duration


@jit(nopython=True, parallel=True)      
def smoothing_function(x,y,mean=True,  window=2, pad = 1):
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



def initializeFigure(xlabel, ylabel, scale= 'loglog',width='1col', height=None):
    '''
    Initialize a single plot for publication.
     
    Creates a figure and an axis object that is set to be the 
    current working axis.
     
    @param width: Width of the figure in cm or either '1col' 
                  (default) or '2col' for single our double 
                  column usage. Single column equals 8.8cm and
                  double column 18cm.
    @type width: float or str (either '1col' or '2col')
    @param height: Height of the figure either in cm. If None
                   (default), will be calculated with an 
                   aspect ratio of 7/10 (~1/1.4).
    @type height: float or None
    @return: figure and axis objects.
    @rtype: tuple (figure, axis)
     
    '''
    # Set up Matplotlib parameters for the figure.
    import matplotlib as mpl
    from matplotlib import pyplot as plt

 
    # make sure defaults are used
    plt.style.use(['science', 'scatter'])
    plt.rcParams['text.usetex'] = True
    import matplotlib



 
    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]
     
    # Get figure width in inch
    if width == '1col':
        width = 8.8 # width [cm]
    elif width == '2col':
        width = 18.0 # width [cm]
    figWidth = width * cm_to_inch # width [inch]
     
 
    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 7.5/10.
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch # height [inch]
     
 
    # Create figure with right resolution for publication
    fig = plt.figure(figsize=(figWidth, figHeight), dpi=300)
     
 
    # Add axis object and select as current axis for pyplot
    ax = fig.add_subplot(111)
    plt.sca(ax)
    
    ax.tick_params(axis='both', which='minor',left=0,right=0,bottom=0, top=0, direction='out', labelsize='medium', pad=2)
    ax.tick_params(axis='both', which='major',left=1,right=0,bottom=1, top=0, direction='out', labelsize='small', pad=2) 
    
    
    if scale=='loglog':
       # ax.loglog(x,y, label =label)
        ax.set_yscale('log')
        ax.set_xscale('log')
    elif scale=='semilogy':
        ax.set_yscale('log')
    elif scale=='semilogx':
        ax.set_xscale('log')
    else:
        pass
    
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    
     
    return fig, ax

def create_colors(hmany):

    ### Create Colormap ### Remove white part of RdBU
    from matplotlib.colors import LinearSegmentedColormap

    interval = np.hstack([np.linspace(0, 0.4), np.linspace(0.60, 1)])
    colors   = plt.cm.RdBu_r(interval)
    cmap     = LinearSegmentedColormap.from_list('name', colors)
    col      = cmap(np.linspace(0,1,hmany))
    
    return col


def heatmap_func(x,  y, z,
                 numb_bins, xlabel, ylabel, colbar_label, min_counts =10, what ='mean', ax_scale ='loglog',
                 min_x= -1e10, min_y= -1e10, min_z= -1e10, 
                 max_x= 1e10, max_y= 1e10, max_z= 1e10,
                 log_colorbar=True,fig_size =(20,18), f_size =35):
    

    
    def binned_quantity(x, y, what, std_or_error_of_mean, mov_aver_window, return_counts=False):
        y_b, x_b, binnumber     = stats.binned_statistic(x, y, what, bins   = mov_aver_window)
        z_b, x_b, binnumber     = stats.binned_statistic(x, y, 'std',   bins  = mov_aver_window)
        points , x_b, binnumber = stats.binned_statistic(x, y, 'count', bins= mov_aver_window)        
        if std_or_error_of_mean==0:
            z_b =z_b/np.sqrt(points)
        x_b = x_b[:-1] + 0.5*(x_b[1:]-x_b[:-1]) 
        #x_b = x_b[1:]  
        if return_counts:
            return x_b, y_b, z_b, points
        else:
            return x_b, y_b, z_b
    
    """Define constants"""
    au_to_km      = 1.496e8
    rsun          = 696340 #km

    
    """Quantities we want to plot"""
    xf, yf, zf = np.array(x),  np.array(y), np.array(z)
    

    index             = (xf>min_x)& (yf>min_y) & (zf>min_z) & (xf<max_x)& (yf<max_y) & (zf<max_z)
    yf1               =  yf[index]
    zf1               =  zf[index]
    xf1               =  xf[index]
    


    """" Create bins """
    numb_x_bins, numb_y_bins  = numb_bins, numb_bins 
    if ax_scale=='loglog':
        xf1_bins                  = np.logspace(np.log10(min(xf1)), np.log10(max(xf1)),numb_x_bins )
        yf1_bins                  = np.logspace(np.log10(min(yf1)), np.log10(max(yf1)),numb_y_bins )
    elif ax_scale=='linear':
        xf1_bins                  = np.linspace((min(xf1)), (max(xf1)),numb_x_bins )
        yf1_bins                  = np.linspace((min(yf1)), (max(yf1)),numb_y_bins )
    elif ax_scale=='semilogx':
        xf1_bins                  = np.linspace((min(xf1)), (max(xf1)),numb_x_bins )
        yf1_bins                  = np.logspace(np.log10(min(yf1)), np.log10(max(yf1)),numb_y_bins )
    elif ax_scale=='semilogy':
        xf1_bins                  = np.logspace(np.log10(min(xf1)), np.log10(max(xf1)),numb_x_bins )
        yf1_bins                  = np.linspace((min(yf1)), (max(yf1)),numb_y_bins )


    """" Estimate mean or median within each bin """
    means = stats.binned_statistic_2d(xf1,
                                      yf1,
                                      values    = zf1,
                                      statistic = what,
                                      bins=[xf1_bins,yf1_bins])[0]

    """" Estimate counts within each bin """
    counts  = stats.binned_statistic_2d(xf1,
                                      yf1,
                                      values    = zf1,
                                      statistic = 'count',
                                      bins=[xf1_bins,yf1_bins])[0]
    
    """" Estimate stds within each bin """
    stds  = stats.binned_statistic_2d(xf1,
                                      yf1,
                                      values    = zf1,
                                      statistic = 'std',
                                      bins=[xf1_bins,yf1_bins])[0]
    
    z         = means
    rows, cols = np.shape(z)

    
    """ Remove bins with less than min_counts counts """
    for i in range(np.shape(counts)[0]):
        for k in range(np.shape(counts)[1]):
            if counts[i,k] < min_counts:
                means[i,k] = np.nan

    ### Create Colormap ### Remove white part of RdBU
    interval = np.hstack([np.linspace(0, 0.5), np.linspace(0.5, 1)])
    colors   = plt.cm.RdBu_r(interval)
    cmap     = LinearSegmentedColormap.from_list('name', colors)
    

    
    fig = plt.figure(figsize=fig_size)
    gs = GridSpec(8, 8)

    ax = fig.add_subplot(gs[0:8, 0:8])
    # ax_hist_x = fig.add_subplot(gs[0,0:7])
    # ax_hist_y = fig.add_subplot(gs[1:8, 7])
 

    # On purpose!!!
    x_centers = yf1_bins
    y_centers = xf1_bins

    current_cmap = matplotlib.cm.get_cmap(cmap)
    current_cmap.set_bad(color='gray')

    colbar_z = z.flatten()
    colbar_z = colbar_z[colbar_z>0]

#     z = interpolate_replace_nans(z)
    

    if log_colorbar:
        normi    =  pltcolors.LogNorm()
        c        = ax.pcolormesh(y_centers, x_centers,  z,cmap=cmap, norm = normi)
    else:
        c        = ax.pcolormesh(y_centers, x_centers,  z,cmap=cmap)
    
    print(np.shape(y_centers))

    
    cax = fig.add_axes([0.91, 0.125, 0.05, 0.755])

    ax1 = fig.colorbar(c,cmap=cmap, cax=cax, orientation='vertical', pad=4)#,ticks=tick_locations_plot, extend='both')

    ax1.ax.tick_params(which='both',left=0,right=0, labelsize=f_size)
    ax.tick_params(which='both',left=1,right=0,bottom=1, top=0, direction='out', labelsize=f_size)



    ax1.ax.set_ylabel(colbar_label,  fontsize =f_size)
    ax.set_xlabel(xlabel , fontsize =f_size)
    ax.set_ylabel(ylabel, fontsize =f_size)
    
    # Set axis scale
    if ax_scale=='loglog':
        ax.set_yscale('log')
        ax.set_xscale('log')  

        
    elif ax_scale=='semilogy':
        ax.set_yscale('log')

    elif ax_scale=='semilogx':
        ax.set_xscale('log')  

    return y_centers, x_centers,  z

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new|_index values."""
    df_out = pd.DataFrame(index=new_index)
    df_out.index.name = df.index.name

    for colname, col in df.iteritems():
        df_out[colname] = np.interp(new_index, df.index, col)

    return df_out



def find_fit(x, y, x0, xf):  
    x = np.array(x)
    y = np.array(y)
    ind1 =np.argsort(x)
     
    x = x[ind1]
    y = y[ind1]
    ind = y>-1e15
    x = x[ind]
    y = y[ind]
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




def binned_quantity(x, y, what, std_or_error_of_mean, mov_aver_window,loglog, return_counts=False):
    x = np.array(x); y =np.array(y)
    x = x.astype(float); y =y.astype(float)
    index =np.array(np.argsort(x).astype(int))
    #print(index)
    x = x[index]
    y = y[index]
    # print(x)
    # print(y)
    
    ind = y>-1e10

    x = np.array(x[ind]); y =np.array(y[ind])
    if loglog:
        mov_aver_window = np.logspace(np.log10(np.nanmin(x)),np.log10(np.nanmax(x)), mov_aver_window)
    y_b, x_b, binnumber     = stats.binned_statistic(x, y, what, bins   = mov_aver_window)  # from scipy stats
    z_b, x_b, binnumber     = stats.binned_statistic(x, y, 'std',   bins  = mov_aver_window)
    points , x_b, binnumber = stats.binned_statistic(x, y, 'count', bins= mov_aver_window)    
   # percentiles , x_b, binnumber = stats.binned_statistic(x, y, lambda y: np.percentile(y, percentile), bins= mov_aver_window) 


    if std_or_error_of_mean==0:
        z_b =z_b/np.sqrt(points)
    x_b = x_b[:-1] + 0.5*(x_b[1:]-x_b[:-1]) 
    #x_b = x_b[1:]  
    if return_counts:
        return x_b, y_b, z_b, points
    else:
        return x_b, y_b, z_b

def progress_bar(jj, length):
    print('Completed', round(100*(jj/length),2))


def mean_manual(xpar, ypar,what,std_or_std_mean, nbins, loglog,  upper_percentile=95, remove_upper_percentile= False):
    xpar = np.array(xpar);     ypar = np.array(ypar)
    ind  =  (xpar>-1e9) & (ypar>-1e9)
    
    xpar = xpar[ind];  ypar = ypar[ind];

    if loglog:
        bins = np.logspace(np.log10(np.nanmin(xpar)),np.log10(np.nanmax(xpar)), nbins)       
    else:
        bins = np.linspace(np.nanmin(xpar),np.nanmax(xpar), nbins)        
    res1   = np.digitize(xpar,bins)

    unique = np.unique(res1)
    #print(bins)


    ypar_mean =[];     ypar_std   = []
    xpar_mean =[];     ypar_count = []
    for i in range(len(unique)):
        xvalues1 = xpar[res1==unique[i]] 
        yvalues1 = ypar[res1==unique[i]]

        if remove_upper_percentile:
            percentile = np.percentile(yvalues1, upper_percentile)
            xvalues1   = xvalues1[yvalues1<percentile] 
            yvalues1   = yvalues1[yvalues1<percentile]          
        if what=='mean':
            ypar_mean.append(np.nanmean(yvalues1))
            xpar_mean.append(np.nanmean(xvalues1))    
        else:
            ypar_mean.append(np.nanmedian(yvalues1))
            xpar_mean.append(np.nanmedian(xvalues1)) 
        ypar_std.append(np.nanstd(yvalues1))
        ypar_count.append(len(yvalues1))
        
            
    if  std_or_std_mean==0:
        z_b = np.array(ypar_std)/np.sqrt(ypar_count)
    else:
        z_b = np.array(ypar_std)
        
        
   # x_b = bins#[:-1] + 0.5*(bins[1:]-bins[:-1]) 

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


def mov_fit_func(xx, yy, w_size,  xmin, xmax, numb_fits, keep_plot):
    keep_err = []; keep_ind = []; keep_x =[];xvals =[]; yvals =[];
    
    index1   = np.where(xx>xmin)[0].astype(int)#[0]

    if index1.size!=0:
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
        if keep_plot:
            return {'xvals': keep_x, 'plaw': keep_ind, 'fit_err': keep_err, "plot_x": xvals, 'plot_y': yvals}
        else:
            return {'xvals': keep_x, 'plaw': keep_ind, 'fit_err': keep_err}
    else:
        return {np.nan}



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
    xs = np.convolve(x, np.ones(n) / n, mode='same')

    return xs


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

    newfilename = oldfilename[:dot_ix] + '.' + newextension.strip('.')

    return newfilename


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

    # create combined index from old and new index arrays
    ix_com = np.unique(np.append(df.index, ix_new))

    # sort the combined index (ascending order)
    ix_com.sort()

    # re-index and interpolate over the non-matching points
    df2 = df.reindex(ix_com).interpolate(method=interp_method)

    # drop all the old index points by re-indexing to new index
    df3 = df2.reindex(ix_new)
    #print(len(df3)), print(len(ix_new))
    return df3


def plot_fixrate(df, df_new, L, r):
    """
    Make a summary plot of the difference between the original data frame *df* and fixed *df_new*.

    Args:
        df: [pandas DataFrame] Original dataframe
        df_new: [pandas DataFrame] Dataframe returned by fix_data_rate()
        L: [float] The sample cycle duration
        r: [pandas DataSeries] Data rate variable of *df*

    Returns:
        999 [int]
    """

    # get the time index in to a non-index variable
    df['t'] = df.index.values

    # running difference of *t* is the *cadence* of the sampling
    df['delta'] = df.t.diff()

    # remove NANs
    df['delta'] = df.delta.fillna(pd.Timedelta(seconds=0))

    # get the time index in to a non-index variable
    df_new['t'] = df_new.index.values

    # running difference of *t* is the *cadence* of the sampling
    df_new['delta'] = df_new.t.diff()

    # remove NANs
    df_new['delta'] = df_new.delta.fillna(pd.Timedelta(seconds=0))

    # open new figure window
    plt.figure()

    # plot the delta_t
    df.delta.plot(marker='.', label='original')

    # plot the delta_t as it should be according to the data rate
    (L / r).mul(1e9).plot(label='acc to datarate')

    # plot the delta_t of the fixed-data-rate dataframe
    df_new.delta.plot(marker='x', ls='', label='fixed')

    # apply figure labels, titles, etc
    plt.ylabel('$\Delta t$ [nanoseconds]')
    plt.xlabel('Time [UT]')
    plt.grid(which='both')
    plt.title(str(df.index[0]) + ' --> ' + str(df.index[-1]))
    plt.tight_layout()
    plt.legend()

    return 999


def listsearch(search_string, input_list):
    """
    Return matching items from a list

    Args:
        search_string: [str] String to search for (starting only)
        input_list: [list] List to search in

    Returns:
        found_list: [list] List of matching items

    """

    found_list = []
    i = 0

    for si in input_list:

        if si.startswith(search_string):
            # print('%d: %s' % (i, si))
            found_list.append(si)

        i += 1

    return found_list


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

    w = signal.windows.get_window(win_name, N)

    return w


def chunkify(ts_in, chunk_duration):
    """
    Divide a given timeseries in to chunks of *chunk_duration*

    Args:
        ts_in: [pd.Timeseries] Input timeseries
        chunk_duration: [float] Duration of the chunks in seconds

    Returns:

    """
    #print('converting to chunks of len %.2f sec' % chunk_duration)

    dchunk_str = str(chunk_duration) + 'S'

    chunks_time = pd.date_range(ts_in[0].ceil('1s'), ts_in[-1].floor('1s'), freq=dchunk_str)

    return chunks_time


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
    if flow == 0.0:
        f = fhigh
    else:
        f = flow
    nterms = int(5./f)
    if nterms > 5000.:
        nterms = 5000.
    out = digital_filter(flow,fhigh,A,nterms)
    
    if len(np.shape(data))>1:
        new_series_x = signal.convolve(data[:,0],out,mode='same',method='direct')
        new_series_y = signal.convolve(data[:,1],out,mode='same',method='direct')
        new_series_z = signal.convolve(data[:,2],out,mode='same',method='direct')
        new_series =  np.transpose(np.vstack((new_series_x,new_series_y,new_series_z)))
    else:
        new_series = signal.convolve(data,out,mode='same',method='direct')

    return new_series

def digital_filter(flow,fhigh,aGibbs,nterms):
    
    if fhigh < flow:
        fstop = float(1)
    else:
        fstop = float(0)
    
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

def initializeFigure_1by_2_noshare_y(xlabel, ylabel, scale= 'loglog',width='1col', height=None,share_y=False):
    '''
    Initialize a single plot for publication.
     
    Creates a figure and an axis object that is set to be the 
    current working axis.
     
    @param width: Width of the figure in cm or either '1col' 
                  (default) or '2col' for single our double 
                  column usage. Single column equals 8.8cm and
                  double column 18cm.
    @type width: float or str (either '1col' or '2col')
    @param height: Height of the figure either in cm. If None
                   (default), will be calculated with an 
                   aspect ratio of 7/10 (~1/1.4).
    @type height: float or None
    @return: figure and axis objects.
    @rtype: tuple (figure, axis)
     
    '''
    # Set up Matplotlib parameters for the figure.
    import matplotlib as mpl
    from matplotlib import pyplot as plt

 
    # make sure defaults are used
    plt.style.use(['science', 'scatter'])
    plt.rcParams['text.usetex'] = True
    import matplotlib
    plt.rc('axes', linewidth=2)
   # plt.rc('xtick_params', width=1, length=6) 
   # plt.rc('ytick_params', width=1, length=6)
   


 
    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]
     
    # Get figure width in inch
    if width == '1col':
        width = 12 # width [cm]
    elif width == '2col':
        width = 48.0 # width [cm]
    elif width == '3col':
        width = 24.0 # width [cm]

    figWidth = width * cm_to_inch # width [inch]
     
 
    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 5/10.
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch # height [inch]
     
    if share_y==0:

        # Create figure with right resolution for publication
        fig, axes = plt.subplots(1,2, figsize=(5*figWidth,4*figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},   dpi=300)
    else:

        # Create figure with right resolution for publication
        fig, axes = plt.subplots(1,2, figsize=(figWidth, figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},  dpi=300)
    for i in range(2):

        ax =axes[i]

        ax.tick_params(axis='both', which='minor',left=0,right=0,bottom=0, top=0, direction='out', labelsize='xx-large', pad=2)
        ax.tick_params(axis='both', which='major',left=1,right=0,bottom=i, top=0, direction='out', labelsize='xx-large', pad=2) 


    if scale=='loglog':
       # ax.loglog(x,y, label =label)
        ax.set_yscale('log')
        ax.set_xscale('log')
    elif scale=='semilogy':
        ax.set_yscale('log')
    elif scale=='semilogx':
        ax.set_xscale('log')
    else:
        print('linear')
    
    # ax.set_ylabel(ylabel)
    # if i ==1:
    #     axes[i, k].set_xlabel(xlabel)
    
     
    return fig, axes
def initializeFigure_1by_2(xlabel, ylabel, scale= 'loglog',width='1col', height=None,share_y=False):
    '''
    Initialize a single plot for publication.
     
    Creates a figure and an axis object that is set to be the 
    current working axis.
     
    @param width: Width of the figure in cm or either '1col' 
                  (default) or '2col' for single our double 
                  column usage. Single column equals 8.8cm and
                  double column 18cm.
    @type width: float or str (either '1col' or '2col')
    @param height: Height of the figure either in cm. If None
                   (default), will be calculated with an 
                   aspect ratio of 7/10 (~1/1.4).
    @type height: float or None
    @return: figure and axis objects.
    @rtype: tuple (figure, axis)
     
    '''
    # Set up Matplotlib parameters for the figure.
    import matplotlib as mpl
    from matplotlib import pyplot as plt

 
    # make sure defaults are used
    plt.style.use(['science', 'scatter'])
    plt.rcParams['text.usetex'] = True
    import matplotlib
    plt.rc('axes', linewidth=2)
   # plt.rc('xtick_params', width=1, length=6) 
   # plt.rc('ytick_params', width=1, length=6)
   


 
    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]
     
    # Get figure width in inch
    if width == '1col':
        width = 12 # width [cm]
    elif width == '2col':
        width = 48.0 # width [cm]
    elif width == '3col':
        width = 24.0 # width [cm]

    figWidth = width * cm_to_inch # width [inch]
     
 
    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 5/10.
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch # height [inch]
     
    if share_y:

        # Create figure with right resolution for publication
        fig, axes = plt.subplots(1,2, figsize=(5*figWidth,4*figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},sharex =True, sharey='row',  dpi=300)
    else:

        # Create figure with right resolution for publication
        fig, axes = plt.subplots(1,2, figsize=(figWidth, figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},  dpi=300)
    for i in range(2):

        ax =axes[i]

        ax.tick_params(axis='both', which='minor',left=0,right=0,bottom=0, top=0, direction='out', labelsize='xx-large', pad=2)
        ax.tick_params(axis='both', which='major',left=1,right=0,bottom=i, top=0, direction='out', labelsize='xx-large', pad=2) 


    if scale=='loglog':
       # ax.loglog(x,y, label =label)
        ax.set_yscale('log')
        ax.set_xscale('log')
    elif scale=='semilogy':
        ax.set_yscale('log')
    elif scale=='semilogx':
        ax.set_xscale('log')
    else:
        print('linear')
    
    # ax.set_ylabel(ylabel)
    # if i ==1:
    #     axes[i, k].set_xlabel(xlabel)
    
     
    return fig, axes

def initializeFigure_2by_3(xlabel, ylabel, scale= 'loglog',width='1col', height=None,share_y=False):
    '''
    Initialize a single plot for publication.
     
    Creates a figure and an axis object that is set to be the 
    current working axis.
     
    @param width: Width of the figure in cm or either '1col' 
                  (default) or '2col' for single our double 
                  column usage. Single column equals 8.8cm and
                  double column 18cm.
    @type width: float or str (either '1col' or '2col')
    @param height: Height of the figure either in cm. If None
                   (default), will be calculated with an 
                   aspect ratio of 7/10 (~1/1.4).
    @type height: float or None
    @return: figure and axis objects.
    @rtype: tuple (figure, axis)
     
    '''
    # Set up Matplotlib parameters for the figure.
    import matplotlib as mpl
    from matplotlib import pyplot as plt

 
    # make sure defaults are used
    plt.style.use(['science', 'scatter'])
    plt.rcParams['text.usetex'] = True
    import matplotlib
    plt.rc('axes', linewidth=2)
   # plt.rc('xtick_params', width=1, length=6) 
   # plt.rc('ytick_params', width=1, length=6)
   


 
    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]
     
    # Get figure width in inch
    if width == '1col':
        width = 12 # width [cm]
    elif width == '2col':
        width = 48.0 # width [cm]
    elif width == '3col':
        width = 24.0 # width [cm]

    figWidth = width * cm_to_inch # width [inch]
     
 
    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 5/10.
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch # height [inch]
     
    if share_y:

        # Create figure with right resolution for publication
        fig, axes = plt.subplots(2,3, figsize=(5*figWidth,5*figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},sharex =True, sharey='row',  dpi=300)
    else:

        # Create figure with right resolution for publication
        fig, axes = plt.subplots(2,3, figsize=(figWidth, figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},  dpi=300)
    for k in range(2):
        for i in range(3):

            ax =axes[k,i]

            ax.tick_params(axis='both', which='minor',left=0,right=0,bottom=0, top=0, direction='out', labelsize='xx-large', pad=2)
            ax.tick_params(axis='both', which='major',left=1,right=0,bottom=i, top=0, direction='out', labelsize='xx-large', pad=2) 

            if scale=='loglog':
               # ax.loglog(x,y, label =label)
                ax.set_yscale('log')
                ax.set_xscale('log')
            elif scale=='semilogy':
                ax.set_yscale('log')
            elif scale=='semilogx':
                ax.set_xscale('log')
            else:
                print('linear')
    
    # ax.set_ylabel(ylabel)
    # if i ==1:
    #     axes[i, k].set_xlabel(xlabel)
    
     
    return fig, axes



def inset_axis_params(size ='xx-large'):
    minor_tick_params = {'axis':'both',
                'which':'minor',
                'left':1,
                'right':0,
                'bottom':0,
                'top':1,
                'direction':'out',
                'labelsize':size,
                'pad':2}

    major_tick_params = {'axis':'both',
                    'which':'major',
                    'left':1,
                    'right':0,
                    'bottom':0,
                    'top':1,
                    'direction':'out',
                    'labelsize':size,
                    'pad':2}
    return minor_tick_params, major_tick_params


def initializeFigure_2by_2(xlabel, ylabel, scale= 'loglog',width='1col', height=None):
    '''
    Initialize a single plot for publication.
     
    Creates a figure and an axis object that is set to be the 
    current working axis.
     
    @param width: Width of the figure in cm or either '1col' 
                  (default) or '2col' for single our double 
                  column usage. Single column equals 8.8cm and
                  double column 18cm.
    @type width: float or str (either '1col' or '2col')
    @param height: Height of the figure either in cm. If None
                   (default), will be calculated with an 
                   aspect ratio of 7/10 (~1/1.4).
    @type height: float or None
    @return: figure and axis objects.
    @rtype: tuple (figure, axis)
     
    '''
    # Set up Matplotlib parameters for the figure.
    import matplotlib as mpl
    from matplotlib import pyplot as plt

 
    # make sure defaults are used
    plt.style.use(['science', 'scatter'])
    plt.rcParams['text.usetex'] = True
    import matplotlib



 
    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]
     
    # Get figure width in inch
    if width == '1col':
        width = 12 # width [cm]
    elif width == '2col':
        width = 18.0 # width [cm]
    figWidth = width * cm_to_inch # width [inch]
     
 
    # Get figure height in inch
    if height is None:
        fig_aspect_ratio = 7.5/10.
        figHeight = figWidth * fig_aspect_ratio  # height [inch]
    else:
        figHeight = height * cm_to_inch # height [inch]
     
 
    # Create figure with right resolution for publication
    fig, axes = plt.subplots(2,2, figsize=(figWidth, figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08, 'height_ratios': [2.5,1.2]},sharex =True, sharey='row',  dpi=300)

    for i in range(2):
        for k in range(2):
            ax =axes[i, k ]
            
            ax.tick_params(axis='both', which='minor',left=0,right=0,bottom=0, top=0, direction='out', labelsize='medium', pad=2)
            ax.tick_params(axis='both', which='major',left=1,right=0,bottom=i, top=0, direction='out', labelsize='medium', pad=2) 
    
    
    if scale=='loglog':
       # ax.loglog(x,y, label =label)
        ax.set_yscale('log')
        ax.set_xscale('log')
    elif scale=='semilogy':
        ax.set_yscale('log')
    elif scale=='semilogx':
        ax.set_xscale('log')
    else:
        print('linear')
    
    # ax.set_ylabel(ylabel)
    # if i ==1:
    #     axes[i, k].set_xlabel(xlabel)
    
     
    return fig, axes

def closest_argmin(A, B):
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]


def find_closest_values_of_2_arrays(a, b):
    dup = np.searchsorted(a, b)
    uni = np.unique(dup)
    uni = uni[uni < a.shape[0]]
    ret_b = np.zeros(uni.shape[0])
    for idx, val in enumerate(uni):
        bw = np.argmin(np.abs(a[val]-b[dup == val]))
        tt = dup == val
        ret_b[idx] = np.where(tt == True)[0][bw]
    return np.column_stack((uni, ret_b))


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
        print(' ')
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
    return [file_names, int(len(file_names)/2)]

def createFilePath(names):
    ''' creatFilePath
    PURPOSE / OUTPUT:
        Turn an ordered list of names and concatinate them into a filepath.
    '''
    return ('/'.join([x for x in names if x != '']))