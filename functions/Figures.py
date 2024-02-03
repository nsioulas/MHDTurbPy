import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import os


# Set up Matplotlib parameters for the figure.
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as pltcolors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

### Create Colormap ### Remove white part of RdBU
from matplotlib.colors import LinearSegmentedColormap

# make sure defaults are used
#plt.style.use(['science', 'scatter'])
plt.rcParams['text.usetex'] = True

import sys

import general_functions as func
import calc_diagnostics as calc
def format_timestamp(timestamp,format_2_return):
    return timestamp.strftime(format_2_return)


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

    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]

    # Get figure width in inch
    if width == '1col':
        width = 8.8 # width [cm]
    elif width == '2col':
        width = 18.0 # width [cm]
    figWidth = width * cm_to_inch # width [inch]


    # Get figure height in inch
    figHeight = figWidth * (7.5/10.) if height is None else height * cm_to_inch
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
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)


    return fig, ax

def create_colors(hmany, which=None):
    
    import colormaps as cmaps
    
    

    if which is None:
        interval = np.hstack([np.linspace(0, 0.45), np.linspace(0.55, 1)])
        colors   = cmaps.w5m4(interval)
    elif which=='bone':
        interval = np.hstack([np.linspace(0, 0.35), np.linspace(0.65, 1)])
        colors   = plt.cm.OrRd(interval)
        colors   = plt.cm.RdGy_r(interval)


    cmap     = LinearSegmentedColormap.from_list('name', colors)

    return cmap(np.linspace(0,1,hmany))





def heatmap_func(x,  y, z,
                 numb_bins, xlabel, ylabel, colbar_label, min_counts =10, what ='mean', ax_scale ='loglog',
                 min_x= -1e10, min_y= -1e10, min_z= -1e10, 
                 max_x= 1e10, max_y= 1e10, max_z= 1e10, min_col = None, max_col =None,
                 log_colorbar=True,fig_size =(20,18), f_size =35, specify_edges= False, xedges =None, yedges =None,plot_contours=True, estimate_mean_median= True, return_figure =False):


    """Quantities we want to plot"""
    xf, yf, zf = np.array(x),  np.array(y), np.array(z)
    

    index             = (xf>min_x)& (yf>min_y) & (zf>min_z) & (xf<max_x)& (yf<max_y) & (zf<max_z) & (~np.isinf(x)) & (~np.isinf(y))& (~np.isinf(z))
    yf1               =  yf[index]
    zf1               =  zf[index]
    xf1               =  xf[index]
    


    """" Create bins """
    numb_x_bins, numb_y_bins  = numb_bins, numb_bins 
    
    if specify_edges:
        xmin, xmax = xedges[0], xedges[1]
        ymin, ymax = yedges[0], yedges[1]
        
    else:
        xmin, xmax = np.nanmin(xf1), np.nanmax(xf1)
        ymin, ymax = np.nanmin(yf1), np.nanmax(yf1)
        
    if ax_scale=='loglog':
        xf1_bins                  = np.logspace(np.log10(xmin), np.log10(xmax),numb_x_bins )
        yf1_bins                  = np.logspace(np.log10(ymin), np.log10(ymax),numb_y_bins )
    elif ax_scale=='linear':
        xf1_bins                  = np.linspace((xmin), (xmax),numb_x_bins )
        yf1_bins                  = np.linspace((ymin), (ymax),numb_y_bins )
    elif ax_scale=='semilogx':
        yf1_bins                  = np.linspace((ymin), (ymax),numb_y_bins )
        xf1_bins                  = np.logspace(np.log10(xmin), np.log10(xmax),numb_x_bins )
    elif ax_scale=='semilogy':
        yf1_bins                  = np.logspace(np.log10(ymin), np.log10(ymax),numb_y_bins )
        xf1_bins                  = np.linspace((xmin), (xmax),numb_x_bins )

    elif ax_scale == 'symlogy':
        xf1_bins                  = np.logspace(np.log10(xmin), np.log10(xmax),numb_x_bins )
        yf1_bins = func.symlogspace(ymin, ymax, numb_y_bins, linthresh=1e-5)


    """" Estimate mean or median within each bin """
    means   = stats.binned_statistic_2d( x= xf1,
                                         y= yf1,
                                         values    = zf1,
                                         statistic = what,
                                         bins=[xf1_bins,yf1_bins])[0]

    """" Estimate counts within each bin """
    counts  = stats.binned_statistic_2d(x= xf1,
                                        y= yf1,
                                        values    = zf1,
                                        statistic = 'count',
                                        bins=[xf1_bins,yf1_bins])[0]
    
    """" Estimate stds within each bin """
    stds  = stats.binned_statistic_2d(  
                                        x= xf1,
                                        y= yf1,
                                        values    = zf1,
                                        statistic = 'std',
                                        bins=[xf1_bins,yf1_bins])[0]
    
    rows, cols = np.shape(means)

    
    """ Remove bins with less than min_counts counts """
    for i in range(np.shape(counts)[0]):
        for k in range(np.shape(counts)[1]):
            if counts[i,k] < min_counts:
                means[i,k] = np.nan

    ### Create Colormap ### Remove white part of RdBU
   # interval = np.hstack([np.linspace(0, 0.5), np.linspace(0.5, 1)])
    #colors   = plt.cm.RdBu_r(interval)
    
    interval = np.hstack([np.linspace(0, 0.5), np.linspace(0.5, 1)])
    #colors   = plt.cm.OrRd(interval)
    colors   = plt.cm.RdGy_r(interval)
    #colors   = plt.cm.Blues(interval)
    cmap     = LinearSegmentedColormap.from_list('name', colors)
    
    
    # On purpose!!!
    xvals  = xf1_bins
    yvals  = yf1_bins
    zvals  = means.T
    counts = counts.T

    if return_figure:
    
        fig = plt.figure(figsize=fig_size)
        gs = GridSpec(8, 8)

        ax = fig.add_subplot(gs[0:8, 0:8])
        grid_thick = 0.2
        ax.xaxis.grid(True, "major", linewidth=grid_thick, ls='-')
        ax.yaxis.grid(True, "major", linewidth=grid_thick, ls='-')
        ax.yaxis.grid(True, "minor", linewidth=grid_thick, ls='-')
        ax.xaxis.grid(True, "minor", linewidth=grid_thick, ls='-')
        
        current_cmap = matplotlib.cm.get_cmap(cmap)
        current_cmap.set_bad(color='slategray')

        colbar_z = zvals.flatten()
        colbar_z = colbar_z


        if log_colorbar:
            normi    =  pltcolors.LogNorm()
            c        = ax.pcolormesh(xvals, yvals,  zvals,cmap=cmap, norm = normi)
        else:
            normi    =  pltcolors.Normalize(vmin =min_col, vmax = max_col)
            if min_col !=None:
                
                c        = ax.pcolormesh(xvals, yvals,  zvals,cmap=cmap, norm = normi)
            else:
                c        = ax.pcolormesh(xvals, yvals,  zvals,cmap=cmap)


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
        elif ax_scale == 'symlogy':
            ax.set_xscale('log')
            ax.set_yscale('symlog')

    if  return_figure:
        return fig, ax, xvals, yvals,  zvals, cmap, c, normi
    else:
        return xvals, yvals,  zvals


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
    figHeight = figWidth * (5/10.) if height is None else height * cm_to_inch
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

    plt.rc('axes', linewidth=2)

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
    figHeight = figWidth * (5/10.) if height is None else height * cm_to_inch
    fig, axes = plt.subplots(1,2, figsize=(5*figWidth,4*figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},sharex =True, sharey='row',  dpi=300)
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

def initializeFigure_1by_3(xlabel, ylabel, scale= 'loglog',width='1col', height=None,share_y=False):
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

    plt.rc('axes', linewidth=1.2)

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
    figHeight = figWidth * (3.5/10.) if height is None else height * cm_to_inch
    fig, axes = plt.subplots(1,3, figsize=(5*figWidth,4*figHeight), gridspec_kw = {'wspace':0.1, 'hspace':0.08},sharex =True, sharey='row',  dpi=300)
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
    figHeight = figWidth * (5/10.) if height is None else height * cm_to_inch
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

    # Prepare figure width and height
    cm_to_inch = 0.393701 # [inch/cm]

    # Get figure width in inch
    if width == '1col':
        width = 12 # width [cm]
    elif width == '2col':
        width = 18.0 # width [cm]
    figWidth = width * cm_to_inch # width [inch]


    # Get figure height in inch
    figHeight = figWidth * (7.5/10.) if height is None else height * cm_to_inch
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


def visualize_downloaded_intervals(
                                  sc,
                                  final_Par,
                                  final_Mag,
                                  nn_df,
                                  rolling_window,
                                  res_rate,
                                  my_dir,
                                  format_2_return ="%Y_%m_%d",
                                  size             = 21,
                                  inset_f_size     = 20,
                                  numb_subplots    = 7,
                                  join_path_figs   = True,
                                  save_fig         = True 

                                 ):

    spacecraft = 'PSP' if sc ==0 else 'SolO'
    # Creat figure name using start, end date and sc.
    f1          = format_timestamp(final_Mag.index[0], format_2_return)
    f2          = format_timestamp(final_Mag.index[-1], format_2_return)
    figure_name = f1+"_"+f2+"_"+str(sc) + '.png'

    # Resample to desired rate
    final_Mag = final_Mag.resample(f'{str(res_rate)}s').mean()
    final_Par = final_Par.resample(f'{str(res_rate)}s').mean()
    nn_df     = nn_df.resample(f'{str(res_rate)}s').mean()
    # Estimate relevant quantitities
   # nn_df       = calc.prepare_particle_data_for_visualization( final_Par, final_Mag, rolling_window)
   # nn_df       = nn_df.resample(f'{str(res_rate)}s').mean()

    # Choose limiting dates
    start_date_lim  = final_Par.index[0]
    end_date_lim    = final_Par.index[-1]

    # Init figure
    fig, axs        = plt.subplots(numb_subplots, sharex=True,figsize=(30,15), gridspec_kw = {'wspace':0.05, 'hspace':0.05})
    minor_tick_params, major_tick_params = inset_axis_params(size ='xx-large')



    #Now plot
    """1st plot"""
    try:
        final_Mag['B_RTN'] = np.sqrt(final_Mag.Br**2 + final_Mag.Bt**2 + final_Mag.Bn**2)

        axs[0].plot(final_Mag['Br'],linewidth=0.4,ls='-', ms=0,color ='darkblue')
        axs[0].plot(final_Mag['Bt'],linewidth=0.4,ls='-', ms=0,color ='darkred')
        axs[0].plot(final_Mag['Bn'],linewidth=0.4,ls='-', ms=0,color ='darkgreen')
        axs[0].plot(final_Mag['B_RTN'],linewidth=0.4,ls='-', ms=0,color ='k')

        RTN_Flag =1

        """2nd plot"""
        axs[1].plot(np.sqrt(final_Par.Vr**2 + final_Par.Vt**2 + final_Par.Vn**2),linewidth=0.8,ls='-', ms=0,color ='C0')#,label='$|B|$')
        ax2 = axs[1].twinx()

    except:
        final_Mag['B_RTN'] = np.sqrt(final_Mag.Bx**2 + final_Mag.By**2 + final_Mag.Bz**2)

        axs[0].plot(final_Mag['Bx'],linewidth=0.4,ls='-', ms=0,color ='darkblue')
        axs[0].plot(final_Mag['By'],linewidth=0.4,ls='-', ms=0,color ='darkred')
        axs[0].plot(final_Mag['Bz'],linewidth=0.4,ls='-', ms=0,color ='darkgreen')
        axs[0].plot(final_Mag['B_RTN'],linewidth=0.4,ls='-', ms=0,color ='k')

        RTN_Flag =0

        """2nd plot"""
        axs[1].plot(np.sqrt(final_Par.Vx**2 + final_Par.Vy**2 + final_Par.Vz**2),linewidth=0.8,ls='-', ms=0,color ='C0')#,label='$|B|$')
        ax2 = axs[1].twinx()   



    ax2.plot(final_Par['Vth'],linewidth=0.8,ls='-', ms=0,color ='k')#,label='$|B|$')
    #dfts[['Vth']].plot(ax = ax, legend=False, style=['C1'], lw = 0.6, alpha = 0.6)
    ax2.legend(['$V_{th}~ [km/s]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.6), loc = 2)


    """3rd plot"""
    axs[2].plot(final_Par.np,linewidth=0.8,ls='-', ms=0,color ='darkred')#,label='$|B|$')

    """3rd plot"""
    axs[3].plot(nn_df.sigma_c, linewidth=0.8,ls='-', ms=0,color ='darkblue')#,label='$|B|$')

    """3rd plot"""
    axs[3].plot(nn_df.sigma_r, linewidth=0.8,ls='-', ms=0,color ='darkred')#,label='$|B|$')

    """3rd plot"""
    axs[4].plot(nn_df.beta, linewidth=0.8,ls='-', ms=0,color ='black')#,label='$|B|$')

    """3rd plot"""
    axs[5].plot(nn_df.VB, linewidth=0.8,ls='-', ms=0,color ='black')#,label='$|B|$')

    """4th plot"""
    axs[6].plot(final_Par.Dist_au, linewidth=0.8,ls='-', ms=0,color ='black')#,label='$|B|$')
    try:
        ax3 = axs[6].twinx()
        ax3.plot(final_Par['carr_lon'],linewidth=0.8,ls='-', ms=0,color ='darkred')#,label='$|B|$')
        #dfts[['Vth']].plot(ax = ax, legend=False, style=['C1'], lw = 0.6, alpha = 0.6)
        ax3.legend(['$Carr. long ~ [^{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 0.6), loc = 2)
    except:
        pass


    ## y Axis labels ##
    if RTN_Flag ==1:
         axs[0].legend([r'$B_{r} ~ [nT]$',r'$B_{t} ~ [nT]$',r'$B_{n} ~ [nT]$',r'$|B| ~ [nT]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
    else:
         axs[0].legend([r'$B_{x} ~ [nT]$',r'$B_{y} ~ [nT]$',r'$B_{z} ~ [nT]$',r'$|B| ~ [nT]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01, 1), loc = 2)
    axs[1].legend(['$V_{sw} ~[km ~s^{-1}$]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    axs[2].legend(['$N_{p}~[(cm^{-3}$]'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    axs[3].legend(['$\sigma_{c}$','$\sigma_{r}$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    #axs[4].legend([], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    axs[4].legend([r'$\beta$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    axs[5].legend([r'$\Theta_{VB} ~[^{\circ}]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    axs[6].legend([r'$R ~[au]$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)


    #axs[3].legend([r'$\sigma_c$'], fontsize='large', frameon=False, bbox_to_anchor=(1.01,1), loc = 2)
    for i in range(numb_subplots ):

        axs[i].xaxis.grid(True, "minor", linewidth=.1, ls='-');  
        axs[i].yaxis.grid(True, "major", linewidth=.1, ls='-');
        axs[i].yaxis.grid(True, "minor", linewidth=.1, ls='-');  
        axs[i].xaxis.grid(True, "major", linewidth=.1, ls='-');    

        axs[i].xaxis.grid(True, "minor", linewidth=.1, ls='-');  
        axs[i].yaxis.grid(True, "major", linewidth=.1, ls='-');
        axs[i].yaxis.grid(True, "minor", linewidth=.1, ls='-');  
        axs[i].xaxis.grid(True, "major", linewidth=.1, ls='-'); 
        axs[i].tick_params(**minor_tick_params)
        axs[i].tick_params(**major_tick_params)



        if i==0:

            axs[i].legend(loc=0, frameon=0, fontsize=20)


        # Set axis limits
        axs[i].set_xlim([start_date_lim, end_date_lim])
    if join_path_figs:
        final_save_path = Path(my_dir).joinpath('figures')
    else:
        final_save_path = Path(my_dir)#.joinpath('figures')
    os.makedirs(str(final_save_path), exist_ok=True)
    if save_fig:
        fig.savefig(str(final_save_path.joinpath(figure_name)), format='png',dpi=300,bbox_inches='tight')
    fig.show()
