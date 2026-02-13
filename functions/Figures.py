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

_FUNCTIONS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FUNCTIONS_DIR.parent
sys.path.insert(1, str(_FUNCTIONS_DIR))

#from  CUSIA.Colors.CUSIA_Colors import mycmap



import general_functions as func

def format_timestamp(timestamp,format_2_return):
    return timestamp.strftime(format_2_return)


def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    #if you don't have LaTeX installed on your laptop and this statement 
    # generates error, comment it out
    plt.rc('text', usetex=True)

    return

def plot_line_points(x, y, figsize=6, xlabel=' ', ylabel=' ', col= 'darkslateblue', 
                     xp = None, yp = None, points = False, pmarker='.', pcol='slateblue',
                     legend=None, plegend = None, legendloc='lower right', 
                     plot_title = None, grid=None, figsave = None):
    """
    A simple helper routine to make plots that involve a line and (optionally)
    a set of points, which was introduced and used during the first two weeks 
    of class.
    """
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()

    if legend:
        plt.plot(x, y, lw = 1., c=col, label = legend)
        if points: 
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        plt.plot(x, y, lw = 1., c=col)
        if points:
            plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)
        
    if grid: 
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')
        
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()

def initializeFigure(xlabel=r'$f ~[sc]$', ylabel=r'$PSD ~[nT^{2} Hz^{-1}]$', scale= 'loglog',width='1col', height=None):
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
    figHeight = figWidth * (8.35/10.) if height is None else height * cm_to_inch
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

def create_colors_new(hmany, which=None, return_cmap=False):
    import colormaps as cmaps

    if which is None:
        # Generate a color map using an interval excluding the middle range
        interval = np.hstack([np.linspace(0, 0.45, num=hmany//2), np.linspace(0.55, 1, num=hmany//2)])
        colors = cmaps.w5m4(interval)  # Assuming w5m4 returns an array of RGBA values
    elif which == 'bone':
        # For 'bone', use a different excluded middle range
        interval = np.hstack([np.linspace(0, 0.35, num=hmany//2), np.linspace(0.65, 1, num=hmany//2)])
        colors = plt.cm.RdGy_r(interval)  # Use matplotlib's colormap

    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    if return_cmap:
        return cmap, colors  # Return the colormap object and colors
    else:
        return cmap  # Return only the colormap object
    
    
def create_colors(hmany, which=None, return_cmap =False, NN =None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import colormaps as cmaps  # Assuming this is a custom colormap module

    if which is None:
        # Create two intervals and combine them for a custom colormap.
        interval = np.hstack([np.linspace(0, 0.45, num=hmany//2),
                              np.linspace(0.55, 1, num=hmany - hmany//2)])
        colors = cmaps.w5m4(interval)
        
    elif which == 'bone':
        # Use two sub-intervals for the OrRd colormap.
        interval = np.hstack([np.linspace(0, 0.35, num=hmany//2),
                              np.linspace(0.65, 1, num=hmany - hmany//2)])
        colors = plt.cm.OrRd(interval)
        
    elif which == 'rdgy':
        # Use the full range of the RdGy colormap.
        interval =  np.hstack([np.linspace(0, 0.35, num=hmany//2),
                              np.linspace(0.65, 1, num=hmany - hmany//2)])
        colors = plt.cm.RdGy(interval)
        
    elif which == 'half_blues':
        # Focus on the lower half of the Blues colormap.
        interval = np.linspace(0, 0.6, hmany)
        colors = plt.cm.PuBu_r(interval)

    elif which == 'half_blues_r':
        # Focus on the lower half of the Blues colormap.
        interval = np.linspace(0.4, 1, hmany)
        colors = plt.cm.PuBu(interval)
        
    elif which == 'cusia':
        # Use a custom NeutralGrey colormap from a custom mapping function.
        interval = np.linspace(0, 1, hmany)
        colors = mycmap(colors='NeutralGrey')(interval)
    else:
        raise ValueError("Unknown colormap option: {}".format(which))


    if  return_cmap:
        # Create the custom colormap from the selected colors.
        if NN==None:
            cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        else:
            cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, NN)            
        return cmap, cmap(np.linspace(0, 1, hmany))
    else:
        # Create the custom colormap from the selected colors.
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        return cmap(np.linspace(0, 1, hmany))


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# def create_colors(hmany, which=None):
#     """
#     Returns an (hmany x 4) array of RGBA colors sampled from one of 15 variations
#     of an "old_rose_misty_blue" gradient. In each variation the gradient is defined
#     by four color stops: a starting old-rose tone, two intermediate colors that add extra
#     nuance and adjust the level of dulness, and an ending misty blue tone.
    
#     Parameters
#     ----------
#     hmany : int
#         Number of colors (each as an RGBA row) to return.
#     which : str, optional
#         One of 15 variant names. If None or an invalid key is passed,
#         the default 'old_rose_misty_blue_3' is used.
        
#         Valid options:
#             'old_rose_misty_blue_1'
#             'old_rose_misty_blue_2'
#             'old_rose_misty_blue_3'
#             'old_rose_misty_blue_4'
#             'old_rose_misty_blue_5'
#             'old_rose_misty_blue_6'
#             'old_rose_misty_blue_7'
#             'old_rose_misty_blue_8'
#             'old_rose_misty_blue_9'
#             'old_rose_misty_blue_10'
#             'old_rose_misty_blue_11'
#             'old_rose_misty_blue_12'
#             'old_rose_misty_blue_13'
#             'old_rose_misty_blue_14'
#             'old_rose_misty_blue_15'
    
#     Returns
#     -------
#     numpy.ndarray
#         An (hmany x 4) array of RGBA values (floats in [0,1]) sampled along the gradient.
#     """
#     # Define 15 variations of the "old_rose_misty_blue" gradient.
#     # Each variant is defined by 4 stops: [start, mid1, mid2, end].
#     # The start is an old rose (redish) tone and the end is a misty blue.
#     # The two intermediate colors adjust the gradient's dulness.
#     variants = {
#         'old_rose_misty_blue_1':  ["#B04A4A", "#A85765", "#A05570", "#7497D0"],
#         'old_rose_misty_blue_2':  ["#AA4949", "#A04D4F", "#9D546C", "#7093C9"],
#         'old_rose_misty_blue_3':  ["#A45252", "#9C5258", "#97506A", "#6C8DC2"],
#         'old_rose_misty_blue_4':  ["#9F4C4C", "#984A59", "#935066", "#6787BB"],
#         'old_rose_misty_blue_5':  ["#9A4646", "#944C55", "#8F4F62", "#6281B4"],
#         'old_rose_misty_blue_6':  ["#954040", "#8D473E", "#8A4B5E", "#5D7BAD"],
#         'old_rose_misty_blue_7':  ["#8F3A3A", "#893C43", "#84475A", "#5775A6"],
#         'old_rose_misty_blue_8':  ["#8A3434", "#853A3F", "#7F414F", "#52709F"],
#         'old_rose_misty_blue_9':  ["#853030", "#803434", "#7A3C4B", "#4D6A98"],
#         'old_rose_misty_blue_10': ["#7F2A2A", "#792C30", "#753645", "#486493"],
#         'old_rose_misty_blue_11': ["#7A2424", "#743C3A", "#703F40", "#43618C"],
#         'old_rose_misty_blue_12': ["#75201F", "#6A3C2F", "#6C3A3C", "#3D5D85"],
#         'old_rose_misty_blue_13': ["#6F1C1A", "#6A1F1D", "#683737", "#38587F"],
#         'old_rose_misty_blue_14': ["#6A1815", "#652421", "#633332", "#34527A"],
#         'old_rose_misty_blue_15': ["#650612", "#602518", "#5E2F2F", "#304D75"],
#     }
    
#     # Use default if key not found.
#     if which not in variants:
#         which = 'old_rose_misty_blue_3'
    
#     # Retrieve the chosen variant's list of 4 color stops.
#     stops = variants[which]
    
#     # Create a LinearSegmentedColormap using these 4 stops.
#     cmap = LinearSegmentedColormap.from_list(name=which, colors=stops, N=256)
    
#     # Sample hmany evenly spaced colors from the colormap.
#     sampled_colors = cmap(np.linspace(0, 1, hmany))
    
#     return sampled_colors

def heatmap_func(x,  y, z,
                 numb_bins, xlabel, ylabel, colbar_label, min_counts =10, what ='mean', ax_scale ='loglog',
                 min_x= -1e10, min_y= -1e10, min_z= -1e10, 
                 max_x= 1e10, max_y= 1e10, max_z= 1e10, min_col = None, max_col =None,
                 log_colorbar=True,fig_size =(20,18), f_size =35, specify_edges= False, xedges =None, yedges =None,plot_contours=True, estimate_mean_median= True, return_figure =False,  norm_2_max = False):


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
    
    if norm_2_max:
        zvals  =zvals/np.nanmax(np.nanmax(zvals))
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
        elif ax_scale == 'linear':
            ax.set_xscale('linear')
            ax.set_yscale('linear')

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
sys.path.insert(1, str(_FUNCTIONS_DIR))

#from  CUSIA.Colors.CUSIA_Colors import mycmap



import general_functions as func

def format_timestamp(timestamp,format_2_return):
    return timestamp.strftime(format_2_return)


def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    #if you don't have LaTeX installed on your laptop and this statement 
    # generates error, comment it out
    plt.rc('text', usetex=True)

    return

def plot_line_points(x, y, figsize=6, xlabel=' ', ylabel=' ', col= 'darkslateblue', 
                     xp = None, yp = None, points = False, pmarker='.', pcol='slateblue',
                     legend=None, plegend = None, legendloc='lower right', 
                     plot_title = None, grid=None, figsave = None):
    """
    A simple helper routine to make plots that involve a line and (optionally)
    a set of points, which was introduced and used during the first two weeks 
    of class.
    """
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()

    if legend:
        plt.plot(x, y, lw = 1., c=col, label = legend)
        if points: 
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        plt.plot(x, y, lw = 1., c=col)
        if points:
            plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)
        
    if grid: 
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')
        
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()

def initializeFigure(xlabel=r'$f ~[sc]$', ylabel=r'$PSD ~[nT^{2} Hz^{-1}]$', scale= 'loglog',width='1col', height=None):
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
    figHeight = figWidth * (8.35/10.) if height is None else height * cm_to_inch
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

def create_colors_new(hmany, which=None, return_cmap=False):
    import colormaps as cmaps

    if which is None:
        # Generate a color map using an interval excluding the middle range
        interval = np.hstack([np.linspace(0, 0.45, num=hmany//2), np.linspace(0.55, 1, num=hmany//2)])
        colors = cmaps.w5m4(interval)  # Assuming w5m4 returns an array of RGBA values
    elif which == 'bone':
        # For 'bone', use a different excluded middle range
        interval = np.hstack([np.linspace(0, 0.35, num=hmany//2), np.linspace(0.65, 1, num=hmany//2)])
        colors = plt.cm.RdGy_r(interval)  # Use matplotlib's colormap

    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    if return_cmap:
        return cmap, colors  # Return the colormap object and colors
    else:
        return cmap  # Return only the colormap object
    
    
def create_colors(hmany, which=None, return_cmap =False, NN =None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import colormaps as cmaps  # Assuming this is a custom colormap module

    if which is None:
        # Create two intervals and combine them for a custom colormap.
        interval = np.hstack([np.linspace(0, 0.45, num=hmany//2),
                              np.linspace(0.55, 1, num=hmany - hmany//2)])
        colors = cmaps.w5m4(interval)
        
    elif which == 'bone':
        # Use two sub-intervals for the OrRd colormap.
        interval = np.hstack([np.linspace(0, 0.35, num=hmany//2),
                              np.linspace(0.65, 1, num=hmany - hmany//2)])
        colors = plt.cm.OrRd(interval)
        
    elif which == 'rdgy':
        # Use the full range of the RdGy colormap.
        interval =  np.hstack([np.linspace(0, 0.35, num=hmany//2),
                              np.linspace(0.65, 1, num=hmany - hmany//2)])
        colors = plt.cm.RdGy(interval)
        
    elif which == 'half_blues':
        # Focus on the lower half of the Blues colormap.
        interval = np.linspace(0, 0.6, hmany)
        colors = plt.cm.PuBu_r(interval)

    elif which == 'half_blues_r':
        # Focus on the lower half of the Blues colormap.
        interval = np.linspace(0.4, 1, hmany)
        colors = plt.cm.PuBu(interval)
        
    elif which == 'cusia':
        # Use a custom NeutralGrey colormap from a custom mapping function.
        interval = np.linspace(0, 1, hmany)
        colors = mycmap(colors='NeutralGrey')(interval)
    else:
        raise ValueError("Unknown colormap option: {}".format(which))


    if  return_cmap:
        # Create the custom colormap from the selected colors.
        if NN==None:
            cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        else:
            cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, NN)            
        return cmap, cmap(np.linspace(0, 1, hmany))
    else:
        # Create the custom colormap from the selected colors.
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors)
        return cmap(np.linspace(0, 1, hmany))


# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# def create_colors(hmany, which=None):
#     """
#     Returns an (hmany x 4) array of RGBA colors sampled from one of 15 variations
#     of an "old_rose_misty_blue" gradient. In each variation the gradient is defined
#     by four color stops: a starting old-rose tone, two intermediate colors that add extra
#     nuance and adjust the level of dulness, and an ending misty blue tone.
    
#     Parameters
#     ----------
#     hmany : int
#         Number of colors (each as an RGBA row) to return.
#     which : str, optional
#         One of 15 variant names. If None or an invalid key is passed,
#         the default 'old_rose_misty_blue_3' is used.
        
#         Valid options:
#             'old_rose_misty_blue_1'
#             'old_rose_misty_blue_2'
#             'old_rose_misty_blue_3'
#             'old_rose_misty_blue_4'
#             'old_rose_misty_blue_5'
#             'old_rose_misty_blue_6'
#             'old_rose_misty_blue_7'
#             'old_rose_misty_blue_8'
#             'old_rose_misty_blue_9'
#             'old_rose_misty_blue_10'
#             'old_rose_misty_blue_11'
#             'old_rose_misty_blue_12'
#             'old_rose_misty_blue_13'
#             'old_rose_misty_blue_14'
#             'old_rose_misty_blue_15'
    
#     Returns
#     -------
#     numpy.ndarray
#         An (hmany x 4) array of RGBA values (floats in [0,1]) sampled along the gradient.
#     """
#     # Define 15 variations of the "old_rose_misty_blue" gradient.
#     # Each variant is defined by 4 stops: [start, mid1, mid2, end].
#     # The start is an old rose (redish) tone and the end is a misty blue.
#     # The two intermediate colors adjust the gradient's dulness.
#     variants = {
#         'old_rose_misty_blue_1':  ["#B04A4A", "#A85765", "#A05570", "#7497D0"],
#         'old_rose_misty_blue_2':  ["#AA4949", "#A04D4F", "#9D546C", "#7093C9"],
#         'old_rose_misty_blue_3':  ["#A45252", "#9C5258", "#97506A", "#6C8DC2"],
#         'old_rose_misty_blue_4':  ["#9F4C4C", "#984A59", "#935066", "#6787BB"],
#         'old_rose_misty_blue_5':  ["#9A4646", "#944C55", "#8F4F62", "#6281B4"],
#         'old_rose_misty_blue_6':  ["#954040", "#8D473E", "#8A4B5E", "#5D7BAD"],
#         'old_rose_misty_blue_7':  ["#8F3A3A", "#893C43", "#84475A", "#5775A6"],
#         'old_rose_misty_blue_8':  ["#8A3434", "#853A3F", "#7F414F", "#52709F"],
#         'old_rose_misty_blue_9':  ["#853030", "#803434", "#7A3C4B", "#4D6A98"],
#         'old_rose_misty_blue_10': ["#7F2A2A", "#792C30", "#753645", "#486493"],
#         'old_rose_misty_blue_11': ["#7A2424", "#743C3A", "#703F40", "#43618C"],
#         'old_rose_misty_blue_12': ["#75201F", "#6A3C2F", "#6C3A3C", "#3D5D85"],
#         'old_rose_misty_blue_13': ["#6F1C1A", "#6A1F1D", "#683737", "#38587F"],
#         'old_rose_misty_blue_14': ["#6A1815", "#652421", "#633332", "#34527A"],
#         'old_rose_misty_blue_15': ["#650612", "#602518", "#5E2F2F", "#304D75"],
#     }
    
#     # Use default if key not found.
#     if which not in variants:
#         which = 'old_rose_misty_blue_3'
    
#     # Retrieve the chosen variant's list of 4 color stops.
#     stops = variants[which]
    
#     # Create a LinearSegmentedColormap using these 4 stops.
#     cmap = LinearSegmentedColormap.from_list(name=which, colors=stops, N=256)
    
#     # Sample hmany evenly spaced colors from the colormap.
#     sampled_colors = cmap(np.linspace(0, 1, hmany))
    
#     return sampled_colors

def heatmap_func(x,  y, z,
                 numb_bins, xlabel, ylabel, colbar_label, min_counts =10, what ='mean', ax_scale ='loglog',
                 min_x= -1e10, min_y= -1e10, min_z= -1e10, 
                 max_x= 1e10, max_y= 1e10, max_z= 1e10, min_col = None, max_col =None,
                 log_colorbar=True,fig_size =(20,18), f_size =35, specify_edges= False, xedges =None, yedges =None,plot_contours=True, estimate_mean_median= True, return_figure =False,  norm_2_max = False):


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
    
    if norm_2_max:
        zvals  =zvals/np.nanmax(np.nanmax(zvals))
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
        elif ax_scale == 'linear':
            ax.set_xscale('linear')
            ax.set_yscale('linear')

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


# ============================================================
#  COMPLETE "START -> FINISH" PIPELINE
#  - loads interval pickles
#  - (optionally) masks ALL "large gaps" per gap-type threshold
#  - applies rolling mean
#  - makes the 7-panel overview plot + optional sub-Alfvénic shading
#  - makes ONE scatter plot per file (colored by sub-Alfvénic interval)
# ============================================================

import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Small utilities (self-contained)
# ============================================================
def format_timestamp(ts, fmt="%Y_%m_%d"):
    return pd.Timestamp(ts).strftime(fmt)


def inset_axis_params(size="xx-large"):
    minor_tick_params = dict(which="minor", length=3, width=0.8, labelsize=size, direction="in")
    major_tick_params = dict(which="major", length=6, width=1.0, labelsize=size, direction="in")
    return minor_tick_params, major_tick_params


def load_files(load_path: str, pattern: str):
    """
    Compatible replacement for func.load_files(load_path, 'final.pkl') style.
    It finds files recursively and returns a sorted list.
    """
    load_path = str(load_path)
    hits = glob(os.path.join(load_path, "**", pattern), recursive=True)
    hits = sorted(hits)
    return hits


# ============================================================
# Gap handling (fast, pythonic)
# ============================================================
def _prep_gap_df(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a gap table to have valid datetime Start/End, sorted.
    Expected columns: ['Start', 'End'].
    """
    if gaps is None or not isinstance(gaps, pd.DataFrame) or len(gaps) == 0:
        return pd.DataFrame(columns=["Start", "End"])

    if ("Start" not in gaps.columns) or ("End" not in gaps.columns):
        return pd.DataFrame(columns=["Start", "End"])

    g = gaps[["Start", "End"]].copy()
    g["Start"] = pd.to_datetime(g["Start"])
    g["End"] = pd.to_datetime(g["End"])
    g = g[g["End"] > g["Start"]].sort_values("Start").reset_index(drop=True)
    return g


def filter_gaps_by_min_duration(gaps, min_gap):
    """
    Keep only gaps with duration >= min_gap.
    This is what you want if "large gaps" should be removed/masked in the data products.
    """
    g = _prep_gap_df(gaps)
    if len(g) == 0:
        return g

    min_gap = pd.to_timedelta(min_gap)
    dt = g["End"] - g["Start"]
    keep = dt >= min_gap
    return g.loc[keep].reset_index(drop=True)


def merge_gap_intervals(gaps, merge_tol ="0s"):
    """
    Merge overlapping or nearly-contiguous gap intervals.
    merge_tol lets you merge gaps separated by <= merge_tol.
    """
    g = _prep_gap_df(gaps)
    if len(g) == 0:
        return g

    merge_tol = pd.to_timedelta(merge_tol)

    s = g["Start"].to_numpy(dtype="datetime64[ns]").astype("int64")
    e = g["End"].to_numpy(dtype="datetime64[ns]").astype("int64")
    tol = int(merge_tol.value)

    out_s = [int(s[0])]
    out_e = [int(e[0])]

    for i in range(1, len(s)):
        if int(s[i]) <= out_e[-1] + tol:
            if int(e[i]) > out_e[-1]:
                out_e[-1] = int(e[i])
        else:
            out_s.append(int(s[i]))
            out_e.append(int(e[i]))

    out = pd.DataFrame(
        {
            "Start": pd.to_datetime(np.array(out_s, dtype="int64")),
            "End": pd.to_datetime(np.array(out_e, dtype="int64")),
        }
    )
    return out


def mask_df_with_gaps(df, gaps, columns=None):
    """
    Set df.loc[start:end, columns] = NaN for each gap interval.

    Intended use:
      - first filter gaps by duration threshold (large gaps)
      - merge them (optional)
      - mask data so plotting/rolling does not bridge them
    """
    if df is None or not isinstance(df, (pd.DataFrame, pd.Series)):
        return df

    if isinstance(df, pd.Series):
        out = df.copy()
        cols = None
    else:
        out = df.copy()
        cols = columns

    g = _prep_gap_df(gaps)
    if len(g) == 0:
        return out

    for t0, t1 in zip(g["Start"].to_numpy(), g["End"].to_numpy()):
        if cols is None:
            out.loc[t0:t1] = np.nan
        else:
            out.loc[t0:t1, cols] = np.nan

    return out


def build_large_gap_masks(
    mag_gaps: pd.DataFrame,
    qtn_gaps: pd.DataFrame,
    par_gaps: pd.DataFrame,
    sc_pot_gaps: pd.DataFrame,
    gap_thresholds: dict,
    merge_tol:  "0s",
):
    """
    Returns dict of merged "large gaps" per type.
    gap_thresholds keys: {'mag','qtn','par','sc_pot'} with timedelta-like values.
    """
    out = {}

    for key, g in [
        ("mag", mag_gaps),
        ("qtn", qtn_gaps),
        ("par", par_gaps),
        ("sc_pot", sc_pot_gaps),
    ]:
        thr = gap_thresholds.get(key, None)
        if thr is None:
            out[key] = pd.DataFrame(columns=["Start", "End"])
            continue

        gf = filter_gaps_by_min_duration(g, thr)
        gm = merge_gap_intervals(gf, merge_tol=merge_tol)
        out[key] = gm

    return out


# ============================================================
# Your plotting function (kept EXACT as current version,
# plus enforced sub-Alfvénic min duration default 10 min)
# ============================================================
def visualize_downloaded_intervals(
    sc,
    final_Par,
    final_Mag,
    nn_df,
    my_dir,
    format_2_return="%Y_%m_%d",
    size=21,
    numb_subplots=7,
    font_size="x-large",
    join_path_figs=True,
    save_fig=True,
    # ==========================================================
    # NEW (all optional)
    # ==========================================================
    fname_tag="",  # appended AFTER the date-range in the main figure filename
    add_vb_ref_lines=False,  # (1) add y=0 and y=180 dashed reference lines in VB panel
    vb_ref_lines=(0.0, 180.0),
    vb_ref_ls="--",
    vb_ref_lw=1.2,
    shade_subalfvenic=False,  # (2) shade intervals where Ma < 1 for >= N minutes (merge gaps <= M)
    subalfvenic_ma_threshold=1.0,
    subalfvenic_window="1min",          # N (kept)
    subalfvenic_gap_tolerance="1min",   # M
    subalfvenic_min_duration="10min",   # enforce minimum event duration (default 10 min)
    subalfvenic_span_alpha=0.2,
    subalfvenic_span_color="darkred",
    make_subalfvenic_scatter=False,   # (3) ONE scatter plot per file with colored intervals + legend dates
    subalfvenic_scatter_folder="subalfvenic_scatter",
    subalfvenic_scatter_dpi=250,
):
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def _format_tag(tag: str) -> str:
        tag = str(tag).strip()
        if tag == "":
            return ""
        tag = tag.replace(" ", "")
        return f"_{tag}"

    def _to_timedelta(x):
        if isinstance(x, pd.Timedelta):
            return x
        return pd.to_timedelta(str(x))

    def _median_dt_ns(tidx: pd.DatetimeIndex) -> int:
        if tidx is None or len(tidx) < 2:
            return 0
        t = tidx.view("int64")
        dt = np.diff(t)
        if dt.size == 0:
            return 0
        dt = dt[dt > 0]
        if dt.size == 0:
            return 0
        return int(np.median(dt))

    def _find_subalfvenic_intervals_fast(
        ma_series: pd.Series,
        ma_thr: float,
        min_dur: pd.Timedelta,
        gap_tol: pd.Timedelta,
    ):
        if ma_series is None or len(ma_series) == 0:
            return []

        idx = ma_series.index
        if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
            return []

        ma = ma_series.to_numpy(dtype=float, copy=False)
        ok = np.isfinite(ma)
        cond = ok & (ma < ma_thr)
        if not np.any(cond):
            return []

        c = cond.astype(np.int8)
        dc = np.diff(c)
        starts = np.where(dc == 1)[0] + 1
        ends = np.where(dc == -1)[0]

        if cond[0]:
            starts = np.r_[0, starts]
        if cond[-1]:
            ends = np.r_[ends, len(cond) - 1]

        if starts.size == 0 or ends.size == 0:
            return []

        dt_nom_ns = _median_dt_ns(idx)
        if dt_nom_ns <= 0:
            dt_nom_ns = int(min_dur.value) if min_dur.value > 0 else int(pd.Timedelta("1min").value)

        t_ns = idx.view("int64")
        t0_ns = t_ns[starts]
        t1_ns = t_ns[ends] + dt_nom_ns

        gap_tol_ns = int(gap_tol.value)
        min_dur_ns = int(min_dur.value)

        merged_t0 = [int(t0_ns[0])]
        merged_t1 = [int(t1_ns[0])]

        for i in range(1, len(t0_ns)):
            a0 = int(t0_ns[i])
            a1 = int(t1_ns[i])
            if a0 <= merged_t1[-1] + gap_tol_ns:
                if a1 > merged_t1[-1]:
                    merged_t1[-1] = a1
            else:
                merged_t0.append(a0)
                merged_t1.append(a1)

        out = []
        for a0, a1 in zip(merged_t0, merged_t1):
            if (a1 - a0) >= min_dur_ns:
                out.append((pd.to_datetime(a0), pd.to_datetime(a1)))
        return out

    f1 = format_timestamp(final_Mag.index[0], format_2_return)
    f2 = format_timestamp(final_Mag.index[-1], format_2_return)
    tag = _format_tag(fname_tag)
    figure_name = f"{f1}_{f2}{tag}_{str(sc)}.png"

    start_date_lim = final_Par.index[0]
    end_date_lim = final_Par.index[-1]

    fig, axs = plt.subplots(
        numb_subplots,
        sharex=True,
        figsize=(0.95 * 30, 0.95 * 15),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )
    minor_tick_params, major_tick_params = inset_axis_params(size="xx-large")

    index = final_Mag.index
    par_index = final_Par.index
    sig_index = nn_df.index

    try:
        final_Mag["B_RTN"] = np.sqrt(final_Mag.Br**2 + final_Mag.Bt**2 + final_Mag.Bn**2)

        axs[0].plot(index, final_Mag["Br"].values, linewidth=0.4, ls="-", ms=0, color="darkblue")
        axs[0].plot(index, final_Mag["Bt"].values, linewidth=0.4, ls="-", ms=0, color="darkred")
        axs[0].plot(index, final_Mag["Bn"].values, linewidth=0.4, ls="-", ms=0, color="darkgreen")
        axs[0].plot(index, final_Mag["B_RTN"].values, linewidth=0.4, ls="-", ms=0, color="k")

        RTN_Flag = 1

        axs[1].plot(
            par_index,
            np.sqrt(final_Par.Vr**2 + final_Par.Vt**2 + final_Par.Vn**2).values,
            linewidth=0.8,
            ls="-",
            ms=0,
            color="C0",
        )
        ax2 = axs[1].twinx()

    except Exception:
        final_Mag["B_RTN"] = np.sqrt(final_Mag.Bx**2 + final_Mag.By**2 + final_Mag.Bz**2)

        axs[0].plot(index, final_Mag["Bx"].values, linewidth=0.4, ls="-", ms=0, color="darkblue")
        axs[0].plot(index, final_Mag["By"].values, linewidth=0.4, ls="-", ms=0, color="darkred")
        axs[0].plot(index, final_Mag["Bz"].values, linewidth=0.4, ls="-", ms=0, color="darkgreen")
        axs[0].plot(index, final_Mag["B_RTN"].values, linewidth=0.4, ls="-", ms=0, color="k")

        axs[0].legend(
            [r"$B_{r} ~ [nT]$", r"$B_{t} ~ [nT]$", r"$B_{n} ~ [nT]$", r"$|B| ~ [nT]$"],
            fontsize=font_size,
            frameon=False,
            bbox_to_anchor=(1.01, 0.6),
            loc=2,
            ncol=4,
        )

        RTN_Flag = 0

        axs[1].plot(
            np.sqrt(final_Par.Vx**2 + final_Par.Vy**2 + final_Par.Vz**2),
            linewidth=0.8,
            ls="-",
            ms=0,
            color="C0",
        )
        ax2 = axs[1].twinx()

    ax2.plot(par_index, final_Par["Vth"].values, linewidth=0.8, ls="-", ms=0, color="k")
    ax2.legend(["$T_{p}~ [eV]$"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 0.6), loc=2)

    axs[2].plot(par_index, final_Par.np.values, linewidth=0.8, ls="-", ms=0, color="darkred")

    axs[3].plot(sig_index, nn_df.sigma_c.values, linewidth=0.8, ls="-", ms=0, color="darkblue")
    axs[3].plot(sig_index, nn_df.sigma_r.values, linewidth=0.8, ls="-", ms=0, color="darkred")

    axs[4].semilogy(sig_index, nn_df.beta.values, linewidth=0.8, ls="-", ms=0, color="black")
    axs[4].axhline(y=1, ls=":", c="k", lw=2)
    axs[4].set_ylim([1 / 2 * np.nanmin(nn_df.beta.values), 2 * np.nanmax(nn_df.beta.values)])

    try:
        ax4 = axs[4].twinx()
        ax4.semilogy(sig_index, nn_df.Ma.values, linewidth=0.8, ls="-", ms=0, color="darkred")
        ax4.legend(["$M_a$"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 0.6), loc=2)
        ax4.axhline(y=1, ls=":", c="darkred", lw=2)
        ax4.set_ylim([1 / 3 * np.nanmin(nn_df.Ma.values), 3 * np.nanmax(nn_df.Ma.values)])
    except Exception:
        pass

    axs[5].plot(sig_index, nn_df.VB.values, linewidth=0.8, ls="-", ms=0, color="black")

    if add_vb_ref_lines:
        y0, y1 = vb_ref_lines
        axs[5].axhline(y=y0, ls=vb_ref_ls, c="k", lw=vb_ref_lw)
        axs[5].axhline(y=y1, ls=vb_ref_ls, c="k", lw=vb_ref_lw)

    axs[6].plot(par_index, 215.043 * final_Par.Dist_au.values, linewidth=0.8, ls="-", ms=0, color="black")

    if sc == "PSP":
        try:
            ax3 = axs[6].twinx()
            ax3.plot(final_Par["carr_lon"], linewidth=0.8, ls="-", ms=0, color="darkred")
            ax3.legend(
                ["$Carr. long ~ [^{\\circ}]$"],
                fontsize=font_size,
                frameon=False,
                bbox_to_anchor=(1.01, 0.6),
                loc=2,
            )
        except Exception:
            pass

    subalfvenic_intervals = []
    if shade_subalfvenic and isinstance(nn_df, pd.DataFrame) and ("Ma" in nn_df.columns):
        try:
            N = _to_timedelta(subalfvenic_window)
            M = _to_timedelta(subalfvenic_gap_tolerance)
            Dmin = _to_timedelta(subalfvenic_min_duration)
            min_dur_eff = max(N, Dmin)

            subalfvenic_intervals = _find_subalfvenic_intervals_fast(
                ma_series=nn_df["Ma"],
                ma_thr=subalfvenic_ma_threshold,
                min_dur=min_dur_eff,
                gap_tol=M,
            )

            for t0, t1 in subalfvenic_intervals:
                t0c = max(pd.Timestamp(t0), pd.Timestamp(start_date_lim))
                t1c = min(pd.Timestamp(t1), pd.Timestamp(end_date_lim))
                if t1c <= t0c:
                    continue
                for ax in axs:
                    ax.axvspan(t0c, t1c, alpha=subalfvenic_span_alpha, color=subalfvenic_span_color, lw=0)
        except Exception:
            subalfvenic_intervals = []

    if make_subalfvenic_scatter:
        if isinstance(nn_df, pd.DataFrame) and ("Ma" in nn_df.columns) and ("VB" in nn_df.columns):
            try:
                if len(subalfvenic_intervals) == 0:
                    N = _to_timedelta(subalfvenic_window)
                    M = _to_timedelta(subalfvenic_gap_tolerance)
                    Dmin = _to_timedelta(subalfvenic_min_duration)
                    min_dur_eff = max(N, Dmin)

                    subalfvenic_intervals = _find_subalfvenic_intervals_fast(
                        ma_series=nn_df["Ma"],
                        ma_thr=subalfvenic_ma_threshold,
                        min_dur=min_dur_eff,
                        gap_tol=M,
                    )

                if len(subalfvenic_intervals) > 0:
                    if join_path_figs:
                        base_save_path = Path(my_dir).joinpath("figures")
                    else:
                        base_save_path = Path(my_dir)

                    scatter_path = base_save_path.joinpath(subalfvenic_scatter_folder)
                    os.makedirs(str(scatter_path), exist_ok=True)

                    fig_sc = plt.figure(figsize=(9.0, 6.0))
                    ax_sc = fig_sc.add_subplot(111)

                    for k, (t0, t1) in enumerate(subalfvenic_intervals, start=1):
                        sub = nn_df.loc[t0:t1]
                        if sub is None or len(sub) == 0:
                            continue

                        m = sub["Ma"].to_numpy(dtype=float, copy=False)
                        th = sub["VB"].to_numpy(dtype=float, copy=False)
                        ok = np.isfinite(m) & np.isfinite(th) & (m < subalfvenic_ma_threshold)
                        if ok.sum() < 2:
                            continue

                        lab0 = format_timestamp(pd.Timestamp(t0), "%Y-%m-%d %H:%M")
                        lab1 = format_timestamp(pd.Timestamp(t1), "%Y-%m-%d %H:%M")
                        ax_sc.scatter(th[ok], m[ok], s=6, label=f"{lab0} → {lab1}")

                    ax_sc.set_xlabel(r"$\Theta_{VB} ~[^{\circ}]$", fontsize=14)
                    ax_sc.set_ylabel(r"$M_a$", fontsize=14)
                    ax_sc.grid(True, which="both", ls=":", lw=0.6)
                    ax_sc.legend(frameon=False, fontsize="small", loc="best")

                    scatter_name = f"scatter_Ma_vs_VB_{f1}_{f2}{tag}_{str(sc)}.png"

                    if save_fig:
                        fig_sc.savefig(
                            str(scatter_path.joinpath(scatter_name)),
                            format="png",
                            dpi=subalfvenic_scatter_dpi,
                            bbox_inches="tight",
                        )
                    fig_sc.show()
            except Exception:
                pass

    if RTN_Flag == 1:
        axs[0].legend(
            [r"$B_{r} ~ [nT]$", r"$B_{t} ~ [nT]$", r"$B_{n} ~ [nT]$", r"$|B| ~ [nT]$"],
            fontsize=font_size,
            frameon=False,
            bbox_to_anchor=(1.01, 1),
            loc=2,
        )
    else:
        axs[0].legend(
            [r"$B_{x} ~ [nT]$", r"$B_{y} ~ [nT]$", r"$B_{z} ~ [nT]$", r"$|B| ~ [nT]$"],
            fontsize=font_size,
            frameon=False,
            bbox_to_anchor=(1.01, 1),
            loc=2,
        )

    axs[1].legend(["$V_{sw} ~[km ~s^{-1}$]"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 1), loc=2)
    axs[2].legend(["$N_{p}~[(cm^{-3}$]"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 1), loc=2)
    axs[3].legend(["$\\sigma_{c}$", "$\\sigma_{r}$"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 1), loc=2)
    axs[4].legend([r"$\beta$"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 1), loc=2)
    axs[5].legend([r"$\Theta_{VB} ~[^{\circ}]$"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 1), loc=2)
    axs[6].legend([r"$R ~[R_{\odot}]$"], fontsize=font_size, frameon=False, bbox_to_anchor=(1.01, 1), loc=2)

    for i in range(numb_subplots):
        axs[i].xaxis.grid(True, "minor", linewidth=0.1, ls=":")
        axs[i].yaxis.grid(True, "major", linewidth=0.1, ls=":")
        axs[i].yaxis.grid(True, "minor", linewidth=0.1, ls=":")
        axs[i].xaxis.grid(True, "major", linewidth=0.1, ls=":")

        axs[i].tick_params(**minor_tick_params)
        axs[i].tick_params(**major_tick_params)
        axs[i].set_xlim([start_date_lim, end_date_lim])

    if join_path_figs:
        final_save_path = Path(my_dir).joinpath("figures")
    else:
        final_save_path = Path(my_dir)

    os.makedirs(str(final_save_path), exist_ok=True)

    if save_fig:
        fig.savefig(
            str(final_save_path.joinpath(figure_name)),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )

    fig.show()


    fig.show()