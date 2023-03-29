import numpy as np
import pandas as pd
from numba import jit,njit, prange
import os
import sys

os.chdir("/Users/nokni/work/MHDTurbPy/")

sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func

from scipy import constants
mu_0            = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
mu0             = constants.mu_0   #
m_p             = constants.m_p    # Proton mass [kg]
kb              = constants.k      # Boltzman's constant     [j/K]
au_to_km        = 1.496e8
T_to_Gauss      = 1e4
km2m            = 1e3
nT2T            = 1e-9
cm2m            = 1e-2    

def estimate_vec_magnitude(xvector):
    """
    Estimate the magnitude of the input vector.

    Parameters:
        xvec (numpy.array): A numpy array representing the input vector.

    Returns:
        numpy.array: A numpy array containing the magnitudes of the input vector.

    """

    return np.linalg.norm(xvector, axis=1,  keepdims=True)

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
    
    V_norm      = estimate_vec_magnitude(V).T[0]
    B_norm      = estimate_vec_magnitude(B).T[0]
    
    
    dot_product = (V * B).sum(axis=1)
 
    if restrict_2_90:
        angle       = np.arccos(np.abs(dot_product) / (V_norm * B_norm)) / np.pi * 180
    else:
        angle       = np.arccos(dot_product / (V_norm * B_norm)) / np.pi * 180       
        
    if return_denom:
        return angle, dot_product, V_norm, B_norm
    else:
        return angle



def perp_vector(a, b):
    """
    This function calculates the component of a vector perpendicular to another vector.

    Parameters:
    a (ndarray) : A 2D numpy array representing the first vector.
    b (ndarray) : A 2D numpy array representing the second vector.

    Returns:
    ndarray     : A 2D numpy array representing the component of the first input vector that is perpendicular to the second input vector.
    """
    b_unit = b / estimate_vec_magnitude(b)
    proj = (np.sum((a * b_unit), axis=1, keepdims=True))* b_unit
    perp = a - proj
    return perp




def est_alignment_angles(
                         xvec,
                         yvec,
                         return_mag_align_correl = False):
    """
    Calculate the sine of the angle between two vectors.

    Parameters:
        xvec (numpy.array): A numpy array representing the first input vector.
        yvec (numpy.array): A numpy array representing the second input vector.

    Returns:
        numpy.array: A numpy array containing the sine values of the angles between the input vectors.
    """
    
    # Estimate cross product of the two vectors
    
    numer          = np.cross(xvec, yvec)

    # Estimate magnitudes of the two vectors:
    xvec_mag       = estimate_vec_magnitude(xvec)
    yvec_mag       = estimate_vec_magnitude(yvec)

    # Estimate sigma (sigma_r for (δv, δb), sigma_c for (δzp, δz-) )
    sigma_mean           = np.nanmean((xvec_mag**2 - yvec_mag**2 )/( xvec_mag**2 + yvec_mag**2 ))
    sigma_median         = np.nanmedian((xvec_mag**2 - yvec_mag**2 )/( xvec_mag**2 + yvec_mag**2 ))   

    # Estimate denominator
    denom          = (xvec_mag*yvec_mag)
    
    # Make sure we dont have inf vals
    numer[np.isinf(numer)] = np.nan
    denom[np.isinf(denom)] = np.nan

    numer          = estimate_vec_magnitude(numer)
    denom          = np.abs(denom)


    # Estimate sine of the  two vectors
    sins              = (numer/denom)
    thetas            = np.arcsin(sins)*180/np.pi
    thetas[thetas>90] = 180 -thetas[thetas>90]
    
    # Regular alignment angle
    reg_align_angle_sin = np.nanmean(sins)
    
    # polarization intermittency angle (Beresnyak & Lazarian 2006):
    polar_int_angle = (np.nanmean(numer)/ np.nanmean(denom))   

    # Weighted angles
    weighted_sins  = np.sin(np.nansum(thetas*(denom / np.nansum(denom)))*np.pi/180)
    #weighted_sins  = np.nansum(((sins)*(denom / np.nansum(denom))))
                               
    if return_mag_align_correl== False:
        sins, xvec_mag, yvec_mag = None, None, None
                               
    return sigma_mean, sigma_median, sins, xvec_mag, yvec_mag, reg_align_angle_sin, polar_int_angle, weighted_sins




def shifted_df_calcs(B,  lag_coefs, coefs):
    """
    This function calculates the shifted dataframe.

    Parameters:
    B (pandas.DataFrame) : The input dataframe.
    lag_coefs (list)     : A list of integers representing the lags.
    coefs (list)         : A list of coefficients for the calculation.

    Returns:
    ndarray              : A 2D numpy array representing the result of the calculation.
    """
    return pd.DataFrame(np.add.reduce([x*B.shift(y) for x, y in zip(coefs, lag_coefs)]),
                        index=B.index, columns=B.columns).values


def fast_unit_vec(a):
    return a.T / estimate_vec_magnitude(a)

def mag_of_projection_ells(l_vector, B_l_vector, db_perp_vector):
    
    # estimate unit vector in parallel and displacement dir
    B_l_vector     = B_l_vector/estimate_vec_magnitude(B_l_vector)
    db_perp_vector = db_perp_vector/estimate_vec_magnitude(db_perp_vector)
    
    # estimate unit vector in pependicular by cross product
    b_perp_vector  = np.cross(B_l_vector, db_perp_vector)
    
    # Calculate dot product in-place
    l_ell     = np.abs(np.nansum(l_vector* B_l_vector, axis=1))
    l_xi      = np.abs(np.nansum(l_vector* db_perp_vector, axis=1))
    l_lambda  = np.abs(np.nansum(l_vector* b_perp_vector, axis=1))
    
    return l_ell, l_xi, l_lambda

def local_structure_function(
                             B,
                             V,
                             Np,
                             tau,
                             dt,
                             return_unit_vecs         = False,
                             five_points_sfunc        = True,
                             estimate_alignment_angle = False,
                             return_mag_align_correl  = False
                            ):
    '''
    Parameters:
    B (pandas dataframe)              : The magnetic field data
    V (pandas dataframe)              : The solar wind velocity data
    Np (pandas dataframe)             : The proton density data
    tau (int)                         : The time lag
    return_unit_vecs (bool)           : Return unit vectors if True, default is False
    five_points_sfunc (bool)          : Use five point structure function if True, default is True
    estimate_alignment_angle (bool)   : Wether to estimate the alignment angle (Using several different methods)
    
    Returns:
    dB (numpy array)                  : The fluctuation of the magnetic field
    VBangle (numpy array)             : The angle between local magnetic field and solar wind velocity
    Phiangle (numpy array)            : The angle between local solar wind velocity perpendicular to local magnetic field and the fluctuation of the magnetic field
    dB_perp_hat (numpy array)       : Unit vector of the fluctuation of the magnetic field perpendicular to the local magnetic field
    B_l_hat (numpy array)             : Unit vector of the local magnetic field
    B_perp_2_hat (numpy array)        : Unit vector perpendicular to both the fluctuation of the magnetic field and the local magnetic field
    V_l_hat (numpy array)             : Unit vector of the local velocity field
    '''
    # added five_point Structure functions
    if five_points_sfunc:
        # define coefs for loc fields
        coefs_loc     = np.array([1, 4, 6, 4, 1])/16
        lag_coefs_loc = np.array([-2*tau, -tau, 0, tau, 2*tau]).astype(int)

        # define coefs for fluctuations
        coefs_db     = np.array([1, -4, +6, -4, 1])/np.sqrt(35)
        lag_coefs_db = np.array([-2*tau, -tau, 0, tau, 2*tau]).astype(int)
        
        #Compute the fluctuation
        dB           = shifted_df_calcs(B, lag_coefs_db, coefs_db )

        #Compute the fluctuation
        du           = shifted_df_calcs(V, lag_coefs_db, coefs_db )

        # Estimate local B
        B_l          = shifted_df_calcs(B, lag_coefs_loc, coefs_loc)

        # Estimate local Vsw
        V_l          = shifted_df_calcs(V, lag_coefs_loc, coefs_loc)  
     
        # Estimate local Vsw
        N_l          = shifted_df_calcs(Np, lag_coefs_loc, coefs_loc)       

    # Estimate regular 2-point Structure functions
    else:
        #Compute the fluctuation
        dB           = (B.iloc[:-tau].values - B.iloc[tau:].values)

        #Compute the fluctuation
        du           = (V.iloc[:-tau].values - V.iloc[tau:].values)

        # Estimate local B
        B_l          = (B.iloc[:-tau].values + B.iloc[tau:].values)/2

        # Estimate local Vsw
        V_l          = (V.iloc[:-tau].values + V.iloc[tau:].values)/2

        # Estimate average of Np to avoid unphysical spikes
        N_l          = (Np.iloc[:-tau].values + Np.iloc[tau:].values)/2
    

    # Estimate local perpendicular displacement direction
    dB_perp          = perp_vector(dB, B_l)

    #Estimate l vector
    l_vec            = V_l*tau*dt


    # Estrimate l's in three directions
    l_ell, l_xi, l_lambda = mag_of_projection_ells(l_vec, B_l, dB_perp)

    #  Estimate the component l perpendicular to Blocal
    l_perp           = perp_vector(l_vec, B_l)

    # Estimate angles needed for 3D decomposition
    VBangle          = angle_between_vectors(l_vec, B_l, restrict_2_90 = True)
    Phiangle         = angle_between_vectors(l_perp, dB_perp,  restrict_2_90 = True)
    
    # Create empty dictionaries
    unit_vecs         = {}
    align_angles_vb   = {}
    align_angles_zpm  = {}                         
    
    if estimate_alignment_angle:
        
        # Constant to normalize mag field in vel units
        kinet_normal     = 1e-15 / np.sqrt(mu0 * N_l * m_p)

        # Kinetic normalization of magnetic field
        dva_perp         =  dB_perp*kinet_normal

        # We need the perpendicular component of the fluctuations
        du_perp          = perp_vector(du, B_l)
        
        # Sign of  background Br
        signB            = - np.sign(B_l.T[0])
        
        # Estimate fluctuations in Elssaser variables
        dzp_perp         = du_perp + (np.array(signB)*dva_perp.T).T
        dzm_perp         = du_perp - (np.array(signB)*dva_perp.T).T
        
        
        #Estimate magnitudes,  angles in three different ways          
        ub_results = est_alignment_angles(du_perp, 
                                          dva_perp,
                                          return_mag_align_correl = return_mag_align_correl)
    
        zpm_results = est_alignment_angles(dzp_perp, 
                                           dzm_perp,
                                           return_mag_align_correl = return_mag_align_correl)   
                               

                               
        # Assign values for va, v, z+, z-
        sigma_r_mean, sigma_r_median, sins_ub,  v_mag, va_mag, reg_align_angle_sin_ub, polar_int_angle_ub, weighted_sins_ub       = ub_results
        sigma_c_mean, sigma_c_median, sins_zpm, zp_mag, zm_mag, reg_align_angle_sin_zpm, polar_int_angle_zpm, weighted_sins_zpm   = zpm_results   
                               

        align_angles_vb      = {     
                                     'sig_r_mean'        : sigma_r_mean,
                                     'sig_r_median'      : sigma_r_median,
                                     'reg_angle'         : reg_align_angle_sin_ub,
                                     'polar_inter_angle' : polar_int_angle_ub,            
                                     'weighted_angle'    : weighted_sins_ub,
                                     'v_mag'             : v_mag,
                                     'va_mag'            : va_mag,
                                     'sins_uva'          : sins_ub,
        }
                               
        align_angles_zpm     = {     
                                     'sig_c_mean'        : sigma_c_mean,
                                     'sig_c_median'      : sigma_c_median,
                                     'reg_angle'         : reg_align_angle_sin_zpm,
                                     'polar_inter_angle' : polar_int_angle_zpm,            
                                     'weighted_angle'    : weighted_sins_zpm,
                                     'zp_mag'            : zp_mag,
                                     'zm_mag'            : zm_mag,
                                     'sins_zpm'          : sins_zpm,
        }
        
    if return_unit_vecs:

        # Estimate unit vectors
        unit_vecs     = {
                         'dB_perp_hat'   : fast_unit_vec(dB_perp), 
                         'B_l_hat'       : fast_unit_vec(fast_unit_vec(B_l)),   
                         'B_perp_2_hat'  : np.cross(B_l_hat, dB_perp_hat),
        }

    return dB, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm



@jit( parallel =True,  nopython=True)
def structure_functions_3D(
                           indices, 
                           qorder,
                           mat
                          ):
    """
    Parameters:
    indices (int array)  :  Indices of the data to be processed.
    qorder (int array)   :  Orders of the structure functions to be calculated.
    mat (2D array)       :  Data matrix with 3 columns for 3D field components.

    Returns:
    result (float array) : Structure functions estimated.
    """
    
    # initiate arrays
    result = np.zeros(len(qorder))
    

    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]

    # Estimate sfuncs
    dB = np.sqrt((ar[indices])**2 + 
                 (at[indices])**2 + 
                 (an[indices])**2)

    for i in prange(len(qorder)):   
        result[i] = np.nanmean(np.abs(dB)**qorder[i])
    return list(result)

def estimate_pdfs_3D(
                     indices,
                     mat
                    ):
    """
    A function to estimate probability density functions for 3D field.

    Parameters:
    indices (int array): Indices of the data to be processed.
    mat (2D array): Data matrix with 3 columns for 3D field components.

    Returns:
    result (dict): A dictionary containing the estimated PDFs for each component.
    """
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]

    xPDF_ar, yPDF_ar, _,_ = func.pdf(ar[indices], 45, False, True,scott_rule =False)
    xPDF_at, yPDF_at, _,_ = func.pdf(at[indices], 45, False, True,scott_rule =False)
    xPDF_an, yPDF_an, _,_ = func.pdf(at[indices], 45, False, True,scott_rule =False)

    return {'ar': [xPDF_ar, yPDF_ar], 'at': [xPDF_at, yPDF_at], 'an': [xPDF_an, yPDF_an] }

@jit(nopython=True, parallel=True)
def fast_dot_product(a,b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@jit( nopython=True)
def save_flucs(
               indices,
               mat, 
               ells
              ):
    # Define field components
    ar = mat.T[0]
    at = mat.T[1]
    an = mat.T[2]
    
    # initiate arrays
    # result = np.zeros(len(qorder))
    
    # Estimate flucs for each component
    d_Br    = ar[indices]
    d_Bt    = at[indices]
    d_Bn    = an[indices]
    ell_fin = ells[indices]
    return d_Br, d_Bt, d_Bn, ell_fin

def estimate_3D_sfuncs(
                       B,
                       V,
                       Np,
                       dt,
                       Vsw,
                       di, 
                       conditions,
                       qorder,
                       tau_values,
                       estimate_PDFS            = False,
                       return_unit_vecs         = False,
                       five_points_sfuncs       = True,
                       estimate_alignment_angle = False,
                       return_mag_align_correl  = False,
                       return_coefs             = False,
                       only_general             = False,
                       theta_thresh_gen         = 0,
                       phi_thresh_gen           = 0
                      ):
    """
    Estimate the 3D structure functions for the data given in `B` and `V`

    Parameters
    ----------
    B (pandas dataframe)              : The magnetic field data
    V (pandas dataframe)              : The solar wind velocity data
    Np (pandas dataframe)             : The proton density data
    
    dt: float
        time step
    Vsw: float
        solar wind speed
    di: float
        ion inertial length
    conditions: dict
        conditions for each structure function
    qorder: array
        order of the structure function
    tau_values: array
        time lags for the structure function
    estimate_PDFS: bool, optional
        whether to estimate the PDFs for each structure function, default False
    return_unit_vecs: bool, optional
        whether to return the unit vectors, default False
    five_points_sfuncs: bool, optional
        whether to use the 5 point stencil, default True
    return_coefs: bool, optional
        whether to return raw fluctuations or the estimated SF's, default False

    Returns
    -------
    l_di: array
        x values in di
    sf_ell_perp.T: array
        transposed sf_ell_perp
    sf_Ell_perp.T: array
        transposed sf_Ell_perp
    sf_ell_par.T: array
        transposed sf_ell_par
    sf_overall.T: array
        transposed sf_overall
    PDF_dict: dict
        a dictionary with the PDFs for each structure function and overall
    """

    #init conditions
    sf_ell_perp_conds       = conditions['ell_perp']
    sf_Ell_perp_conds       = conditions['Ell_perp']
    sf_ell_par_conds        = conditions['ell_par']
    sf_ell_par_rest_conds   = conditions['ell_par_rest']

    # Initialize arrays
    sf_ell_perp             = np.zeros(( len(tau_values), len(qorder)))
    sf_Ell_perp             = np.zeros(( len(tau_values), len(qorder)))
    sf_ell_par              = np.zeros(( len(tau_values), len(qorder)))
    sf_ell_par_rest         = np.zeros(( len(tau_values), len(qorder)))
    sf_overall              = np.zeros(( len(tau_values), len(qorder)))
       
    # Initialize dictionaries    
    thetas                  = {}
    phis                    = {}
    ub_polar                = []
    ub_reg                  = []
    ub_weighted             = []
    zpm_polar               = []
    zpm_reg                 = []
    zpm_weighted            = []
    u_norms                 = {}
    b_norms                 = {}
    sig_c_mean              = []
    sig_r_mean              = []
    sig_c_median            = []
    sig_r_median            = []

    if return_coefs:
        dBell_perpR = {};   dBEll_perpR = {};  dBell_parR = {}; dBell_par_restR = {}; dB_all_R = {};  
        dBell_perpT = {};   dBEll_perpT = {};  dBell_parT = {}; dBell_par_restT = {}; dB_all_T = {};
        dBell_perpN = {};   dBEll_perpN = {};  dBell_parN = {}; dBell_par_restN = {}; dB_all_N = {};
        lambdas     = {};   xis         = {};  ells       = {}; ells_rest       = {};

    # Run main loop
    for jj, tau_value in enumerate(tau_values):
                               
                               
        # Main function                     
        dB, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm = local_structure_function(
                                                                                                         B,
                                                                                                         V,
                                                                                                         Np,
                                                                                                         int(tau_value),
                                                                                                         dt            ,
                                                                                                         return_unit_vecs         = return_unit_vecs,
                                                                                                         five_points_sfunc        = five_points_sfuncs,
                                                                                                         estimate_alignment_angle = estimate_alignment_angle,
                                                                                                         return_mag_align_correl  = return_mag_align_correl
                                                                                                        )
                               
        # Now store the angles estimated in previous function
        thetas[str(jj)] = VBangle
        phis[str(jj)]   = Phiangle   
                             
        if estimate_alignment_angle:
            # Va, v average angles
            ub_polar.append(align_angles_vb['polar_inter_angle'])
            ub_reg.append(align_angles_vb['reg_angle'])
            ub_weighted.append(align_angles_vb['weighted_angle'])
            sig_r_mean.append(align_angles_vb['sig_r_mean'])
            sig_r_median.append(align_angles_vb['sig_r_median'])            

            # Zp, Zm average angles
            zpm_polar.append(align_angles_zpm['polar_inter_angle'])
            zpm_reg.append(align_angles_zpm['reg_angle'])
            zpm_weighted.append(align_angles_zpm['weighted_angle'])
            sig_c_mean.append(align_angles_zpm['sig_c_mean'])
            sig_c_median.append(align_angles_zpm['sig_c_median'])            

        
        if only_general        == False:

            # for sf_ell_perp

            indices                      = np.where((VBangle>sf_ell_perp_conds['theta']) & (Phiangle>sf_ell_perp_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_lambda_fin = save_flucs(indices, dB, l_lambda)

                dBell_perpR[str(jj)]     =  d_Br
                dBell_perpT[str(jj)]     =  d_Bt
                dBell_perpN[str(jj)]     =  d_Bn
                lambdas[str(jj)]         =  l_lambda_fin

            else:
                sf_ell_perp[jj, :]       = structure_functions_3D(indices, qorder, dB)

            if estimate_PDFS:
                PDF_ell_perp             = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_perp = None

            # for sf_Ell_perp
            indices                      = np.where((VBangle>sf_Ell_perp_conds['theta']) & (Phiangle<sf_Ell_perp_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_xi_fin         = save_flucs(indices, dB, l_xi)

                dBEll_perpR[str(jj)]     =  d_Br
                dBEll_perpT[str(jj)]     =  d_Bt
                dBEll_perpN[str(jj)]     =  d_Bn
                xis[str(jj)]             =  l_xi_fin
            else:
                sf_Ell_perp[jj, :]       = structure_functions_3D(indices, qorder, dB)

            if estimate_PDFS:        
                PDF_Ell_perp             = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_Ell_perp = None

            # for sf_ell_par
            indices                      = np.where((VBangle<sf_ell_par_conds['theta']) & (Phiangle<sf_ell_par_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_ell_fin = save_flucs(indices, dB, l_ell)

                dBell_parR[str(jj)]      =  d_Br
                dBell_parT[str(jj)]      =  d_Bt
                dBell_parN[str(jj)]      =  d_Bn
                ells[str(jj)]            =  l_ell_fin
            else:
                sf_ell_par[jj, :]        = structure_functions_3D(indices, qorder, dB)    

            if estimate_PDFS:
                PDF_ell_par              = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_par = None

            # for sf_ell_par_restricted
            indices                      = np.where((VBangle<sf_ell_par_rest_conds['theta']) & (Phiangle<sf_ell_par_rest_conds['phi']))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn, l_ell_fin_rest    = save_flucs(indices, dB, l_ell)

                dBell_par_restR[str(jj)] =  d_Br
                dBell_par_restT[str(jj)] =  d_Bt
                dBell_par_restN[str(jj)] =  d_Bn
                ells_rest[str(jj)]       =  l_ell_fin_rest
            else:
                sf_ell_par_rest[jj, :]   = structure_functions_3D(indices, qorder, dB)    

            if estimate_PDFS:
                PDF_ell_par_rest         = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_par_rest = None 
        else:

            # for sf general
            indices                      = np.where((VBangle>theta_thresh_gen) & (Phiangle>phi_thresh_gen))[0]

            if return_coefs:
                d_Br, d_Bt, d_Bn         = save_flucs(indices, dB)

                dB_all_R[str(jj)]        =  d_Br
                dB_all_T[str(jj)]        =  d_Bt
                dB_all_N[str(jj)]        =  d_Bn
            else:
                sf_overall[jj, :]        = structure_functions_3D(indices, qorder, dB) 

            if estimate_PDFS:
                PDF_all                  = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_all = None    

    # Also estimate x values in di
    l_di    = (tau_values*dt*Vsw)/di
    
    # Return fluctuations
    if  return_coefs:
        if only_general:
            flucts = {
                       'ell_all'          :  {'R': dB_all_R,         'T' : dB_all_T,          'N': dB_all_N    },               
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw,
                       'di'                :  di,
                       'dt'                :  dt
                     }            
        else:
            flucts = {
                       'ell_perp'          :  {'R': dBell_perpR,      'T' : dBell_perpT,       'N': dBell_perpN,     'lambdas'  : lambdas  },
                       'Ell_perp'          :  {'R': dBEll_perpR,      'T' : dBEll_perpT,       'N': dBEll_perpN,     'xis'      : xis      },
                       'ell_par'           :  {'R': dBell_parR,       'T' : dBell_parT,        'N': dBell_parN ,     'ells'     : ells     },
                       'ell_par_rest'      :  {'R': dBell_par_restR,  'T' : dBell_par_restT,   'N': dBell_par_restN, 'ells_rest': ells_rest},          
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw,
                       'di'                :  di,
                       'dt'                :  dt
                     }
    else:
        flucts = None
    
    Sfunctions     = {
                      'ell_perp'     : sf_ell_perp.T,
                      'Ell_perp'     : sf_Ell_perp.T,
                      'ell_par'      : sf_ell_par.T,
                      'ell_par_rest' : sf_ell_par_rest.T,
                      'ell_overall'  : sf_overall.T
    }
    

    try:
        PDFs            =  {
                        'All'          : PDF_all,
                        'ell_par'      : PDF_ell_par,
                        'ell_par_rest' : PDF_ell_par_rest,
                        'ell_perp'     : PDF_Ell_perp,
                        'ell_perp'     : PDF_ell_perp
       }
    except:
        PDFs            = None   
    if estimate_alignment_angle:
        overall_align_angles ={
                                'VB' :  {'reg': ub_reg, 'polar':  ub_polar, 'weighted': ub_weighted, 'sig_r_mean': sig_r_mean, 'sig_r_median': sig_r_median},
                                'Zpm':  {'reg': zpm_reg, 'polar': zpm_polar, 'weighted': zpm_weighted,'sig_c_mean': sig_c_mean, 'sig_c_median': sig_c_median}            
                              }
    else:
        overall_align_angles = None
                            
    
    return thetas, phis,  flucts, l_di, Sfunctions, PDFs, overall_align_angles
                

def find_closest_values_in_arrays(arr_list, L_list, limited_window=False, xlims=None):
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