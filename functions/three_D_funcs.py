import numpy as np
import pandas as pd
from numba import jit,njit, prange
import os
import sys
from joblib import Parallel, delayed
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func
import TurbPy as turb
import traceback

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



def est_alignment_angles(
                         xvec,
                         yvec,
                         return_mag_align_correl = False,
                         est_sigma_c             = False):
    """
    Calculate the sine of the angle between two vectors.

    Parameters:
        xvec (numpy.array): A numpy array representing the first input vector.
        yvec (numpy.array): A numpy array representing the second input vector.

    Returns:
        numpy.array: A numpy array containing the sine values of the angles between the input vectors.
    """
    
    # Use the cross product and calculate the norm for sine calculation along the last axis
    numer        = np.sqrt(np.nansum(np.cross(xvec, yvec, axis=1)**2, axis=1))
    
    # Calculate dot product for cosine calculation along the last axis (axis=1 for 2D arrays)
    numer_cos = np.nansum(xvec * yvec, axis=1)
    
    
    # Calculate the magnitude of vectors directly on the original vectors
    xvec_mag = func.estimate_vec_magnitude(xvec)
    yvec_mag = func.estimate_vec_magnitude(yvec)


    # Estimate sigma (sigma_r for (δv, δb), sigma_c for (δzp, δz-) )
    sigma_ts             = (xvec_mag**2 - yvec_mag**2 )/( xvec_mag**2 + yvec_mag**2 )
    
    if est_sigma_c:
        sigma_mean           = (np.nanmean(xvec_mag**2 - yvec_mag**2))/(np.nanmean(xvec_mag**2 + yvec_mag**2))        
        
    else:
        sigma_mean           = (np.nanmean(xvec_mag**2) - np.nanmean(yvec_mag**2))/(np.nanmean(xvec_mag**2) + np.nanmean(yvec_mag**2))
    sigma_median         = np.nan # We don't need it!  
    
    # Estimate denominator
    denom          = (xvec_mag*yvec_mag)
    
    # Make sure we dont have inf vals
    numer[np.isinf(numer)] = np.nan
    denom[np.isinf(denom)] = np.nan


    # Estimate Counts!
    counts             = len(numer[numer>-1e10])


    # Regular alignment angle
    reg_align_angle_sin = np.nanmean(numer/denom)
    
    # polarization intermittency angle (Beresnyak & Lazarian 2006):
    polar_int_angle_sin     = np.nanmean(numer)/ np.nanmean(denom)  

    # Weighted angles
    weighted_sins       = np.nan # We don't need it either
                      
    if return_mag_align_correl== False:
        sins, xvec_mag, yvec_mag = None, None, None
                               
    return counts, sigma_ts, sigma_mean, sigma_median, numer, numer_cos,  denom, xvec_mag, yvec_mag, reg_align_angle_sin, polar_int_angle_sin, weighted_sins


def fast_unit_vec(a):
    return a.T / func.estimate_vec_magnitude(a)



def mag_of_ell_projections_and_angles(
                                        l_vector,
                                        B_l_vector, 
                                        db_perp_vector,
                                        est_proj_ells  =  True
                                    ):
    
    try:
        # estimate unit vector in parallel and displacement dir
        B_l_vector     = (B_l_vector.T/func.estimate_vec_magnitude(B_l_vector)).T
        db_perp_vector = (db_perp_vector.T/func.estimate_vec_magnitude(db_perp_vector)).T

        if est_proj_ells:
            # estimate unit vector in pependicular by cross product
            b_perp_vector  = np.cross(B_l_vector, db_perp_vector)

            # Calculate dot product in-place
            l_ell    = np.abs(func.dot_product(l_vector, B_l_vector))
            l_xi     = np.abs(func.dot_product(l_vector, db_perp_vector))
            l_lambda = np.abs(func.dot_product(l_vector, b_perp_vector))
            
           # print('Shapes:', np.shape(l_ell), np.shape(l_xi), np.shape(l_lambda))
            
        else:
            l_ell, l_xi, l_lambda = np.nan, np.nan,np.nan
            
        #  Estimate the component l perpendicular to Blocal
        l_perp         = func.perp_vector(l_vector, B_l_vector)

        # Estimate angles needed for 3D decomposition
        VBangle        = func.angle_between_vectors(l_vector, B_l_vector, restrict_2_90 = True)
        Phiangle       = func.angle_between_vectors(l_perp, db_perp_vector,  restrict_2_90 = True)

    except:
        traceback.print_exc()

    return l_ell, l_xi, l_lambda, VBangle, Phiangle


def local_structure_function(
                             B,
                             V,
                             V_sc,
                             Np,
                             tau,
                             dt,
                             return_unit_vecs         = False,
                             five_points_sfunc        = True,
                             estimate_alignment_angle = False,
                             return_mag_align_correl  = False,
                             fix_sign                 = True, 
                             return_B_in_vel_units    = False,
                             turb_amp_analysis        = False,
                             also_return_db_nT        = False,
                             use_local_polarity       = True,
                             use_np_factor            = 1,
                             est_proj_ells            =  True
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

    # Constant to normalize mag field in vel units
    kinet_normal = (1e-15 / np.sqrt(use_np_factor * mu0 * Np.rolling('1min', center=True).mean() * m_p)).values

    # Directly multiply each column in B by kinet_normal, broadcasting the operation
    Va = B.multiply(kinet_normal, axis=0).interpolate()

    # Keep flag 
    normal_flag = 'B_in_vel_units' if return_B_in_vel_units else'B_in_nT_units'

    if five_points_sfunc:
        
        # define coefs for loc fields
        coefs_loc     = np.array([1, 4, 6, 4, 1])/16
        lag_coefs_loc = np.array([-2*tau, -tau, 0, tau, 2*tau]).astype(int)

        # define coefs for fluctuations
        coefs_db     = np.array([1, -4, +6, -4, 1])/np.sqrt(35)
        lag_coefs_db = np.array([-2*tau, -tau, 0, tau, 2*tau]).astype(int)
        
        #Compute the fluctuations in B
        dB           = turb.shifted_df_calcs(B, lag_coefs_db, coefs_db, return_df = True)
        
        #Compute the fluctuations in Va
        dVa          = (turb.shifted_df_calcs(Va, lag_coefs_db, coefs_db, return_df = True)).values

        needed_index = dB.index
        dB           = dB.values

        #Compute the fluctuations in V
        du           = turb.shifted_df_calcs(V - V_sc.values, lag_coefs_db, coefs_db )
        
        #Compute the fluctuations in Np
        dN           = turb.shifted_df_calcs(Np, lag_coefs_db, coefs_db )

        # Estimate local B
        B_l          = turb.shifted_df_calcs(B, lag_coefs_loc, coefs_loc)

        # Estimate local Vsw
        #V_l          = turb.shifted_df_calcs(V func.perp_vector(dB, B_l, return_paral_comp = True- V_sc.values +  Va.values, lag_coefs_loc, coefs_loc)
        V_l          = turb.shifted_df_calcs(V - V_sc.values, lag_coefs_loc, coefs_loc) + turb.shifted_df_calcs( Va, lag_coefs_loc, coefs_loc)
        
        # Est d
        di_array     = turb.shifted_df_calcs(228 / np.sqrt(Np), lag_coefs_loc, coefs_loc)


    # Estimate regular 2-point Structure functions
    else:
        #Compute the fluctuations in B
        dB           = B.iloc[:-tau].values - B.iloc[tau:].values
        
        #Compute the fluctuations in Va
        dVa          = Va.iloc[:-tau].values - Va.iloc[tau:].values

        #Compute the fluctuations in V
        du           = (V - V_sc.values).iloc[:-tau].values - (V - V_sc.values).iloc[tau:].values
        
        #Compute the fluctuations in Np
        dN           = Np.iloc[:-tau].values - Np.iloc[tau:].values
        
        # Estimate local B
        B_l          = (B.iloc[:-tau].values + B.iloc[tau:].values)/2

        # Estimate local Vsw
        V_l          = ((V - V_sc.values +  Va.values).iloc[:-tau].values + (V - V_sc.values +  Va.values).iloc[tau:].values)/2

        # Est d
        quant        = 228 / np.sqrt(Np)
        di_array     = (quant.iloc[:-tau].values + quant.iloc[tau:].values)/2


        #print('bl', np.shape(B_l)), print('dB', np.shape(dB))
        # Keep df index
        needed_index = B.iloc[:-tau,:].index
                

    # Estimate local perpendicular displacement direction
    dB_perp, dB_parallel     = func.perp_vector(dB, B_l, return_paral_comp = True)
    dVa_perp, dVa_parallel   = func.perp_vector(dVa, B_l, return_paral_comp = True)
    
    #Estimate l vector
    l_vec                    = V_l*tau*dt
    
    #Estimate Vsw
    Vsw                      = np.nanmean(np.sqrt(V_l.T[0]**2 + V_l.T[1]**2 + V_l.T[2]**2) )

    # Estrimate l's in three directions
    
    l_ell, l_xi, l_lambda,  VBangle, Phiangle = mag_of_ell_projections_and_angles(l_vec,
                                                                                  B_l,
                                                                                  dB_perp,
                                                                                  est_proj_ells  =  est_proj_ells)
    
    #print(type(l_xi), np.shape(l_xi), type(di_array), np.shape(di_array) )
    #di_array_flat =   # Flatten di_array to (367660,)
    l_ell, l_xi, l_lambda = l_ell / di_array.ravel(), l_xi / di_array.ravel(), l_lambda / di_array.ravel()
    lmag                  = func.estimate_vec_magnitude(l_vec)/ di_array.ravel()

    #l_ell, l_xi, l_lambda = l_ell / di_array[:, None], l_xi / di_array[:, None], l_lambda / di_array[:, None]
    print('Got here 2')
    
    # Estimate magntidtues of the par and perp components of increments
    dB_perp_amp        = np.sqrt(dB_perp.T[0]**2 + dB_perp.T[1]**2 + dB_perp.T[2]**2)
    dB_parallel_amp    = np.sqrt(dB_parallel.T[0]**2 + dB_parallel.T[1]**2 + dB_parallel.T[2]**2)
    
    # Create empty dictionaries
    unit_vecs         = {}
    align_angles_vb   = {}
    align_angles_zpm  = {}                         
    
    if estimate_alignment_angle:

        # We need the perpendicular component of the fluctuations
        du_perp = func.perp_vector(du, B_l)

        # Determine the sign of background Br
        polarity        = np.sign(func.newindex(B['Br'].rolling('30min', center=True).mean().interpolate(), needed_index).values)
        signBx          = - polarity if fix_sign else polarity
        local_polarity  = - np.sign(B_l.T[0]) if fix_sign else np.sign(B_l.T[0])
        
        # Estimate fluctuations in Elssaser variables
        
        if use_local_polarity:
            dzp_perp = du_perp + signBx[:, None] * dVa_perp
            dzm_perp = du_perp - signBx[:, None] * dVa_perp
        else:
            dzp_perp = du_perp + local_polarity[:, None] * dVa_perp
            dzm_perp = du_perp - local_polarity[:, None] * dVa_perp           

        if turb_amp_analysis:
            try:
                dB_mod           = turb.shifted_df_calcs(pd.DataFrame(np.sqrt(B.Br**2 + B.Bt**2 + B.Bn**2)), lag_coefs_db, coefs_db)

                keep_turb_amp = {
                                 'dva_perp'          : dVa_perp,
                                 'du_perp'           : du_perp,
                                 'dzp_perp'          : dzp_perp,
                                 'dzm_perp'          : dzm_perp,
                                 'dB_nT'             : dB,
                                 'dB_mod_nT'         : dB_mod,
                                 'dB_perp_amp_nT'    : dB_perp_amp,
                                 'dB_parallel_amp_nT': dB_parallel_amp,
                                 'B_l'               : B_l,

                                }

            except:
                traceback.print_exc()
        else:
            keep_turb_amp = {None}            
        
        #Estimate magnitudes,  angles in three different ways          
        ub_results = est_alignment_angles(du_perp, 
                                          dVa_perp,
                                          return_mag_align_correl = return_mag_align_correl)
    
        zpm_results = est_alignment_angles(dzp_perp, 
                                           dzm_perp,
                                           return_mag_align_correl = return_mag_align_correl,
                                           est_sigma_c             = True)   
                               
        # Assign values for va, v, z+, z-
        countsvb, sigma_r_ts,  sigma_r_mean, sigma_r_median, sins_ub_num, cos_ub_num, sins_ub_den,  v_mag, va_mag, reg_align_angle_sin_ub, polar_int_angle_sin_ub, weighted_sins_ub      = ub_results
        
        countszp, sigma_c_ts,  sigma_c_mean, sigma_c_median, sins_zp_num, cos_zp_num, sins_zp_den, zp_mag, zm_mag, reg_align_angle_sin_zpm, polar_int_angle_sin_zpm, weighted_sins_zpm   = zpm_results  

        
        align_angles_vb      = {     
                                     'sig_r_ts'          : sigma_r_ts,
                                     'sig_r_mean'        : sigma_r_mean,
                                     'sig_r_median'      : sigma_r_median,
                                     'reg_angle'         : reg_align_angle_sin_ub,
                                     'polar_inter_angle' : polar_int_angle_sin_ub,            
                                     'weighted_angle'    : weighted_sins_ub,
                                     'v_mag'             : v_mag,
                                     'va_mag'            : va_mag,
                                     'sins_ub_num'       : sins_ub_num,
                                     'cos_ub_num'        : cos_ub_num,
                                     'sins_ub_den'       : sins_ub_den,
                                     'counts'            : countsvb
        }
                               
        align_angles_zpm     = {     
                                     'sig_c_ts'          : sigma_c_ts,
                                     'sig_c_mean'        : sigma_c_mean,
                                     'sig_c_median'      : sigma_c_median,
                                     'reg_angle'         : reg_align_angle_sin_zpm,
                                     'polar_inter_angle' : polar_int_angle_sin_zpm,            
                                     'weighted_angle'    : weighted_sins_zpm,
                                     'zp_mag'            : zp_mag,
                                     'zm_mag'            : zm_mag,
                                     'sins_zp_num'       : sins_zp_num,                                    
                                     'cos_zp_num'        : cos_zp_num,
                                     'sins_zp_den'       : sins_zp_den,                                     
                                     'counts'            : countszp
        }
        
    if return_unit_vecs:

        # Estimate unit vectors
        unit_vecs     = {
                         'dB_perp_hat'   : fast_unit_vec(dB_perp), 
                         'B_l_hat'       : fast_unit_vec(fast_unit_vec(B_l)),   
                         'B_perp_2_hat'  : np.cross(B_l_hat, dB_perp_hat),
        }

    if return_B_in_vel_units:
        dVa_perp_amp = np.sqrt(dVa_perp.T[0]**2 + dVa_perp.T[1]**2 + dVa_perp.T[2]**2)
        dVa_par_amp  = np.sqrt(dVa_parallel.T[0]**2 + dVa_parallel.T[1]**2 + dVa_parallel.T[2]**2)
        
        return Vsw, B_l, keep_turb_amp, dVa, dVa_perp_amp, dVa_par_amp, du, dN, kinet_normal, signBx,  normal_flag,  lmag, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb,align_angles_zpm, needed_index, local_polarity
        
        
    else:
        return Vsw, B_l, keep_turb_amp, dB, dB_perp_amp, dB_parallel_amp, du, dN, kinet_normal, signBx,normal_flag, lmag, l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm, needed_index, local_polarity



@jit( parallel =True,  nopython=True)
def structure_functions_3D(
                           indices, 
                           qorder,
                           mat,
                           max_std=12
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
    ar = np.abs(mat.T[0][indices])
    at = np.abs(mat.T[1][indices])
    an = np.abs(mat.T[2][indices])
    
    # Estimate Standard deviation!
    std_r = np.nanstd(ar)
    std_t = np.nanstd(at)
    std_n = np.nanstd(an)
    
    # Remove very extreme events
    index = (ar<max_std*std_r) & (at<max_std*std_t) & (an<max_std*std_n)
    ar    = ar[index]
    at    = at[index]
    an    = an[index]
    
    dbtot = np.sqrt(ar**2 + at**2 +an**2)
    for i in prange(len(qorder)):   
        result[i] = np.nanmean((dbtot)**qorder[i])

    sdk   = result[3] /result[1]**2
    return list(result), sdk

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

    
def vars_2_estimate(ts_list=None):
    
    # Load default variables
    default_vars = ['R', 'T', 'N']
    return default_vars if ts_list is None else ts_list+ default_vars



def quants_2_estimate(
                    B_l,
                    local_polarity,
                    dB,
                    dB_perp,
                    dB_parallel,
                    dV,
                    dzp,
                    dzm,
                    dN,
                    Np, 
                    keep_turb_amp,
                    kinet_normal,
                    sign_Bx,
                    B,
                    V,
                    phis,
                    thetas,
                    align_angles_zpm,
                    align_angles_vb,
                    tau_value,
                    needed_index,
                    di,
                    Vsw,
                    five_points_sfunc = True,
                    av_hours          = None,
                    ts_list           = None):

    if av_hours==None:
        av_hours=1/60

    # What to estimate
    quants = vars_2_estimate(ts_list=ts_list)

    #Initialize variables dict
    variables = {}
    
    if 'db_perp_amp' in quants:
        variables['db_perp_amp'] = dB_perp
        
    if 'db_par_amp' in quants:
        variables['db_par_amp']  = dB_parallel
               
    
    if 'PVI_vec_zp' in quants:
    
        dzp        = pd.DataFrame({'DateTime': needed_index,
                                  'Zpr'      : dzp.T[0],
                                  'Zpt'      : dzp.T[1],
                                  'Zpn'      : dzp.T[2]}).set_index('DateTime')

        # Estimate PVI of \vec{Zp} 
        variables['PVI_vec_zp'] = func.newindex(turb.estimate_PVI( 
                                                                 dzp.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 keys              = ['Zpr', 'Zpt', 'Zpn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1,                 
                                                                 input_flucts      = True,
                                                                 dbs               = dzp), needed_index).values.T[0]
    if 'PVI_vec_zm' in quants:
    
        dzm        = pd.DataFrame({'DateTime': needed_index,
                                  'Zmr'      : dzm.T[0],
                                  'Zmt'      : dzm.T[1],
                                  'Zmn'      : dzm.T[2]}).set_index('DateTime')

        # Estimate PVI of \vec{Zp} 
        variables['PVI_vec_zm'] = func.newindex(turb.estimate_PVI( 
                                                                 dzm.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 keys              = ['Zmr', 'Zmt', 'Zmn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1,                 
                                                                 input_flucts      = True,
                                                                 dbs               = dzm), needed_index).values.T[0]
        
    if 'PVI_vec' in quants:
        
            
        dbb        = pd.DataFrame({'DateTime': needed_index,
                                  'Br'      : dB.T[0],
                                  'Bt'      : dB.T[1],
                                  'Bn'      : dB.T[2]}).set_index('DateTime')
        
        
        
        variables['PVI_vec'] = func.newindex(turb.estimate_PVI( 
                                                                 dbb.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 keys              = ['Br', 'Bt', 'Bn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1,                 
                                                                 input_flucts      = True,
                                                                 dbs               = dbb), needed_index).values.T[0]

    if 'PVI_vec_V' in quants:
          
        dvv       = pd.DataFrame({'DateTime': needed_index,
                                  'Vr'      : dV.T[0],
                                  'Vt'      : dV.T[1],
                                  'Vn'      : dV.T[2]}).set_index('DateTime')
        
        
        
        variables['PVI_vec_V'] = func.newindex(turb.estimate_PVI( 
                                                                 dvv.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 keys              = ['Vr', 'Vt', 'Vn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1,                 
                                                                 input_flucts      = True,
                                                                 dbs               = dvv), needed_index).values.T[0]

    if 'PVI_Np' in quants:
        
        
        # Estimate PVI of \vec{Zp} 
        variables['PVI_Np'] = func.newindex(turb.estimate_PVI( 
                                                                 Np.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 keys              = list(Np.keys()),
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'mod',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1), needed_index).values.T[0]
    if 'R' in quants:
        variables['R']             =  dB.T[0]  
        
    if 'T' in quants:
        variables['T']             =  dB.T[1] 
        
    if 'N' in quants:
        variables['N']             =  dB.T[2] 
        
        del dB
        
    if 'B_l_R' in quants:
        variables['B_l_R']             =  B_l.T[0]  
        
    if 'B_l_T' in quants:
        variables['B_l_T']             =  B_l.T[1] 
        
    if 'B_l_N' in quants:
        variables['B_l_N']             =  B_l.T[2]
        
        del B_l
        
    if 'V_R' in quants:
        variables['V_R']             =  dV.T[0]  
        
    if 'V_T' in quants:
        variables['V_T']             =  dV.T[1] 
        
    if 'V_N' in quants:
        variables['V_N']             =  dV.T[2] 
        
        del dV
         

    if 'sign_Bx' in quants:
        variables['sign_Bx']         =  sign_Bx 
        
        del sign_Bx 
        
    if 'local_polarity' in quants:
        variables['local_polarity']  =  local_polarity
        
        del local_polarity 
        
    if 'N_p' in quants:
        
        variables['N_p']             =  dN.T[0]
        
        
    if 'db_index' in quants:
        variables['db_index']       = needed_index
        
        
    if 'kinet_normal' in quants:
        
        variables['kinet_normal']   =  kinet_normal
        
        del kinet_normal
        
    if 'phis' in quants:
        variables['phis']          =  phis
        del phis
        
    if 'thetas' in quants:
        variables['thetas']        =  thetas   
        del thetas
        
    if 'Vsw' in quants:
        variables['Vsw']           =  func.newindex(np.sqrt(V.Vr**2 + V.Vt**2 + V.Vn**2), needed_index).values

    if 'Bmod' in quants:
        variables['Bmod']          =  func.newindex(np.sqrt(B.Br**2 + B.Bt**2 + B.Bn**2), needed_index).values  
        
    if 'VBangle_big' in quants:
        variables['VBangle_big']   =  func.newindex(pd.DataFrame({'DateTime':B.index,
                                                                  'values'  :func.angle_between_vectors(B.values,
                                                                                                        V.values)}).set_index('DateTime'), needed_index).values.T[0]
    if 'sig_c' in quants:
        variables['sig_c']         = align_angles_zpm['sig_c_ts']

    if 'sig_r' in quants:
        variables['sig_r']         = align_angles_vb['sig_r_ts'] 

    if 'sins_ub_num' in quants: 
        variables['sins_ub_num']   = align_angles_vb['sins_ub_num'] 
        
    if 'cos_ub_num' in quants: 
        variables['cos_ub_num']   = align_angles_vb['cos_ub_num'] 
        
    if 'sins_ub_den' in quants:
        variables['sins_ub_den']   = align_angles_vb['sins_ub_den'] 

    if 'sins_zp_num' in quants:
        variables['sins_zp_num']  = align_angles_zpm['sins_zp_num']
        
    if 'cos_zp_num' in quants: 
        variables['cos_zp_num']   = align_angles_zpm['cos_zp_num'] 
        
    if 'sins_zp_den' in quants:
        variables['sins_zp_den']   = align_angles_zpm['sins_zp_den']

        
    if 'sins_zp' in quants:
        variables['sins_zp']       = align_angles_zpm['sins_zp_num']/align_angles_zpm['sins_zp_den']
        
    if 'sins_ub' in quants:
        variables['sins_ub']        = align_angles_vb['sins_ub_num']/ align_angles_vb['sins_ub_den']

    if 'zp_mag' in quants:
        variables['zp_mag']        = align_angles_zpm['zp_mag'] 

    if 'zm_mag' in quants:
        variables['zm_mag']        = align_angles_zpm['zm_mag']  

    if 'compress_squire' in quants:
        variables['compress_squire'] = func.newindex(turb.compressibility_complex_squire( 
                                                                                       tau_value,
                                                                                       B.copy(),
                                                                                       av_hours =av_hours),needed_index).values.T[0]
    if 'compress_squire_V' in quants:
        variables['compress_squire_V'] = func.newindex(turb.compressibility_complex_squire( 
                                                                                       tau_value,
                                                                                       V.copy(),
                                                                                       keys     = ['Vr', 'Vt', 'Vn'],
                                                                                       av_hours =av_hours),needed_index).values.T[0]
    if 'compress_chen' in quants:
        variables['compress_chen']   = func.newindex(turb.compressibility_complex_chen( 
                                                                                       tau_value,
                                                                                       B.copy(),
                                                                                       av_hours =av_hours),needed_index).values.T[0]
        
    if 'compress_chen_V' in quants:
        variables['compress_chen_V']   = func.newindex(turb.compressibility_complex_chen( 
                                                                                       tau_value,
                                                                                       V.copy(),
                                                                                       keys     = ['Vr', 'Vt', 'Vn'],
                                                                                       av_hours =av_hours),needed_index).values.T[0]
    if 'compress_simple' in quants:

        variables['compress_simple'] = func.newindex(turb. calculate_compressibility( 
                                                                                   tau_value,
                                                                                   B.copy(),
                                                                                   five_points_sfunc=five_points_sfunc),needed_index).values.T[0]
    if 'compress_simple_V' in quants:

        variables['compress_simple_V'] = func.newindex(turb. calculate_compressibility( 
                                                                                   tau_value,
                                                                                   V.copy(),
                                                                                   keys     = ['Vr', 'Vt', 'Vn'],
                                                                                   five_points_sfunc=five_points_sfunc),needed_index).values.T[0]
    if 'variance' in quants:
        # Estimate expansion factor
        variables['variance']     = func.newindex(turb.variance_anisotropy_verdini(
                                                                                   tau_value,
                                                                                   B.copy(),
                                                                                   av_hours =av_hours), needed_index).values
        
    if 'norm_turb_amplitude' in quants:
        # Estimate expansion factor
        variables['norm_turb_amplitude']= func.newindex(turb.norm_fluct_amplitude(
                                                                                   tau_value,
                                                                                   B.copy(),
                                                                                   av_hours =av_hours,
                                                                                   denom_av_hours ='4H'), needed_index).values.T[0]  

    if 'PVI_mod' in quants:
        # Estimate PVI of |B|
        variables['PVI_mod'] = func.newindex(turb.estimate_PVI(
                                                                 B.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'mod',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs=-1), needed_index).values.T[0]
        


        
    if 'PVI_mod_V' in quants:
        # Estimate PVI of \vec{B} 
        variables['PVI_mod_V'] = func.newindex(turb.estimate_PVI( 
                                                                 V.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 1,
                                                                 keys              = ['Vr', 'Vt', 'Vn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'mod',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs=-1), needed_index).values.T[0]        

        
    return variables


def save_flucs(indices,
               final_variables,
               ells,
               ell_identifier
              ):
    
    # Extract keys from final_variables
    var_keys = list(final_variables.keys())
    
    if len(indices)>0:
        # Initialize selected_points dictionary using dictionary comprehension
        selected_points                    = {var_key: final_variables[var_key][indices] for var_key in var_keys}
    
        #Also select ell values!
        selected_points[ell_identifier]    =  ells[indices]
    else:
        # Initialize selected_points dictionary using dictionary comprehension
        selected_points                    = {var_key: [np.nan] for var_key in var_keys}
    
        #Also select ell values!
        selected_points[ell_identifier]    =[np.nan]   

    return selected_points


def init_dicts_for_SFs(step,
                       tau_values,
                       qorder,
                       create_dicts=True):
    
    thetas_bin_array = np.arange(0, 91, step)
    phis_bin_array = np.arange(0, 91, step)

    if create_dicts:
        
        sf_data_V = {
            f'th_{prev_theta}_{theta}_ph_{prev_phi}_{phi}': {
                                                             'xvals'     : [], 
                                                             'yvals'     : np.nan*np.ones((len(tau_values), len(qorder))), 
                                                             'counts'    : [],
                                                             'identifier': []
            }
            for prev_theta, theta in zip(thetas_bin_array, thetas_bin_array[1:])
            for prev_phi, phi in zip(phis_bin_array, phis_bin_array[1:])
        }

        sf_data_B = {
            f'th_{prev_theta}_{theta}_ph_{prev_phi}_{phi}': {
                                                             'xvals'     : [], 
                                                             'yvals'     : np.nan*np.ones((len(tau_values), len(qorder))), 
                                                             'counts'    : [],
                                                             'identifier': []
            }
            for prev_theta, theta in zip(thetas_bin_array, thetas_bin_array[1:])
            for prev_phi, phi in zip(phis_bin_array, phis_bin_array[1:])
        }

        return thetas_bin_array, phis_bin_array, sf_data_V, sf_data_B
    else:
        return thetas_bin_array, phis_bin_array, None, None
    
    
    
    
def finner_bins_SFs(  dB,
                      dV,
                      qorder,
                      outer_count,
                      tau_values,
                      step,
                      VBangle,
                      Phiangle, 
                      di,
                      Vsw,
                      dt,
                      sf_data_V,
                      sf_data_B):
    
    thetas, phis, _, _ = init_dicts_for_SFs(step,
                                            tau_values,
                                            qorder,
                                            create_dicts=False)
    
    thetas_bin_array = np.arange(0, 91, step)
    phis_bin_array = np.arange(0, 91, step)


    bin_keys = list(sf_data_V.keys())

    counts = 0
    for ii in range(1, len(thetas)):
        the_min, the_max = thetas[ii - 1], thetas[ii]
        the_cond = (VBangle > the_min) & (VBangle < the_max)

        for jj in range(1, len(phis)):

            phi_min, phi_max = phis[jj - 1], phis[jj]
            phi_cond = (Phiangle > phi_min) & (Phiangle < phi_max)
            indices = np.where(the_cond & phi_cond)[0]

             # Append values to the yvals arrays
            sf_data_V[bin_keys[counts]]['yvals'][outer_count, :] = np.array(structure_functions_3D(indices, qorder, dV))
            sf_data_B[bin_keys[counts]]['yvals'][outer_count, :] = np.array(structure_functions_3D(indices, qorder, dB))
      
            sf_data_B [bin_keys[counts]]['counts'].append(len(indices))
            sf_data_V [bin_keys[counts]]['counts'].append(len(indices))
                                                                             
            sf_data_B [bin_keys[counts]]['xvals'].append((tau_values[outer_count]*dt*Vsw)/di)
            sf_data_V [bin_keys[counts]]['xvals'].append((tau_values[outer_count]*dt*Vsw)/di)
            
            sf_data_B [bin_keys[counts]]['identifier'].append(f'th_min_{the_min}_th_max_{the_max}_ph_min{phi_min}_ph_max{phi_max}')
            sf_data_V [bin_keys[counts]]['identifier'].append(f'th_min_{the_min}_th_max_{the_max}_ph_min{phi_min}_ph_max{phi_max}')
            
            counts = counts + 1

    return sf_data_B, sf_data_V 





def estimate_3D_sfuncs(
                       B,
                       V,
                       V_sc,
                       Np,
                       dt,
                      # Vsw,
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
                       phi_thresh_gen           = 0,
                       extra_conditions         = False,
                       fix_sign                 = True,
                       ts_list                  = None,
                       thetas_phis_step         = 10,
                       return_B_in_vel_units    = False,
                       turb_amp_analysis        = False,
                       estimate_dzp_dzm         = False,
                       also_return_db_nT        = False,
                       use_local_polarity       = True,
                       use_np_factor            = True,
                       est_proj_ells            = True):
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


    # Function to initialize arrays
    def init_nan_array(shape):
        return np.full(shape, np.nan)

    # Initialize 2D arrays
    sf_ell_perp_B, sf_Ell_perp_B, sf_ell_par_B, sf_ell_par_rest_B, sf_overall_B = \
        (init_nan_array((len(tau_values), len(qorder))) for _ in range(5))

    sf_ell_perp_V, sf_Ell_perp_V, sf_ell_par_V, sf_ell_par_rest_V, sf_overall_V = \
        (init_nan_array((len(tau_values), len(qorder))) for _ in range(5))

    sf_ell_perp_Zp, sf_Ell_perp_Zp, sf_ell_par_Zp, sf_ell_par_rest_Zp, sf_overall_Zp, \
    sf_ell_perp_Zm, sf_Ell_perp_Zm, sf_ell_par_Zm, sf_ell_par_rest_Zm, sf_overall_Zm = \
        (init_nan_array((len(tau_values), len(qorder))) for _ in range(10))

    # Initialize 1D arrays
    counts_ell_perp, counts_Ell_perp, counts_ell_par, counts_ell_par_rest, counts_overall, \
    sdk_ell_perp_B, sdk_Ell_perp_B, sdk_ell_par_B, sdk_ell_par_rest_B, sdk_overall_B, \
    l_ell_perp, l_Ell_perp, l_ell_par, l_ell_par_rest, l_overall = \
        (init_nan_array(len(tau_values)) for _ in range(15))

    # Initialize dictionaries    
    thetas, phis, u_norms, b_norms                     = {}, {}, {}, {}
    ub_polar, ub_reg, ub_weighted                      = [], [], []
    zpm_polar, zpm_reg, zpm_weighted                   = [], [], []
    sig_c_mean, sig_r_mean, sig_c_median, sig_r_median = [], [], [], []
    counts_vb, counts_zp                               = [], []
    l_ell_arr, l_xi_arr, l_lambda_arr                  = [], [], []
    l_mags                                             = []
    #checks                                             = {}
    
    
    if only_general ==2:
        _, _, sf_data_V, sf_data_B =  init_dicts_for_SFs(
                                                         thetas_phis_step,
                                                         tau_values,
                                                         qorder,
                                                         create_dicts=True)
    
    if return_coefs:
        lambda_dict = {}; xi_dict = {}; ell_par_dict = {}; ell_par_rest_dict = {}; ell_all_dict ={}
    
    
    
    # Run main loop
    for jj, tau_value in enumerate(tau_values):
        
        try:
       

            # Call the function with keyword arguments directly
            Vsw, B_l, keep_turb_amp,  dB, dB_perp, dB_parallel, dV, dN,  kinet_normal, sign_Bx, normal_flag,  l_mag,  l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm, needed_index, local_polarity = local_structure_function(
                           B.copy(),
                           V.copy(),
                           V_sc.copy(),
                           Np.copy(),
                           int(tau_value),
                           dt,
                           return_unit_vecs         = return_unit_vecs,
                           five_points_sfunc        = five_points_sfuncs,
                           estimate_alignment_angle = estimate_alignment_angle,
                           return_mag_align_correl  = return_mag_align_correl,
                           fix_sign                 = fix_sign,
                           return_B_in_vel_units    = return_B_in_vel_units,
                           turb_amp_analysis        = turb_amp_analysis,
                           also_return_db_nT        = also_return_db_nT,
                           use_local_polarity       = use_local_polarity,
                           use_np_factor            = use_np_factor,
                           est_proj_ells            =  est_proj_ells
            )
            
            
            l_mags.append(np.nanmean(l_mag))

            
            
            
            if estimate_alignment_angle:
                # Va, v average angles
                ub_polar.append(align_angles_vb['polar_inter_angle'])
                ub_reg.append(align_angles_vb['reg_angle'])
                ub_weighted.append(align_angles_vb['weighted_angle'])
                sig_r_mean.append(align_angles_vb['sig_r_mean'])
                sig_r_median.append(align_angles_vb['sig_r_median'])   
                counts_vb.append(align_angles_vb['counts'])

                # Zp, Zm average angles
                zpm_polar.append(align_angles_zpm['polar_inter_angle'])
                zpm_reg.append(align_angles_zpm['reg_angle'])
                zpm_weighted.append(align_angles_zpm['weighted_angle'])
                sig_c_mean.append(align_angles_zpm['sig_c_mean'])
                sig_c_median.append(align_angles_zpm['sig_c_median']) 
                counts_zp.append(align_angles_zpm['counts'])
                
            if jj==0:
                Vsw_fin = Vsw
            
            if use_local_polarity:
                sign_B_back  = local_polarity
            else:
                sign_B_back  = sign_Bx               
            

            # Estimate elssaser variables
            if return_B_in_vel_units:
                
                d_Zp = dV + dB *  sign_B_back[:, None]  
                d_Zm = dV - dB *  sign_B_back[:, None]
            else:
                combined = sign_B_back * kinet_normal[:, 0]

                # Element-wise operations for d_Zp and d_Zm
                d_Zp = dV + dB * combined[:, None]  
                d_Zm = dV - dB * combined[:, None]

                del combined
                
                

            # Estimate extra quantities     
            if return_coefs:
                final_variables = quants_2_estimate(
                                                    B_l,
                                                    local_polarity,
                                                    dB,
                                                    dB_perp, 
                                                    dB_parallel,
                                                    dV,
                                                    d_Zp,
                                                    d_Zm,
                                                    dN,
                                                    Np,
                                                    keep_turb_amp,
                                                    np.concatenate(kinet_normal),
                                                    sign_Bx,
                                                    B.copy(),
                                                    V.copy(),
                                                    Phiangle,
                                                    VBangle,
                                                    align_angles_zpm,
                                                    align_angles_vb,
                                                    int(tau_value),
                                                    needed_index,
                                                    di,
                                                    Vsw,
                                                    five_points_sfunc        = five_points_sfuncs,
                                                    av_hours  = 1/120,
                                                    ts_list   = ts_list)


            if only_general ==1:

                """ For General """
                if return_coefs:
                    indices                   = np.where((final_variables['thetas'] > theta_thresh_gen) & (final_variables['phis'] > phi_thresh_gen))[0]
                    ell_all_dict[str(jj)]     = save_flucs(indices, final_variables, l_mag,'lambda')
                else:
                    indices                     = np.where((VBangle > theta_thresh_gen) & (Phiangle > phi_thresh_gen))[0]
                    sf_overall_B[jj, :]         = structure_functions_3D(indices, qorder, dB)
                    sf_overall_V[jj, :]         = structure_functions_3D(indices, qorder, dV)
                    sf_overall_Zp[jj, :]        = structure_functions_3D(indices, qorder, d_Zp)
                    sf_overall_Zm[jj, :]        = structure_functions_3D(indices, qorder, d_Zm)
                    counts_overall[jj]          = len(indices)

                if estimate_PDFS:
                    PDF_all                  = estimate_pdfs_3D(indices,  dB)
                else:
                    PDF_all = None   
            elif only_general ==0:

                """ For Perpendicular """
                if return_coefs:
                    indices              = np.where((final_variables['thetas'] > sf_ell_perp_conds['theta']) & (final_variables['phis'] > sf_ell_perp_conds['phi']))[0]
                    lambda_dict[str(jj)] = save_flucs(indices, final_variables, l_lambda,'lambdas')

                else:
                    indices                      = np.where((VBangle > sf_ell_perp_conds['theta']) & (Phiangle > sf_ell_perp_conds['phi']))[0]
                    sf_ell_perp_B[jj, :], sdk_ell_perp_B[jj]            = structure_functions_3D(indices, qorder, dB)
                    sf_ell_perp_V[jj, :],_                              = structure_functions_3D(indices, qorder, dV)
                    sf_ell_perp_Zp[jj, :],_                             = structure_functions_3D(indices, qorder, d_Zp)
                    sf_ell_perp_Zm[jj, :],_                             = structure_functions_3D(indices, qorder, d_Zm)
                    l_ell_perp[jj]                                      = np.nanmean(l_lambda[indices]) 


                    counts_ell_perp[jj]      = len(indices)


                if estimate_PDFS:
                    PDF_ell_perp             = estimate_pdfs_3D(indices,  dB)
                else:
                    PDF_ell_perp = None

                """ For Displacement """
                if return_coefs:
                    indices                  = np.where((final_variables['thetas'] > sf_Ell_perp_conds['theta']) & (final_variables['phis'] < sf_Ell_perp_conds['phi']))[0]
                    xi_dict[str(jj)]         = save_flucs(indices, final_variables, l_xi,'xis')

                else:
                    indices                  = np.where((VBangle > sf_Ell_perp_conds['theta']) & (Phiangle < sf_Ell_perp_conds['phi']))[0]
                    sf_Ell_perp_B[jj, :], sdk_Ell_perp_B[jj]     = structure_functions_3D(indices, qorder, dB)
                    sf_Ell_perp_V[jj, :],_                       = structure_functions_3D(indices, qorder, dV)
                    sf_Ell_perp_Zp[jj, :],_                      = structure_functions_3D(indices, qorder, d_Zp)
                    sf_Ell_perp_Zm[jj, :],_                      = structure_functions_3D(indices, qorder, d_Zm)
                    l_Ell_perp[jj]                               = np.nanmean(l_xi[indices]) 
                    counts_Ell_perp[jj]                          = len(indices)

                if estimate_PDFS:        
                    PDF_Ell_perp             = estimate_pdfs_3D(indices,  dB)

                else:
                    PDF_Ell_perp = None


                """ For Parallel """
                if return_coefs:
                    indices                  = np.where((final_variables['thetas'] < sf_ell_par_conds['theta']) & (final_variables['phis'] < sf_ell_par_conds['phi']))[0]
                    ell_par_dict[str(jj)]    = save_flucs(indices, final_variables, l_ell,'ells')


                else:
                    indices                  = np.where((VBangle < sf_ell_par_conds['theta']) & (Phiangle < sf_ell_par_conds['phi']))[0] 
                    sf_ell_par_B[jj, :], sdk_ell_par_B[jj]         = structure_functions_3D(indices, qorder, dB)
                    sf_ell_par_V[jj, :], _                         = structure_functions_3D(indices, qorder, dV)
                    sf_ell_par_Zp[jj, :], _                        = structure_functions_3D(indices, qorder, d_Zp)
                    sf_ell_par_Zm[jj, :], _                        = structure_functions_3D(indices, qorder, d_Zm)
                    l_ell_par[jj]                                  = np.nanmean(l_ell[indices]) 

                    counts_ell_par[jj]                             = len(indices)

                if estimate_PDFS:
                    PDF_ell_par              = estimate_pdfs_3D(indices,  dB)
                else:
                    PDF_ell_par = None

                """ For Parallel but restricted """
                if return_coefs:
                    indices        = np.where((final_variables['thetas'] < sf_ell_par_rest_conds['theta']) & (final_variables['phis'] < sf_ell_par_rest_conds['phi']))[0]
                    ell_par_rest_dict[str(jj)]                        = save_flucs(indices, final_variables, l_ell,'ells_rest')
                else:
                    indices                                           = np.where((VBangle < sf_ell_par_rest_conds['theta']) & (Phiangle < sf_ell_par_rest_conds['phi']))[0]
                    
                    sf_ell_par_rest_B[jj, :], sdk_ell_par_rest_B[jj]  = structure_functions_3D(indices, qorder, dB)
                    sf_ell_par_rest_V[jj, :], _                       = structure_functions_3D(indices, qorder, dV)
                    sf_ell_par_rest_Zp[jj, :], _                      = structure_functions_3D(indices, qorder, d_Zp)
                    sf_ell_par_rest_Zm[jj, :], _                      = structure_functions_3D(indices, qorder, d_Zm)
                    l_ell_par_rest[jj]                                = np.nanmean(l_ell[indices]) 
                    counts_ell_par_rest[jj]                           = len(indices)

                if estimate_PDFS:
                    PDF_ell_par_rest         = estimate_pdfs_3D(indices,  dB)
                else:
                    PDF_ell_par_rest = None 


                """ For General """
                if return_coefs ==0:

                    indices                    = np.where((VBangle > 0) & (Phiangle > 0))[0]
                    sf_overall_B[jj, :], sdk_overall_B[jj]        = structure_functions_3D(indices, qorder, dB)
                    sf_overall_V[jj, :], _                        = structure_functions_3D(indices, qorder, dV)
                    sf_overall_Zp[jj, :], _                       = structure_functions_3D(indices, qorder, d_Zp)
                    sf_overall_Zm[jj, :], _                       = structure_functions_3D(indices, qorder, d_Zm)
                    counts_overall[jj]                            = len(indices)



            else:                                                     

                if return_coefs ==0:                                                   

                    sf_data_B, sf_data_V  = finner_bins_SFs(  dB,
                                                              dV,
                                                              qorder,
                                                              jj,
                                                              tau_values,
                                                              thetas_phis_step,
                                                              VBangle,
                                                              Phiangle,
                                                              di,
                                                              Vsw,
                                                              dt,
                                                              sf_data_V,
                                                              sf_data_B)

        except:
            traceback.print_exc()
            pass
    
    
    # Also estimate x values in di
    #l_di    = (tau_values*dt*Vsw_fin)/di
    l_di     = np.array(l_mags)
    
    # Return fluctuations
    if  return_coefs:
        l_mag     = l_mag[np.where((final_variables['thetas'] > 0) & (final_variables['phis'] > 0))[0]]
        l_ell     = l_ell[np.where((final_variables['thetas'] > 0) & (final_variables['phis'] > 0))[0]]
        l_xi      = l_xi[np.where((final_variables['thetas'] > 0) & (final_variables['phis'] > 0))[0]]
        l_lambda  = l_lambda[np.where((final_variables['thetas'] > 0) & (final_variables['phis'] > 0))[0]]
        if only_general:
            flucts = {
                       'ell_all'           :  pd.DataFrame(ell_all_dict).T.apply(lambda col: pd.Series([item for sublist in col for item in sublist])),   
                       'turb_amp'          : keep_turb_amp,
                
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw_fin,
                       'di'                :  di,
                       'dt'                :  dt,
                       'B_flag'            : normal_flag
                     }            
        else:
            flucts = {
                        'ell_perp'    : pd.DataFrame(lambda_dict).T, 
                        'Ell_perp'    : pd.DataFrame(xi_dict).T, 
                        'ell_par'     : pd.DataFrame(ell_par_dict).T, 
                        'tau_lags'    : tau_values,
                        'l_di'        : l_di,
                        'Vsw'         : Vsw_fin,
                        'di'          : di,
                        'dt'          : dt,
                        'B_flag'      : normal_flag
                      }
           
    else:
        flucts = None
                                                                             
                                                                             
    if only_general!=2:
        Sfunctions     = {
                            'B':{
                                  'ell_perp'            : sf_ell_perp_B.T,
                                  'Ell_perp'            : sf_Ell_perp_B.T,
                                  'ell_par'             : sf_ell_par_B.T,
                                  'ell_par_rest'        : sf_ell_par_rest_B.T,
                                  'ell_overall'         : sf_overall_B.T,
                                  'sdk_ell_perp'            : sdk_ell_perp_B,
                                  'sdk_Ell_perp'            : sdk_Ell_perp_B,
                                  'sdk_ell_par'             : sdk_ell_par_B,
                                  'sdk_ell_par_rest'        : sdk_ell_par_rest_B,
                                  'sdk_ell_overall'         : sdk_overall_B},

                            'V' :{
                                  'ell_perp'            : sf_ell_perp_V.T,
                                  'Ell_perp'            : sf_Ell_perp_V.T,
                                  'ell_par'             : sf_ell_par_V.T,
                                  'ell_par_rest'        : sf_ell_par_rest_V.T,
                                  'ell_overall'         : sf_overall_V.T}, 
                            'Zp' :{
                                  'ell_perp'            : sf_ell_perp_Zp.T,
                                  'Ell_perp'            : sf_Ell_perp_Zp.T,
                                  'ell_par'             : sf_ell_par_Zp.T,
                                  'ell_par_rest'        : sf_ell_par_rest_Zp.T,
                                  'ell_overall'         : sf_overall_Zp.T}, 
                            'Zm' :{
                                  'ell_perp'            : sf_ell_perp_Zm.T,
                                  'Ell_perp'            : sf_Ell_perp_Zm.T,
                                  'ell_par'             : sf_ell_par_Zm.T,
                                  'ell_par_rest'        : sf_ell_par_rest_Zm.T,
                                  'ell_overall'         : sf_overall_Zm.T},  

                              'counts_ell_perp'     : counts_ell_perp.T,
                              'counts_Ell_perp'     : counts_Ell_perp.T,
                              'counts_ell_par'      : counts_ell_par.T,
                              'counts_ell_par_rest' : counts_ell_par_rest.T,
                              'counts_ell_overall'  : counts_overall.T,
                              'l_ell_perp'          : l_ell_perp, 
                              'l_Ell_perp'          : l_Ell_perp, 
                              'l_ell_par'           : l_ell_par,
                              'l_ell_par_rest'      : l_ell_par_rest,
                              'B_flag'              : normal_flag,
                              #'checks'              : checks
                        }    
                  
    else:                                                            
        Sfunctions     = {
                            'B'      : sf_data_B,
                            'V'      : sf_data_V,
                            'B_flag' : normal_flag
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
        overall_align_angles ={ 'l_di' :   l_di,
                                'VB'   :  {'reg': ub_reg, 'polar':  ub_polar, 'weighted': ub_weighted, 'sig_r_mean': sig_r_mean, 'sig_r_median': sig_r_median, 'counts': counts_vb},
                                'Zpm'  :  {'reg': zpm_reg, 'polar': zpm_polar, 'weighted': zpm_weighted,'sig_c_mean': sig_c_mean, 'sig_c_median': sig_c_median, 'counts': counts_zp}            
                              }
    else:
        overall_align_angles = None                           
    
    return  l_mag, l_lambda, l_xi, l_ell, VBangle, Phiangle,  flucts, l_di, Sfunctions, PDFs, overall_align_angles
                                      