import numpy as np
import pandas as pd
from numba import jit,njit, prange
import os
import sys
from joblib import Parallel, delayed
sys.path.insert(1, os.path.join(os.getcwd(), 'functions'))
import general_functions as func
import TurbPy as turb

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
    numer_cos      = np.nansum(xvec* yvec, axis=1)# np.nansum(, axis=1)

    # Estimate magnitudes of the two vectors:
    xvec_mag       = func.estimate_vec_magnitude(xvec)
    yvec_mag       = func.estimate_vec_magnitude(yvec)

    # Estimate sigma (sigma_r for (δv, δb), sigma_c for (δzp, δz-) )
    sigma_ts             = (xvec_mag**2 - yvec_mag**2 )/( xvec_mag**2 + yvec_mag**2 )
    sigma_mean           = np.nanmean(sigma_ts)
    sigma_median         = np.nanmedian(sigma_ts)   
    
    # Estimate denominator
    denom          = (xvec_mag*yvec_mag)
    
    # Make sure we dont have inf vals
    numer[np.isinf(numer)] = np.nan
    denom[np.isinf(denom)] = np.nan

    numer          = func.estimate_vec_magnitude(numer)
    denom          = np.abs(denom)
    
    # Counts!
    
    counts        = len(numer[numer>-1e10])

    # Estimate sine of the  two vectors
    sins                = (numer/denom)
    thetas              = np.arcsin(sins)*180/np.pi
    thetas[thetas>90]   = 180 -thetas[thetas>90]
    
    # Regular alignment angle
    reg_align_angle_sin = np.nanmean(sins)
    
    # polarization intermittency angle (Beresnyak & Lazarian 2006):
    polar_int_angle     = (np.nanmean(numer)/ np.nanmean(denom))   

    # Weighted angles
    weighted_sins       = np.sin(np.nansum(thetas*(denom / np.nansum(denom)))*np.pi/180)
    #weighted_sins  = np.nansum(((sins)*(denom / np.nansum(denom))))
                               
    if return_mag_align_correl== False:
        sins, xvec_mag, yvec_mag = None, None, None
                               
    return counts, sigma_ts, sigma_mean, sigma_median, numer, numer_cos,  denom, xvec_mag, yvec_mag, reg_align_angle_sin, polar_int_angle, weighted_sins


def fast_unit_vec(a):
    return a.T / func.estimate_vec_magnitude(a)



def mag_of_ell_projections_and_angles(
                                        l_vector,
                                        B_l_vector, 
                                        db_perp_vector
                                    ):
    
    # estimate unit vector in parallel and displacement dir
    B_l_vector     = (B_l_vector.T/func.estimate_vec_magnitude(B_l_vector)).T
    db_perp_vector = (db_perp_vector.T/func.estimate_vec_magnitude(db_perp_vector)).T

    # estimate unit vector in pependicular by cross product
    b_perp_vector  = np.cross(B_l_vector, db_perp_vector)

    # Calculate dot product in-place
    l_ell          = np.abs(np.nansum(l_vector* B_l_vector, axis=1))
    l_xi           = np.abs(np.nansum(l_vector* db_perp_vector, axis=1))
    l_lambda       = np.abs(np.nansum(l_vector* b_perp_vector, axis=1))

    #  Estimate the component l perpendicular to Blocal
    l_perp         = func.perp_vector(l_vector, B_l_vector)

    # Estimate angles needed for 3D decomposition
    VBangle        = func.angle_between_vectors(l_vector, B_l_vector, restrict_2_90 = True)
    Phiangle       = func.angle_between_vectors(l_perp, db_perp_vector,  restrict_2_90 = True)

    return l_ell, l_xi, l_lambda, VBangle, Phiangle

def local_structure_function(
                             B,
                             V,
                             Np,
                             tau,
                             dt,
                             return_unit_vecs         = False,
                             five_points_sfunc        = True,
                             estimate_alignment_angle = False,
                             return_mag_align_correl  = False,
                             fix_sign                 = True, 
                             return_B_in_vel_units    = False,
                             turb_amp_analysis        = False
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
        
        #Compute the fluctuations in B
        dB           = turb.shifted_df_calcs(B, lag_coefs_db, coefs_db, return_df = True)
        needed_index = dB.index
        dB           = dB.values

        #Compute the fluctuations in V
        du           = turb.shifted_df_calcs(V, lag_coefs_db, coefs_db )
        
        #Compute the fluctuations in Np
        dN           = turb.shifted_df_calcs(Np, lag_coefs_db, coefs_db )

        # Estimate local B
        B_l          = turb.shifted_df_calcs(B, lag_coefs_loc, coefs_loc)

        # Estimate local Vsw
        V_l          = turb.shifted_df_calcs(V, lag_coefs_loc, coefs_loc)

        # Estimate average of Np to avoid unphysical spikes
        N_l          = turb.shifted_df_calcs(Np, lag_coefs_loc, coefs_loc)     

        
    # Estimate regular 2-point Structure functions
    else:
        #Compute the fluctuations in B
        dB           = B.iloc[:-tau].values - B.iloc[tau:].values#(B.iloc[tau:].values - B.iloc[:-tau].values)

        #Compute the fluctuations in V
        du           = V.iloc[:-tau].values - V.iloc[tau:].values#(V.iloc[tau:].values - V.iloc[:-tau].values)
        
        #Compute the fluctuations in Np
        dN           = Np.iloc[:-tau].values - Np.iloc[tau:].values# (Np.iloc[tau:].values - Np.iloc[:-tau].values)
        
        # Estimate local B
        B_l          = (B.iloc[:-tau].values + B.iloc[tau:].values)/2

        # Estimate local Vsw
        V_l          = (V.iloc[:-tau].values + V.iloc[tau:].values)/2

        # Estimate average of Np to avoid unphysical spikes
        N_l          = (Np.iloc[:-tau].values + Np.iloc[tau:].values)/2
        
#         needed_index = B[tau:].index
        
#         dB_shape                = B.shape
#         dB_filled               = pd.DataFrame(np.nan, index=B.index, columns=B.columns)
#         dB_filled.iloc[:-tau,:] = dB
        needed_index = B.iloc[:-tau,:].index
                
   
    # Estimate local perpendicular displacement direction
    dB_perp, dB_parallel   = func.perp_vector(dB, B_l, return_paral_comp = True)
    
    # Estimate amplitudes of perp, par
    dB_perp_amp        = np.sqrt(dB_perp.T[0]**2 + dB_perp.T[1]**2 + dB_perp.T[2]**2  )
    dB_parallel_amp = np.sqrt(dB_parallel.T[0]**2 + dB_parallel.T[1]**2 + dB_parallel.T[2]**2  )
    
    
    #Estimate l vector
    l_vec            = V_l*tau*dt

    # Estrimate l's in three directions
    l_ell, l_xi, l_lambda,  VBangle, Phiangle = mag_of_ell_projections_and_angles(l_vec,
                                                                                  B_l,
                                                                                  dB_perp)
    # Constant to normalize mag field in vel units
    kinet_normal     = 1e-15 / np.sqrt(mu0 * N_l * m_p)
    
    if return_B_in_vel_units:
        dB         =  dB*kinet_normal
        normal_flag = 'B_in_vel_units'
    else:
        normal_flag = 'B_in_nT_units'
        
    # Create empty dictionaries
    unit_vecs         = {}
    align_angles_vb   = {}
    align_angles_zpm  = {}                         
    
    if estimate_alignment_angle:

        # Kinetic normalization of magnetic field
        dva_perp         =  dB_perp*kinet_normal

        # We need the perpendicular component of the fluctuations
        du_perp          = func.perp_vector(du, B_l)

        
        # Sign of  background Br
        if fix_sign:
            signBx            = - np.sign(B_l.T[0])
            
            # Estimate fluctuations in Elssaser variables
            dzp_perp         = du_perp + (np.array(signBx)*dva_perp.T).T
            dzm_perp         = du_perp - (np.array(signBx)*dva_perp.T).T
            
        else:
            #print('Here')
            signBx            =  np.sign(func.newindex( B['Br'].rolling('10min', center=True).mean().interpolate(), needed_index).values)  #np.abs(np.sign(B_l.T[0]))     

            # Estimate fluctuations in Elssaser variables
            dzp_perp         = du_perp + (np.array(signBx)*dva_perp.T).T
            dzm_perp         = du_perp - (np.array(signBx)*dva_perp.T).T
            

        

        if turb_amp_analysis:
            
            keep_turb_amp = {
                             'dva_perp'  : dva_perp,
                             'du_perp'   : du_perp,
                             'dzp_perp'  : dzp_perp,
                             'dzm_perp'  : dzm_perp
                            }
        else:
            keep_turb_amp = {
                             'dva_perp'  : None,
                             'du_perp'   : None,
                             'dzp_perp'  : None,
                             'dzm_perp'  : None
                            }            
        
        #Estimate magnitudes,  angles in three different ways          
        ub_results = est_alignment_angles(du_perp, 
                                          dva_perp,
                                          return_mag_align_correl = return_mag_align_correl)
    
        zpm_results = est_alignment_angles(dzp_perp, 
                                           dzm_perp,
                                           return_mag_align_correl = return_mag_align_correl)   
                               
        # Assign values for va, v, z+, z-
        countsvb, sigma_r_ts,  sigma_r_mean, sigma_r_median, sins_ub_numer, cos_ub_numer, sins_ub_denom,  v_mag, va_mag, reg_align_angle_sin_ub, polar_int_angle_ub, weighted_sins_ub         = ub_results
        countszp, sigma_c_ts,  sigma_c_mean, sigma_c_median, sins_zpm_numer, cos_zpm_numer, sins_zpm_denom, zp_mag, zm_mag, reg_align_angle_sin_zpm, polar_int_angle_zpm, weighted_sins_zpm   = zpm_results  

        
        align_angles_vb      = {     
                                     'sig_r_ts'          : sigma_r_ts,
                                     'sig_r_mean'        : sigma_r_mean,
                                     'sig_r_median'      : sigma_r_median,
                                     'reg_angle'         : reg_align_angle_sin_ub,
                                     'polar_inter_angle' : polar_int_angle_ub,            
                                     'weighted_angle'    : weighted_sins_ub,
                                     'v_mag'             : v_mag,
                                     'va_mag'            : va_mag,
                                     'sins_uva_num'      : sins_ub_numer,
                                     'cos_uva_numer'     : cos_ub_numer,
                                     'sins_uva_den'      : sins_ub_denom,
                                     'counts'            : countsvb
        }
                               
        align_angles_zpm     = {     
                                     'sig_c_ts'          : sigma_c_ts,
                                     'sig_c_mean'        : sigma_c_mean,
                                     'sig_c_median'      : sigma_c_median,
                                     'reg_angle'         : reg_align_angle_sin_zpm,
                                     'polar_inter_angle' : polar_int_angle_zpm,            
                                     'weighted_angle'    : weighted_sins_zpm,
                                     'zp_mag'            : zp_mag,
                                     'zm_mag'            : zm_mag,
                                     'sins_zpm_num'      : sins_zpm_numer,                                    
                                     'cos_zpm_numer'     : cos_zpm_numer,
                                     'sins_zpm_den'      : sins_zpm_denom,                                     
                                     'counts'            : countszp
        }
        
    if return_unit_vecs:

        # Estimate unit vectors
        unit_vecs     = {
                         'dB_perp_hat'   : fast_unit_vec(dB_perp), 
                         'B_l_hat'       : fast_unit_vec(fast_unit_vec(B_l)),   
                         'B_perp_2_hat'  : np.cross(B_l_hat, dB_perp_hat),
        }

    return keep_turb_amp, dB, dB_perp_amp, dB_parallel_amp, du, dN, kinet_normal, signBx, normal_flag,  func.estimate_vec_magnitude(l_vec), l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm, needed_index

@jit( parallel =True,  nopython=True)
def structure_functions_3D(
                           indices, 
                           qorder,
                           mat,
                           max_std=18
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

    for i in prange(len(qorder)):   
        result[i] = np.nanmean(ar**qorder[i]) +  np.nanmean(at**qorder[i]) +  np.nanmean(an**qorder[i])
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

    
def vars_2_estimate(ts_list=None):
    
    # Load default variables
    default_vars = ['R', 'T', 'N']
    return default_vars if ts_list is None else ts_list+ default_vars



def quants_2_estimate(
                    dB,
                    dB_perp,
                    dB_parallel,
                    dV,
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
        variables['db_perp_amp'] =dB_perp
        
    if 'db_par_amp' in quants:
        variables['db_par_amp'] =dB_parallel
               
    
    if 'PVI_vec_zp' in quants:
    
        # First estimate dzp
        dzp        = dV + (dB.T*kinet_normal*np.array(sign_Bx)).T
        
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
                                                                 hours             = 2,
                                                                 keys              = ['Zpr', 'Zpt', 'Zpn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1,                 
                                                                 input_flucts      = True,
                                                                 dbs               = dzp), needed_index).values.T[0]
    if 'PVI_vec_zm' in quants:
    
        # First estimate dzp
        dzm        = dV - (dB.T*kinet_normal*np.array(sign_Bx)).T
        
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
                                                                 hours             = 2,
                                                                 keys              = ['Zmr', 'Zmt', 'Zmn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs            =-1,                 
                                                                 input_flucts      = True,
                                                                 dbs               = dzm), needed_index).values.T[0]

    if 'PVI_Np' in quants:
        
        
        # Estimate PVI of \vec{Zp} 
        variables['PVI_Np'] = func.newindex(turb.estimate_PVI( 
                                                                 Np.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 2,
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
        
    if 'V_R' in quants:
        variables['V_R']             =  dV.T[0]  
        
    if 'V_T' in quants:
        variables['V_T']             =  dV.T[1] 
        
    if 'V_N' in quants:
        variables['V_N']             =  dV.T[2] 
        
        del dV
        
#     if 'dva_perp' in quants:
#         variables['dva_perp']             =  keep_turb_amp['dva_perp']
        
#     if 'du_perp' in quants:
#         variables['du_perp']             =  keep_turb_amp['du_perp']
        
#     if 'dzp_perp' in quants:
#         variables['dzp_perp']             =  keep_turb_amp['dzp_perp']
        
#     if 'dzm_perp' in quants:
#         variables['dzm_perp']             =  keep_turb_amp['dzm_perp']  

    if 'sign_Bx' in quants:
        variables['sign_Bx']         =  sign_Bx 
        
        del sign_Bx 
        
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
        variables['sins_ub_num']   = align_angles_vb['sins_uva_num'] 
        
    if 'cos_uva_numer' in quants: 
        variables['cos_uva_numer']   = align_angles_vb['cos_uva_numer'] 
        
    if 'sins_ub_den' in quants:
        variables['sins_ub_den']   = align_angles_vb['sins_uva_den'] 

    if 'sins_zpm_num' in quants:
        variables['sins_zpm_num']  = align_angles_zpm['sins_zpm_num']
        
    if 'cos_zpm_numer' in quants: 
        variables['cos_zpm_numer']   = align_angles_zpm['cos_zpm_numer'] 
        
    if 'sins_zpm_den' in quants:
        variables['sins_zpm_den']   = align_angles_zpm['sins_zpm_den']

        
    if 'sins_zpm' in quants:
        variables['sins_zpm']       = align_angles_zpm['sins_zpm_num']/align_angles_zpm['sins_zpm_den']
        
    if 'sins_ub' in quants:
        variables['sins_ub']        = align_angles_vb['sins_uva_num']/ align_angles_vb['sins_uva_den']

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
                                                                 hours             = 2,
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'mod',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs=-1), needed_index).values.T[0]
        

    if 'PVI_vec' in quants:
        
        # Estimate PVI of \vec{B} 
        variables['PVI_vec'] = func.newindex(turb.estimate_PVI( 
                                                                 B.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 2,
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
                                                                 use_taus          = True,
                                                                 return_only_PVI   = True,
                                                                 n_jobs=-1), needed_index).values.T[0]
    if 'PVI_vec_V' in quants:
        # Estimate PVI of \vec{B} 
        variables['PVI_vec_V'] = func.newindex(turb.estimate_PVI( 
                                                                 V.copy(),
                                                                 [1],
                                                                 [tau_value],
                                                                 di,
                                                                 Vsw,
                                                                 hours             = 2,
                                                                 keys              = ['Vr', 'Vt', 'Vn'],
                                                                 five_points_sfunc = five_points_sfunc,
                                                                 PVI_vec_or_mod    = 'vec',
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
                                                                 hours             = 2,
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
                       phi_thresh_gen           = 0,
                       extra_conditions         = False,
                       fix_sign                 = True,
                       ts_list                  = None,
                       thetas_phis_step         = 10,
                       return_B_in_vel_units    = False,
                       turb_amp_analysis        = False):
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
    sf_ell_perp_B             = np.zeros(( len(tau_values), len(qorder))); counts_ell_perp             = np.zeros( len(tau_values));
    sf_Ell_perp_B             = np.zeros(( len(tau_values), len(qorder))); counts_Ell_perp             = np.zeros( len(tau_values));
    sf_ell_par_B              = np.zeros(( len(tau_values), len(qorder))); counts_ell_par              = np.zeros( len(tau_values));
    sf_ell_par_rest_B         = np.zeros(( len(tau_values), len(qorder))); counts_ell_par_rest         = np.zeros( len(tau_values)); 
    sf_overall_B              = np.zeros(( len(tau_values), len(qorder))); counts_overall              = np.zeros( len(tau_values))
    
    sf_ell_perp_V             = np.zeros(( len(tau_values), len(qorder))); 
    sf_Ell_perp_V             = np.zeros(( len(tau_values), len(qorder))); 
    sf_ell_par_V              = np.zeros(( len(tau_values), len(qorder))); 
    sf_ell_par_rest_V         = np.zeros(( len(tau_values), len(qorder))); 
    sf_overall_V              = np.zeros(( len(tau_values), len(qorder))); 
    
    # Initialize dictionaries    
    thetas                  = {}; phis                    = {}
    u_norms                 = {}; b_norms                 = {}
    ub_polar                = []; ub_reg                  = [];  ub_weighted             = []
    zpm_polar               = []; zpm_reg                 = [];  zpm_weighted            = []
    sig_c_mean              = []; sig_r_mean              = []; sig_c_median             = [];  sig_r_median            = []
    
    counts_vb               = []; counts_zp               = [];
    
    
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
       

        # Call the function with keyword arguments directly
        keep_turb_amp, dB, dB_perp, dB_parallel, dV, dN,  kinet_normal, sign_Bx, normal_flag,  l_mag,  l_ell, l_xi, l_lambda, VBangle, Phiangle, unit_vecs, align_angles_vb, align_angles_zpm, needed_index = local_structure_function(
                                                                                                                                                                       B.copy(),
                                                                                                                                                                       V.copy(),
                                                                                                                                                                       Np.copy(),
                                                                                                                                                                       int(tau_value),
                                                                                                                                                                       dt,
                                                                                                                                                                       return_unit_vecs         = return_unit_vecs,
                                                                                                                                                                       five_points_sfunc        = five_points_sfuncs,
                                                                                                                                                                       estimate_alignment_angle = estimate_alignment_angle,
                                                                                                                                                                       return_mag_align_correl  = return_mag_align_correl,
                                                                                                                                                                       fix_sign                 = fix_sign,
                                                                                                                                                                       return_B_in_vel_units    = return_B_in_vel_units,
                                                                                                                                                                       turb_amp_analysis        = turb_amp_analysis)
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

        # Estimate extra quantities     
        if return_coefs:
            final_variables = quants_2_estimate(
                                                dB,
                                                dB_perp, 
                                                dB_parallel,
                                                dV,
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
                indices                    = np.where((VBangle > theta_thresh_gen) & (Phiangle > phi_thresh_gen))[0]
                sf_overall_B[jj, :]        = structure_functions_3D(indices, qorder, dB)
                sf_overall_V[jj, :]        = structure_functions_3D(indices, qorder, dV)
                counts_overall[jj]         = len(indices)

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
                indices                  = np.where((VBangle > sf_ell_perp_conds['theta']) & (Phiangle > sf_ell_perp_conds['phi']))[0]
                sf_ell_perp_B[jj, :]         = structure_functions_3D(indices, qorder, dB)
                sf_ell_perp_V[jj, :]         = structure_functions_3D(indices, qorder, dV)
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
                sf_Ell_perp_B[jj, :]     = structure_functions_3D(indices, qorder, dB)
                sf_Ell_perp_V[jj, :]     = structure_functions_3D(indices, qorder, dV)
                counts_Ell_perp[jj]      = len(indices)

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
                sf_ell_par_B[jj, :]      = structure_functions_3D(indices, qorder, dB)
                sf_ell_par_V[jj, :]      = structure_functions_3D(indices, qorder, dV)
                counts_ell_par[jj]       = len(indices)

            if estimate_PDFS:
                PDF_ell_par              = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_par = None

            """ For Parallel but restricted """
            if return_coefs:
                indices                   = np.where((final_variables['thetas'] < sf_ell_par_rest_conds['theta']) & (final_variables['phis'] < sf_ell_par_rest_conds['phi']))[0]
                ell_par_rest_dict[str(jj)]= save_flucs(indices, final_variables, l_ell,'ells_rest')
            else:
                indices                   = np.where((VBangle < sf_ell_par_rest_conds['theta']) & (Phiangle < sf_ell_par_rest_conds['phi']))[0]
                sf_ell_par_rest_B[jj, :]  = structure_functions_3D(indices, qorder, dB)
                sf_ell_par_rest_V[jj, :]  = structure_functions_3D(indices, qorder, dV)
                counts_ell_par_rest[jj]       = len(indices)

            if estimate_PDFS:
                PDF_ell_par_rest         = estimate_pdfs_3D(indices,  dB)
            else:
                PDF_ell_par_rest = None 
                
                
            """ For General """
            if return_coefs ==0:
   
                indices                    = np.where((VBangle > 0) & (Phiangle > 0))[0]
                sf_overall_B[jj, :]        = structure_functions_3D(indices, qorder, dB)
                sf_overall_V[jj, :]        = structure_functions_3D(indices, qorder, dV)
                counts_overall[jj]         = len(indices)
                                                        
                                                                           
                                                                           
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


    # Also estimate x values in di
    l_di    = (tau_values*dt*Vsw)/di
    
    # Return fluctuations
    if  return_coefs:
        if only_general:
            flucts = {
                       'ell_all'           :  pd.DataFrame(ell_all_dict).T.apply(lambda col: pd.Series([item for sublist in col for item in sublist])),   
                       'turb_amp'          : keep_turb_amp,
                
                       'tau_lags'          :  tau_values,
                       'l_di'              :  l_di,
                       'Vsw'               :  Vsw,
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
                        'Vsw'         : Vsw,
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
                                  'ell_overall'         : sf_overall_B.T},

                            'V' :{
                                  'ell_perp'            : sf_ell_perp_V.T,
                                  'Ell_perp'            : sf_Ell_perp_V.T,
                                  'ell_par'             : sf_ell_par_V.T,
                                  'ell_par_rest'        : sf_ell_par_rest_V.T,
                                  'ell_overall'         : sf_overall_V.T},        

                          'counts_ell_perp'     : counts_ell_perp.T,
                          'counts_Ell_perp'     : counts_Ell_perp.T,
                          'counts_ell_par'      : counts_ell_par.T,
                          'counts_ell_par_rest' : counts_ell_par_rest.T,
                          'counts_ell_overall'  : counts_overall.T,
                          'B_flag'              : normal_flag 
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
    
    return VBangle, Phiangle,  flucts, l_di, Sfunctions, PDFs, overall_align_angles
                                      