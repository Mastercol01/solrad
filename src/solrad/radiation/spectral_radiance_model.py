#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions related to the computation of the 
spatial and pectral distribution of direct and diffuse irradiance, also called
spectral radiance. This is accomplished by a combination of various models.
More specifically, one model for the spatial distribution of the diffuse component,
one model for the spatial distribution of the direct component and one model
for the spectral distribution of both.

"""

#%%                        IMPORTATION OF LIBRARIES

import numpy as np
import pandas as pd
from solrad.radiation.direct_radiance_model import compute_direct_radiance
from solrad.radiation.diffuse_radiance_model import compute_diffuse_radiance
from solrad.radiation.spectrl2 import compute_direct_and_diffuse_normalized_spectra


#%%

def compute_spectral_radiance(Az, El, dAz, dEl, 
                              DatetimeIndex_obj, sun_apel, sun_az,
                              Gh, extra_Gbn, Gbn, Gdh,
                              SP, rel_airmass, H2O, O3, 
                              aod_500nm, alpha_500nm, 
                              spectrally_averaged_aaf,
                              single_scattering_albedo, 
                              ground_albedo = 0, 
                              mean_surface_tilt = 0, 
                              num_iterations=500):
    
    """
    Compute direct and diffuse components of spectral radiance across time.
    
    
    Parameters
    ----------
    Az : numpy.array of floats with shape (E,A)
       Azimuth array of meshgrid of Azimuth, Elevation values. It contains
       the azimuth (in degrees) of each sky element to be considered in 
       the calculation of spectral sky radiance. The values of 'Az' should vary 
       along axis 1. Values should be between 0 and 360.
    
    El : numpy.array of floats with shape (E,A)
       Elevation array of meshgrid of Azimuth, Elevation values. It contains
       the elevation (in degrees) of each sky element to be considered in 
       the calculation of spectral sky radiance. The values of 'El' should vary 
       along axis 0. Values should be between 0 and 90.
       
    dAz : float
        Angular resolution of 'Az' in degrees.
        
    dEl : float
        Angular resolution of 'El' in degrees.
        
    DatetimeIndex_obj : pandas.DatetimeIndex object with shape (T,) 
        An index of Timestamp values detailing the times at which each of the
        samples of the time-dependent variables were taken. We denote its 
        length as T.
    
    sun_apel : numpy.array of floats with shape (T,) 
        Sun's apparent elevation (in degrees) across time. Values must lie 
        between 0 and 90 (inclusive).
        
    sun_az : numpy.array of floats with shape (T,) 
        Suns's azimuth (in degrees) across time. Values must lie 
        between 0 and 360 (inclusive).
        
    Gh : numpy.array of floats with shape (T,)  
       Global horizontal irradiance [W/m^2] across time. Must be a
       non-negative array of numbers. 
         
    extra_Gbn : numpy.array of floats with shape (T,) 
        Extraterrestrial normal irradiance [W/m^2] across time. Must be a
        non-negative array of numbers. 
        
    Gbn : numpy.array of floats with shape (T,)  
       Direct normal irradiance [W/m^2] across time. Must be a
       non-negative array of numbers. 
    
    Gdh : numpy.array of floats with shape (T,) 
        Diffuse horizontal irradiance[W/m^2] across time. Must be a
        non-negative array of numbers.
    
    SP : numpy.array of floats with shape (T,) 
        Surface Pressure [Pa] across time.
        
    rel_airmass : numpy.array of floats with shape (T,) 
        Relative airmass [unitless] acorss time.
        
    H2O : numpy.array of floats with shape (T,) 
        Atmospheric water vapor content [cm] across time.
        
    O3 : numpy.array of floats with shape (T,) 
        Atmospheric ozone content [atm-cm] across time. 
        
    aod_500nm : numpy.array of floats with shape (T,) 
        Aerosol turbidity at 500 nm [unitless] across time.
        
    alpha_500nm : numpy.array of floats with shape (T,) 
        Angstrom turbidity exponent at 500nm [unitless] across time.
        
    spectrally_averaged_aerosol_asymmetry_factor : numpy.array of floats with shape (T,)
        Average across selected range of wavelengths of the Aerosol asymmetry 
        factor (mean cosine of scattering angle) [unitless], across time. 
        
    single_scattering_albedo : numpy.array of floats with shape (T,122)
        Aerosol single scattering albedo at multiple wavelengths. It is matrix 
        of size Tx122 where the second dimension spans the wavelength range and
        the first one spans the number of simulations (i.e, length of 
        'DatetimeIndex_obj') [unitless]. 
        
    ground_albedo : float or numpy.array of floats with shape (T,122), optional
        Albedo [0-1] of the ground surface. Can be provided as a scalar value
        if albedo is not spectrally-dependent, or as a Tx122 matrix where
        the second dimension spans the wavelength range and the first one spans
        the number of simulations (i.e, length of 'DatetimeIndex_obj').
        [unitless]. Default is 0.
        
    mean_surface_tilt : float or numpy.array of floats with shape (T,), optional
        Mean panel tilt from horizontal [degrees] across time. Default is 0.
        
    num_iterations : int, optional
        Number of iterations to use when filling NaN data. Default is 500.
        
        
    Returns
    -------
    res : dict
        Dictionary containing result variables. It has the following Key-Value
        pairs:
            
            Keys : Values
            -------------
            "Siv" : numpy.array of floats with shape (T,)   
                Igawa's 'Sky Index' parameter across time.
            
            "Kc" : numpy.array of floats with shape (T,) 
                Igawa's 'Clear Sky Index' parameter across time.
                
            "Cle" : numpy.array of floats with shape (T,) 
                Igawa's 'Cloudless Index' parameter across time.
                
            "wavelengths" : numpy.array of floats with shape (122,)
                Wavelengths in nanometers.
                
            "DatetimeIndex_obj" : pandas.Series of pandas.Timestamp objects.
                Series of Timestamp values detailing the times at which each of the
                samples of the time-dependent variables were taken. We denote its 
                length as T.
                
            "direct" : List with length T of numpy.arrays of floats with shape (E,A,122)
                Direct component of spectral radiance across time.
                
            "dffuse" : List with length T of numpy.arrays of floats with shape (E,A,122)
                Diffuse component of sepctral radiance across time.
                
                
    Notes
    -----
    1) "mean_surface_tilt" variable really only affects the computation of
       the spectral distribution of diffuse radiance. It has no effect on 
       the actual value. 
    """
    
    
    

    # ---------- COMPUTE DIRECT AND DIFFUSE NORMALIZED SPECTRA ACROSS TIME ----------
    
    res_spectral =\
    compute_direct_and_diffuse_normalized_spectra(
        
    sun_apzen                = pd.Series(data = 90 - sun_apel,           index = DatetimeIndex_obj),
    SP                       = pd.Series(data = SP,                      index = DatetimeIndex_obj), 
    rel_airmass              = pd.Series(data = rel_airmass,             index = DatetimeIndex_obj),
    H2O                      = pd.Series(data = H2O,                     index = DatetimeIndex_obj),
    O3                       = pd.Series(data = O3,                      index = DatetimeIndex_obj), 
    aod_500nm                = pd.Series(data = aod_500nm,               index = DatetimeIndex_obj), 
    alpha_500nm              = pd.Series(data = alpha_500nm,             index = DatetimeIndex_obj), 
    spectrally_averaged_aaf  = pd.Series(data = spectrally_averaged_aaf, index = DatetimeIndex_obj),
    dayofyear                = None,
    ground_albedo            = ground_albedo,
    mean_surface_tilt        = mean_surface_tilt,
    single_scattering_albedo = single_scattering_albedo, 
    )
    
    # -------------- COMPUTE DIFFUSE RADIANCE ACROSS TIME-------------
    
    res_diffuse =\
    compute_diffuse_radiance(
        
    Az              = Az.reshape(list(Az.shape) + [1]),
    El              = El.reshape(list(El.shape) + [1]),
    dAz             = dAz, 
    dEl             = dEl,
    Gh              = Gh.reshape(          1, 1, len(Gh)),
    Gdh             = Gdh.reshape(         1, 1, len(Gdh)),
    extra_Gbn       = extra_Gbn.reshape(   1, 1, len(extra_Gbn)),
    sun_az          = sun_az.reshape(      1, 1, len(sun_az)),
    sun_apel        = sun_apel.reshape(    1, 1, len(sun_apel)),
    rel_airmass     = rel_airmass.reshape( 1, 1, len(rel_airmass)),
    num_iterations  = num_iterations
    )
    
    
    # -------------- COMPUTE DIRECT RADIANCE ACROSS TIME-------------
    
    res_direct =\
    compute_direct_radiance(
        
    Az       = Az.reshape(list(Az.shape) + [1]), 
    El       = El.reshape(list(El.shape) + [1]), 
    Gbn      = Gbn.reshape(     1, 1, len(Gbn)),
    sun_az   = sun_az.reshape(  1, 1, len(sun_az)),
    sun_apel = sun_apel.reshape(1, 1, len(sun_apel))
    )
    
    
    
    
    # ------- COMPUTE DIRECT AND DIFFUSE SPECTRAL RADIANCES ACROSS TIME -------
    
    spectral_direct_radiances  = []
    spectral_diffuse_radiances = []
    
    for nt in range(len(DatetimeIndex_obj)):
        
        # --- COMPUTE DIRECT SPECTRAL RADIANCE AT ONE TIME --------
        
        spectral_direct_radiance =\
        res_direct[:,:,nt].reshape(res_direct.shape[0], 
                                   res_direct.shape[1], 1)
        
        spectral_direct_radiance =\
        spectral_direct_radiance*res_spectral["direct"][nt,:].reshape(1, 1, 122)
        
        spectral_direct_radiances.append(spectral_direct_radiance)
        
        
        
        # --- COMPUTE DIFFUSE SPECTRAL RADIANCE AT ONE TIME --------
        
        spectral_diffuse_radiance =\
        res_diffuse["Lea"][:,:,nt].reshape(res_diffuse["Lea"].shape[0],
                                           res_diffuse["Lea"].shape[1], 1)
        
        spectral_diffuse_radiance =\
        spectral_diffuse_radiance*res_spectral["diffuse"][nt,:].reshape(1, 1, 122)
        
        spectral_diffuse_radiances.append(spectral_diffuse_radiance)
        
        
        
    # ---- PACKAGE RESULTS ------
    
    total_res = {}
    total_res["direct"]  = spectral_direct_radiances
    total_res["diffuse"] = spectral_diffuse_radiances
    total_res["Siv"]     = res_diffuse["Siv"].flatten()
    total_res["Kc"]      = res_diffuse["Kc"].flatten() 
    total_res["Cle"]     = res_diffuse["Cle"].flatten()
    total_res["wavelengths"] = res_spectral["wavelengths"]
    
        
        
    return total_res











