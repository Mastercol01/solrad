#%%                      DESCRIPTION OF MODULE
"""
Pvlib's 'spectrl2' module that implements the Bird Simple Spectral Model.

This module is almost the same as that of the pvlib source:
https://pvlib-python.readthedocs.io/en/v0.8.1/_modules/pvlib/spectrum/spectrl2.html#spectrl2

However, some minor have been made to it. First, some of the comments and
descriptions were changed to better suit the standard being used in other 
modules of this package. Second, the 'spectrl2' function was slightly altered
in order to accomodate the use of the Shettel and Fenn models (see reference [1]) for the
Aerosol Asymmetry Factor and the Single Scattering Albedo. Third, one extra
function has been added: :func:`~solrad.radiation.spectrl2.compute_direct_and_diffuse_normalized_spectra` which
basically just uses spectrl2 and then normalizes the obtained irradiance spectra.
To be absolutely clear, most of this code is not mine (just the changes
mentioned above) and was taken from the source mentioned above. Credit for the
spctrl2 implementation belongs to them.

References 
----------
[1] Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the Lower 
Atmosphere and the Effects of Humidity Variations on their Optical Properties. Environ. Res.. 94. 

"""
#%%                      IMPORTATION OF LIBRARIES
import pvlib
import numpy as np
import pandas as pd
from pvlib.tools import cosd
from scipy.integrate import simpson
#%%                      DEFINITION OF CONSTANTS

# SPECTRL2 extraterrestrial spectrum and atmospheric absorption coefficients
_SPECTRL2_COEFFS = np.zeros(122, dtype=np.dtype([
    ('wavelength', 'float64'),
    ('spectral_irradiance_et', 'float64'),
    ('water_vapor_absorption', 'float64'),
    ('ozone_absorption', 'float64'),
    ('mixed_absorption', 'float64'),
]))
_SPECTRL2_COEFFS['wavelength'] = [  # nm
    300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 330.0, 335.0, 340.0, 345.0,
    350.0, 360.0, 370.0, 380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0,
    450.0, 460.0, 470.0, 480.0, 490.0, 500.0, 510.0, 520.0, 530.0, 540.0,
    550.0, 570.0, 593.0, 610.0, 630.0, 656.0, 667.6, 690.0, 710.0, 718.0,
    724.4, 740.0, 752.5, 757.5, 762.5, 767.5, 780.0, 800.0, 816.0, 823.7,
    831.5, 840.0, 860.0, 880.0, 905.0, 915.0, 925.0, 930.0, 937.0, 948.0,
    965.0, 980.0, 993.5, 1040.0, 1070.0, 1100.0, 1120.0, 1130.0, 1145.0,
    1161.0, 1170.0, 1200.0, 1240.0, 1270.0, 1290.0, 1320.0, 1350.0, 1395.0,
    1442.5, 1462.5, 1477.0, 1497.0, 1520.0, 1539.0, 1558.0, 1578.0, 1592.0,
    1610.0, 1630.0, 1646.0, 1678.0, 1740.0, 1800.0, 1860.0, 1920.0, 1960.0,
    1985.0, 2005.0, 2035.0, 2065.0, 2100.0, 2148.0, 2198.0, 2270.0, 2360.0,
    2450.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0,
    3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0
]
_SPECTRL2_COEFFS['spectral_irradiance_et'] = [  # W/m^2/nm
    0.5359, 0.5583, 0.622, 0.6927, 0.7151, 0.8329, 0.9619, 0.9319, 0.9006,
    0.9113, 0.9755, 0.9759, 1.1199, 1.1038, 1.0338, 1.4791, 1.7013, 1.7404,
    1.5872, 1.837, 2.005, 2.043, 1.987, 2.027, 1.896, 1.909, 1.927, 1.831,
    1.891, 1.898, 1.892, 1.84, 1.768, 1.728, 1.658, 1.524, 1.531, 1.42,
    1.399, 1.374, 1.373, 1.298, 1.269, 1.245, 1.223, 1.205, 1.183, 1.148,
    1.091, 1.062, 1.038, 1.022, 0.9987, 0.9472, 0.8932, 0.8682, 0.8297,
    0.8303, 0.814, 0.7869, 0.7683, 0.767, 0.7576, 0.6881, 0.6407, 0.6062,
    0.5859, 0.5702, 0.5641, 0.5442, 0.5334, 0.5016, 0.4775, 0.4427, 0.44,
    0.4168, 0.3914, 0.3589, 0.3275, 0.3175, 0.3073, 0.3004, 0.2928, 0.2755,
    0.2721, 0.2593, 0.2469, 0.244, 0.2435, 0.2348, 0.2205, 0.1908, 0.1711,
    0.1445, 0.1357, 0.123, 0.1238, 0.113, 0.1085, 0.0975, 0.0924, 0.0824,
    0.0746, 0.0683, 0.0638, 0.0495, 0.0485, 0.0386, 0.0366, 0.032, 0.0281,
    0.0248, 0.0221, 0.0196, 0.0175, 0.0157, 0.0141, 0.0127, 0.0115, 0.0104,
    0.0095, 0.0086
]
_SPECTRL2_COEFFS['water_vapor_absorption'] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.075, 0.0, 0.0, 0.0, 0.0, 0.016, 0.0125, 1.8, 2.5, 0.061,
    0.0008, 0.0001, 1e-05, 1e-05, 0.0006, 0.036, 1.6, 2.5, 0.5, 0.155, 1e-05,
    0.0026, 7.0, 5.0, 5.0, 27.0, 55.0, 45.0, 4.0, 1.48, 0.1, 1e-05, 0.001, 3.2,
    115.0, 70.0, 75.0, 10.0, 5.0, 2.0, 0.002, 0.002, 0.1, 4.0, 200.0, 1000.0,
    185.0, 80.0, 80.0, 12.0, 0.16, 0.002, 0.0005, 0.0001, 1e-05, 0.0001, 0.001,
    0.01, 0.036, 1.1, 130.0, 1000.0, 500.0, 100.0, 4.0, 2.9, 1.0, 0.4, 0.22,
    0.25, 0.33, 0.5, 4.0, 80.0, 310.0, 15000.0, 22000.0, 8000.0, 650.0, 240.0,
    230.0, 100.0, 120.0, 19.5, 3.6, 3.1, 2.5, 1.4, 0.17, 0.0045
]
_SPECTRL2_COEFFS['ozone_absorption'] = [
    10.0, 4.8, 2.7, 1.35, 0.8, 0.38, 0.16, 0.075, 0.04, 0.019, 0.007, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003, 0.006, 0.009, 0.014, 0.021, 0.03,
    0.04, 0.048, 0.063, 0.075, 0.085, 0.12, 0.119, 0.12, 0.09, 0.065, 0.051,
    0.028, 0.018, 0.015, 0.012, 0.01, 0.008, 0.007, 0.006, 0.005, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]
_SPECTRL2_COEFFS['mixed_absorption'] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,
    0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.3,
    0.02, 0.0002, 0.00011, 1e-05, 0.05, 0.011, 0.005, 0.0006, 0.0, 0.005, 0.13,
    0.04, 0.06, 0.13, 0.001, 0.0014, 0.0001, 1e-05, 1e-05, 0.0001, 0.001, 4.3,
    0.2, 21.0, 0.13, 1.0, 0.08, 0.001, 0.00038, 0.001, 0.0005, 0.00015,
    0.00014, 0.00066, 100.0, 150.0, 0.13, 0.0095, 0.001, 0.8, 1.9, 1.3, 0.075,
    0.01, 0.00195, 0.004, 0.29, 0.025
]

#%%                    DEFINITION OF FUNCTIONS

def _spectrl2_transmittances(apparent_zenith, relative_airmass,
                             surface_pressure, precipitable_water, ozone,
                             optical_thickness, scattering_albedo, dayofyear):
    """
    Calculate transmittance factors from Section 2 of Bird and Riordan 1984.

    Parameters
    ----------
    apparent_zenith, relative_airmass, surface_pressure, precipitable_water,
    ozone, dayofyear: float or 1d np.array
        One value per timestamp
    optical_thickness, scattering_albedo: np.ndarray
        Array with shape (122, N) where N is either 1 or len(apparent_zenith)

    Returns
    -------
    earth_sun_distance_correction: float or 1d np.array
        Same shape/type as apparent_zenith
    rayleigh_transmittance, aerosol_transmittance, vapor_transmittance,
    ozone_transmittance, mixed_transmittance, aerosol_scattering,
    aerosol_absorption: np.ndarray
        Array with shape (122, N) where N is len(apparent_zenith)
    """
    # add a dimension so that each ndarray is 2d with shape (122, 1)
    wavelength = _SPECTRL2_COEFFS['wavelength'][:, np.newaxis]
    vapor_coeff = _SPECTRL2_COEFFS['water_vapor_absorption'][:, np.newaxis]
    ozone_coeff = _SPECTRL2_COEFFS['ozone_absorption'][:, np.newaxis]
    mixed_coeff = _SPECTRL2_COEFFS['mixed_absorption'][:, np.newaxis]

    # ET spectral irradiance correction for earth-sun distance seasonality.
    # Note that we only want the distance correction coefficient, so set
    # solar_constant=1:
    earth_sun_distance_correction = \
        pvlib.irradiance.get_extra_radiation(dayofyear, method='spencer',
                                             solar_constant=1)  # Eq 2-2, 2-3
    # Rayleigh scattering
    # note: 101300 is used for consistentcy with reference; can't use
    # atmosphere.get_absolute_airmass because it uses 101325
    airmass = relative_airmass * surface_pressure / 101300
    wavelength_um = wavelength / 1000
    rayleigh_transmittance = np.exp(
        # Note: the report uses 1.335 but spectrl2_2.c uses 1.3366
        # -airmass / (wavelength_um**4 * (115.6406 - 1.335 / wavelength_um**2))
        -airmass / (wavelength_um**4 * (115.6406 - 1.3366 / wavelength_um**2))
    )  # Eq 2-4

    # Aerosol scattering and absorption, Eq 2-6
    aerosol_transmittance = np.exp(-optical_thickness * relative_airmass)

    # Water vapor absorption, Eq 2-8
    aWM = vapor_coeff * precipitable_water * relative_airmass
    vapor_transmittance = np.exp(-0.2385 * aWM / (1 + 20.07 * aWM)**0.45)

    # Ozone absorption
    ozone_max_height = 22
    h0_norm = ozone_max_height / 6370
    ozone_mass_numerator = (1 + h0_norm)
    ozone_mass_denominator = np.sqrt(cosd(apparent_zenith)**2 + 2 * h0_norm)
    ozone_mass = ozone_mass_numerator / ozone_mass_denominator  # Eq 2-10
    ozone_transmittance = np.exp(-ozone_coeff * ozone * ozone_mass)  # Eq 2-9

    # Mixed gas absorption, Eq 2-11
    aM = mixed_coeff * airmass
    # Note: the report uses 118.93, but spectrl2_2.c uses 118.3
    # mixed_transmittance = np.exp(-1.41 * aM / (1 + 118.93 * aM)**0.45)
    mixed_transmittance = np.exp(-1.41 * aM / (1 + 118.3 * aM)**0.45)

    # split out aerosol components for diffuse irradiance calcs
    aerosol_scattering = np.exp(
        -scattering_albedo * optical_thickness * relative_airmass
    )  # Eq 3-9

    aerosol_absorption = np.exp(
        -(1 - scattering_albedo) * optical_thickness * relative_airmass
    )  # Eq 3-10

    return (
        earth_sun_distance_correction,
        rayleigh_transmittance,
        aerosol_transmittance,
        vapor_transmittance,
        ozone_transmittance,
        mixed_transmittance,
        aerosol_scattering,
        aerosol_absorption,
    )



def spectrl2(apparent_zenith, aoi, surface_tilt, ground_albedo,
             surface_pressure, relative_airmass, precipitable_water, ozone,
             aerosol_turbidity_500nm, 
             alpha, scattering_albedo,
             spectrally_averaged_aerosol_asymmetry_factor, dayofyear=None,):
    """
    Estimate spectral irradiance using the Bird Simple Spectral Model
    (SPECTRL2).

    The Bird Simple Spectral Model [1]_ produces terrestrial spectra between
    300 and 4000 nm with a resolution of approximately 10 nm. Direct and
    diffuse spectral irradiance are modeled for horizontal and tilted surfaces
    under cloudless skies. SPECTRL2 models radiative transmission, absorption,
    and scattering due to atmospheric aerosol, water vapor, and ozone content.

    Parameters
    ----------
    apparent_zenith : float, numpy.array of floats or pandas.Series of floats
        Solar zenith angle  of the sun [degrees].
        
    aoi : float, numpy.array of floats or pandas.Series of floats
        Angle of incidence of the solar vector on the panel [degrees]. If it 
        is a numpy.array or a pandas.Series, it must be the same length as
        'apparent_zenith' and, additionally, if it is a pandas.Series, it
        should have the same index as 'apparent_zenith'.
        
    surface_tilt : float, numpy.array of floats or pandas.Series of floats
        Panel tilt from horizontal [degrees]. If it is a numpy.array or a
        pandas.Series, it must be the same length as 'apparent_zenith' and,
        additionally, if it is a pandas.Series, it should have the same index 
        as 'apparent_zenith'.
        
    ground_albedo : float or 2D-numpy.array of floats
        Albedo [0-1] of the ground surface. Can be provided as a scalar value
        if albedo is not spectrally-dependent, or as a 122xN matrix where
        the first dimension spans the wavelength range and the second one spans
        the number of simulations (i.e, length of 'apparent zenith')
        [unitless].
        
    surface_pressure : float, numpy.array of floats or pandas.Series of floats
        Surface pressure [Pa]. If it is a numpy.array or a pandas.Series, 
        it must be the same length as 'apparent_zenith' and, additionally, if
        it is a pandas.Series, it should have the same index as 
        'apparent_zenith'.
        
    relative_airmass : float, numpy.array of floats or pandas.Series of floats
        Relative airmass [unitless]. If it is a numpy.array or a pandas.Series, 
        it must be the same length as 'apparent_zenith' and, additionally, if
        it is a pandas.Series, it should have the same index as 
        'apparent_zenith'.
        
    precipitable_water : float, numpy.array of floats or pandas.Series of floats
        Atmospheric water vapor content [cm]. If it is a numpy.array or a
        pandas.Series, it must be the same length as 'apparent_zenith' and,
        additionally, if it is a pandas.Series, it should have the same index 
        as 'apparent_zenith'.
        
    ozone : float, numpy.array of floats or pandas.Series of floats
        Atmospheric ozone content [atm-cm]. If it is a numpy.array or a
        pandas.Series, it must be the same length as 'apparent_zenith' and,
        additionally, if it is a pandas.Series, it should have the same index 
        as 'apparent_zenith'.
        
    aerosol_turbidity_500nm : float, numpy.array of floats or pandas.Series of floats
        Aerosol turbidity at 500 nm [unitless]. If it is a numpy.array or a
        pandas.Series, it must be the same length as 'apparent_zenith' and,
        additionally, if it is a pandas.Series, it should have the same index 
        as 'apparent_zenith'.
        
    scattering_albedo : 2D-numpy.array of floats
        Aerosol single scattering albedo at multiple wavelengths. It is matrix 
        of size Nx122 where the second dimension spans the wavelength range and
        the  first one spans the number of simulations (i.e, length of 
        'apparent zenith') [unitless]. That is, it should be a matrix
        of size matrix.
        
    alpha : float, numpy.array of floats or pandas.Series of floats
        Angstrom turbidity exponent at 500nm [unitless]. If it is a numpy.array or a
        pandas.Series, it must be the same length as 'apparent_zenith' and,
        additionally, if it is a pandas.Series, it should have the same index 
        as 'apparent_zenith'.
        
    spectrally_averaged_aerosol_asymmetry_factor : float, numpy.array of floats or pandas.Series of floats
        Average across selected range of wavelengths of the Aerosol asymmetry 
        factor (mean cosine of scattering angle) [unitless]. If it is a 
        numpy.array or a pandas.Series, it must be the same length as 
        'apparent_zenith' and, additionally, if it is a pandas.Series, 
        it should have the same index as 'apparent_zenith'.
        
    dayofyear : float or numpy.array of floats, optional.
        The day of year [1-365].  Must be provided if `apparent_zenith` is
        not a pandas Series.

    Returns
    -------
    spectra : dict
        A dict of arrays.  With the exception of `wavelength`, which has length
        122, each array has shape (122, N) where N is the length of the
        input ``apparent_zenith``.  All values are spectral irradiance
        with units W/m^2/nm except for `wavelength`, which is in nanometers.

            * wavelength         : Wavelengths in nanometers.
            * dni_extra          : Direct normal extraterrestrial solar spectrum.
            * dhi                : Diffuse horizontal irradiance.
            * dni                : Direct normal irradiance.
            * poa_sky_diffuse    : Diffuse irradiance from the sky on a tilted surface. 
            * poa_ground_diffuse : Diffuse irradiance from the ground reflections on a tilted surface. 
            * poa_direct         : Direct irradiance on tilted surface.
            * poa_global         : Global irradiance on a tilted surface.

    Notes
    -----
    NREL's C implementation ``spectrl2_2.c`` [2]_ of the model differs in
    several ways from the original report [1]_.  The report itself also has
    a few differences between the in-text equations and the code appendix.
    The list of known differences is shown below.  Note that this
    implementation follows ``spectrl2_2.c``.

    =================== ========== ========== ===============
    Equation            Report     Appendix   spectrl2_2.c
    =================== ========== ========== ===============
    2-4                 1.335      1.335      1.3366
    2-11                118.93     118.93     118.3
    3-8                 To'        Tu'        Tu'
    3-5, 3-6, 3-7, 3-1  double Cs  single Cs  single Cs
    2-5                 kasten1966 kasten1966 kastenyoung1989
    =================== ========== ========== ===============

    This implementation also deviates from the reference by including a
    check for angles of incidence greater than 90 degrees; without this,
    the model might return negative spectral irradiance values when the
    sun is behind the plane of array.

    References
    ----------
    .. [1] Bird, R, and Riordan, C., 1984, "Simple solar spectral model for
       direct and diffuse irradiance on horizontal and tilted planes at the
       earth's surface for cloudless atmospheres", NREL Technical Report
       TR-215-2436 doi:10.2172/5986936.
    .. [2] Bird Simple Spectral Model: spectrl2_2.c.
       https://www.nrel.gov/grid/solar-resource/spectral.html
    """
    # values need to be np arrays for broadcasting, so unwrap Series if needed:
    is_pandas = isinstance(apparent_zenith, pd.Series)
    if is_pandas:
        original_index = apparent_zenith.index
        (apparent_zenith, aoi, surface_tilt, ground_albedo, surface_pressure,
         relative_airmass, precipitable_water, ozone, aerosol_turbidity_500nm,
         scattering_albedo, alpha, spectrally_averaged_aerosol_asymmetry_factor) = \
            tuple(map(np.asanyarray, [
                apparent_zenith, aoi, surface_tilt, ground_albedo,
                surface_pressure, relative_airmass, precipitable_water, ozone,
                aerosol_turbidity_500nm, scattering_albedo, alpha,
                spectrally_averaged_aerosol_asymmetry_factor]))

        dayofyear = original_index.dayofyear.values

    if not is_pandas and dayofyear is None:
        raise ValueError('dayofyear must be specified if not using pandas '
                         'Series inputs')

    # add a dimension so that each ndarray is 2d with shape (122, 1)
    wavelength = _SPECTRL2_COEFFS['wavelength'][:, np.newaxis]
    spectrum_et = _SPECTRL2_COEFFS['spectral_irradiance_et'][:, np.newaxis]

    optical_thickness = \
        pvlib.atmosphere.angstrom_aod_at_lambda(aod0=aerosol_turbidity_500nm,
                                                lambda0=500, alpha=alpha,
                                                lambda1=wavelength)  # Eq 2-7



    spectrl2 = _spectrl2_transmittances(apparent_zenith, relative_airmass,
                                        surface_pressure, precipitable_water,
                                        ozone, optical_thickness,
                                        scattering_albedo, dayofyear)
    D, Tr, Ta, Tw, To, Tu, Tas, Taa = spectrl2

    spectrum_et_adj = spectrum_et * D
    # spectrum of direct irradiance, Eq 2-1
    Id = spectrum_et_adj * Tr * Ta * Tw * To * Tu

    cosZ = cosd(apparent_zenith)
    # Eq 3-17
    Cs = np.where(wavelength <= 450, ((wavelength + 550)/1000)**1.8, 1.0)
    ALG = np.log(1 - spectrally_averaged_aerosol_asymmetry_factor)  # Eq 3-14
    BFS = ALG * (0.0783 + ALG * (-0.3824 - ALG * 0.5874))  # Eq 3-13
    AFS = ALG * (1.459 + ALG * (0.1595 + ALG * 0.4129))  # Eq 3-12
    Fs = 1 - 0.5 * np.exp((AFS + BFS * cosZ) * cosZ)  # Eq 3-11
    Fsp = 1 - 0.5 * np.exp((AFS + BFS / 1.8) / 1.8)  # Eq 3.15

    # evaluate the "primed terms" -- transmittances evaluated at airmass=1.8
    primes = _spectrl2_transmittances(apparent_zenith, 1.8,
                                      surface_pressure, precipitable_water,
                                      ozone, optical_thickness,
                                      scattering_albedo, dayofyear)
    _, Trp, Tap, Twp, Top, Tup, Tasp, Taap = primes

    # Note: not sure what the correct form of this equation is.
    # The first coefficient is To' in Eq 3-8 but Tu' in the code appendix.
    # spectrl2_2.c uses Tu'.
    sky_reflectivity = (
        # Top * Twp * Taap * (0.5 * (1-Trp) + (1-Fsp) * Trp * (1-Tasp))
        Tup * Twp * Taap * (0.5 * (1-Trp) + (1-Fsp) * Trp * (1-Tasp))
    )  # Eq 3-8

    # a common factor for 3-5 and 3-6
    common_factor = spectrum_et_adj * cosZ * To * Tu * Tw * Taa
    # Note: spectrl2_2.c differs from the report in how the Cs value is used.
    # The two commented out lines match the report, while the following match
    # spectrl2_2.c. With regard to Cs, the equations in the report and
    # spectrl12_2.c are algebraically equivalent.
    # Ir = common_factor * (1 - Tr**0.95) * 0.5 * Cs  # Eq 3-5
    # Ia = common_factor * Tr**1.5 * (1 - Tas) * Fs * Cs  # Eq 3-6
    Ir = common_factor * (1 - Tr**0.95) * 0.5  # Eq 3-5
    Ia = common_factor * Tr**1.5 * (1 - Tas) * Fs  # Eq 3-6

    rs = sky_reflectivity
    rg = ground_albedo
    Ig = (Id * cosZ + Ir + Ia) * rs * rg / (1 - rs * rg)  # Eq 3-7

    # total scattered irradiance
    # Note: see discussion about Cs above.
    # Is = Ir + Ia + Ig  # Eq 3-1
    Is = (Ir + Ia + Ig) * Cs  # Eq 3-1

    # calculate spectral irradiance on a tilted surface, Eq 3-18
    # Note: clipping cosd(aoi) to >=0 is not in the reference, but is necessary
    # to prevent nonsense values when the sun is behind the plane of array.
    # The same constraint is applied in irradiance.haydavies when not
    # supplying `projection_ratio`.
    aoi_projection_nn = np.maximum(cosd(aoi), 0)  # GH 1348
    Ibeam = Id * aoi_projection_nn

    # don't need surface_azimuth if we provide projection_ratio.
    # Also constrain cos zenith to avoid blowup, as in irradiance.haydavies
    projection_ratio = aoi_projection_nn / np.maximum(cosZ, 0.01745)
    Isky = pvlib.irradiance.haydavies(surface_tilt=surface_tilt,
                                      surface_azimuth=None,
                                      dhi=Is,
                                      dni=Id,
                                      dni_extra=spectrum_et_adj,
                                      projection_ratio=projection_ratio)

    ghi = Id * cosZ + Is
    Iground = pvlib.irradiance.get_ground_diffuse(surface_tilt, ghi, albedo=rg)

    Itilt = Ibeam + Isky + Iground
    wavelength_1d = wavelength.reshape(-1)  # only needs 1 dimension
    return {
        'wavelength': wavelength_1d,
        'dni_extra': spectrum_et_adj,
        'dhi': Is,
        'dni': Id,
        'poa_sky_diffuse': Isky,
        'poa_ground_diffuse': Iground,
        'poa_direct': Ibeam,
        'poa_global': Itilt,
    }





def compute_direct_and_diffuse_normalized_spectra(sun_apzen, SP, rel_airmass, H2O, O3, aod_500nm, alpha_500nm, 
                                                  single_scattering_albedo, spectrally_averaged_aaf, 
                                                  ground_albedo = 0, mean_surface_tilt = 0, dayofyear = None):
    
    """
    Estimate spectral irradiance using the Bird Simple Spectral Model
    (pvlib's SPECTRL2) and then return the normalized spectra for direct/beam
    and diffuse irradiance.

    Parameters
    ----------
    sun_apzen : float, numpy.array of floats or pandas.Series of floats of length N
        Solar zenith angle of the sun [degrees].
                
    SP : float, numpy.array of floats or pandas.Series of floats of length N
        Surface Pressure [Pa]. If it is a pandas.Series, it should have the same index as 
        *apzen*.
        
    rel_airmass : float, numpy.array of floats or pandas.Series of floats of length N
        Relative airmass [unitless]. If it is a pandas.Series, it should have the same 
        index as *apzen*.
        
    H2O : float, numpy.array of floats or pandas.Series of floats of length N
        Atmospheric water vapor content [cm]. If it is a pandas.Series, 
        it should have the same index as *apzen*.
        
    O3 : float, numpy.array of floats or pandas.Series of floats of length N
        Atmospheric ozone content [atm-cm]. If it is a pandas.Series, 
        it should have the same index as *apzen*.
        
    aod_500nm : float, numpy.array of floats or pandas.Series of floats of length N
        Aerosol turbidity at 500 nm [unitless]. If it is a pandas.Series, 
        it should have the same index as *apzen*.
        
    alpha_500nm : float, numpy.array of floats or pandas.Series of floats of length N
        Angstrom turbidity exponent at 500nm [unitless]. If it is a pandas.Series, 
        it should have the same index as *apzen*.
        
    single_scattering_albedo : 2D-numpy.array of floats with shape (N,122)
        Aerosol single scattering albedo at multiple wavelengths. It is matrix 
        of size Nx122 where the second dimension spans the wavelength range and
        the first one spans the number of simulations (i.e, length of 
        *apzen*) [unitless]. 
        
    spectrally_averaged_aerosol_asymmetry_factor : float, numpy.array of floats or pandas.Series of floats of length N
        Average across selected range of wavelengths of the Aerosol asymmetry 
        factor (mean cosine of scattering angle) [unitless]. If it is a pandas.Series, 
        it should have the same index as *apzen*.
        
    ground_albedo : float or 2D-numpy.array of floats with shape (N,122)
        Albedo [0-1] of the ground surface. Can be provided as a scalar value
        if albedo is not spectrally-dependent, or as a Nx122 matrix where
        the second dimension spans the wavelength range and the first one spans
        the number of simulations (i.e, length of *apzen*).
        [unitless]. Default is 0.
        
    mean_surface_tilt : float, numpy.array of floats or pandas.Series of floats of length N
        Mean panel tilt from horizontal [degrees]. IIf it is a pandas.Series, 
        it should have the same index as *apzen*.
        
    dayofyear : float or numpy.array of floats of length N, optional.
        The day of year [1-365].  Must be provided if `apzen` is
        not a pandas Series. Default is None.
        
        
    Returns
    -------
    res : dict
        Dictionary containing result variables. It has the following Key-Value
        pairs:
            
            Keys : Values
            -------------
        
            "direct" : numpy.array of floats with shape (122,) or 2D numpy.array of floats with (N, 122)
                Normalized spectrum of direct/beam irradiance. The shape of the output
                depends on the shape of the inputs. If *apzen* is a single value,
                *res["direct"]* will have shape (122,). Else, it will have shape
                (N, 122) where N is the length of *apzen*. The 122 makes reference to 
                the number of wavelengths at which the results are given.
    
            "diffuse" : numpy.array of floats with shape (122,) or 2D numpy.array of floats with (N, 122)   
                Normalized spectrum of diffuse irradiance. The shape of the output
                depends on the shape of the inputs. If *apzen* is a single value,
                *res["diffuse"]* will have shape (122,). Else, it will have shape
                (N, 122) where N is the length of *apzen*. The 122 makes reference to 
                the number of wavelengths at which the results are given.
                
            "wavelengths" : numpy.array of floats with shape (122,)
                Wavelengths in nanometers, at which the normalized spectra are 
                given.

    
    Raises
    ------
    1) TypeError : 
        The following TypeError was raised: 'loop of ufunc does not support argument 0 of type 
        float which has no callable exp method'. This usually happens when *single_scattering_albedo*
        is not an numpy.array of floats. Attempt to pass ``numpy.array(single_scattering_albedo).astype(float)`` instead."
        
            
    Notes
    -----
    1) The *mean_surface_tilt* argument really only affects the computation of
       the spectral distribution of diffuse radiance. It has no effect on 
       the actual value. 
                
            
    """
    
    # We compute the transpose of ground albedo if ground albedo is a numpy.array.
    # This is because requires that the shape for ground albedo be 122xN, where N
    # is the number of simulations (length of 'apzen').

    if not isinstance(ground_albedo, int) or isinstance(ground_albedo, float):
        ground_albedo_ = ground_albedo.T
    else:
        ground_albedo_ = ground_albedo
        
    
    # Compute Beam and diffuse irradiance spectra.
    try:
        res =\
        spectrl2(apparent_zenith                             = sun_apzen, 
                aoi                                          = 0, 
                surface_tilt                                 = mean_surface_tilt,
                ground_albedo                                = ground_albedo_,
                surface_pressure                             = SP,
                relative_airmass                             = rel_airmass,
                precipitable_water                           = H2O, 
                ozone                                        = O3, 
                aerosol_turbidity_500nm                      = aod_500nm, 
                alpha                                        = alpha_500nm, 
                scattering_albedo                            = single_scattering_albedo.T,
                spectrally_averaged_aerosol_asymmetry_factor = spectrally_averaged_aaf,
                dayofyear                                    = dayofyear)

    except TypeError as e:
        msg_trigger = "loop of ufunc does not support argument 0 of type float which has no callable exp method"
        if msg_trigger in str(e):
            msg = f"The following TypeError was raised: '{e}'"
            msg = f"{msg}. This usually happens when 'single_scattering_albedo' is not an numpy.array of floats"
            msg = f"{msg}. Attempt to pass 'numpy.array(single_scattering_albedo).astype(float)' instead."
            raise TypeError(msg)
        else:
            raise TypeError(e)


        
    
    # Normalize the computed spectra and return them.
        
    Gb_spectrum = res["dni"].T
    Gb_normalization_factor =\
    simpson(y = Gb_spectrum,
            x = _SPECTRL2_COEFFS['wavelength'],
            axis = 1).reshape(Gb_spectrum.shape[0], 1)
    Gb_normalized_spectrum = Gb_spectrum / Gb_normalization_factor
    
    
    Gd_spectrum = res["dhi"].T
    Gd_normalization_factor =\
    simpson(y = Gd_spectrum,
            x = _SPECTRL2_COEFFS['wavelength'],
            axis = 1).reshape(Gd_spectrum.shape[0], 1)
    Gd_normalized_spectrum = Gd_spectrum / Gd_normalization_factor
    
    
    # We package the results and return them.
    res = {"direct"      : Gb_normalized_spectrum,
           "diffuse"     : Gd_normalized_spectrum,
           "wavelengths" : np.array(_SPECTRL2_COEFFS['wavelength'])}
    
    
    return res
    





    






