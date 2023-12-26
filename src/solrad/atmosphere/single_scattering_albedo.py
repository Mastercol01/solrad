#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of the Single-Scattering Albedo of a site.
"""
#%%                  IMPORTATION OF LIBRARIES

import numpy as np
from scipy.interpolate import griddata


#%%            DEFINITION OF CONSTANTS FOR SHETTEL AND FENN MODEL

# We obtain these values for the single scattering albedo from the
# Shettel & Fenn Aerosol model established in the paper:
# "Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the
# Lower Atmosphere and the Effects of Humidity Variations on their Optical
# Properties. Environ. Res.. 94."


_RELATIVE_HUMIDITIES_SF = [0, 50, 70, 80, 90, 95, 98, 99]
_WAVELENGTHS_SF = np.array([0.200, 0.300, 0.337, 0.550, 0.694, 1.060, 1.536, 2.000,
                         2.250, 2.500, 2.700, 3.000, 3.392, 3.750, 4.500])*1e3

_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF = np.meshgrid(_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF)
_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF = _WAVELENGTHS_SF.flatten(), _RELATIVE_HUMIDITIES_SF.flatten()

_SINGLE_SCATTERING_ALBEDO_SF = {}
_SINGLE_SCATTERING_ALBEDO_SF["Rural"] =\
np.array([0.6789, 0.9316, 0.9470, 0.9407, 0.9315, 0.8613, 0.7920, 0.8661, 0.8442, 0.8201, 0.5773, 0.7513, 0.8778, 0.9127, 0.8410,
          0.6862, 0.9341, 0.9487, 0.9427, 0.9336, 0.8653, 0.7978, 0.8751, 0.8402, 0.8324, 0.5757, 0.6459, 0.8662, 0.9127, 0.8395,
          0.6995, 0.9379, 0.9521, 0.9462, 0.9378, 0.8714, 0.8144, 0.8769, 0.8603, 0.8423, 0.5821, 0.5060, 0.8509, 0.9161, 0.8369,
          0.7494, 0.9517, 0.9632, 0.9592, 0.9531, 0.9039, 0.8571, 0.9079, 0.8989, 0.8765, 0.6060, 0.3261, 0.8062, 0.9271, 0.8277,
          0.8034, 0.9657, 0.9740, 0.9720, 0.9682, 0.9343, 0.9013, 0.9307, 0.9285, 0.9010, 0.6176, 0.2711, 0.7781, 0.9256, 0.8154,
          0.8274, 0.9713, 0.9785, 0.9772, 0.9743, 0.9471, 0.9200, 0.9397, 0.9418, 0.9113, 0.6225, 0.2612, 0.7689, 0.9262, 0.8097,
          0.8507, 0.9767, 0.9830, 0.9829, 0.9811, 0.9624, 0.9652, 0.9484, 0.9593, 0.9289, 0.6457, 0.2848, 0.7465, 0.9229, 0.8220,
          0.8679, 0.9808, 0.9861, 0.9866, 0.9855, 0.9716, 0.9589, 0.9509, 0.9666, 0.9283, 0.6480, 0.2969, 0.7339, 0.9176, 0.7915])


_SINGLE_SCATTERING_ALBEDO_SF["Urban"] =\
np.array([0.5886, 0.6389, 0.6426, 0.6382, 0.6243, 0.5565, 0.4787, 0.4237, 0.4095, 0.3991, 0.3491, 0.3787, 0.4076, 0.4155, 0.4097,
          0.5947, 0.6488, 0.6525, 0.6484, 0.6345, 0.5672, 0.4893, 0.4361, 0.4220, 0.4187, 0.3557, 0.3579, 0.4203, 0.4292, 0.4207,
          0.6471, 0.7008, 0.7049, 0.7026, 0.6899, 0.6270, 0.5542, 0.5080, 0.4937, 0.4777, 0.3950, 0.2950, 0.4867, 0.5055, 0.4835,
          0.7208, 0.7738, 0.7706, 0.7805, 0.7710, 0.7197, 0.6565, 0.6158, 0.6015, 0.5782, 0.4517, 0.2503, 0.5650, 0.6081, 0.5648,
          0.7803, 0.8311, 0.8362, 0.8422, 0.8366, 0.7993, 0.7900, 0.7146, 0.7030, 0.6747, 0.5807, 0.2506, 0.6262, 0.6986, 0.6342,
          0.8241, 0.8713, 0.8765, 0.8852, 0.8826, 0.8581, 0.8228, 0.7928, 0.7861, 0.7552, 0.5529, 0.2564, 0.6697, 0.7697, 0.6862,
          0.8662, 0.9080, 0.9130, 0.9240, 0.9240, 0.9120, 0.8928, 0.8674, 0.8691, 0.8357, 0.5978, 0.2759, 0.7056, 0.8363, 0.7298,
          0.8869, 0.9253, 0.9302, 0.9421, 0.9441, 0.9378, 0.9256, 0.9002, 0.9004, 0.8716, 0.6150, 0.2950, 0.7165, 0.8628, 0.7423])


_SINGLE_SCATTERING_ALBEDO_SF["Maritime"] =\
np.array([0.7900, 0.9672, 0.9772, 0.9820, 0.9833, 0.9749, 0.9720, 0.9823, 0.9727, 0.9562, 0.9055, 0.9211, 0.9778, 0.9846, 0.9800,
          0.8014, 0.9697, 0.9789, 0.9835, 0.9849, 0.9776, 0.9750, 0.9826, 0.9756, 0.9592, 0.8890, 0.7197, 0.9535, 0.9818, 0.9647,
          0.8293, 0.9751, 0.9829, 0.9870, 0.9884, 0.9873, 0.9817, 0.9810, 0.9818, 0.9651, 0.8541, 0.5357, 0.9097, 0.9763, 0.9350,
          0.8935, 0.9864, 0.9939, 0.9936, 0.9944, 0.9924, 0.9931, 0.9837, 0.9893, 0.9739, 0.8031, 0.4516, 0.8541, 0.9668, 0.8968,
          0.9170, 0.9907, 0.9937, 0.9955, 0.9961, 0.9946, 0.9920, 0.9799, 0.9912, 0.9775, 0.7900, 0.4423, 0.8390, 0.9670, 0.8869,
          0.9354, 0.9932, 0.9954, 0.9968, 0.9972, 0.9961, 0.9940, 0.9776, 0.9904, 0.9667, 0.7796, 0.4471, 0.8220, 0.9587, 0.8777,
          0.9556, 0.9956, 0.9971, 0.9980, 0.9982, 0.9975, 0.9946, 0.9725, 0.9896, 0.9642, 0.7644, 0.4575, 0.7980, 0.9511, 0.8637,
          0.9678, 0.9970, 0.9980, 0.9985, 0.9988, 0.9982, 0.9944, 0.9671, 0.9882, 0.9587, 0.7495, 0.4664, 0.7760, 0.9432, 0.8498])





#%%                   DEFINITION OF FUNCTIONS

def compute_single_scattering_albedo_using_SF(RH, wavelength, model, interp_method = "linear"):

    """
    Compute the single scattering albedo using the Shettel and Fenn (SF) model
    as described in [1].

    Parameters
    ----------
    RH : float or array_like of floats
      Relative Humidity of the air in %. Must be a non-negative number or array
      of numbers between 0 and 100.

    wavelength : float or array_like of floats
       Wavelength in nanometers for which the single scattering albedo
       is to be computed. Must be a number or array of numbers between 200
       and 4500.

    model : {“Rural”, “Urban”, "Maritime"}
       Model to be used in the computation of the single scattering albedo.

    interp_method : {"linear", "nearest", "cubic"}, optional
        Method of interpolation to use on the data. Default is "linear".

    Returns
    -------
    single_scattering_albedo : numpy.array of floats
      Single scattering albedo.

    Notes
    -----
    1) 'RH'' can be an array of any length, while 'wavelength' is a float;
    and viceversa. However, if 'RH' and 'wavelength' are both arrays,
    they should be the same length.

    References
    ----------
    [1] Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the
    Lower Atmosphere and the Effects of Humidity Variations on their Optical
    Properties. Environ. Res.. 94.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from Solrad.Atmosphere.single_scattering_albedo import compute_single_scattering_albedo_using_SF
    >>>
    >>> wavelengths = np.linspace(280, 4000, 1000)
    >>> RHs = np.array([0, 50, 70, 80, 90, 95, 98, 99])
    >>>
    >>> for model in ["Rural", "Urban", "Maritime"]:
    >>>
    >>>     data = np.zeros((len(RHs), len(wavelengths)))
    >>>     fig = plt.figure(figsize=(15,10))
    >>>
    >>>     for i, RH in enumerate(RHs):
    >>>         data[i,:] = compute_single_scattering_albedo_using_SF(RH, wavelengths, model, interp_method="linear")
    >>>         plt.plot(wavelengths, data[i,:], label = f"{RH}%")
    >>>
    >>>
    >>>     plt.xlim(280, 4000)
    >>>     plt.xlabel("Wavelengths [nm]")
    >>>     plt.ylabel("Single Scattering Albedo [-]")
    >>>     plt.title(f"Model= {model}")
    >>>     plt.legend(title = "Relative Humidity")
    >>>     plt.grid()
    >>>     plt.show()

    """


    points = np.stack([_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF], axis=1)
    values = _SINGLE_SCATTERING_ALBEDO_SF[model]

    # We modify the variables so they can be used with current data.
    RH0 = np.minimum(99, RH)
    lmbda = wavelength.copy()

    # The function should work both for arrays and floats.
    try: len_RH0 = len(RH0)
    except TypeError: len_RH0 = 0

    try: len_lmbda = len(lmbda)
    except TypeError: len_lmbda = 0

    if  len_RH0 == 0 and len_lmbda > 0:
      RH0 = np.full(len_lmbda, RH0)

    elif len_RH0 > 0 and len_lmbda == 0:
      lmbda = np.full(len_RH0, lmbda)

    elif len_RH0 != len_lmbda:
      raise Exception("Length of RH and length wavelength do not match")

    xi = np.stack([lmbda, RH0], axis=1)
    single_scattering_albedo = griddata(points=points,
                                        values=values,
                                        xi=xi,
                                        method=interp_method)

    return single_scattering_albedo












# %%
