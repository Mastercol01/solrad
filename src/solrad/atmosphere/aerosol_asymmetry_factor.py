#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of the aerosol asymmetry factor of a site.
"""

#%%                  IMPORTATION OF LIBRARIES

import numpy as np
from scipy.interpolate import griddata

#%%            DEFINITION OF CONSTANTS FOR SHETTEL AND FENN MODEL

# We obtain these values for the aerosol asymmetry factor from the
# Shettel & Fenn Aerosol model established in the paper:
# "Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the
# Lower Atmosphere and the Effects of Humidity Variations on their Optical
# Properties. Environ. Res.. 94."

_RELATIVE_HUMIDITIES_SF = [0, 50, 70, 80, 90, 95, 98, 99]
_WAVELENGTHS_SF = np.array([0.200, 0.300, 0.337, 0.550, 0.694, 1.060, 1.536, 2.000,
                            2.250, 2.500, 2.700, 3.000, 3.392, 3.750, 4.500])*1e3

_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF = np.meshgrid(_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF)
_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF = _WAVELENGTHS_SF.flatten(), _RELATIVE_HUMIDITIES_SF.flatten()

_AEROSOL_ASYMMETRY_FACTOR_SF = {}
_AEROSOL_ASYMMETRY_FACTOR_SF["Rural"] =\
np.array([0.7581, 0.6785, 0.6712, 0.6479, 0.6342, 0.6176, 0.6334, 0.7063, 0.7271, 0.7463, 0.7788, 0.7707, 0.7424, 0.7312, 0.7442,
          0.7598, 0.6836, 0.6762, 0.6534, 0.6392, 0.6220, 0.6365, 0.7070, 0.7277, 0.7478, 0.7821, 0.7759, 0.7407, 0.7311, 0.7460,
          0.7632, 0.6928, 0.6865, 0.6630, 0.6498, 0.6314, 0.6440, 0.7098, 0.7303, 0.7522, 0.7953, 0.7816, 0.7380, 0.7310, 0.7530,
          0.7725, 0.7240, 0.7197, 0.6997, 0.6850, 0.6650, 0.6702, 0.7181, 0.7378, 0.7651, 0.8168, 0.7661, 0.7286, 0.7336, 0.7654,
          0.7759, 0.7505, 0.7473, 0.7311, 0.7178, 0.6934, 0.6877, 0.7167, 0.7338, 0.7638, 0.8252, 0.7289, 0.7101, 0.7248, 0.7662,
          0.7758, 0.7606, 0.7581, 0.7438, 0.7312, 0.7062, 0.6956, 0.7168, 0.7322, 0.7625, 0.8276, 0.7130, 0.7014, 0.7198, 0.7848,
          0.7773, 0.7726, 0.7712, 0.7612, 0.7508, 0.7307, 0.7227, 0.7346, 0.7533, 0.7853, 0.8590, 0.7444, 0.7222, 0.7366, 0.7862,
          0.7778, 0.7798, 0.7786, 0.7717, 0.7628, 0.7444, 0.7365, 0.7491, 0.7609, 0.7921, 0.8688, 0.7577, 0.7294, 0.7413, 0.7920])


_AEROSOL_ASYMMETRY_FACTOR_SF["Urban"] =\
np.array([0.7785, 0.7182, 0.7067, 0.6617, 0.6413, 0.6166, 0.6287, 0.6883, 0.707,  0.7243, 0.737,  0.7446, 0.7391, 0.7371, 0.7414,
          0.7811, 0.7236, 0.7126, 0.6685, 0.6482, 0.6231, 0.6342, 0.6919, 0.7106, 0.7286, 0.7440, 0.7453, 0.7388, 0.7392, 0.7455,
          0.7906, 0.7476, 0.7385, 0.6998, 0.6803, 0.6536, 0.6590, 0.7066, 0.7250, 0.7484, 0.7769, 0.7406, 0.7351, 0.7459, 0.7625,
          0.7849, 0.7713, 0.7650, 0.7342, 0.7162, 0.6873, 0.6820, 0.7131, 0.7312, 0.7563, 0.8030, 0.7171, 0.7125, 0.7400, 0.7698,
          0.7924, 0.7847, 0.7807, 0.7574, 0.7417, 0.7120, 0.6999, 0.7115, 0.7330, 0.7619, 0.8183, 0.6984, 0.7031, 0.7301, 0.7696,
          0.7883, 0.7922, 0.7897, 0.7730, 0.7597, 0.7328, 0.7160, 0.7241, 0.7365, 0.7650, 0.8287, 0.6912, 0.6946, 0.7224, 0.7678,
          0.7834, 0.7974, 0.7966, 0.7874, 0.7775, 0.7550, 0.7371, 0.7376, 0.7461, 0.7724, 0.8415, 0.6924, 0.6951, 0.7195, 0.7682,
          0.7814, 0.7993, 0.7995, 0.7948, 0.7870, 0.7682, 0.7751, 0.7501, 0.7565, 0.7609, 0.8516, 0.7137, 0.7039, 0.7241, 0.7728])


_AEROSOL_ASYMMETRY_FACTOR_SF["Maritime"] =\
np.array([0.7516, 0.6960, 0.692,  0.6756, 0.6767, 0.6844, 0.6936, 0.7055, 0.7112, 0.7177, 0.7367, 0.6287, 0.6779, 0.6784, 0.6599,
          0.7593, 0.7065, 0.7005, 0.6915, 0.6907, 0.698,  0.7105, 0.7227, 0.7289, 0.738,  0.7642, 0.6953, 0.6907, 0.6919, 0.6808,
          0.7788, 0.7288, 0.7242, 0.7214, 0.7211, 0.733,  0.7445, 0.7579, 0.7649, 0.779,  0.8192, 0.7671, 0.7171, 0.7275, 0.7235,
          0.7954, 0.7782, 0.7752, 0.7717, 0.7721, 0.7777, 0.7872, 0.8017, 0.8089, 0.8301, 0.8844, 0.8332, 0.7557, 0.7597, 0.7823,
          0.8007, 0.7922, 0.7917, 0.7865, 0.7847, 0.7847, 0.7974, 0.806,  0.8147, 0.8387, 0.8966, 0.8466, 0.7644, 0.7671, 0.7949,
          0.8079, 0.8035, 0.8034, 0.7986, 0.7955, 0.7927, 0.7994, 0.8117, 0.8193, 0.8444, 0.9068, 0.8525, 0.7778, 0.7736, 0.8056,
          0.8146, 0.8182, 0.8167, 0.8103, 0.8082, 0.8011, 0.8051, 0.8167, 0.8237, 0.8495, 0.9167, 0.8816, 0.7862, 0.7802, 0.8168,
          0.8203, 0.827,  0.826,  0.8196, 0.8176, 0.8096, 0.8196, 0.8202, 0.8255, 0.852,  0.9228, 0.895,  0.7965, 0.7847, 0.8242])



#%%                 DEFINITION OF FUNCTIONS


def compute_aerosol_asymmetry_factor_using_SF(RH, wavelength, model, interp_method = "linear"):

    """
    Compute the aerosol asymmetry factor using the Shettel and Fenn (SF) model
    as described in [1].

    Parameters
    ----------
    RH : float or array_like of floats
      Relative Humidity of the air in %. Must be a non-negative number or array
      of numbers between 0 and 100.

    wavelength : float or array_like of floats
       Wavelength in nanometers for which the aerosol asymmetry factor
       is to be computed. Must be a number or array of numbers between 200
       and 4500.

    model : {“Rural”, “Urban” and "Maritime"}
       Model to be used in the computation of the Aerosol Asymmetry Factor.

    interp_method : {"linear", "nearest", "cubic"}, optional
        Method of interpolation to use on the data. Default is "linear".

    Returns
    -------
    aerosol_asymmetry_factor : numpy.array of floats
      Aerosol asymmetry factor.

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
    >>> from atmosphere.aerosol_asymmetry_factor import compute_aerosol_asymmetry_factor_using_SF
    >>>
    >>> wavelengths = np.linspace(280, 4000, 1000)
    >>> RHs = np.array([0, 50, 70, 80, 90, 95, 98, 99])
    >>>
    >>> for model in ["Rural", "Urban", "Maritime"]:
    >>>
    >>>       data = np.zeros((len(RHs), len(wavelengths)))
    >>>       fig = plt.figure(figsize=(15,10))
    >>>
    >>>       for i, RH in enumerate(RHs):
    >>>           data[i,:] = compute_aerosol_asymmetry_factor_using_SF(RH, wavelengths, model, interp_method="linear")
    >>>           plt.plot(wavelengths, data[i,:], label = f"{RH}%")
    >>>
    >>>       plt.xlim(280, 4000)
    >>>       plt.xlabel("Wavelengths [nm]")
    >>>       plt.ylabel("Aerosol Asymmetry Factor [-]")
    >>>       plt.title(f"Model= {model}")
    >>>       plt.legend(title = "Relative Humidity")
    >>>       plt.grid()
    >>>       plt.show()
    """

    points = np.stack([_WAVELENGTHS_SF, _RELATIVE_HUMIDITIES_SF], axis=1)
    values = _AEROSOL_ASYMMETRY_FACTOR_SF[model]

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
    aerosol_asymmetry_factor = griddata(points=points,
                                        values=values,
                                        xi=xi,
                                        method=interp_method)

    return aerosol_asymmetry_factor


