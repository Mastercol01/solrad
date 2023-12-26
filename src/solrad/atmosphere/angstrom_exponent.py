#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of the Angstrom Turbidity Exponent of a site.
"""

#%%                  IMPORTATION OF LIBRARIES

import numpy as np
import pandas as pd

#%%             DEFINITION OF CONSTANTS FOR SHETTEL AND FENN MODEL

# We define the constants of the Shettel & Fenn Aerosol model for computing the
# angstrom turbidity exponent, as described in the paper "SMARTS2, a simple
# model of the atmospheric radiative transfer of sunshine: algorithms and
# performance assessment" pages 16-18.

_INDEX = ["Rural", "Urban", "Maritime"]
_COLUMNS = ["C1", "C2", "C3", "D1", "D2", "D3", "D4"]

_ANGSTROM_EXP_COEFFS_SF = pd.DataFrame(index = _INDEX , columns = _COLUMNS).astype(float)
_ANGSTROM_EXP_COEFFS_SF.loc["Rural",:]    = [0.581, 16.823, 17.539, 0.8547, 78.696, 0, 54.416]
_ANGSTROM_EXP_COEFFS_SF.loc["Urban",:]    = [0.2595, 33.843, 39.524, 1.0, 84.254, -9.1, 65.458]
_ANGSTROM_EXP_COEFFS_SF.loc["Maritime",:] = [0.1134, 0.8941, 1.0796, 0.04435, 1.6048, 0, 1.5298]


#%%               DEFINITION OF FUNCTIONS


def compute_angstrom_exponent_using_SF(RH, wavelength = 500, model = "Urban"):

    """
    Compute the Ansgtrom turbidity exponent suing the Shettel and Fenn model,
    as detailed in [1].

    Parameters
    ----------

    RH : float or array_like of floats
      Relative Humidity of the air in %. Must be a non-negative number or array
      of numbers between 0 and 100.

    wavelength : float, optional
       Wavelength in nanometers for which the Angstrom turbidity exponent
       is to be computed. Must be a non-ngeative number. Default is 500.

    model : {“Rural”, “Urban” and "Maritime"}, optional
       Model to be used in the computation of the the Angstrom exponent.

    Returns
    -------
    alpha : float or numpy.array of floats
        Angstrom turbidity coefficient.

    References
    ----------
    [1] Gueymard, Chris. (1995). SMARTS2, a simple model of the atmospheric
    radiative transfer of sunshine: algorithms and performance assessment.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from atmosphere.aerosol_asymmetry_factor import compute_angstrom_exponent_using_SF
    >>>
    >>> RHs = np.linspace(0, 100, 101)
    >>> for model in ["Rural", "Urban", "Maritime"]:

    >>>     fig = plt.figure(figsize=(15,10))
    >>>     alpha1 = compute_angstrom_exponent_using_SF(RHs, 499, model=model)
    >>>     alpha2 = compute_angstrom_exponent_using_SF(RHs, 500, model=model)
    >>>     plt.plot(RHs, alpha1, label="λ < 500 nm")
    >>>     plt.plot(RHs, alpha2, label="λ ≥ 500 nm")
    >>>     plt.grid()
    >>>     plt.xlim(0,100)
    >>>     plt.xlabel("RH [%]")
    >>>     plt.ylabel("Angstrom Exponent")
    >>>     plt.title(f"Model = {model}")
    >>>     plt.legend(title = "Spectral Range")
    >>>     plt.show()

    """

    C1, C2, C3, D1, D2, D3, D4 = _ANGSTROM_EXP_COEFFS_SF.loc[model, :]


    Xrh = np.cos(np.deg2rad(0.9*np.array(RH)))

    # According to the Shettel & Fenn model, alpha is computed differently,
    # depending on the specific spectral region it is intended for.

    if wavelength < 500:
        alpha = (C1 + C2*Xrh) / (1 + C3*Xrh)

    else:
        alpha = (D1 + D2*Xrh + D3*Xrh**2) / (1 + D4*Xrh)

    if len(alpha) == 1:
      alpha = alpha[0]

    return alpha




