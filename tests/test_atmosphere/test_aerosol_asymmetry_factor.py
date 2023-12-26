#%%                       IMPORTATION OF LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from solrad.atmosphere.aerosol_asymmetry_factor import *

#%%                           OTHER TESTS

# Let us check the variation of the Aerosol Asymmetry Factor with respect
# to relative humidity and wavelength, for all 3 models.

# Range of Wavelengths and Relative Humidities to try.

wavelengths = np.linspace(280, 4000, 1000)
RHs = np.array([0, 50, 70, 80, 90, 95, 98, 99])

for model in ["Rural", "Urban", "Maritime"]:

    data = np.zeros((len(RHs), len(wavelengths)))
    fig = plt.figure(figsize=(15,10))

    for i, RH in enumerate(RHs):
        data[i,:] = compute_aerosol_asymmetry_factor_using_SF(RH, wavelengths, model, interp_method="linear")
        plt.plot(wavelengths, data[i,:], label = f"{RH}%")

    plt.xlim(280, 4000)
    plt.xlabel("Wavelengths [nm]")
    plt.ylabel("Aerosol Asymmetry Factor [-]")
    plt.title(f"Model= {model}")
    plt.legend(title = "Relative Humidity")
    plt.grid()
    plt.show()
# %%
