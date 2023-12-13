#%%
# We import libraries
import os
import solrad
import numpy as np
from solrad.Site import Site
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from solrad.auxiliary_funcs import load_obj_with_pickle
from solrad.radiation.spectrl2 import spectrl2, compute_direct_and_diffuse_normalized_spectra

#%%
# ---- LOAD SITE_OBJ ----
# As a test, we get the atmospheric data required for this module from
# precomputed Site obj.

TESTS_PATH = os.path.join(os.path.dirname(solrad.__path__[0]), "tests")
Site_obj_path = os.path.join(TESTS_PATH, "test_Site_obj.pkl")
Site_obj = load_obj_with_pickle(path = Site_obj_path)

#%%
# Scpecify year, day and month of the data.
# year, month, day = 2023, 1, 1
year, month, day   = 2023, 1, 2


sun_apzen                = Site_obj.sun_data[(year, month, day)]["apzen"]
SP                       = Site_obj.climate_and_air_data[(year, month, day)]["SP"]
rel_airmass              = Site_obj.sun_data[(year, month, day)]["rel_airmass"]
H2O                      = Site_obj.climate_and_air_data[(year, month, day)]["H2O"]
O3                       = Site_obj.climate_and_air_data[(year, month, day)]["O3"]
aod_500nm                = Site_obj.climate_and_air_data[(year, month, day)]["aod_500nm"]
alpha_500nm              = Site_obj.climate_and_air_data[(year, month, day)]["alpha_500nm"]
single_scattering_albedo = np.array(Site_obj.single_scattering_albedo[(year, month, day)].iloc[:,1:]).astype(float).T
spectrally_averaged_aaf  = Site_obj.climate_and_air_data[(year, month, day)]["spectrally_averaged_aaf"]
ground_albedo            = 0

index = sun_apzen.index


# Compute Normalized Spectra
res =\
compute_direct_and_diffuse_normalized_spectra(sun_apzen                = sun_apzen,
                                              SP                       = SP,
                                              rel_airmass              = rel_airmass,
                                              H2O                      = H2O, 
                                              O3                       = O3, 
                                              aod_500nm                = aod_500nm,  
                                              alpha_500nm              = alpha_500nm,
                                              single_scattering_albedo = single_scattering_albedo.T,
                                              spectrally_averaged_aaf  = spectrally_averaged_aaf)


#%%     ------ PLOT NORMALIZED DIRECT SPECTRAL IRRADIANCE ------

fig = plt.figure(figsize=(16,12))
for nt in [15*i for i in range(10)]:

    plt.plot(res["wavelengths"], res["direct"][nt,:], label = index[nt])
    
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Spectrum [-]")
plt.title("Normalized Direct Spectral Irradiance")
plt.xlim(res["wavelengths"][0], res["wavelengths"][-1])
plt.show()

    
#%%    ------ PLOT NORMALIZED DIFFUSE SPECTRAL IRRADIANCE ------

fig = plt.figure(figsize=(16,12))
for nt in [15*i for i in range(10)]:

    plt.plot(res["wavelengths"], res["diffuse"][nt,:], label = index[nt])
    
plt.legend()
plt.xlabel("Wavelength [nm]")
plt.ylabel("Normalized Spectrum [-]")
plt.title("Normalized Diffuse Spectral Irradiance")
plt.xlim(res["wavelengths"][0], res["wavelengths"][-1])
plt.show()

#%%     ------- CHECKING NORMALIZATION ------
integral_values_direct = simpson(y = res["direct"],
                                    x = res["wavelengths"],
                                    axis = 1)

integral_values_diffuse = simpson(y = res["diffuse"],
                                    x = res["wavelengths"],
                                    axis = 1)

print(integral_values_direct)
print(integral_values_diffuse)
# %%
