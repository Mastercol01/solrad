
#%%   
# We import libraries
import os
import solrad
import time as tm
import numpy as np
from solrad.Site import Site
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from solrad.auxiliary_funcs import load_obj_with_pickle
from solrad.radiation.spectral_radiance_model import *

#%%
# ---- DEFINE EVALUATION POINTS FOR SKY RADIANCE DISTRIBUTION ----
angular_resolution = 2 #[degrees]
num_pts_el = round(90/angular_resolution) + 1
num_pts_az = 4*(num_pts_el - 1) + 1

az = np.linspace(0, 360, num_pts_az)
el = np.linspace(0,  90, num_pts_el)

Az, El = np.meshgrid(az, el)
Theta = np.deg2rad(Az)
dAz, dEl = 360/(num_pts_az - 1), 90/(num_pts_el - 1)
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
year, month, day = 2023, 1, 2


DatetimeIndex_obj = Site_obj.simulation_time_data[(year, month, day)]    
sun_apel  = np.array(Site_obj.sun_data[(year, month, day)]["apel"])
sun_az    = np.array(Site_obj.sun_data[(year, month, day)]["az"])

Gh          = np.array(Site_obj.climate_and_air_data[(year, month, day)]["G(h)"])
extra_Gbn   = np.array(Site_obj.climate_and_air_data[(year, month, day)]["extra_Gb(n)"])
Gbn         = np.array(Site_obj.climate_and_air_data[(year, month, day)]["Gb(n)"])
Gdh         = np.array(Site_obj.climate_and_air_data[(year, month, day)]["Gd(h)"])

SP          = np.array(Site_obj.climate_and_air_data[(year, month, day)]["SP"])
rel_airmass = np.array(Site_obj.sun_data[(year, month, day)]["rel_airmass"])
H2O         = np.array(Site_obj.climate_and_air_data[(year, month, day)]["H2O"])
O3          = np.array(Site_obj.climate_and_air_data[(year, month, day)]["O3"])

aod_500nm   = np.array(Site_obj.climate_and_air_data[(year, month, day)]["aod_500nm"])
alpha_500nm = np.array(Site_obj.climate_and_air_data[(year, month, day)]["alpha_500nm"])

spectrally_averaged_aaf = np.array(Site_obj.climate_and_air_data[(year, month, day)]["spectrally_averaged_aaf"])

single_scattering_albedo = np.array(Site_obj.single_scattering_albedo[(year, month, day)].iloc[:,1:]).astype(float)

ground_albedo = 0
mean_surface_tilt = 0
num_iterations = 500

#%%   ---- COMPUTE SPECTRAL RADIANCE DISTRIBUTION FOR ALL TIMES ----

t = tm.time()
res =\
compute_spectral_radiance(Az                          = Az,
                            El                        = El,
                            dAz                       = dAz,
                            dEl                       = dEl, 
                            DatetimeIndex_obj           = DatetimeIndex_obj,
                            sun_apel                  = sun_apel,
                            sun_az                    = sun_az,
                            Gh                        = Gh,
                            extra_Gbn                 = extra_Gbn,
                            Gbn                       = Gbn, 
                            Gdh                       = Gdh,
                            SP                        = SP, 
                            rel_airmass               = rel_airmass,
                            H2O                       = H2O, 
                            O3                        = O3, 
                            aod_500nm                 = aod_500nm,
                            alpha_500nm               = alpha_500nm, 
                            spectrally_averaged_aaf   = spectrally_averaged_aaf, 
                            single_scattering_albedo  = single_scattering_albedo,  
                            ground_albedo             = ground_albedo, 
                            mean_surface_tilt         = mean_surface_tilt, 
                            num_iterations            = num_iterations
                            )
dt = tm.time() - t
#%%      ---- PLOT DIFFUSE SPECTRAL RADIANCE DISTRIBUTION FOR ALL TIMES AT CERTAIN WAVELENGTHS ----

component = "diffuse"
wavelengths_idxs = np.array([[15, 30, 45], [50, 65, 80]])

for nt in range(len(DatetimeIndex_obj)):
    fig, axs = plt.subplots(nrows=2, ncols=3, 
    subplot_kw=dict(projection='polar'))
    fig.set_figheight(12)
    fig.set_figwidth(16)

    title = f"{Site_obj.name}: Spectral {component} radiance at time {DatetimeIndex_obj[nt]}"
    plt.suptitle(f"{title}.\n (N = 0°, E = 90°, S = 180°, W = 270°)")

    for nr, nc in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]:

        wavelength_idx = wavelengths_idxs[nr, nc]
        wavelength     = res["wavelengths"][wavelength_idx]
        Color          = res[component][nt][:,:,wavelength_idx]

        max_ = np.max(Color)
        if(max_ > 0):
            Color_ = Color/max_
        else:
            Color_ = Color

        axs[nr,nc].contourf(Theta, 90-El, Color_, levels = 25, cmap = plt.cm.hot)
        axs[nr,nc].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])
        axs[nr,nc].set_title(f"{wavelength} nm")
        axs[nr,nc].tick_params(axis='y', colors='gray')
        axs[nr,nc].set_theta_zero_location("N")
        fig.set_figheight(12)
        fig.set_figwidth(16)
        axs[nr,nc].grid(False)

        m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
        m.set_array(Color)
        cbar = plt.colorbar(m, ax=axs[nr,nc])
        cbar.ax.set_title('W/m^2/sr/nm')
    plt.show()

    
      
#%%      ---- PLOT DIFFUSE SPECTRAL RADIANCE DISTRIBUTION FOR ALL WAVELENGTHS AT A CERTAIN POINT FOR ALL TIMES ----


i,j = 10, 10
component = "diffuse"
maximum_val = max([res[component][nt][i,j,:].max() for nt in range(len(DatetimeIndex_obj))])
for nt in range(len(DatetimeIndex_obj)):
    az_ = Az[i,j]
    el_ = El[i,j]

    title = f"{Site_obj.name}: Spectral {component} radiance at time {DatetimeIndex_obj[nt]}"
    plt.suptitle(f"{title}.\n Azimuth = {az_}°, Elevation = {el_}°.")

    values = res[component][nt][i,j,:]
    plt.plot(res["wavelengths"], values)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Spectral Radiance [W/m^2/sr/nm]")
    plt.ylim(0, maximum_val)
    plt.show()


# %%
