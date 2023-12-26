#%%                             EXAMPLES       
    
# We import libraries 
import os
import solrad
import time as tm
import numpy as np
from solrad.Site import Site
import matplotlib.pyplot as plt
from solrad.auxiliary_funcs import load_obj_with_pickle
from solrad.radiation.diffuse_radiance_model import *

#%%

# ---- DEFINE EVALUATION POINTS FOR SKY RADIANCE DISTRIBUTION ----
angular_resolution = 1 #[degrees]
num_pts_el = round(90/angular_resolution) + 1
num_pts_az = 4*(num_pts_el - 1) + 1

az = np.linspace(0, 360, num_pts_az)
el = np.linspace(0,  90, num_pts_el)

Az, El = np.meshgrid(az, el)

# Reshape Az, El into the required shapes.
Az = Az.reshape(list(Az.shape) + [1])
El = El.reshape(list(El.shape) + [1])

#%%

# ---- LOAD SITE_OBJ ----
# As a test, we get the atmospheric data required for this module from
# precomputed Site obj.

TESTS_PATH = os.path.join(os.path.dirname(solrad.__path__[0]), "tests")
Site_obj_path = os.path.join(TESTS_PATH, "test_Site_obj.pkl")
Site_obj = load_obj_with_pickle(path = Site_obj_path)

# Scpecify year, day and month of the data.
# year, month, day = 2023, 1, 1
year, month, day = 2023, 1, 2

Gh        = Site_obj.climate_and_air_data[(year, month, day)]["G(h)"]
Gdh       = Site_obj.climate_and_air_data[(year, month, day)]["Gd(h)"]
extra_Gbn = Site_obj.climate_and_air_data[(year, month, day)]["extra_Gb(n)"]
sun_apel    = Site_obj.sun_data[(year, month, day)]["apel"]
sun_az      = Site_obj.sun_data[(year, month, day)]["az"]
rel_airmass = Site_obj.sun_data[(year, month, day)]["rel_airmass"]

index = Gh.index
# Transform data from pandas.Series to numpy.arrays

Gh          = np.array(Gh)
Gdh         = np.array(Gdh)
extra_Gbn   = np.array(extra_Gbn)
sun_apel    = np.array(sun_apel)
sun_az      = np.array(sun_az)
rel_airmass = np.array(rel_airmass)

# Reshape data into the required shapes.

Gh           = Gh.reshape(1, 1, len(Gh))
Gdh          = Gdh.reshape(1, 1, len(Gdh))
extra_Gbn    = extra_Gbn.reshape(1, 1, len(extra_Gbn))
sun_apel     = sun_apel.reshape(1, 1, len(sun_apel))
sun_az       = sun_az.reshape(1, 1 ,len(sun_az))
rel_airmass  = rel_airmass.reshape(1, 1, len(rel_airmass))

# Plot Irradiances
fig = plt.figure(figsize=(16,12))
plt.plot(index, Gdh.flatten(), label = "Diffuse Horizontal Irradiance")
plt.plot(index, Gh.flatten(), label = "Global Horizontal Irradiance")
plt.title(f"Irradiances for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("Irradiance [W/m^2]")
plt.xlabel("Time [month-day hour]")
plt.legend()
plt.grid()
plt.show()



#%%       ----- COMPUTE DIFFUSE SKY RADIANCE DISTRIBUTION ----


t = tm.time()
res = compute_diffuse_radiance(Az              = Az,  
                                El              = El,
                                dAz     = angular_resolution,
                                dEl     = angular_resolution,
                                Gh              = Gh,
                                Gdh             = Gdh, 
                                extra_Gbn       = extra_Gbn, 
                                sun_az          = sun_az,
                                sun_apel        = sun_apel,
                                rel_airmass     = rel_airmass,
                                num_iterations  = 500)

dt = tm.time() - t

#%%    ---- PLOT SKY RADIANCE DISTRIBUTION FOR ALL TIMES (3D) ----


Phi = np.deg2rad(Az).reshape(Az.shape[0], Az.shape[1])
Theta = np.deg2rad(90 - El).reshape(El.shape[0], El.shape[1]) 

X = np.cos(Phi)*np.sin(Theta)
Y = np.sin(Phi)*np.sin(Theta)
Z = np.cos(Theta)

for nt in range(Gdh.shape[-1]):
    
    Color = res["Lea"][:,:,nt]
    
    max_ = np.max(res["Lea"][:,:,nt])
    if(max_ > 0):
        Color_ = Color/max_
    else:
        Color_ = Color
    
        
    title = f"{Site_obj.name}: Diffuse Radiance at time: {index[nt]}"
    
    
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(45, 180)
    ax.set_xlabel("X (↑ == N, ↓ == S)")
    ax.set_ylabel("Y (↑ == E, ↓ == W)")
    ax.set_title(title)
    ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))
    
    m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
    m.set_array(Color)
    cbar = plt.colorbar(m, ax=ax)
    cbar.ax.set_title('W/m^2/sr')
    
    plt.show()

#%%    ---- PLOT SKY RADIANCE DISTRIBUTION FOR ALL TIMES (2D) ----
        
for nt in range(Gdh.shape[-1]):

    Color = res["Lea"][:,:,nt]
    
    max_ = np.max(res["Lea"][:,:,nt])
    if(max_ > 0):
        Color_ = Color/max_
    else:
        Color_ = Color
    
        
    title = f"{Site_obj.name}: Diffuse Radiance at time: {index[nt]}"

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.contourf(Phi, np.rad2deg(Theta), Color_, levels = 25, cmap = plt.cm.hot)
    ax.set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])
    ax.set_title("(N = 0°, E = 90°, S = 180°, W = 270°)")
    ax.tick_params(axis='y', colors='gray')
    ax.set_theta_zero_location("N")
    fig.set_figheight(12)
    fig.set_figwidth(16)
    ax.grid(False)

    plt.suptitle(title)
    m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
    m.set_array(Color)
    cbar = plt.colorbar(m, ax=ax)
    cbar.ax.set_title('W/m^2/sr')

    plt.show()
    
#%%           ---- CHECKING CONSERVATION OF DIFFUSE IRRADIANCE -----   
    
# Energy must be conserved. As such, the diffuse irradiance on a horizontal
# plane should be constant. In other words, we should be able to recompute 
# Gdh from Lea and still get the exact same values.

Lea = res["Lea"]
integrand = Lea*(np.sin(Theta)*np.cos(Theta)).reshape(Az.shape)

recomputed_Gdh =\
0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])

ang_res = np.deg2rad(angular_resolution)
recomputed_Gdh = recomputed_Gdh.sum(axis=(0,1))*ang_res*ang_res

fig = plt.figure(figsize=(16,12))
plt.plot(index, Gdh.flatten(), label = "Original")
plt.plot(index, recomputed_Gdh, label = "Recomputed")
plt.title(f"Diffuse horizontal irradiance for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("Diffuse horizontal irradiance [W/m^2]")
plt.legend()
plt.grid()
plt.show()


#%%      ---- CHECKING UNFITNESS OF CONSTANTS AS A METHOD FOR COMPUTING LzED 1------

# Let us compute LzEd numerically and via Igawa's non linear regression in 
# order to compare them.

# ------ NUMERICAL ------
Le = res["Le"]
integrand = Le*(np.sin(Theta)*np.cos(Theta)).reshape(Az.shape)
numerical_LzEd =\
0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])

ang_res = np.deg2rad(angular_resolution)
numerical_LzEd = numerical_LzEd.sum(axis=(0,1))*ang_res*ang_res
numerical_LzEd = 1/numerical_LzEd

# ------ REGRESSION ------

# Plotting
cts_LzEd = compute_inverse_of_the_integration_value_of_relative_sky_radiance_distribution_using_constants_LzEd(res["Kc"], res["Cle"], sun_apel)
fig = plt.figure(figsize=(16,12))
plt.plot(index, cts_LzEd.flatten(), label = "Igawa's non-linear Regression")
plt.plot(index, numerical_LzEd, label = "Numerical Computation")
plt.title(f"LzEd for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("LzEd")
plt.legend()
plt.grid()
plt.show()

#%%      ---- CHECKING UNFITNESS OF CONSTANTS AS A METHOD FOR COMPUTING LzED 2------

# Let's take the last excercise a step further and check that using Igawa's
# regresio Gdh is not conserved.

new_Lea = res["Le"]*Gdh*cts_LzEd
integrand = new_Lea*(np.sin(Theta)*np.cos(Theta)).reshape(Az.shape)
recomputed_Gdh =\
0.25*(integrand[:-1,:-1,:] + integrand[:-1, 1:,:] + integrand[1:,:-1,:] + integrand[1:,1:,:])

ang_res = np.deg2rad(angular_resolution)
recomputed_Gdh = recomputed_Gdh.sum(axis=(0,1))*ang_res*ang_res

fig = plt.figure(figsize=(16,12))
plt.plot(index, Gdh.flatten(), label = "Original")
plt.plot(index, recomputed_Gdh, label = "Recomputed, using Igawa's non-linear regression")
plt.title(f"Diffuse horizontal irradiance for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("Diffuse horizontal irradiance [W/m^2]")
plt.legend()
plt.grid()
plt.show()
    

#%%           ------ PLOT OF SKY CONDITIONS -------

fig = plt.figure(figsize=(16,12))
plt.plot(index, res["Siv"].flatten())
plt.axhline(y = 1.15, color = 'gray', linestyle = ':', label="Intermidiate Overcast limit")
plt.axhline(y = 0.9, color = 'gray', linestyle = '-.', label="Intermidiate Sky limit")
plt.axhline(y = 0.3, color = 'gray', linestyle = '--', label="Intermidiate Clear Sky limit")
plt.axhline(y = 0.15, color = 'gray', linestyle = '-', label="Clear Sky limit")
plt.title(f"Sky Index for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("Sky Index")
plt.xlabel("Time [month-day hour]")
plt.legend()
plt.grid()
plt.show()
# %%
