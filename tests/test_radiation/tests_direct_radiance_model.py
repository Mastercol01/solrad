#%%
    
# We import libraries
import os
import solrad
import time as tm
import numpy as np
from solrad.Site import Site
import matplotlib.pyplot as plt
from solrad.radiation.direct_radiance_model import *
from solrad.auxiliary_funcs import load_obj_with_pickle


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

#%%

# Scpecify year, day and month of the data.
# year, month, day = 2023, 1, 1
year, month, day = 2023, 1, 25

Gbn  = Site_obj.climate_and_air_data[(year, month, day)]["Gb(n)"]

index = Gbn.index

sun_apel    = Site_obj.sun_data[(year, month, day)]["apel"]
sun_az      = Site_obj.sun_data[(year, month, day)]["az"]

# Transform data from pandas.Series to numpy.arrays

Gbn         = np.array(Gbn)
sun_apel    = np.array(sun_apel)
sun_az      = np.array(sun_az)

# Reshape data into the required shapes.

Gbn          = Gbn.reshape(1, 1, len(Gbn))
sun_apel     = sun_apel.reshape(1, 1, len(sun_apel))
sun_az       = sun_az.reshape(1, 1 ,len(sun_az))

# Plot Irradiances
fig = plt.figure(figsize=(16,12))
plt.plot(index, Gbn.flatten(), label = "Direct Normal Irradiance")
plt.title(f"Irradiances for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("Irradiance [W/m^2]")
plt.legend()
plt.grid()
plt.show()



#%%       ----- COMPUTE DIFFUSE SKY RADIANCE DISTRIBUTION ----

t = tm.time()
Lea = compute_direct_radiance(Az               = Az,  
                                El              = El,
                                Gbn             = Gbn,
                                sun_az          = sun_az,
                                sun_apel        = sun_apel)

dt = tm.time() - t

#%%    ---- PLOT CUMMULATIVE SKY RADIANCE DISTRIBUTION (3D) ----

    
Phi = np.deg2rad(Az).reshape(Az.shape[0], Az.shape[1])
Theta = np.deg2rad(90 - El).reshape(El.shape[0], El.shape[1]) 

X = np.cos(Phi)*np.sin(Theta)
Y = np.sin(Phi)*np.sin(Theta)
Z = np.cos(Theta)

total_Lea = Lea.sum(axis=-1)

Color = total_Lea

max_ = np.max(total_Lea)
if(max_ > 0):
    Color_ = Color/max_
else:
    Color_ = Color

    
title = f"{Site_obj.name}: Cummulative Direct Irradiance at time {year},{month},{day}"


fig = plt.figure(figsize=(16, 12))
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


#%%    ---- PLOT CUMMULATIVE SKY RADIANCE DISTRIBUTION (2D) ----
        
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

        
#%%           ---- CHECKING CONSERVATION OF DIRECT IRRADIANCE -----   
      
# Energy must be conserved. As such, the direct irradiance on a normal
# plane should be constant. In other words, we should be able to recompute 
# Gbn from Lea and still get the exact same values.

recomputed_Gbn = Lea.sum(axis=(0,1))

fig = plt.figure(figsize=(16,12))
plt.plot(index, Gbn.flatten(), label = "Original")
plt.plot(index, recomputed_Gbn, label = "Recomputed")
plt.title(f"Direct normal irradiance for {Site_obj.name} on the {year}, {month}, {day}")
plt.ylabel("Diffuse irradiance [W/m^2]")
plt.legend()
plt.grid()
plt.show()
    


#%%
