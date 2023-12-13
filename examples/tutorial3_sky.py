#%%                     DESCRIPTION
"""
The provided Python tutorial module serves as a comprehensive guide to utilizing
the solrad library for solar radiation simulation. Specifically, how to calculate
solar radiation quantities. The module is designed to teach you how to use the 
'Sky' class in order to perform your simualtion. This tutorial builds upon the Site 
object initialized in a previous tutorial.
"""
#%%                IMPORTATION OF LIBRARIES
import os
import solrad
import numpy as np
from solrad.Sky import Sky
from solrad.Site import Site
import matplotlib.pyplot as plt
import solrad.auxiliary_funcs as aux
from scipy.interpolate import interp1d

#%%                  LOAD Site_obj

#%%     1) LOAD Site_obj
# We require a 'Site' class instance in order to initialize a 'Sky' class instance.
# Therefore, we load the previously compute 'Site_obj' instance from tutorial 2.

# Change this path if you saved the 'example_Site_obj.pkl' object elsewhere.
EXAMPLES_PATH = os.path.join(os.path.dirname(solrad.__path__[0]), "examples")
Site_obj_path = os.path.join(EXAMPLES_PATH, "example_Site_obj.pkl")
Site_obj = aux.load_obj_with_pickle(path = Site_obj_path)


#%%     2) INITIALIZATION OF SKY OBJECT
# We may now initialize the sky object by passing in the loaded site object as well
# as the number of division we would like the sky to be discretised in.

Sky_obj = Sky(Site_obj, num_divisions = 50)

# VISUALIZE SKY VAULT DISCRETIZATION IN 2D
Sky_obj.plot_disk_patches(figsize=(12,12))

# VISUALIZE SKY VAULT DISCRETIZATION IN 3D 
Sky_obj.plot_sphere_patches(figsize=(12,12), axis_view=(25, 30))

# (NOTE: Naturally, the higher the number of divisions, the better results.)


#%%     3) DISCRETISATION OF THE SKY-VAULT
# Although, the sky is automatically discretised once we initialize the Sky_obj,
# we can always rediscretize it by calling the 'self.discretise' method.

Sky_obj.discretise(num_divisions = 400)

# VISUALIZE SKY VAULT DISCRETIZATION IN 2D
Sky_obj.plot_disk_patches(figsize=(12,12))

# VISUALIZE SKY VAULT DISCRETIZATION IN 3D 
Sky_obj.plot_sphere_patches(figsize=(12,12), axis_view=(25, 30))

# NOTE: Doing calling this method after initialization will also errase all other radiation quantities computed up to that point.

#%%     4) ZONE_DATA AND PATCH_DATA
# After initialization, two important attributes within Sky_obj are created.
# These are 'self.zone_data' and 'self.patch_data'. These are dictionaries which
# store all relevant data of each zone and patch of the discretised sky.

# We call a 'zone', each sky ring that contains one or more sky patches.
# We call a 'patch', each cell that subdivides the sky.

# Let us see the keys of 'self.zone_data' attribute.
print("--- ZONE KEYS ---") 
print(Sky_obj.zone_data.keys())

# These keys are integers which identify each sky zone uniquely.
# The greater the number, the further away the zone is from the sky zenith and the closer
# it is to the sky horizon. We call these numbers 'zone_num'. Let us see the content for one
# of its keys.

print("--- ZONE[KEY] CONTENT ---") 
print(Sky_obj.zone_data[5])

# Now, let us see they keys of 'self.patch_data':
print("--- PATCH KEYS ---") 
print(Sky_obj.patch_data.keys())

# These keys are 2-tuples of integers which together identify each sky patch uniquely.
# The first number is the 'zone_num' each patch belongs to and the second number is its 'patch_num',
# i.e, a number identifying that patch (only) uniquely within that zone. Let us see the content for one
# of its keys.

print("--- PATCH[KEY] CONTENT ---") 
print(Sky_obj.patch_data[(5,1)])

# If you whish to learn more about the specific contents of each dictionary, please
# see the documentation for the method "self.discretise".


#%%    5)    RADIANCES
# With this, we can now fairly easily compute an approximation of the 
# sky's spectral radiance, as well as its regular radiance, for a given date.
# To do this, we make use of the 'self.compute_radiances_for_a_date' method.

# (NOTE: the date chosen must be present in 'Site_obj.simulation_time_data.keys()')
Sky_obj.compute_radiances_for_a_date(year = 2023, 
                                     month = 1, 
                                     day = 10, 
                                     nel = 46, 
                                     naz = 181, 
                                     num_iterations = 500, 
                                     use_site_horizon=True)

# The results are stored in the attribute 'self.radiance_res':
print(Sky_obj.radiances)

# We can also plot the results for a particular time of the date simulated.

# PLOT SPECTRAL RADIANCE
for nt in [15, 50, 100]: 
    Sky_obj.plot_spectral_radiance_for_a_date(component="direct",  nt=nt)
    Sky_obj.plot_spectral_radiance_for_a_date(component="diffuse", nt=nt)

# PLOT RADIANCE
for nt in [15, 50, 100]: 
    Sky_obj.plot_radiance_for_a_date(component="direct",  nt=nt)
    Sky_obj.plot_radiance_for_a_date(component="diffuse", nt=nt)
    Sky_obj.plot_radiance_for_a_date(component="diffuse", nt=nt, projection = "sphere")

# PLOT IGAWA'S SKY INDEX
fig = plt.figure(figsize=(14,12))
plt.plot(Sky_obj.radiances["DatetimeIndex_obj"], Sky_obj.radiances["Siv"])
plt.xlabel("Time")
plt.ylabel("Siv [-]")
plt.title("Igawa's clear sky index across time")


#%%     6) (SEPCTRAL) RADIANT EXPOSURE VECTORS
# We can now compute the spectral and regular exposure vectors for the discretised
# sky. Each sky patch has one spectral and one regular radiant exposure vectors.
# The magnitude of these vectors (be it spectral or regular) tells us the total
# amount energy (per unit area per unit wavelength in the first case, while and just per unit area
# in the second case) that is emitted by that sky patch, for a given time interval. The direction
# of these vectors points to the 'center of radiation' of the sky patch. That is, the weighted average 
# of emission direction. We can use these vectors (be it spectral or regular) in order to perform 
# computations of absorption and direction optimization, among other things.

# We first delete the 'self.radiances' attribute in order to conserve memory space.
# Depending on your hardware this may or may not be necessary.
Sky_obj.radiances = None

# COMPUTE RADIANT EXPOSURE VECTORS (SPECRAL AND NOT SPECTRAL)
start_date = (2023, 1, 1)
end_date   = (2023, 1, 31)
Sky_obj.compute_exposure_vectors_for_a_date_interval( start_date  = start_date,
                                                        end_date    = end_date,
                                                        nel               = 46,
                                                        naz               = 181,
                                                        num_iterations    = 500,
                                                        use_site_horizon  = False,
                                                        int_nzen          = 20,
                                                        int_naz           = 30)

# (NOTE: the dates chosen must be present in 'Site_obj.simulation_time_data.keys()')

# The results can be found in 2 places:
# 1) 'self.exposure_vectors' attribute
# 2) 'self.patch_data["exposure"]' attribute

print("--- self.exposure_vectors KEYS ---")
print(Sky_obj.exposure_vectors.keys())

print("--- self.patch_data['exposure'] KEYS ---")
print(Sky_obj.patch_data[(5,1)]["exposure"].keys())

# If you whish to learn more about the specific contents of each dictionary, please
# see the documentation for the method "self.compute_exposure_vectors_for_a_date_interval".

# PLOT RADIANT EXPOSURE (VECTOR MAGNITUDE, NOT SPECTRAL) FOR ALL SKY-PATCHES
for projection in ["disk", "sphere"]:
    for mode in ["global", "direct", "diffuse"]:
        config = {"projection":projection, "mode":mode, "n":1000}
        Sky_obj.plot_exposures(config)


# PLOT RADIANT EXPOSURE (VECTOR MAGNITUDE, SPECTRAL) FOR SOME SKY PATCHES
wavelengths = Sky_obj.exposure_vectors["wavelengths"]
start_date  = Sky_obj.time_integrated_spectral_radiance["start_date"]
end_date    = Sky_obj.time_integrated_spectral_radiance["end_date"]

for key in [(0, 0), (1, 1), (3, 0), (3, 1), (3, 2), (3, 3), (5, 8), (5, 9), (5, 10), (5, 11), (7, 11), (7, 12), (7, 13)]:
    zone_num, patch_num = key
    direct_spectral_irrad_mag  = Sky_obj.patch_data[key]["exposure"]["spectral_direct"]["magnitude"]
    diffuse_spectral_irrad_mag = Sky_obj.patch_data[key]["exposure"]["spectral_diffuse"]["magnitude"]
    global_spectral_irrad_mag  = Sky_obj.patch_data[key]["exposure"]["spectral_global"]["magnitude"]

    az_lims  = [round(Sky_obj.patch_data[key]['inf_az'],2),  round(Sky_obj.patch_data[key]['sup_az'],2)]
    zen_lims = [round(Sky_obj.patch_data[key]['inf_zen'],2), round(Sky_obj.patch_data[key]['sup_zen'],2)]

    fig = plt.figure(figsize=(16,12))
    plt.plot(wavelengths, direct_spectral_irrad_mag,  label="direct")
    plt.plot(wavelengths, diffuse_spectral_irrad_mag, label="diffuse")
    plt.plot(wavelengths, global_spectral_irrad_mag,  label="global")
    plt.suptitle(f"From date {start_date} to {end_date}")
    plt.title(f"Spectral exposure of patch ({key[0]}, {key[1]}) with az_lims: {az_lims}°, zen_lims: {zen_lims}°")
    plt.ylabel("Spectral exposure [Wh/m^2/nm]")
    plt.xlabel("Wavelengths [nm]")
    plt.xlim(300, 4000)
    plt.ylim(0, 18)
    plt.grid()
    plt.legend()
    plt.show()


#%%     7) ABSORBED ENERGY
# We can also compute the incident and absorbed energies for a unit plane, given an 
# an absorption fucntion, for a given radiation component.

# DUMMY ABSORPTION FUNCTION FOR EXAMPLE
wavelengths = Sky_obj.exposure_vectors["wavelengths"]
func_array  = np.logical_and(375 <= wavelengths, wavelengths <= 1000).astype(float)
func        = interp1d(x = wavelengths, y = func_array, kind = "linear")

def absorption_func(arguments):
    cos_aoi = np.cos(np.deg2rad(arguments[:,0]))
    cos_aoi[cos_aoi<0] = 0
    return cos_aoi*func(arguments[:,1])


aoi = np.full(len(wavelengths), 50)
arguments = np.stack([aoi,wavelengths], axis=1)
res = absorption_func(arguments)
plt.plot(wavelengths, res, label = f"aoi = {aoi[0]}°")
plt.ylim(0,1)
plt.show()

# UNIT NORMAL OF PLANE TO EVALUATE
n_uvec=[0,0,1] # (horizontal plane in this case)

# COMPUTE INCIDENT ENERGY (ABSORPTION = 1, FOR EVERY WAVELENGTH)
total_incident_energy =\
Sky_obj.compute_absorbed_energy_by_unit_plane(n_uvec=n_uvec, absorption_func=1, component="global")

# COMPUTE INCIDENT ENERGY (ABSORPTION VARIES WITH WAVELENGTH)
total_absorbed_incident_energy =\
Sky_obj.compute_absorbed_energy_by_unit_plane(n_uvec=n_uvec, absorption_func=absorption_func, component="global")

print(f"Total incident energy on unit horizontal plane: {total_incident_energy/1000} kWh")
print(f"Total absorbed incident energy on unit horizontal plane: {total_absorbed_incident_energy/1000} kWh")
#%%     8) OPTIMAL ORIENTATION
# Finally, we can find the plane orientation that maximizes energy absorption for a unit plane, given an 
# an absorption fucntion, for a given radiation component. 

res =\
Sky_obj. compute_optimal_plane_orientation(absorption_func = 1, component = "global")
"""
res =\
Sky_obj. compute_optimal_plane_orientation(absorption_func = absorption_func, component = "global")
"""

print("--- OPTIMAL ORIENTATION RESULTS --- ")
print(res)
# %%
