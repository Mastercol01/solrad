#%%

import os 
import solrad
from solrad.Site import Site
import matplotlib.pyplot as plt
import solrad.auxiliary_funcs as aux
from scipy.interpolate import interp1d
from solrad.Sky import *

#%%                  OTHER TESTS

# ---- LOAD SITE_OBJ ----
# As a test, we get the atmospheric data required for this module from
# precomputed Site obj.

TESTS_PATH = os.path.join(os.path.dirname(solrad.__path__[0]), "tests")
Site_obj_path = os.path.join(TESTS_PATH, "test_Site_obj.pkl")
Site_obj = aux.load_obj_with_pickle(path = Site_obj_path)

# --- INITIALIZE SKY OBJ ---
Sky_obj = Sky(Site_obj, num_divisions = 400)   


# --- VISUALIZE SKY VAULT DISCRETIZATION IN 2D ---
Sky_obj.plot_disk_patches(figsize=(12,12))

# --- VISUALIZE SKY VAULT DISCRETIZATION IN 3D ---
Sky_obj.plot_sphere_patches(figsize=(12,12), axis_view=(25, 30))
    
# --- EXTRACT ZONE, PATCH DATA ---
zone_data, patch_data = Sky_obj.zone_data, Sky_obj.patch_data

# --- BIN SKY PTS INTO THE CORRECT SKY PATCHES -----
sky_pts = np.array([[30, 90],
                    [45, 180],
                    [60, 270],
                    [77.25, 35.4]])

zone_nums, local_patch_nums =\
Sky_obj.sky_points_to_zones_patches(sky_pts[:,0], sky_pts[:, 1])


#%%      ---- RADIANCES ----

# COMPUTE BOTH SPECTRAL RADIANCE AND RADIANCE FOR A DATE
year, month, day = 2023, 1, 1
Sky_obj.compute_radiances_for_a_date(year, month, day, nel = 46, naz = 181, num_iterations = 500, use_site_horizon=True)

# PLOT SPECTRAL RADIANCE
nt = 15
config = {"wavelength_idxs" : np.array([30, 60])}
Sky_obj.plot_spectral_radiance_for_a_date(component="direct",  nt=nt)
Sky_obj.plot_spectral_radiance_for_a_date(component="diffuse", nt=nt)
Sky_obj.plot_spectral_radiance_for_a_date(component="diffuse", nt=nt, config=config)

# PLOT RADIANCE
Sky_obj.plot_radiance_for_a_date(component="direct",  nt=nt)
Sky_obj.plot_radiance_for_a_date(component="diffuse", nt=nt)
Sky_obj.plot_radiance_for_a_date(component="diffuse", nt=nt, projection = "sphere")

# PLOT IGAWA'S SKY INDEX
fig = plt.figure(figsize=(14,12))
plt.plot(Sky_obj.radiances["DatetimeIndex_obj"], Sky_obj.radiances["Siv"])
plt.xlabel("Time")
plt.ylabel("Siv [-]")
plt.title("Igawa's clear sky index across time")

    
    
#%%    ---- TIME-INTEGRATED SPECTRAL RADIANCE ----

# COMPUTE TIME-INTEGRATED SPECTRAL RADIANCE FOR A DATE INTERVAL 
start_date = (2023, 1, 1)
end_date   = (2023, 1, 10)
Sky_obj._compute_time_integrated_spectral_radiance_for_a_date_interval( start_date        = start_date,
                                                                        end_date          = end_date,
                                                                        nel               = 46,
                                                                        naz               = 181,
                                                                        num_iterations    = 500,
                                                                        use_site_horizon  = True)

# PLOT the TIME-INTEGRATED SPECTRAL RADIANCE FOR THE CHOSEN DATE INTERVAL 
Sky_obj._plot_time_integrated_spectral_radiance_for_a_date_interval(component="direct")
Sky_obj._plot_time_integrated_spectral_radiance_for_a_date_interval(component="diffuse")



#%%    ---- (SEPCTRAL) RADIANT EXPOSURE VECTORS ----

# COMPUTE RADIANT EXPOSURE VECTORS (SPECRAL AND NOT SPECTRAL)
start_date = (2023, 1, 1)
end_date   = (2023, 1, 10)
Sky_obj.compute_exposure_vectors_for_a_date_interval( start_date  = start_date,
                                                        end_date    = end_date,
                                                        nel               = 46,
                                                        naz               = 181,
                                                        num_iterations    = 500,
                                                        use_site_horizon  = False,
                                                        int_nzen          = 20,
                                                        int_naz           = 30)

# PLOT RADIANT EXPOSURE (VECTOR MAGNITUDE, NOT SPECTRAL) FOR ALL SKY-PATCHES
for projection in ["disk", "sphere"]:
    for mode in ["global", "direct", "diffuse"]:
        config = {"projection":projection, "mode":mode, "n":1000}
        Sky_obj.plot_exposures(config)


# PLOT RADIANT EXPOSURE (VECTOR MAGNITUDE, SPECTRAL) FOR ALL SKY-PATCHES
wavelengths = Sky_obj.exposure_vectors["wavelengths"]
start_date  = Sky_obj.time_integrated_spectral_radiance["start_date"]
end_date    = Sky_obj.time_integrated_spectral_radiance["end_date"]

for key in Sky_obj.patch_data.keys():
    zone_num, patch_num = key
    direct_spectral_irrad_mag  = Sky_obj.patch_data[key]["exposure"]["spectral_direct"]["magnitude"]
    diffuse_spectral_irrad_mag = Sky_obj.patch_data[key]["exposure"]["spectral_diffuse"]["magnitude"]
    global_spectral_irrad_mag  = Sky_obj.patch_data[key]["exposure"]["spectral_global"]["magnitude"]

    az_lims  = [round(patch_data[key]['inf_az'],2),  round(patch_data[key]['sup_az'],2)]
    zen_lims = [round(patch_data[key]['inf_zen'],2), round(patch_data[key]['sup_zen'],2)]

    fig = plt.figure(figsize=(16,12))
    plt.plot(wavelengths, direct_spectral_irrad_mag,  label="direct")
    plt.plot(wavelengths, diffuse_spectral_irrad_mag, label="diffuse")
    plt.plot(wavelengths, global_spectral_irrad_mag,  label="global")
    plt.suptitle(f"From date {start_date} to {end_date}")
    plt.title(f"Spectral exposure of patch ({key[0]}, {key[1]}) with az_lims: {az_lims}°, zen_lims: {zen_lims}°")
    plt.ylabel("Spectral exposure [Wh/m^2/nm]")
    plt.xlabel("Wavelengths [nm]")
    plt.xlim(300, 4000)
    plt.ylim(0,10)
    plt.grid()
    plt.legend()
    plt.show()


    

#%%              CHECK CONSERVATION OF ENERGY

start_date  = Sky_obj.time_integrated_spectral_radiance["start_date"]
end_date    = Sky_obj.time_integrated_spectral_radiance["end_date"]

start_date_ts = pd.Timestamp(f"{start_date[0]}-{start_date[1]}-{start_date[2]}")
end_date_ts   = pd.Timestamp(f"{end_date[0]}-{end_date[1]}-{end_date[2]}")
one_day       = pd.Timedelta("1D")

# Energy is in units of Watt-Hours.
total_energy_on_the_horizontal_plane_original_approach = 0

# NOTE: WE ASSUME A PANEL OF UNIT AREA. THAT'S WHY WE MULTIPLY BY NOTHING
# BUT THE UNITS ARE POWER AND ENERGY INSTEAD OF POWER/m^2 AND ENERGY/m^2.

current_date_ts = start_date_ts
while current_date_ts <= end_date_ts:
    key = (current_date_ts.year, current_date_ts.month, current_date_ts.day)
    apzen     = np.array(Sky_obj.Site_obj.sun_data[key]["apzen"]).astype(float)
    Gbn       = np.array(Sky_obj.Site_obj.climate_and_air_data[key]["Gb(n)"]).astype(float)
    Gdh       = np.array(Sky_obj.Site_obj.climate_and_air_data[key]["Gd(h)"]).astype(float)
    hms_float = np.array(Sky_obj.Site_obj.climate_and_air_data[key]["hms_float"]).astype(float)
    
    cos_aoi = np.cos(np.deg2rad(apzen))
    cos_aoi[cos_aoi < 0] = 0
    
    power = Gdh + Gbn*cos_aoi 
    energy = simpson(y = power, x = hms_float)
    total_energy_on_the_horizontal_plane_original_approach += energy
    current_date_ts += one_day

    
# Energy is in units of Watt-Hours.
total_energy_on_the_horizontal_plane_new_approach = 0

for key, patch_dict in Sky_obj.patch_data.items():
    
    global_radiant_exposure_vector =\
    patch_dict["exposure"]["global"]["vector"] 
    
    energy = np.dot([0,0,1], global_radiant_exposure_vector)
    
    total_energy_on_the_horizontal_plane_new_approach += energy
    

percent_error  = total_energy_on_the_horizontal_plane_new_approach
percent_error -= total_energy_on_the_horizontal_plane_original_approach
percent_error  = 100*abs(percent_error)
percent_error /= total_energy_on_the_horizontal_plane_original_approach


print(f"--- CALCULATION OF THE TOTAL ENERGY THAT FALLS ONTO A 1m^2 HORIZONTAL PANEL FROM THE SKY: FROM {start_date} TO {end_date} ---")   
print(f"ORIGINAL ENERGY APPROACH [Wh]: {total_energy_on_the_horizontal_plane_original_approach}")
print(f"NEW ENERGY APPROACH [Wh]: {total_energy_on_the_horizontal_plane_new_approach}")    
print(f"PERCENTAGE ERROR: {percent_error}") 

# We can see that both numbers are extremely close. The difference
# in Wh may be explained by the difference in integration methods.
# It satnds to reason that a higher spatial and time resolution would 
# porduce closer results. But even so, the resoluts are extremely close.
# The differences may also lie in the partial omission of the effect of the
# horizon, when computing the diffuse energy in the new approach.
# This is something for future work.

#%%
func_array = np.logical_and(450 <= wavelengths, wavelengths <= 1000).astype(float)
func       = interp1d(x = wavelengths, y = func_array, kind = "linear")

def absorption_func(arguments):
    cos_aoi = np.cos(np.deg2rad(arguments[:,0]))
    cos_aoi[cos_aoi<0] = 0
    return cos_aoi*func(arguments[:,1])

# aoi = np.full(len(wavelengths), 50)
# arguments = np.stack([aoi,wavelengths], axis=1)
# res = absorption_func(arguments)
# plt.plot(wavelengths, res)
# plt.ylim(0,1)
# plt.show()

total_absorbed_incident_energy =\
Sky_obj.compute_absorbed_energy_by_unit_plane(n_uvec=[0,0,1], absorption_func=1, component="global")

total_absorbed_incident_energy =\
Sky_obj.compute_absorbed_energy_by_unit_plane(n_uvec=[0,0,1], absorption_func=1, component="direct")

total_absorbed_incident_energy =\
Sky_obj.compute_absorbed_energy_by_unit_plane(n_uvec=[0,0,1], absorption_func=1, component="diffuse")

total_absorbed_incident_energy =\
Sky_obj.compute_absorbed_energy_by_unit_plane(n_uvec=[0,0,1], absorption_func=absorption_func, component="global")

#%%

res =\
Sky_obj. compute_optimal_plane_orientation(min_res = 0.5, naz = 13, nel = 4, absorption_func = 1, component = "global")

res =\
Sky_obj. compute_optimal_plane_orientation(min_res = 0.5, naz = 13, nel = 4, absorption_func = absorption_func, component = "global")


# %%
