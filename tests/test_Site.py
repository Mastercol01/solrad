#%%
import os
import solrad
from solrad.Site import *
import solrad.auxiliary_funcs as aux

#%%
# 1) --- INITIALIZE SITE OBJECT FOR SIMULATION ---
Site_obj = Site(latitude  = 6.230833,
                longitude = -75.590553,
                altitude  = 1500,
                tz        = '-05:00',
                name      = 'Medellín',
                SF_model  = 'Rural' )

#%%
# 2) --- DEFINE TERRAIN HORIZON ---
az   = np.linspace(0, 360, 361)
el1  = 15*np.sin(az*np.pi/180)**2
el2  = 15*np.sin(az*np.pi/45)**2
func = interp1d(x=az,y=el2)

Site_obj.set_horizon_from_arrays(azimuth=az, elevation=el1)
Site_obj.plot_horizon(az)

Site_obj.set_horizon_from_func(func)
Site_obj.plot_horizon(az)

Site_obj.reset_horizon()
Site_obj.plot_horizon(az)

Site_obj.set_horizon_from_pvgis()
Site_obj.plot_horizon(az)


#%%
# 3) --- DEFINE TIME PERIOD OF SIMULATION ---
Site_obj.define_simulation_time_data(start_time = "2023-1-1 00:00:00",
                                    end_time   = "2023-1-31 23:59:59.9", 
                                    freq       = "5min", 
                                    min_hms    = "sunrise",
                                    max_hms    = "sunset",  
                                    inclusive = False)


#%%
# 4) --- SET AND COMPUTE CLIMATE DATA ---
Site_obj.set_climate_data_from_pvgis_tmy_data(startyear=2005, endyear=2015)
Site_obj.compute_extraterrestrial_normal_irradiance()
Site_obj.compute_cummulative_time_integral_of_irradiances()

#%%
# 5) --- COMPUTE SUN DATA ---
Site_obj.compute_sun_data()

#%%
# 6) --- COMPUTE AIR DATA ---
# ozone column
Site_obj.compute_ozone_column_using_van_Heuklon_model()
Site_obj.compute_ozone_column_using_satelite_data()

# water column
Site_obj.compute_water_column_using_gueymard94_model()
Site_obj.compute_water_column_using_satelite_data()

# Angstrom exponent 
Site_obj.compute_angstrom_turbidity_exponent_500nm_using_SF_model()

# AOD_550nm
Site_obj.compute_aod_500nm_using_satelite_data()

#%%
# 7) --- COMPUTE AEROSOL PROPERTIES ---
Site_obj.compute_single_scattering_albedo_using_SF_model()
Site_obj.compute_aerosol_asymmetry_factor_using_SF_model()
Site_obj.compute_spectrally_averaged_aerosol_asymmetry_factor()

#%%
# --- PLOT DATA ---
Site_obj.plot_data(col = "Gb(n)",
                years = None, 
                months = None, 
                days   = None, 
                hours  = [6.5, 17.5],
                mode=2, 
                interp_method = "linear", 
                figsize = (16, 12))

#%%
# --- INTERPOLATE DATA ---
Site_obj.time_interpolate_variable(col   = "Gb(n)",
                                year  = 2023,
                                month = 1,
                                day   = 1, 
                                new_hms_float = np.linspace(6.5, 17.5, 100),
                                interp_method = "linear")


#%%
TESTS_PATH = os.path.join(os.path.dirname(solrad.__path__[0]), "tests")
Site_obj_path = os.path.join(TESTS_PATH, "test_Site_obj.pkl")

# Save Site obj
aux.save_obj_with_pickle(Site_obj, path = Site_obj_path)
# %%
