#%%                     DESCRIPTION
"""
The provided Python tutorial module serves as a comprehensive guide to utilizing
the solrad library for solar radiation simulation. Specifically, how geographical and
climate data mut be prepared for the later simulation of radiation. The module is 
designed to teach you how to use the 'Site' class in order to prepare your simualtion.

"""
#%%                     IMPORTATION OF LIBRARIES
import os
import solrad
import numpy as np
from solrad.Site import Site
import solrad.auxiliary_funcs as aux

#%%                     DEFINITION PATH CONSTANTS

# You must change these paths to mtched where you have actually stored your data in your system 
# (see tutorial 1) Don't leave as is.
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(solrad.__path__[0])), "data")
OZONE_DATABASE_PATH = os.path.join(DATA_PATH, "ozone")
WATER_DATABASE_PATH = os.path.join(DATA_PATH, "water")
AOD_550NM_DATABASE_PATH = os.path.join(DATA_PATH, "aod_550nm")


#%%     1) INITIALIZATION OF SITE OBJECT

# In order to initialize a Site object we require some information about the geographical
# location that we are tying to simulate. For this example, we'll use the city of Medllín, Colombia.
# As a point of reference. 

MEDELLIN_LATITUDE      = 6.230833    # (degrees)
MEDELLIN_LONGITUDE     = -75.590553  # (degrees)
MEDELLIN_ALTITUDE      = 1500        # (meters)
MEDELLIN_TIMEZONE      ='-05:00'     # (GMT-5)
MEDELLIN_AEROSOL_MODEL = "Urban"     # (Urban 'Shettel and Fenn' aerosol model)


Site_obj = Site(latitude  = MEDELLIN_LATITUDE,
                longitude = MEDELLIN_LONGITUDE,
                altitude  = MEDELLIN_ALTITUDE,
                tz        = MEDELLIN_TIMEZONE,
                name      = 'Medellín',
                SF_model  = MEDELLIN_AEROSOL_MODEL)


#%%    2) TERRAIN AND HORIZON

# The horizon by default is set to 0 degrees everywhere for the site object.
# However, as a user, you can define your own custom horizon profile and plot it:
azimuths   = np.linspace(0, 360, 361)            #(degrees)
elevations = 10*np.sin(np.deg2rad(azimuths))**2  #(degrees)
Site_obj.set_horizon_from_arrays(azimuth=azimuths, elevation=elevations)
Site_obj.plot_horizon(azimuth=azimuths)

# You can also go back to the default horizon settings by restting the horizon profile.
Site_obj.reset_horizon()
Site_obj.plot_horizon(azimuth=azimuths)

# Finally, you can let Site_obj set the horizon profile for you. This step
# uses pvgis' API horizon profile from their geographical database.
Site_obj.set_horizon_from_pvgis()
Site_obj.plot_horizon(azimuth=azimuths)


#%%    3) DEFINE SIMULATION TIME

# Next we would like to define the period of time for which the simulation is to be run.
# For this example, we'd like to simulate the whole month of January. Hence, we set our start time 
# to be Jan 1st 2023, at midnight and our end time to be Jan 31st 2023, 6 seconds before midnight.
# We would also like that time data generated be sampled with a frequency of 5 minutes.
# Finally, since our end goal with this library is simulating sun radiation, there's not point 
# in keeping the time data for the hours before sunrise and after sunset. 

simulation_start_time = "2023-1-1 00:00:00"
simulation_end_time   = "2023-1-31 23:59:59.9"
simulation_time_frequency  = "5min"
minimum_hour_minute_second_allowed = "sunrise"
maximum_hour_minute_second_allowed = "sunset"

Site_obj.define_simulation_time_data(start_time = simulation_start_time,
                                     end_time   = simulation_end_time, 
                                     freq       = simulation_time_frequency, 
                                     min_hms    = minimum_hour_minute_second_allowed,
                                     max_hms    = maximum_hour_minute_second_allowed)

# This command generates an attribute called 'self.simulation_time_data'.
# It is a dict where each key is a date of simulation:
print(Site_obj.simulation_time_data.keys())

# Let's check out the contents for Jan 10th 2023:
print(Site_obj.simulation_time_data[(2023, 1, 10)])

# As we can see, we have a DatetimeIndex with a frequency of 5 minutes, which starts
# on the (approximate) time of sunrise and ends on the (approximate) time of sunset. 
# Finally, the 'define_simulation_time_data' method is actually quite flexible and many different
# time intervals with other conditions can be defined. Please check the documentation if
# you want to learn more.

#%%    3.5)  Site_obj's PRINCIPAL ATTRIBUTES

# After generating the 'self.simulation_time_data' attribute, other important attributes are generated.
# Namely, the 'self.climate_and_air_data', 'self.sun_data', 'self.single_scattering_albedo' and 'self.aerosol_asymmetry_factor'.
# These are dicts (the same keys as 'self.simulation_time_data') containing pandas.DataFrames which hold relevant information about 
# the site being modeled at each point in time. In broad terms:

# self.climate_and_air_data     : Contains information about the site's climate and air-related quantities.
# self.sun_data                 : Contains information about the position of the sun and relative airmass.
# self.single_scattering_albedo : Contains information about the single scattering albedo property of aerosols for different wavelengths.
# self.aerosol_asymmetry_factor : Contains information about the aerosol asymmetry factor property of aerosols for different wavelengths.

# Let's see check out how each attribute looks like for a particular date (Jan 10th 2023):
print("--------------------------- ")
print("        DATAFRAMES          ")
print("--------------------------- ")

print("---- self.climate_and_air_data ---- ")
print(Site_obj.climate_and_air_data[(2023, 1, 10)])

print("---- self.sun-data ---- ")
print(Site_obj.sun_data[(2023, 1, 10)])

print("---- self.single_scattering_albedo ---- ")
print(Site_obj.single_scattering_albedo[(2023, 1, 10)])

print("---- self.aerosol_asymmetry_factor ---- ")
print(Site_obj.aerosol_asymmetry_factor[(2023, 1, 10)])


# Let's print the columns in each DataFrame.
print("--------------------------- ")
print("        COLUMNS             ")
print("--------------------------- ")

print("---- self.climate_and_air_data ---- ")
print(Site_obj.CLIMATE_AND_AIR_DATA_COLS)

print("---- self.sun-data ---- ")
print(Site_obj.SUN_DATA_COLS)

print("---- self.single_scattering_albedo ---- ")
print(Site_obj.AEROSOL_COLS)

print("---- self.aerosol_asymmetry_factor ---- ")
print(Site_obj.AEROSOL_COLS)


# And also know their meaning and units.
print("--------------------------- ")
print("   DESCRIPTIONS AND UNITS   ")
print("--------------------------- ")

print(Site_obj.variables_info["descriptions"])
print(Site_obj.variables_info["units"])


#%%    4) FILLING Site_obj's MAIN ATTRIBUTE'S

# In order to perform a simulation, the data for each of Site_obj's main attributes must be provided.
# You can do this manually, if you have your own data that you'd like to use. In that case, just fill
# each of the DataFrames with it, for each date. However, the 'Site' class, already provides an arrange
# of methods that can be used for acquiring and filling all the required data as easy as possible.
# For this example, we shall follow this latter approach.
# Finally, while the execution of this tutorial should preferably be followed in the order intended, 
# here it is more or less required, since many quantities require the previous computation of other quantities.

# 4.1) --- SET AND COMPUTE CLIMATE DATA ---
# Compute and fill site data related to climate.
Site_obj.set_climate_data_from_pvgis_tmy_data(startyear=2005, endyear=2015)
Site_obj.compute_extraterrestrial_normal_irradiance()
Site_obj.compute_cummulative_time_integral_of_irradiances()

# 4.2) --- COMPUTE SUN DATA ---
# Compute and fill site data related sun position and relative airmass.
Site_obj.compute_sun_data()

# 4.3) --- COMPUTE AIR DATA ---
# Compute and fill site data related to air quantities.

# 4.31) -- ATMOSPHERIC OZONE COLUMN --
# There are two options for computing the ozone column.
# We can use the van heuklon model or, if we have the data
# (see tutorial 1), we can use satellite data.
"""
Site_obj.compute_ozone_column_using_van_Heuklon_model()
"""
Site_obj.compute_ozone_column_using_satelite_data(path = OZONE_DATABASE_PATH)

# 4.32) -- ATMOSPHERIC WATER COLUMN --
# There are two options for computing the water column.
# We can use the gueymard model or, if we have the data
# (see tutorial 1), we can use satellite data.
"""
Site_obj.compute_water_column_using_gueymard94_model()
"""
Site_obj.compute_water_column_using_satelite_data(path = WATER_DATABASE_PATH)

# 4.33) -- ANGSTROM EXPONENT AT 500 nm--
Site_obj.compute_angstrom_turbidity_exponent_500nm_using_SF_model()

# 4.34) -- AERSOL OPTICAL DEPTH AT 500 nm --
# For this step we do have to previously set up
# # the satellite data we need to use (see tutorial 1).
Site_obj.compute_aod_500nm_using_satelite_data(path = AOD_550NM_DATABASE_PATH)

# 4.4) --- COMPUTE AEROSOL PROPERTIES ---
Site_obj.compute_single_scattering_albedo_using_SF_model()
Site_obj.compute_aerosol_asymmetry_factor_using_SF_model()
Site_obj.compute_spectrally_averaged_aerosol_asymmetry_factor()


#%%    5) VISUALIZE AND MANIPULATE DATA
# We can very easily plot data from the main attributes and interpolated if need be.

# -- PLOT DATA --

# Normal direct irradiance.
Site_obj.plot_data(col    = "Gb(n)",
                   years  = None, 
                   months = None, 
                   days   = None, 
                   hours  = [6.5, 17.5],
                   mode   = 2, 
                   interp_method = "linear", 
                   figsize = (16, 12))

# Temperature.
Site_obj.plot_data(col = "T2m",
                years = None, 
                months = None, 
                days   = None, 
                hours  = [6.5, 17.5],
                mode   = 2, 
                interp_method = "linear", 
                figsize = (16, 12))

# Humidity.
Site_obj.plot_data(col = "RH",
                   years = None, 
                   months = None, 
                   days   = None, 
                   hours  = [6.5, 17.5],
                   mode   = 2, 
                   interp_method = "linear", 
                   figsize = (16, 12))

# ANGSTROM EXPONENT AT 500 nm
Site_obj.plot_data(col = "alpha_500nm",
                   years = None, 
                   months = None, 
                   days   = None, 
                   hours  = [6.5, 17.5],
                   mode   = 2, 
                   interp_method = "linear", 
                   figsize = (16, 12))


# -- INTERPOLATE DATA --

# Interpolate temperature data.
interpolated_data =\
Site_obj.time_interpolate_variable(col   = "T2m",
                                   year  = 2023,
                                   month = 1,
                                   day   = 1, 
                                   new_hms_float = np.linspace(6.5, 17.5, 100),
                                   interp_method = "linear")

print(interpolated_data)


#%%    6) SAVE Site_obj
# You can easily save the object as a pickle format for later use.

# You must change this path to hwere you want the pickle obj to be stored. Don't leave as is.
EXAMPLES_PATH = os.path.join(os.path.dirname(solrad.__path__[0]), "examples")
Site_obj_path = os.path.join(EXAMPLES_PATH, "example_Site_obj.pkl")
aux.save_obj_with_pickle(Site_obj, path = Site_obj_path)

#%%