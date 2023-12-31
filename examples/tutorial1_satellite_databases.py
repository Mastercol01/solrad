#%%                     DESCRIPTION
"""
In this tutorial example we request information from the Climate Data Store (CDS) for the following databases
of atmospheric quantities:

(OPTIONAL)
[1] Copernicus Climate Change Service, Climate Data Store, (2020): 
Ozone monthly gridded data from 1970 to present derived from satellite observations.
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.4ebfe4eb 

(OPTIONAL)
[2] Preusker, R., El Kassar, R. (2022): Monthly total column water vapour over 
land and ocean from 2002 to 2012 derived from satellite observation. Copernicus
Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.8e0e4724 

(REQUIRED)
[3] Copernicus Climate Change Service, Climate Data Store, (2019): Aerosol
properties gridded data from 1995 to present derived from satellite observation. 
Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.239d815c 

We request, download and process said information. This quantities are useful/required for the 
computation of the spectral component of radiance. 

Finally, if followed to completion, the steps carried out in this tutorial need only be followed once.


                  ---- CLIMATE DATA STORE ----

"The Copernicus - Climate Data Store (CDS) is an online open and free service
that allows users to browse and access a wide range of climate datasets via a 
searchable catalogue. It also allows users to build their own applications,
maps and graphs."

1) CDS webpage at:
https://cds.climate.copernicus.eu/cdsapp#!/home

2) More info about its API, at:
https://cds.climate.copernicus.eu/api-how-to

3) Useful tutorial on how to use the API, at
https://youtu.be/cVtiVTSVdlo
    

NOTE: As described by the links in 2) and 3), it is necessary to have a CDS 
      account (and be currently logged in) account in order to be able to use
      the API. Furtheremore, the user's key and the API website link should be 
      stored in a place, recognisable by the system being used. We recommend 
      watching the video tutorial linked in 3), as it is a good explanation on 
      what steps need to be followed in order to work with cdsapi.

"""
#%%                     IMPORTATION OF LIBRARIES
import os
import solrad
import numpy as np
import matplotlib.pyplot as plt
import solrad.atmosphere.ozone_column as oz
import solrad.atmosphere.water_column as wat
import solrad.atmosphere.aod_550nm as aod550

#%%                     DEFINITION DEFAULT PATH CONSTANTS

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(solrad.__path__[0])), "data")
OZONE_DATABASE_PATH = os.path.join(DATA_PATH, "ozone")
WATER_DATABASE_PATH = os.path.join(DATA_PATH, "water")
AOD_550NM_DATABASE_PATH = os.path.join(DATA_PATH, "aod_550nm")

#%%                     DEFINITION DEFAULT PATH VARIABLES

# Define the directory where the data files for each atmospheric quantity will be stored.
# (you must change the value of these paths or set them to None).
# Finally, there are other methods for computing the atmospheric water and the ozone columns. Therefore,
# these fields are optional. However, the aerosol optical depth at 550nm (aod_550nm) does require the 
# use of satellite data.

# (OPTIONAL)
ozone_database_path = OZONE_DATABASE_PATH

# (OPTIONAL)
water_database_path = WATER_DATABASE_PATH

# (REQUIRED)
aod_550nm_database_path = AOD_550NM_DATABASE_PATH

#%%                 RETRIEVE DATA

# We retrieve the database files for each solicited atmospheric quantity.

if ozone_database_path is not None:
    years = ["2019", "2020", "2021"]
    oz.get_CDS_ozone_column_data(path = ozone_database_path,
                                 year = years,
                                 month = None,
                                 file_format = "numpy")
    
if water_database_path is not None:
    years = ["2009", "2010", "2011"]
    wat.get_CDS_water_column_data(path = water_database_path,
                                  year = years,
                                  month = None,
                                  file_format = "numpy")
    
if aod_550nm_database_path is not None:
    years = ["2020", "2021", "2022"]
    aod550.get_CDS_aod_550nm_data(path = aod_550nm_database_path,
                                  year = years,
                                  month = None,
                                  file_format = "numpy")
    
#%%                  FILL DATA NaNs

# We interpolate and fill the NaNs in the data for all required files.
# (Only water column and aod_550nm data require this treatment.) This step
# may actually take quite a while.    

if water_database_path is not None:
    wat.fill_CDS_water_column_data_nans(path = water_database_path, 
                                        iterations = 20000,
                                        show_progress = True,
                                        replace = True)
    
if aod_550nm_database_path is not None:
    aod550.fill_CDS_aod_550nm_data_nans(path = aod_550nm_database_path, 
                                        iterations = 20000,
                                        show_progress = True,
                                        replace = True)

#%%                  PLOT PROCESSED DATA AND RAW DATA (IF AVAILABLE)

for var in ["ozone", "water", "aod550"]:

    if var == "ozone":
        if ozone_database_path is None:
            continue
        else:
            res = oz.process_CDS_ozone_column_data(ozone_database_path, percentile = 0.5, interp_method = "linear")
            year, month = 2021, 5
            avg_data = res["avg_data"][(month)]
            percentile_data = res["percentile_data"][(month)]
            lon, lat = res["longitude"], res["latitude"]
            Lon, Lat = np.meshgrid(lon, lat)
            label = 'ozone column [atm-cm]'

    elif var == "water":
        if water_database_path is None:
            continue
        else:
            res = wat.process_CDS_water_column_data(water_database_path, percentile = 0.5, interp_method = "linear")
            year, month = 2011, 5
            avg_data = res["avg_data"][(month)]
            percentile_data = res["percentile_data"][(month)]
            lon, lat = res["longitude"], res["latitude"]
            Lon, Lat = np.meshgrid(lon, lat)   
            label = 'water column [atm-cm]'         

    elif var == "aod550":
        if aod_550nm_database_path is None:
            continue
        else:
            res = aod550.process_CDS_aod_550nm_data(aod_550nm_database_path, percentile = 0.5, interp_method = "linear")
            year, month = 2022, 5
            avg_data = res["avg_data"][(month)]
            percentile_data = res["percentile_data"][(month)]
            lon, lat = res["longitude"], res["latitude"]
            Lon, Lat = np.meshgrid(lon, lat) 
            label = 'AOD 550nm [-]'  


    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, avg_data, levels = np.linspace(avg_data.min(),avg_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"avg_data @ month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label(label)
    plt.show()

    fig = plt.figure(figsize=(15,10))
    plt.contourf(Lon, Lat, percentile_data, levels = np.linspace(percentile_data.min(),percentile_data.max(), 100))
    cbar = plt.colorbar()
    plt.title(f"percentile_data @ month = {month}")
    plt.ylabel("Latitude [°]")
    plt.xlabel("Longitude [°]")
    cbar.set_label(label)
    plt.show()

    try:
        raw_data = res["raw_data"][(year, month)]
        fig = plt.figure(figsize=(15,10))
        plt.contourf(Lon, Lat, raw_data, levels = np.linspace(np.nanmin(raw_data), np.nanmax(raw_data), 100) )
        cbar = plt.colorbar()
        plt.title(f"raw_data @ year = {year}, month = {month}")
        plt.ylabel("Latitude [°]")
        plt.xlabel("Longitude [°]")
        cbar.set_label(label)
        plt.show()
    except KeyError:
        pass



# %%
