
#%%

import solrad
import matplotlib.pyplot as plt
from solrad.atmosphere.aod_550nm import *


# We define the path of the folder where we want to save the retrieved ozone data.
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(solrad.__path__[0])), "data")
AOD_550NM_DATABASE_PATH = os.path.join(DATA_PATH, "aod_550nm")

#%%
# We retrieve the aod data for all of the months of "2020", "2021" and
# "2022" as an example.
year = ['2020', '2021', '2022']
get_CDS_aod_550nm_data(path = AOD_550NM_DATABASE_PATH, 
                       year = year,
                       file_format = "numpy")

#%%
# Let us compute the filled-NaN values of all .nc filles present in 
# the local aod_550 databse. This may take a while if there are many.
fill_CDS_aod_550nm_data_nans(path = AOD_550NM_DATABASE_PATH, 
                             iterations = 20000,
                             show_progress=True,
                             replace = True)

#%%
# Having our local ozone databse well established, we process the retrieved data.
res = process_CDS_aod_550nm_data(AOD_550NM_DATABASE_PATH, percentile = 0.5, interp_method = "linear")

#%%
# Let us plot some of the data.

year, month = 2020, 6

raw_data = res["raw_data"][(year, month)]
filled_nans_data = res["filled_nans_data"][(year, month)]
avg_data = res["avg_data"][month] 
percentile_data = res["percentile_data"][month]


lat = res["latitude"]
lon = res["longitude"]
# Let us plot some of the data.
Lon, Lat = np.meshgrid(lon, lat)

fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, raw_data, levels = np.linspace(np.nanmin(raw_data), np.nanmax(raw_data), 100))
cbar = plt.colorbar()
plt.title(f"raw_data @ year = {year}, month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('AOD 550nm [-]')
plt.show()

#%%
fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, filled_nans_data, levels = np.linspace(filled_nans_data.min(), filled_nans_data.max(), 100))
cbar = plt.colorbar()
plt.title(f"filled_nans_data @ year = {year}, month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('AOD 550nm [-]')
plt.show()


fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, avg_data, levels = np.linspace(avg_data.min(),avg_data.max(), 100))
cbar = plt.colorbar()
plt.title(f"avg_data @ month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('AOD 550nm [-]')
plt.show()


fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, percentile_data, levels = np.linspace(percentile_data.min(),percentile_data.max(), 100))
cbar = plt.colorbar()
plt.title(f"percentile_data @ month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('AOD 550nm [-]')
plt.show()

#%%
# And also interpolate it.
lat_, lon_ = 6.230833, -75.590553 # Medellín coordinates
filled_nans_data_ = res["filled_nans_data_funcs"][(year, month)]([lat_, lon_])
avg_data_ = res["avg_data_funcs"][(month)]([lat_, lon_])
percentile_data_ = res["percentile_data_funcs"][(month)]([lat_, lon_])

print(f"filled_nans_data @ year = {year}, month = {month} is {filled_nans_data_}")
print(f"avg_data @ month = {month} is {avg_data_}")
print(f"percentile_data @ month = {month} is {percentile_data_}")


#%%