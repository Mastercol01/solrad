#%%                        IMPORTATION OF LIBRARIES
import os
import solrad
import pandas as pd
import matplotlib.pyplot as plt
from solrad.atmosphere.ozone_column import *

# We define the path of the folder where we want to save the retrieved ozone data.
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(solrad.__path__[0])), "data")
OZONE_DATABASE_PATH = os.path.join(DATA_PATH, "ozone")


#%%
# We retrieve the ozone data for all of the months of "2019", "2020" and 
# "2021". Actually, for this level of data, this are the only years we
# can retrieve.

year = ['2019', '2020', '2021']
get_CDS_ozone_column_data(path = OZONE_DATABASE_PATH,
                          year = year,
                          file_format = "numpy")

#%%
# Having our local ozone databse well established, we process the retrieved data.
res = process_CDS_ozone_column_data(OZONE_DATABASE_PATH, percentile = 0.5, interp_method = "linear")

#%%
# Let us plot some of the data.
year, month = 2021, 5
raw_data = res["raw_data"][(year, month)]
avg_data = res["avg_data"][(month)]
percentile_data = res["percentile_data"][(month)]

lon, lat = res["longitude"], res["latitude"]
Lon, Lat = np.meshgrid(lon, lat)

fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, raw_data, levels = np.linspace(np.nanmin(raw_data), np.nanmax(raw_data), 100) )
cbar = plt.colorbar()
plt.title(f"raw_data @ year = {year}, month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('ozone column [atm-cm]')
plt.show()

fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, avg_data, levels = np.linspace(avg_data.min(),avg_data.max(), 100))
cbar = plt.colorbar()
plt.title(f"avg_data @ month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('ozone column [atm-cm]')
plt.show()

fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, percentile_data, levels = np.linspace(percentile_data.min(),percentile_data.max(), 100))
cbar = plt.colorbar()
plt.title(f"percentile_data @ month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('ozone column [atm-cm]')
plt.show()

#%%
# And also interpolate it.
lat_, lon_ = 6.230833, -75.590553 # Medellín coordinates
raw_data_ = res["raw_data_funcs"][(year, month)]([lat_, lon_])
avg_data_ = res["avg_data_funcs"][(month)]([lat_, lon_])
percentile_data_ = res["percentile_data_funcs"][(month)]([lat_, lon_])

print(f"raw_data @ year = {year}, month = {month} is {raw_data_}")
print(f"avg_data @ month = {month} is {avg_data_}")
print(f"percentile_data @ month = {month} is {percentile_data_}")

#%%

# Let us also put van_Heuklon's model to the test and plotted.

MONTH_DAYS =\
{0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

if month < 10: month_ = f"0{month}"
else: month_ = str(month)
    
    
# So that the comparison be fair, we also compute the ozone monthly
# average using van heuklen.
van_heuklen_data = np.zeros(raw_data.shape)  
for day in range(1, MONTH_DAYS[month]+1):
    
    timestamp = pd.Timestamp(f"{year}-{month_}-{day}")
    
    for i, lat_ in enumerate(lat):
            van_heuklen_data[i,:] += compute_van_Heuklon_ozone(lat_, lon, timestamp)
            
    print(f"Days computed: {day}")

van_heuklen_data /= MONTH_DAYS[month]


fig = plt.figure(figsize=(15,10))
plt.contourf(Lon, Lat, van_heuklen_data, levels = np.linspace(van_heuklen_data.min(),van_heuklen_data.max(), 100))
cbar = plt.colorbar()
plt.title(f"van_heuklen_data @ year ={year}, month = {month}")
plt.ylabel("Latitude [°]")
plt.xlabel("Longitude [°]")
cbar.set_label('ozone column [atm-cm]')
plt.show()

# %%
