#%%                 IMPORTATION OF LIBRARIES

from solrad.geotime import *

#%%                 OTHER TESTS

latitude = 6.230833
longitude = -75.56359
tz = '-05:00'
start_time = "2023-1-1 00:00:00"
end_time   = "2023-1-31 23:59:59.999"
freq = "5min"
min_hms = None
max_hms = None
skip_polar_nights = True
inclusive = False
res = geo_date_range(latitude, longitude, tz, start_time, end_time, freq, min_hms, max_hms, skip_polar_nights, inclusive)

#%%