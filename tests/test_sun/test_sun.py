#%%              IMPORTATION OF LIBRARIES

import numpy as np
import solrad.geotime as gtm
from solrad.sun.sun import *

#%%           DEFINITION OF CONSTANTS

MEDELLIN_LAT = 6.25184
MEDELLIN_LON = -75.56359
MEDELLIN_TZ = "-05:00"


#%%                OTHER TESTS
   
# Define time range. We take the first and second of january,
# from sunrise to sunset.
time_data = gtm.geo_date_range(latitude    = MEDELLIN_LAT,
                                longitude  = MEDELLIN_LON,
                                tz         = MEDELLIN_TZ,
                                start_time = "2023-01-01 00:00:00",
                                end_time   = "2023-01-02 23:59:59.999",
                                freq       = "5min",
                                min_hms    = "sunrise",
                                max_hms    = "sunset")


# We initialize the pesure and temperature data.
len_day1 = len(time_data[(2023,1,1)])
len_day2 = len(time_data[(2023,1,2)])

pressure_data    = {(2023, 1, 1): np.full(len_day1, 101325),
                    (2023, 1, 2): np.full(len_day2, 101270)}
temperature_data = {(2023, 1, 1): np.full(len_day1, 25),
                    (2023, 1, 2): np.full(len_day2, 26.5)}

# We compute the sun data.
sun_data = compute_sun_data(latitude         = 6.2518,
                            longitude        = -75.5636,
                            altitude         = 10,
                            time_data        = time_data,
                            pressure_data    = pressure_data,
                            temperature_data = temperature_data)
# %%
