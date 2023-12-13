#%%              IMPORTATION OF LIBRARIES

import numpy as np
import solrad.geotime as gtm
from solrad.climate.pvgis_tmy import *

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
                                end_time   = "2023-02-01 23:59:59.999",
                                freq       = "5min",
                                min_hms    = "sunrise",
                                max_hms    = "sunset")

# We get the original tmy-dataframe for the site.
pvgis_tmy_data = get_pvgis_tmy_dataframe(latitude  = 6.2518,
                                        longitude = -75.5636,
                                        tz        = "-05:00",
                                        startyear = 2005,
                                        endyear   = 2015)

# We get the interpolated climate data for the site.
climate_data = climate_data_from_pvgis_tmy_dataframe(time_data = time_data,
                                                     tmy_data   = pvgis_tmy_data)
# %%
