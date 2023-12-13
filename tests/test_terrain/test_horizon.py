#%%                 IMPORTATION OF LIBRARIES
import pytest
import matplotlib
import numpy as np
from solrad.terrain.horizon import *

#%%                 DEFINITION OF CONSTANTS

MEDELLIN_LAT, MEDELLIN_LON = 6.25184, -75.56359

MEDELLIN_HORIZON_AZIMUTH =\
np.array([  0. ,   7.5,  15. ,  22.5,  30. ,  37.5,  45. ,  52.5,  60. ,
           67.5,  75. ,  82.5,  90. ,  97.5, 105. , 112.5, 120. , 127.5,
          135. , 142.5, 150. , 157.5, 165. , 172.5, 180. , 187.5, 195. ,
          202.5, 210. , 217.5, 225. , 232.5, 240. , 247.5, 255. , 262.5,
          270. , 277.5, 285. , 292.5, 300. , 307.5, 315. , 322.5, 330. ,
          337.5, 345. , 352.5, 360. ])

MEDELLIN_HORIZON_ELEVATION =\
np.array([ 4.2,  4.6,  4.2,  3.1,  5. ,  8. , 10.3, 11.8, 11.8, 12.2, 12.6,
       10.7, 10.7,  9.9,  8.8,  7.6,  8. ,  9.5, 10.7, 10.3,  9.5,  7.6,
        6.5,  5. ,  5. ,  3.8,  2.7,  1.5,  1.9,  3.1,  4.2,  4.6,  4.2,
        3.8,  4.6,  5.3,  5.3,  5.7,  6.1,  5. ,  4.6,  5.7,  7.6,  8. ,
        6.5,  5.3,  3.8,  3.8,  4.2])

FUNC = horizon_func_from_arrays(MEDELLIN_HORIZON_AZIMUTH,
                                MEDELLIN_HORIZON_ELEVATION, 
                                interp_method = 'linear')


#%%                            PYTESTS

def test_horizon_arrays_from_pvgis():
   azimuth, elevation = horizon_arrays_from_pvgis(latitude  =  MEDELLIN_LAT,
                                                  longitude =  MEDELLIN_LON)
   logic_az = azimuth   == pytest.approx(MEDELLIN_HORIZON_AZIMUTH)
   logic_el = elevation == pytest.approx(MEDELLIN_HORIZON_ELEVATION)
   assert logic_az and logic_el


def test_compute_fraction_of_unblocked_sky_patch_by_horizon():
   az0, az1 = 50, 150
   for el0, el1, true_val in [(0, 7, 0), (7, 14, 0.5454174242220962), (14, 21, 1)]:
      val = compute_fraction_of_unblocked_sky_patch_by_horizon(FUNC, el0, el1, az0, az1)
      assert val == pytest.approx(true_val)

                                    

#%%                         OTHER TESTS

plot_horizon(FUNC, MEDELLIN_HORIZON_AZIMUTH)
plot_horizon(FUNC, MEDELLIN_HORIZON_AZIMUTH, config={"projection": "cartesian"})

# %%
