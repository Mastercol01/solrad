#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of solar position data and related parameters.
"""

#%%                       IMPORTATION OF LIBRARIES
import warnings
import numpy as np
import pvlib as pv
import pandas as pd

#%%                       DEFINITION OF FUNCTIONS

def compute_sun_data(latitude, longitude, altitude, time_data, pressure_data, temperature_data, NaN_handling = "strict"):

  """
  Compute solar position data and related parameters for a specific location and time.

  Parameters
  ----------
  latitude : float
    Latitude of the location in degrees.

  longitude : float
    Longitude of the location in degrees.

  altitude : float or None
    Altitude of the location in meters. If None, 0 meters above sea level are asumed.

  time_data : dict
    A dictionary containing the time series for the simulation, separated by date.
    Its strucure is as follows. Each key must be a 3-tuple of (year : int, month : int, day :int) and each corresponding value has to be a
    pandas.DatetimeIndex object containing the time series of the date for which the solar position data is to be calculated.

  pressure_data : dict, float or None
    Normally, it is dictionary containing atmospheric pressure data [Pa] for each time step, separated by year, month, and day.
    Its keys must be the same as that as that of *time_data*. Its correponding values must be of type 'array_like', where
    each array contains the atmospheric pressure values associated with each DatetimeIndex object of *time_data*.
    If float, it means the pressure is constant throughout all timesteps and the same value is used.

  temperature_data : dict, float or None
    Normally, it is dictionary air temperature data [°C] for each time step, separated by year, month, and day.
    Its keys must be the same as that as that of *time_data*. Its correponding values must be of type 'array_like', where
    each array contains the air temperature values associated with each DatetimeIndex object of *time_data*.
    If float, it means the air temperature is constant throughout all timesteps and the same value is used.

  NaN_handling : {"strict", "loose", "null"}, optional
    How to handle NaN and None values when present in *pressure_data* and *temperature_data*.
    If "strict" an Exception is raised.
    If "loose", default values are used instead (see notes for more info).
    If "null", nothing is done about it and NaN/None values are directly passed onto the calculation, which may
    produce NaN results or raise another Exception. 
    Default is "strict".

  Returns
  --------
  sun_data : dict
      A dictionary containing solar data and related parameters for each specific year, month, and day.
      The keys are the same as for *time_data*. The corresping values are pandas.DataFrames whose
      index are the pandas.DatetimeIndex objects contained in *time_data* and whose columns
      contain the calculated sun position variables, time step. 
      Its columns, along with their descriptions, are as follows:

        1) "apzen" : Apparent zenith [°].

        2) "zen" : Zenith [°].

        3) "apel" : Apparent elevation [°].

        4) "el" : Elevation [°].

        5) "az" : Azimuth [°].

        6) "i" : x-component of Sun's unit direction vector [-].

        7) "j" : y-component of Sun's unit direction vector [-].

        8) "k" : z-component of Sun's unit direction vector [-].

        9) "rel_airmass": Relative airmass [-].

  Raises
  ------
  1) Exception 
    "NaN/None values present in pressure_data[date]"

  2) Exception 
    "NaN/None values present in temperature_data[date]"

  Warns
  -----
  1) Warning 
    "NaN/None values present in pressure_data[date]. Using default values instead."

  2) Warning 
    "NaN/None values present in temperature_data[date]. Using default values instead."

  See Also
  --------
  pvlib.solarposition.get_solarposition

  Notes
  -----
  1) In case that NaN_handling is "loose", the default value of temperature used is 15°C and the default
  value of pressure is computed from altitude using the function ``pvlib.atmosphere.alt2pres``.

  References
  ----------
  https://pvlib-python.readthedocs.io/en/v0.4.2/generated/pvlib.solarposition.get_solarposition.html

  Examples
  --------
  >>> import solrad.geotime as tm
  >>> from solrad.sun.sun import compute_sun_data
  >>>
  >>> time_data = tm.geo_date_range(latitude   = 6.2518,
  >>>                               longitude  = -75.5636,
  >>>                               tz         = "-05:00",
  >>>                               start_time = "2023-01-01 00:00:00",
  >>>                               end_time   = "2023-01-02 23:59:59.999",
  >>>                               freq       = "5min",
  >>>                               min_hms    = "sunrise",
  >>>                               max_hms    = "sunset")
  >>>
  >>> len_day1 = len(time_data[(2023,1,1)])
  >>> len_day2 = len(time_data[(2023,1,2)])
  >>>
  >>> pressure_data    = {(2023, 1, 1): np.full(len_day1, 101325),
  >>>                     (2023, 1, 2): np.full(len_day2, 101270)}
  >>> temperature_data = {(2023, 1, 1): np.full(len_day1, 25),
  >>>                     (2023, 1, 2): np.full(len_day2, 26.5)}
  >>>
  >>> sun_data = compute_sun_data(latitude         = 6.2518,
  >>>                             longitude        = -75.5636,
  >>>                             altitude         = 10,
  >>>                             time_data        = time_data,
  >>>                             pressure_data    = pressure_data,
  >>>                             temperature_data = temperature_data)
  """

  pressure_from_altitude = pv.atmosphere.alt2pres(altitude)
  # 'pressure_from_altitude' suposes:
  # - Temperature at sea level    = 15°C 
  # - Pressure at sea level       = 101325 Pa
  # - Gravitational acceleration  = 9.80665 m/s^2
  # - Gas constant for air        = 287.053 J/(kg K)
  # - Relative Humidity           = 0%


  sun_data = {}
  for date, DatetimeIndex_obj in time_data.items():
      try:
          current_pressure    = pressure_data[date]
          current_temperature = temperature_data[date]

      except TypeError:
          current_pressure    = pressure_data
          current_temperature = temperature_data


      current_pressure_is_NaN    = np.array(pd.isnull(current_pressure))
      current_temperature_is_NaN = np.array(pd.isnull(current_temperature))

      if NaN_handling == "strict":
          if current_pressure_is_NaN.any():
              raise Exception(f"NaN/None values present in pressure_data[{date}]")

          if current_temperature_is_NaN.any():
              raise Exception(f"NaN/None values present in temperature_data[{date}]")

      elif NaN_handling == "loose":
          if current_pressure_is_NaN.any():
              msg = f"NaN/None values present in pressure_data[{date}]"
              msg = f"{msg}. Using default values instead."
              current_pressure[current_pressure_is_NaN] = pressure_from_altitude
              warnings.warn(msg)

          if current_temperature_is_NaN.any():
              msg = f"NaN/None values present in temperature_data[{date}]"
              msg = f"{msg}. Using default values instead."
              current_temperature[current_temperature_is_NaN] = 15
              warnings.warn(msg)
          
      elif NaN_handling != "null":
          raise Exception("Value of 'NaN_handling' is invalid.")


      local_sun_data =\
      pv.solarposition.get_solarposition(time        = DatetimeIndex_obj,
                                          latitude    = latitude,
                                          longitude   = longitude,
                                          altitude    = altitude,
                                          pressure    = current_pressure,
                                          method      = 'nrel_numpy',
                                          temperature = current_temperature)

      hms_float  = DatetimeIndex_obj.hour
      hms_float += DatetimeIndex_obj.minute/60
      hms_float += DatetimeIndex_obj.second/3600


      sun_data[date] =\
      pd.DataFrame(index = DatetimeIndex_obj, columns = ["hms_float", "apzen",
                                                         "zen", "apel", "el", "az",
                                                         "i", "j", "k", "rel_airmass"])

      sun_data[date]["hms_float"] = hms_float
      sun_data[date]["apzen"]     = local_sun_data["apparent_zenith"]
      sun_data[date]["zen"]       = local_sun_data["zenith"]
      sun_data[date]["apel"]      = local_sun_data["apparent_elevation"]
      sun_data[date]["el"]        = local_sun_data["elevation"]
      sun_data[date]["az"]        = local_sun_data["azimuth"]

      theta = np.deg2rad(np.array(local_sun_data["apparent_zenith"]).astype(float))
      phi   = np.deg2rad(np.array(local_sun_data["azimuth"]).astype(float))

      sun_data[date]["i"] = np.cos(phi)*np.sin(theta)
      sun_data[date]["j"] = np.sin(phi)*np.sin(theta)
      sun_data[date]["k"] = np.cos(theta)


      sun_data[date]["rel_airmass"] =\
      pv.atmosphere.get_relative_airmass(theta, model='kastenyoung1989')


  return sun_data
  

