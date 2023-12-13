#%%                       IMPORTATION OF LIBRARIES
import numpy as np
import pvlib as pv
import pandas as pd
from scipy.interpolate import interp1d

#%%                 DEFINITION OF FUNCTIONS

def get_pvgis_tmy_dataframe(latitude, longitude, tz, startyear, endyear, usehorizon = False, userhorizon = None):

    """
    Get Typical Meteorological Year (TMY) data from PVGIS.

    Parameters
    -----------
    latitude : float
        Site's latitude in degrees. Must be a number between -90 and 90.

    longitude : float
        Site's longitude in degrees. Must be a number between -180 and 180.

    tz : str
        Time zone string accepted by pandas.

    startyear: int or None
        First year to calculate TMY.

    endyear : int or None
        Last year to calculate TMY, must be at least 10 years from first year.

    usehorizon : bool, optional
      Wether to include effects of horizon. Default is False.

    userhorizon : list of float or None, optional
      Optional user specified elevation of horizon in degrees, at equally spaced azimuth clockwise 
      from north, only valid if usehorizon is true, if usehorizon is true but userhorizon is None then 
      PVGIS will calculate the horizon.

      
    Returns
    -------
    tmy_data : pandas.DataFrame obj of floats
        DataFrame contining the PVGIS TMY data for the whole year.
        Its index is a multiindex of (month, day, hour). Its columns, along
        with their descriptions, are as follows:

        1) "T2m": 2-m air temperature (degree Celsius)

        2) "RH": relative humidity (%)

        3) "G(h)": Global irradiance on the horizontal plane (W/m2)

        4) "Gb(n)": Beam/direct irradiance on a plane always normal to sun rays (W/m2)

        5) "Gd(h)": Diffuse irradiance on the horizontal plane (W/m2)

        6) "IR(h)": Surface infrared (thermal) irradiance on a horizontal plane (W/m2)

        7) "WS10m": 10-m total wind speed (m/s)

        8) "WD10m": 10-m wind direction (0 = N, 90 = E) (degree)

        9) "SP": Surface (air) pressure (Pa)



    Notes
    -----
    1) Latitude of -90° corresponds to the geographic South pole, while a
       latitude of 90° corresponds to the geographic North Pole.

    2) A negative longitude correspondes to a point west of the greenwhich
       meridian, while a positive longitude means it is east of the greenwhich
       meridian.

    3) The PVGIS website uses 10 years of data to generate the TMY, whereas the
       API accessed by this function defaults to using all available years.
       This means that the TMY returned by this function may not be identical
       to the one generated by the website. To replicate the website requests,
       specify the corresponding 10 year period using startyear and endyear.
       Specifying endyear also avoids the TMY changing when new data becomes
       available.


    Examples
    --------
    >>> from climate.pvgis_tmy import get_pvgis_tmy_dataframe
    >>> pvgis_tmy_data = get_pvgis_tmy_dataframe(latitude  = 6.2518,
    >>>                                          longitude = -75.5636,
    >>>                                          tz        = "-05:00",
    >>>                                          startyear = 2005,
    >>>                                          endyear   = 2015)

    """



    #          CONSTRUCTION OF TMY_DATA DATAFRAME


    # Get Typical Meteorological Year (TMY) data of the site in question,
    # from the PVGIS database. This is done through the 'pvlib' library
    # function 'iotools.get_pvgis_tmy'.

    # CHECK THE LINKS BELOW FOR MORE INFO:
    # 1) https://pvlib-python.readthedocs.io/en/v0.9.0/generated/pvlib.iotools.get_pvgis_tmy.html?msclkid=3aacaceecfc211ec8bec2645c1a03011
    # 2) https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/pvgis-tools/tmy-generator_en
    # 3) https://re.jrc.ec.europa.eu/pvg_tools/en/#TMY
    # 4) https://joint-research-centre.ec.europa.eu/pvgis-photovoltaic-geographical-information-system/getting-started-pvgis/pvgis-data-sources-calculation-methods_en


    # T2m: 2-m air temperature (degree Celsius)
    # RH: relative humidity (%)
    # G(h): Global irradiance on the horizontal plane (W/m2)
    # Gb(n): Beam/direct irradiance on a plane always normal to sun rays (W/m2)
    # Gd(h): Diffuse irradiance on the horizontal plane (W/m2)
    # IR(h): Surface infrared (thermal) irradiance on a horizontal plane (W/m2)
    # WS10m: 10-m total wind speed (m/s)
    # WD10m: 10-m wind direction (0 = N, 90 = E) (degree)
    # SP: Surface (air) pressure (Pa)

    tmy_data = pv.iotools.get_pvgis_tmy(latitude = latitude,
                                        longitude = longitude,
                                        usehorizon = usehorizon,
                                        userhorizon = userhorizon,
                                        startyear = startyear,
                                        endyear = endyear,
                                        outputformat = 'csv',
                                        map_variables = True)[0]


    # At this point 'tmy_data' is a DataFrame consisting of 8760 rows.
    # Each row has the TMY info of a a particular hour of a particular day
    # of the year (24*365 = 8760). The thing is, the index of the DataFrame
    # (which is given in hours), was given in terms of UTC=0, that is, in
    # terms of greenwhich median time. We then have to convert it to local
    # time:

    new_tmy_index = tmy_data.index.tz_convert(tz)
    tmy_data = tmy_data.reindex(new_tmy_index)

    # After the time index conversion, we sort the values, in hierarchical
    # order, by Month, Day and Hour:

    tmy_data["Date"] = tmy_data.index

    tmy_data["Month"] = tmy_data["Date"].apply(lambda x:x.month)
    tmy_data["Day"]   = tmy_data["Date"].apply(lambda x:x.day)
    tmy_data["Hour"]  = tmy_data["Date"].apply(lambda x:x.hour)

    tmy_data = tmy_data.sort_values(by=["Month", "Day", "Hour"])
    tmy_data.drop(columns=["Date"], inplace=True)

    # We then create a Multiindex for the DataFrame. In this way we may
    # access any value by specifying the Month, Day and Hour.

    tmy_data = tmy_data.set_index(["Month", "Day", "Hour"])

    # We change the names of the columns to give them more standard names:
    new_cols_dict = {"temp_air":"T2m", "relative_humidity":"RH", "ghi":"G(h)",
                     "dni":"Gb(n)", "dhi":"Gd(h)", "wind_speed":"WS10m",
                     "wind_direction":"WD10m", "pressure":"SP"}

    tmy_data = tmy_data.rename(columns=new_cols_dict)


    return tmy_data




def climate_data_from_pvgis_tmy_dataframe(time_data, tmy_data, interp_method = "linear"):

    """
    Generate climate data from PVGIS TMY (Time Meteorological Year) DataFrame,
    using an interpolation method of user choice.

    Parameters:
    -----------
    time_data : dict
        A dictionary containing the time series for simulation, separated by date.
        Its strucure is as follows. Each key must be a 3-tuple of (year : int, month : int, day :int) and each corresponding value has to be a
        pandas.DatetimeIndex object containing the time series of the date for which the climate data is to be calculated.

    tmy_data : pandas.DataFrame
        A DataFrame containing pvgis' TMY (Time Meteorological Year) data.

    interp_method : {'linear', 'quadratic', 'cubic'}, optional
        The interpolation method to be used. Defaults is 'linear'.

    Returns:
    --------
    climate_data : dict
        A dictionary containing climate data for each specific year, month, and day.
        The keys are the same as for 'time_data'. The corresping values are pandas.DataFrames whose
        index are the pandas.DatetimeIndex objects contained in 'time_data' and whose columns
        contain the climate variables from 'tmy_data', interpolated and evaluated at each
        time step.

    See Also
    --------
    1) get_pvgis_tmy_dataframe
    2) Solrad.Time2.geo_date_range

    Examples:
    ---------

    >>> import geotime as tm
    >>> from climate.pvgis_tmy import get_pvgis_tmy_dataframe
    >>> from climate.pvgis_tmy import climate_data_from_pvgis_tmy_dataframe
    >>>
    >>> time_data = tm.geo_date_range(latitude   = 6.2518,
    >>>                               longitude  = -75.5636,
    >>>                               tz         = "-05:00",
    >>>                               start_time = "2023-01-01 00:00:00",
    >>>                               end_time   = "2023-02-01 23:59:59.999",
    >>>                               freq       = "5min",
    >>>                               min_hms    = "sunrise",
    >>>                               max_hms    = "sunset")
    >>>
    >>> pvgis_tmy_data = get_pvgis_tmy_dataframe(latitude  = 6.2518,
    >>>                                          longitude = -75.5636,
    >>>                                          tz        = "-05:00",
    >>>                                          startyear = 2005,
    >>>                                          endyear   = 2015)
    >>>
    >>> climate_data = climate_data_from_pvgis_tmy_dataframe(time_data = time_data,
    >>>                                                      tmy_data   = pvgis_tmy_data)
    """

    climate_data = {}
    x_interp = np.arange(25)

    for (year, month, day), DatetimeIndex_obj in time_data.items():

      hms_float  = DatetimeIndex_obj.hour
      hms_float += DatetimeIndex_obj.minute/60
      hms_float += DatetimeIndex_obj.second/3600

      climate_data[(year, month, day)] = pd.DataFrame(index   = DatetimeIndex_obj,
                  columns = ["hms_float"] + list(tmy_data.columns))

      climate_data[(year, month, day)]["hms_float" ] = hms_float


      day_ = day
      if (month, day) == (2, 29):
        day_ = 28


      for col in tmy_data.columns:
        y_interp = list(tmy_data.loc[(month, day_), col])

        try:
          y_interp.append(tmy_data.loc[(month, day_+1, 0), col])

        except KeyError:
          try:
            y_interp.append(tmy_data.loc[(month+1, 1, 0), col])

          except KeyError:
            try:
              y_interp.append(tmy_data.loc[(1, 1, 0), col])

            except Exception as m:
              raise(m)

        interp_func = interp1d(x_interp, y_interp, kind = interp_method)
        climate_data[(year, month, day)][col] = interp_func(hms_float)

    return climate_data




# %%