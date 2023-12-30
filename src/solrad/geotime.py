#%%                MODULE DESCRIPTION AND/OR INFO
"""
This is a module containing functions, methods and classes related to the
computation of time and date quantities.
"""

#%%                                   IMPORTATION OF LIBRARIES
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from suntimes import SunTimes
from pandas.core.indexes.datetimes import DatetimeIndex
from pvlib.solarposition import sun_rise_set_transit_spa

#%%                              DEFINITION OF CONSTANTS

# Number of days each month posesses.
MONTH_DAYS =\
{0:0, 1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

# Dict for timestamp to float conversion.
TIMESTAMP_HMS_TO_FLOAT_DICT = { "d" : [1/24, 1/1440, 1/86400],
                                "h" : [1, 1/60, 1/3600],
                                "m" : [60, 1, 1/60],
                                "s" : [3600, 60, 1] }



#%%                           DEFINITION OF FUNCTIONS

def utc_hour_to_tz_name(utc_hour):
    """
    Turns float representing the time zone into a string representing the time
    zone which is accepted by pandas.

    Parameters
    ----------
    utc_hour : float
        Timezone number. Must be anumber between -12 and 12.

    Returns
    -------
    tz_name : str
        Time zone string accepted by pandas.

    Notes
    -----
    1) For more information about the time zone strings accepted by pandas, see
       the link: https://pvlib-python.readthedocs.io/en/v0.3.0/timetimezones.html

    """
    utc_hour = int(utc_hour)
    tz_name = "Etc/GMT"
    if(utc_hour <= 0):  tz_name = "".join([tz_name, "+"])
    tz_name = "".join([tz_name, str(-utc_hour)])

    return tz_name



def timestamp_hms_to_float(timestamp, unit = "h"):

    """
    Convert Timestamp Hour:Minutes:Seconds information to float.

    Example: timestamp_hms_to_float(timestamp, unit = "h"), where
    timestamp = pd.Timestamp("2023-03-08 14:25:36") returns 14.426667. That is,
    it turns the 14h 25min 36s of the timestamp to an equivalent number
    of hours. Had we used timestamp_hms_to_float(timestamp, unit = "s"),
    the result would have been 51936. That is, the equivalent of 14h 25min 36s
    in seconds.


    Parameters
    ----------
    timestamp : pandas.Timestamp object
        Timestamp to convert to float.

    unit : str, optional
        Time unit to which the timestamp is to be converted. It can either be
        'd' (day), 'h' (hour), 'm' (minute) or 's' (second). Default is 'h'.

    tz_name : str
        Time zone string accepted by pandas.


    Returns
    -------
    res : float
        timestamp converted to float to the specified unit.


    """

    conversion = TIMESTAMP_HMS_TO_FLOAT_DICT[unit]

    res = timestamp.hour*conversion[0]
    res += timestamp.minute*conversion[1]
    res += timestamp.second*conversion[2]

    return res


#%%



def compute_sunrise_sunset(latitude, longitude, tz, start, end = None):

    """
    Compute sunrise and sunset times for a given location and time period, mainly using the NREL SPA algorithm [1].

    Parameters
    ----------
    latitude : float
        The latitude of the location in degrees. Must be a number between -90 and 90.

    longitude : float
        The longitude of the location in degrees. Must be a number between -180 and 180.

    tz : str
        Timezone information of the location in the format of +/-HHMM.

    start : str or pandas.DatetimeIndex
        The starting date or datetime index for which to compute sunrise and sunset times.
        If providing a string, it should be in the format 'YYYY-MM-DD'.

    end : None or str, default is None
        If providing a string, it is ending date for which to compute sunrise and sunset times (in the format 'YYYY-MM-DD')
        and can only be used if 'start' is also a string. If None, only the sunrise and sunset times for the start date will be computed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the computed sunrise, sunset, and day duration times for the specified location and time period.

    Examples
    --------
    >>> # Return DataFrame with computed sunrise and sunset times (using local time) from January 1st 2023, to January 10th 2023, for the city of Medellín, Colombia
    >>> compute_sunrise_sunset(6.25184, -75.56359, '-05:00', '2023-01-01', '2023-01-10')
    >>>
    >>> # Return DataFrame with computed sunrise and sunset times (using local time) for June 1st 2023, for the city of Sydney, Australia.
    >>> compute_sunrise_sunset(-33.86785, 151.20732, '+11:00', '2023-06-01')
    >>>
    >>> # Return DataFrame with computed sunrise and sunset times (using local time) for January 1st 2023, January 2nd 2023 and January 3rd 2023, for Medellín, Colombia.
    >>> idx = pandas.DatetimeIndex(['2023-01-01', '2023-01-02', '2023-01-03'])
    >>> compute_sunrise_sunset(6.25184, -75.56359, '-05:00', idx)
    >>>
    >>> # Return DataFrame with computed sunrise and sunset times (using local time) for the whole year of 2023 for a place inside the antartic circle.
    >>> compute_sunrise_sunset(-82, -75.56359, '-05:00', "2023-01-01", "2023-12-31")
    

    Notes
    -----
    1) Latitude of -90° corresponds to the geographic South pole, while a
       latitude of 90° corresponds to the geographic North Pole.

    2) A negative longitude correspondes to a point west of the greenwhich
       meridian, while a positive longitude means it is east of the greenwhich
       meridian.

    3) A sunrise/sunset equal to "PD" means that the place in question is experiencing a polar day. Meanwhile,
       a sunrise/sunset equalt to "PN" stands for polar night.

    4) This algorithm is based on the NREL SPA algorithm . As such, it calculates sunrise and sunset times without
       taking the altitude of the location into account. A higher altitude on a location translates to an earlier sunrise and
       and a later sunset compared to that same location if it were situated at sea level. Nevertheless, the effects of altitude
       are quite small. For every 1500 meters in elevation, a site's sunrise occurs 1 minute earlier and its sunset occurs
       1 minute later than normal [2].

    5) This function also does not take into account the effect of mountains or surrounding terrain/structures on the time
       when the sun first becomes visible to an observer.

    References
    ----------
    [1] https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.solarposition.sun_rise_set_transit_spa.html

    [2] https://www.chicagotribune.com/weather/ct-wea-0928-asktom-20160927-column.html

    [3] https://pypi.org/project/suntimes/
    """

    # Check types of input parameters and defineevaluation times.
    if isinstance(start, str) and end is None:
        start_time = start + f" 12:00:00"
        times = pd.DatetimeIndex([start_time], tz=tz)

    elif isinstance(start, DatetimeIndex) and end is None:
        times = start
        times = times.tz_localize(tz=tz)

    elif isinstance(start, str) and isinstance(end, str):
        start_time = start + f" 12:00:00"
        end_time   = end   + f" 12:00:00"
        times = pd.date_range(start=start_time, end=end_time, freq="1D", tz=tz)

    # Compute sunrise and sunset times.
    data = sun_rise_set_transit_spa(times     = times,
                                    latitude  = latitude,
                                    longitude = longitude,
                                    how       = 'numpy',
                                    delta_t   = 67.0)

    # Repackage that same data into a handier format.
    times_years  = pd.Series(times).apply(lambda x:int(str(x).split("-")[0]))
    times_months = pd.Series(times).apply(lambda x:int(str(x).split("-")[1]))
    times_days   = pd.Series(times).apply(lambda x:int(str(x).replace(" ", "-").split("-")[2]))
    new_index    = pd.MultiIndex.from_tuples(list(zip(times_years, times_months, times_days)))

    data.index = new_index
    data.rename(columns={"transit":"day duration"}, inplace = True)
    data["day duration"] = data["sunset"] - data["sunrise"]


    # --- DEAL WITH POLAR NIGHTS AND DAYS ---
    for year, month, day in data.index[pd.isnull(data["day duration"])]:

        suntimes_obj = SunTimes(longitude=longitude, latitude=latitude, altitude=0)
        tz_name = utc_hour_to_tz_name(int(tz.split(":")[0]))
        info = suntimes_obj.risewhere(datetime(year, month, day), tz_name)

        if info == "PD":
            data.loc[(year, month, day), "sunrise"] = "PD"
            data.loc[(year, month, day), "sunset"]  = "PD"
            data.loc[(year, month, day), "day duration"] = pd.Timedelta(24, "h")

        elif info == "PN":
            data.loc[(year, month, day), "sunrise"] = "PN"
            data.loc[(year, month, day), "sunset"]  = "PN"
            data.loc[(year, month, day), "day duration"] = pd.Timedelta(0, "h")

        else:
            msg = f"Discrepancy in calculation of polar Night/Day on date {year}-{month}-{day}"
            msg = f"{msg}. Approximation had to be carried out."
            warnings.warn(msg)

            approx_summer_solstice = pd.Timestamp(f"{year}-06-21 12:00:00", tz=tz)
            approx_winter_solstice = pd.Timestamp(f"{year}-12-21 12:00:00", tz=tz)

            distance_from_summer_solstice  = approx_summer_solstice
            distance_from_summer_solstice -= pd.Timestamp(f"{year}-{month}-{day} 12:00:00", tz=tz)
            distance_from_summer_solstice  = abs(distance_from_summer_solstice.days)

            distance_from_winter_solstice  = approx_winter_solstice
            distance_from_winter_solstice -= pd.Timestamp(f"{year}-{month}-{day} 12:00:00", tz=tz)
            distance_from_winter_solstice  = abs(distance_from_winter_solstice.days)

            if  latitude >= 0 and distance_from_summer_solstice < distance_from_winter_solstice:
                data.loc[(year, month, day), "sunrise"] = "PD"
                data.loc[(year, month, day), "sunset"]  = "PD"
                data.loc[(year, month, day), "day duration"] = pd.Timedelta(24, "h")

            elif latitude >= 0 and distance_from_winter_solstice <= distance_from_summer_solstice:
                data.loc[(year, month, day), "sunrise"] = "PN"
                data.loc[(year, month, day), "sunset"]  = "PN"
                data.loc[(year, month, day), "day duration"] = pd.Timedelta(0, "h")

            if  latitude < 0 and distance_from_summer_solstice <= distance_from_winter_solstice:
                data.loc[(year, month, day), "sunrise"] = "PN"
                data.loc[(year, month, day), "sunset"]  = "PN"
                data.loc[(year, month, day), "day duration"] = pd.Timedelta(0, "h")

            elif latitude < 0 and distance_from_winter_solstice < distance_from_summer_solstice:
                data.loc[(year, month, day), "sunrise"] = "PD"
                data.loc[(year, month, day), "sunset"]  = "PD"
                data.loc[(year, month, day), "day duration"] = pd.Timedelta(24, "h")

    return data



#%%



def geo_date_range(latitude, longitude, tz,
                   start_time, end_time, freq,
                   min_hms, max_hms,
                   skip_polar_nights = True,
                   inclusive = False):

    """
    Generate a date range based on geographical coordinates and specified time parameters, with optional filtering for each day
    based on user input or sunrise and sunset times.

    Parameters
    ----------
    latitude : float
        The latitude of the location in degrees. Must be a number between -90 and 90.

    longitude : float
        The longitude of the location in degrees. Must be a number between -180 and 180.

    tz : str
        Timezone information of the location in the format of +/-HH:MM.

    start_time : str
        The starting date and time in the format 'YYYY-MM-DD HH:MM:SS'.

    end_time : str
        The ending date and time in the format 'YYYY-MM-DD HH:MM:SS'.

    freq : str
        The frequency at which the date range should be generated.
        Any frequency accepted by pandas.date_range is valid for geo_date_range.

    min_hms : str or None
        A string representing the minimum hour-minute-second (HH:MM:SS) value for a Timestamp within each day's time series.
        If the hms values are below this threshold, they are removed. It can also be set to None to ignore this condition,
        or to "sunrise" to use the computed sunrise time for the location as the value for *min_hms*.

    max_hms: str or None
        A string representing the maximum hour-minute-second (HH:MM:SS) value for a Timestamp within each day's time series.
        If the hms values are above this threshold, they are removed. It can also be set to None to ignore this condition,
        or to "sunset" to use the computed sunset time for the location as the value for *max_hms*.

    skip_polar_nights : bool, optional
        Whether to skip polar night periods during filtering. Defaults to True.

    inclusive : bool, optional
        Whether to forcibly include the end_time in the generated date range, in case it's left out. Defaults to False.

    Returns
    -------
    res : dict
        A dictionary containing the filtered date ranges/time series, separated by day, based on the specified parameters.
        Its strucure is as follows: Each key is a 3-tuple of (year : int, month : int, day :int) and each corresponding value is a
        pandas.DatetimeIndex object containing the time series associated to that date.

    Notes
    -----
    This function depends on the 'compute_sunrise_sunset' function to compute the sunrise and sunset times if required.

    Examples
    -------
    >>> # Generates a date range from '2023-1-1 00:00:00' to '2023-12-31 23:59:59.999' UTC-5 time, using a frequency of 5 min.
    >>> # No filtering of the day hours is carried out.
    >>> latitude = 6.230833
    >>> longitude = -75.56359
    >>> tz = '-05:00'
    >>> start_time = "2023-1-1 00:00:00"
    >>> end_time   = "2023-12-31 23:59:59.999"
    >>> freq = "5min"
    >>> min_hms = None
    >>> max_hms = None
    >>> skip_polar_nights = True
    >>> inclusive = False
    >>> res = geo_date_range(latitude, longitude, tz, start_time, end_time, freq, min_hms, max_hms, skip_polar_nights, inclusive)
    >>>
    >>>
    >>> # Generates a date range from '2023-1-1 00:00:00' to '2023-12-31 23:59:59.999' UTC-5 time, using a frequency of 5 min.
    >>> # Filtering of the day hours is carried out using sunrise and sunset times calculated for Medellín, Colombia.
    >>> latitude = 6.230833
    >>> longitude = -75.56359
    >>> tz = '-05:00'
    >>> start_time = "2023-1-1 00:00:00"
    >>> end_time   = "2023-12-31 23:59:59.999"
    >>> freq = "5min"
    >>> min_hms = "sunrise"
    >>> max_hms = "sunset"
    >>> skip_polar_nights = True
    >>> inclusive = False
    >>> res = geo_date_range(latitude, longitude, tz, start_time, end_time, freq, min_hms, max_hms, skip_polar_nights, inclusive)
    >>>
    >>>
    >>> # Generates a date range from '2023-1-1 00:00:00' to '2023-12-31 23:59:59.999' UTC-5 time, using a frequency of 5 min.
    >>> # Filtering of the day hours is carried out using the range specified by the user.
    >>> latitude = 6.230833
    >>> longitude = -75.56359
    >>> tz = '-05:00'
    >>> start_time = "2023-1-1 00:00:00"
    >>> end_time   = "2023-12-31 23:59:59.999"
    >>> freq = "5min"
    >>> min_hms = "06:23:50"
    >>> max_hms = "17:50:00"
    >>> skip_polar_nights = True
    >>> inclusive = False
    >>> res = geo_date_range(latitude, longitude, tz, start_time, end_time, freq, min_hms, max_hms, skip_polar_nights, inclusive)

    """

    # Compute unfiltered date range.
    data = pd.date_range(start = start_time,
                         end   = end_time,
                         freq  = freq,
                         tz    = tz)

    # Extend date range in case that 'end_time' gets cut-off.
    if inclusive and data[-1].tz_localize(None) < pd.Timestamp(end_time):
        data = data.append(DatetimeIndex([data[-1] + pd.Timedelta(freq)]))

    # Compute sunrise and sunset times if required.
    if min_hms == "sunrise" or max_hms == "sunset":
        sunrise_sunset_data = compute_sunrise_sunset(latitude  = latitude,
                                                     longitude = longitude,
                                                     tz = tz,
                                                     start = start_time.split(" ")[0],
                                                     end   = end_time.split(" ")[0])


    # --- PERFORM DATA FILTERING AND REPACKAGING ---

    new_data = {}
    for year, month, day in list(zip(data.year, data.month, data.day)):

        if min_hms is None:
            day_init = pd.Timestamp(f"{year}-{month}-{day} 00:00:00", tz=tz)

        elif min_hms != "sunrise":
            day_init = pd.Timestamp(f"{year}-{month}-{day} {min_hms}", tz=tz)

        elif min_hms == "sunrise":
            day_init = sunrise_sunset_data.loc[(year, month, day), "sunrise"]

            if day_init == "PD":
                day_init = pd.Timestamp(f"{year}-{month}-{day} 00:00:00", tz=tz)

            elif day_init == "PN" and skip_polar_nights:
                continue

            elif day_init == "PN" and not skip_polar_nights:
                new_data[(year, month, day)] = np.nan
                continue


        if max_hms is None:
            day_fin  = pd.Timestamp(f"{year}-{month}-{day} 23:59:59.999", tz=tz)

        elif max_hms != "sunset":
            day_fin = pd.Timestamp(f"{year}-{month}-{day} {max_hms}", tz=tz)

        elif max_hms == "sunset":
            day_fin = sunrise_sunset_data.loc[(year, month, day), "sunset"]

            if day_fin == "PD":
                day_fin  = pd.Timestamp(f"{year}-{month}-{day} 23:59:59.999", tz=tz)

            elif day_fin == "PN" and skip_polar_nights:
                continue

            elif day_fin == "PN" and not skip_polar_nights:
                new_data[(year, month, day)] = np.nan
                continue


        new_data[(year, month, day)] = data[np.logical_and(day_init <= data, data <= day_fin)]


    return new_data



# %%
