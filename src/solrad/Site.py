#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
computation and manipulation of most of a site's geographical and
metheorological data. 
"""

#%%                       IMPORTATION OF LIBRARIES
import os
import scipy
import solrad
import warnings
import numpy as np
import pvlib as pv
import pandas as pd
import solrad.geotime as gtm
import solrad.sun.sun as sun
import solrad.terrain.horizon as hrz
import matplotlib.pyplot as plt
import solrad.climate.pvgis_tmy as pvgty
import solrad.atmosphere.water_column as wat
import solrad.atmosphere.ozone_column as oz
import solrad.atmosphere.aod_550nm as aod550
from scipy.interpolate import interp1d
import solrad.atmosphere.angstrom_exponent as angsexp
import solrad.atmosphere.single_scattering_albedo as ssa
import solrad.atmosphere.aerosol_asymmetry_factor as aaf

#%%                       DEFINITION OF CONSTANTS

_SPECTRL2_WAVELENGTHS = [# nm
    300.0,  305.0,  310.0,  315.0,  320.0,  325.0,  330.0,  335.0,  340.0,  345.0,
    350.0,  360.0,  370.0,  380.0,  390.0,  400.0,  410.0,  420.0,  430.0,  440.0,
    450.0,  460.0,  470.0,  480.0,  490.0,  500.0,  510.0,  520.0,  530.0,  540.0,
    550.0,  570.0,  593.0,  610.0,  630.0,  656.0,  667.6,  690.0,  710.0,  718.0,
    724.4,  740.0,  752.5,  757.5,  762.5,  767.5,  780.0,  800.0,  816.0,  823.7,
    831.5,  840.0,  860.0,  880.0,  905.0,  915.0,  925.0,  930.0,  937.0,  948.0,
    965.0,  980.0,  993.5,  1040.0, 1070.0, 1100.0, 1120.0, 1130.0, 1145.0,
    1161.0, 1170.0, 1200.0, 1240.0, 1270.0, 1290.0, 1320.0, 1350.0, 1395.0,
    1442.5, 1462.5, 1477.0, 1497.0, 1520.0, 1539.0, 1558.0, 1578.0, 1592.0,
    1610.0, 1630.0, 1646.0, 1678.0, 1740.0, 1800.0, 1860.0, 1920.0, 1960.0,
    1985.0, 2005.0, 2035.0, 2065.0, 2100.0, 2148.0, 2198.0, 2270.0, 2360.0,
    2450.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0,
    3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3800.0, 3900.0, 4000.0]

#%%

class Site:

    # 1) --- INITIALIZATION FUNCS ---

    def __init__(self, latitude, longitude, altitude, tz, name, SF_model = "Rural"):

        """
        Class for calculating and storing all relevant geographical and metheorological information
        of a site, for the later computation of radiance and irradiance data.
        
        Parameters
        ----------
                    
        longitude : float
            Site's longitude in degrees. Must be a number between -180 and 180.
        
        latitude : float
            Site's latitude in degrees. Must be a number between -90 and 90.
            
        altitude : float
            Site's elevation above sea level in meters. Must be non-negative.
            
        tz : str
            Timezone information of the location in the format of +/-HH:MM.
            
        name : str
            Custom name for the site being modelled. 

        SF_model : {"Rural", "Urban", "Maritime"}, optional
            Model to be used in the computation of aerosol properties.
            It is taken from Shettel & Fenn's worked referenced in [1].
            Default is "Rural".
            
            
        Notes
        -----
        1) Latitude of -90° corresponds to the geographic South pole, while a 
        latitude of 90° corresponds to the geographic North Pole.
        
        2) A negative longitude correspondes to a point west of the greenwhich 
        meridian, while a positive longitude means it is east of the greenwhich 
        meridian.  
        
        3) +/-HH:MM format means HOUR-MINUTE. Eg.: "-05:00" means 5 hours,
           0 minutes and 0 seconds, west of the greenwhich meridian or, equivalently,
           GMT-5.
            
        References
        ----------
        [1] Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the
        Lower Atmosphere and the Effects of Humidity Variations on their Optical
        Properties. Environ. Res.. 94.
            
        """

        # Initialize attributes.
        self.tz = tz
        self.name = name
        self.latitude  = latitude
        self.longitude = longitude
        self.altitude  = altitude
        self.SF_model  = SF_model

        # Initialize horizon
        self.horizon =\
        {"func"     : hrz.clear_horizon_func,
         "is_clear" : True,
         "is_pvgis" : False,
         "was_used_for_climate_data" : False
        }


        # Initialize main attributes.
        self.sun_data = {}
        self.climate_and_air_data = {}
        self.single_scattering_albedo = {}
        self.aerosol_asymmetry_factor = {}

        
        self.SUN_DATA_COLS =\
        ["hms_float", "apzen", "zen", "apel", "el", "az", "i", "j",
         "k", "rel_airmass"]

        self.CLIMATE_AND_AIR_DATA_COLS =\
        ["hms_float","G(h)", "Gb(n)", "Gd(h)", "T2m", "SP", "RH", 
         "O3", "H2O", "alpha_500nm", "aod_500nm", "spectrally_averaged_aaf",
         "extra_Gb(n)"]
        
        self.AEROSOL_COLS =\
        ["hms_float"] + [f"{int(i)}nm" for i in _SPECTRL2_WAVELENGTHS]


        # Define an attribute which will contain all descriptions and units
        # of the site variables.
        self.variables_info = {"descriptions":{}, "units":{}}
        
        self.variables_info["descriptions"]  = {
        'T2m': 'Air temperature at 2m above ground',
        'RH': 'Relative humidity',
        'G(h)':'Global horizontal irradiance',
        'Gb(n)':'Beam (direct) normal irradiance',
        'Gd(h)':'Diffuse horizontal irradiance',
        'SP':'Surface Pressure',
        'int G(h)'  :'Time integral of global horizontal irradiance',
        'int Gb(n)':'Time integral of beam (direct) normal irradiance',
        'int Gd(h)':'Time integral of diffuse horizontal irradiance',
        'H2O' : 'Precipitable water column',
        'O3' : 'Atmospheric ozone column',
        'alpha_500nm':'Angstrom turbidity exponent at 500nm',
        'aod_500nm':'Aerosol optical depth at 500nm',
        'spectrally_averaged_aaf': 'Aerosol Asymmetry Factor averaged over specified spectral range',
        'apzen' : 'apparent zenith angle of the Sun',
        'zen' : 'zenith angle of the Sun',
        'apel': 'apparent elevation angle of the Sun',
        'el' : 'elevation angle of the Sun',
        'az' : 'azimuth angle of the Sun',
        'rel_airmass' : 'Relative Airmass',
        'single_scattering_albedo' : 'Single Scattering Albedo',
        'aerosol_asymmetry_factor' : 'Aerosol Asymmetry Factor',
        'extra_Gb(n)' : 'Extraterrestrial Irradiance'}
        
        self.variables_info["units"]  = {
        'T2m': '[°C]',
        'RH': '[%]',
        'G(h)':'[W/m^2]',
        'Gb(n)':'[W/m^2]',
        'Gd(h)':'[W/m^2]',
        'SP':'[Pa]',
        'int G(h)'   :'[Wh/m^2]',
        'int Gb(n)' : '[Wh/m^2]',
        'int Gd(h)' : '[Wh/m^2]',
        'H2O' : '[cm]',
        'O3' : '[atm-cm]',
        'alpha_500nm':'[-]',
        'aod_500nm':'[-]',
        'spectrally_averaged_aaf': '[-]',
        'apzen' : '[°]',
        'zen' : '[°]',
        'apel': '[°]',
        'el' : '[°]',
        'az' : '[°]',
        'rel_airmass' : '[-]',
        'single_scattering_albedo' : '[-]',
        'aerosol_asymmetry_factor' : '[-]',
        'extra_Gb(n)' : '[W/m^2]'}
        
        return None
    

    def _init_main_attrs(self):
        """
        Initialize main attributes of 'Site' obj for storing information.
        These attributes are: 'self.climate_an_air_data', 'self.sun_data',
        'self.aerosol_asymetry_factor', 'self.single_scattering_albedo'.
        """

        for date, DatetimeIndex_obj in self.simulation_time_data.items():

            self.sun_data[date] =\
            pd.DataFrame(index = DatetimeIndex_obj, columns = self.SUN_DATA_COLS)

            self.climate_and_air_data[date] =\
            pd.DataFrame(index = DatetimeIndex_obj, columns = self.CLIMATE_AND_AIR_DATA_COLS)
            
            self.single_scattering_albedo[date] =\
            pd.DataFrame(index = DatetimeIndex_obj, columns = self.AEROSOL_COLS)

            self.aerosol_asymmetry_factor[date] =\
            pd.DataFrame(index = DatetimeIndex_obj, columns = self.AEROSOL_COLS)

            hms_float  = DatetimeIndex_obj.hour
            hms_float += DatetimeIndex_obj.minute/60
            hms_float += DatetimeIndex_obj.second/3600

            self.sun_data[date]["hms_float"] = hms_float
            self.climate_and_air_data[date]["hms_float"] = hms_float
            self.single_scattering_albedo[date]["hms_float"] = hms_float
            self.aerosol_asymmetry_factor[date]["hms_float"] = hms_float

        return None
    
    

    # 2) --- HORIZON FUNCS ---
    
    def set_horizon_from_pvgis(self, interp_method = "linear", timeout = 30):
        """
        Get a site's horizon profile from PVGIS's API and use it for the current site.

        Parameters
        ----------
        interp_methd : {'linear', 'quadratic', 'cubic'}, optional
            Order of the spline interpolator to use. Default is 'linear'.

        timeout : float
            Number of seconds after which the requests library will stop waiting
            for a response of the server. That is, if the requests library does not
            receive a response in the specified number of seconds, it will raise a
            Timeout error.

        Returns
        -------
        None

        Produces
        --------
        self.horizon : dict
            Dictionary with information about the horizon. It has the following 
            key-value pairs:
        
                Key-Values
                ----------
                'func' : Callable
                    Horizon function. Its input is an array of azimuth values (in degrees) 
                    and its output is an array of horizon elevation angle values (in degrees).

                'is_clear' : bool
                    Whether the current horizon is clear or not (i.e, null elevation everywhere).

                'is_pvgis' : bool
                    Whether the current horizon was obtained from pvgis.

                'was_used_for_climate_data' : bool
                    Whether the current horizon was used for the computation of 
                    climate data.

        Notes
        -----
        1) Horizon height is the angle between the local horizontal plane and the
        horizon. In other words, the Horizon height is equal to the horizon's
        elevation angle.
        """

        azimuth, elevation =\
        hrz.horizon_arrays_from_pvgis(latitude  = self.latitude,
                                      longitude = self.longitude,
                                      timeout = timeout)
        
        horizon_func = hrz.horizon_func_from_arrays(azimuth     = azimuth,
                                                    elevation   = elevation,
                                                    interp_method = interp_method)
        

        self.horizon.update({
        "func"     : horizon_func,
        "is_clear" : False,
        "is_pvgis" : True 
        })
        return None
    
    def set_horizon_from_arrays(self, azimuth, elevation, interp_method = "linear"):
            
        """
        Set site's horizon function by interpolating provided data points.

        Parameters
        ----------
        azimuth : array_like (npoints,) 
            Array of azimuth angle values in degrees, from 0 to 360. Must be monotonic-increasing.

        elevation : array_like (npoints,) 
            Array of horizon elevation angle values in degrees. Elevation values must
            lie between 0 and 90.

        interp_method : {"linear", "quadratic", "cubic"}, optional
            Order of the spline interpolator to use. Default is 'linear'.

        Returns
        -------
        None

        Produces
        --------
        self.horizon attribute : dict
            Dictionary with information about the horizon. It has the following 
            key-value pairs:
        
                Key-Values
                ----------
                'func' : Callable
                    Horizon function. Its input is an array of azimuth values (in degrees) 
                    and its output is an array of horizon elevation angle values (in degrees).

                'is_clear' : bool
                    Whether the current horizon is clear or not (i.e, null elevation everywhere).

                'is_pvgis' : bool
                    Whether the current horizon was obtained from pvgis.

                'was_used_for_climate_data' : bool
                    Whether the current horizon was used for the computation of 
                    climate data.

        Notes
        -----
        1) Horizon height is the angle between the local horizontal plane and the
        horizon. In other words, the Horizon height is equal to the horizon's
        elevation angle.
        """

        horizon_func = hrz.horizon_func_from_arrays(azimuth     = azimuth,
                                                    elevation   = elevation,
                                                    interp_method = interp_method)
        
        self.horizon.update({
        "func"     : horizon_func,
        "is_clear" : False,
        "is_pvgis" : False 
        })
        return None
        
    def set_horizon_from_func(self, func):

        """
        Set site's horizon function by directly passing a function.

        Parameters
        ----------
        func : callable
            Horizon profile function. It should take in an array of azimuth values
            (in degrees) and return an array of elevation angle values (in degrees).

        Returns
        -------
        None

        Produces
        --------
        self.horizon attribute : dict
            Dictionary with information about the horizon. It has the following 
            key-value pairs:
        
                Key-Values
                ----------
                'func' : Callable
                    Horizon function. Its input is an array of azimuth values (in degrees) 
                    and its output is an array of horizon elevation angle values (in degrees).

                'is_clear' : bool
                    Whether the current horizon is clear or not (i.e, null elevation everywhere).

                'is_pvgis' : bool
                    Whether the current horizon was obtained from pvgis.

                'was_used_for_climate_data' : bool
                    Whether the current horizon was used for the computation of 
                    climate data.

        Notes
        -----
        1) Horizon height is the angle between the local horizontal plane and the
        horizon. In other words, the Horizon height is equal to the horizon's
        elevation angle.
        """

        azimuth = [0, 90, 180, 270, 360]
        for az in azimuth:
            try:
                elevation = func(az)
                if isinstance(elevation, int) or isinstance(elevation, float):
                    msg = "int/float input values still have to produce"
                    msg = f"{msg} numpy.array output values."
                    raise Exception(msg)
                hrz._check_for_elevation_array_compliance(elevation)
            except Exception as e:
                raise Exception(e)
            
        azimuth = np.linspace(0, 360, 361)
        try:
            elevation = func(azimuth)
        except Exception as e:
            raise Exception(e)
        

        self.horizon.update({
        "func"     : func,
        "is_clear" : False,
        "is_pvgis" : False 
        })
        return None
    
    def reset_horizon(self):
        """
        Resets horizon profile, such that elevation is 0° everywhere.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Produces
        --------
        self.horizon attribute : dict
        Dictionary with information about the horizon. It has the following 
        key-value pairs:
        
            Key-Values
            ----------
            'func' : Callable
                Horizon function. Its input is an array of azimuth values (in degrees) 
                and its output is an array of horizon elevation angle values (in degrees).

            'is_clear' : bool
                Whether the current horizon is clear or not (i.e, null elevation everywhere).

            'is_pvgis' : bool
                Whether the current horizon was obtained from pvgis.

            'was_used_for_climate_data' : bool
                Whether the current horizon was used for the computation of 
                climate data.

        Notes
        -----
        1) Horizon height is the angle between the local horizontal plane and the
        horizon. In other words, the Horizon height is equal to the horizon's
        elevation angle.
        """

        self.horizon.update({
        "func"     : hrz.clear_horizon_func,
        "is_clear" : False,
        "is_pvgis" : False 
        })

        return None
    
    def plot_horizon(self, azimuth, config = None):

        """
        Plots site's horizon profile based on the provided azimuth data.

        Parameters
        ----------
        config : None or dict, optional
            Configuration settings of the plot. When equal to None (which is
            the default) the default plot settings are used. When not equal to None,
            it must be a dict containing some or all of the following key-value
            pairs:

                Keys-Values
                -----------
                "projection" : "polar" or "cartesian"
                    If equal to "polar", the Horizon profile is plotted using a polar
                    plot. If equal to "cartesian", it is plotted using a cartesian plot.
                    "Default is "polar".

                "show_polar_elevation" : bool
                    If true, it shows the elevation angle makers for the polar
                    plot. If False, it does not. Default is False.

                "title" : str
                    Title of the plot. Default is 'Horizon Profile'.

                "facecolor" : str
                    Background color of the Horizon Height part of the plot.
                    Must be equal to str(x), where x is a float between 0 and 1.
                    0 means that the background color is black. 1 means that it
                    is white. Any value in between represents a shade of gray.
                    Default is "0.5".

                "figsize" : tuple of float
                    Figure size of the plot.

        Returns
        -------
        None

        """

        hrz.plot_horizon(y = self.horizon["func"],
                         azimuth = azimuth,
                         config = config)
        return None
    

    # 2) --- SIMULATION TIME FUNCS ---
    
    def define_simulation_time_data(self, start_time, end_time, 
                                    freq, min_hms, max_hms,  
                                    inclusive = False): 
        
        """
        Define time data used for simulation.

        It generates a date range based on geographical coordinates and specified time parameters,
        with optional filtering for each day based on user input or sunrise and sunset times.

        Parameters
        ----------

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

        inclusive : bool, optional
            Whether to forcibly include the end_time in the generated date range, in case it's left out. Defaults to False.

        Returns
        -------
        None

        Produces
        -------
        self.simulation_time_data : dict
            A dictionary containing the filtered date ranges/time series, separated by day, based on the specified parameters.
            Its strucure is as follows: Each key is a 3-tuple of (year : int, month : int, day :int) and each corresponding value is a
            pandas.DatetimeIndex object containing the time series associated to that date.

        Notes
        -----
        1) This function also initializes other attributes such as: *self.climate_and_air_data*,
        *self.sun_data*, *self.single_scattering_albedo*, *self.aerosol_asymmetry_factor*. 
        
        """

        self.simulation_time_data_freq = freq
        
        self.simulation_time_data =\
        gtm.geo_date_range(latitude          = self.latitude,
                           longitude         = self.longitude,
                           tz                = self.tz,
                           start_time        = start_time,
                           end_time          = end_time,
                           freq              = freq,
                           min_hms           = min_hms,
                           max_hms           = max_hms, 
                           skip_polar_nights = True, 
                           inclusive         = inclusive)
        
        self._init_main_attrs()
        
        return None
    

    # 3) --- CLIMATE FUNCS ---
    
    def set_climate_data_from_pvgis_tmy_data(self, startyear, endyear, interp_method="linear", use_site_horizon = False):

        """
        Use the Typical Meteorological Year (TMY) data from PVGIS to partially
        fill the *self.climate_and_air_data*. 

        Parameters
        -----------
        startyear: int or None
            First year to calculate TMY.

        endyear : int or None
            Last year to calculate TMY, must be at least 10 years from first year.

        interp_method : {'linear', 'quadratic', 'cubic'}, optional
            The interpolation method to be used. Defaults is 'linear'.

        use_site_horizon : bool, optional
            Wether to include effects of the site's horizon. Default is False.

        Returns
        --------
        None

        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it
        fills the "G(h)", "Gb(n)", "Gd(h)", "T2m", "SP", "RH" columns of all the 
        DataFrames contained by the *self.climate_and_air_data* dict.

        See Also
        --------
        ``pvlib.iotools.get_pvgis_tmy``

        """

        if not use_site_horizon or self.horizon["is_clear"]:
            usehorizon  = False
            userhorizon = None
            self.horizon["was_used_for_climate_data"] = False

        elif self.horizon["is_pvgis"]:
            usehorizon  = True
            userhorizon = None
            self.horizon["was_used_for_climate_data"] = True

        else:
            usehorizon  = True
            azimuths    = np.linspace(0, 360, 361)
            userhorizon = list(self.horizon["funcs"](azimuths))
            self.horizon["was_used_for_climate_data"] = True


        PVGIS_COLS = ["G(h)", "Gb(n)", "Gd(h)", "T2m", "SP", "RH"]

        pvgis_tmy_data =\
        pvgty.get_pvgis_tmy_dataframe(latitude    = self.latitude, 
                                      longitude   = self.longitude,
                                      tz          = self.tz,
                                      startyear   = startyear, 
                                      endyear     = endyear,
                                      usehorizon  = usehorizon,
                                      userhorizon = userhorizon)
        
        climate_data =\
        pvgty.climate_data_from_pvgis_tmy_dataframe(time_data = self.simulation_time_data,
                                                    tmy_data  = pvgis_tmy_data,
                                                    interp_method=interp_method)

        for date, climate_df in climate_data.items():
            self.climate_and_air_data[date][PVGIS_COLS] = climate_df[PVGIS_COLS]

        return None
    
    def compute_extraterrestrial_normal_irradiance(self, method = "nrel"):

        """
        Determines extraterrestrial radiation from day of year, using pvlib's
        ``get_extra_radiation`` function.
        
        Parameters
        ----------
        method : {"pyephem", "spencer", "asce", "nrel"}, opional
            The method by which the extraterrestrial radiation should be
            calculated. The default is "nrel".

        Returns
        -------
        None.
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it
        fills the "extra_Gbn" column of all the DataFrames contained by the
        *self.climate_and_air_data* dict.

        """

        for date, DatetimeIndex_obj in self.simulation_time_data.items():
            extra_Gbn = pv.irradiance.get_extra_radiation(datetime_or_doy = DatetimeIndex_obj,
                                                          method = method,
                                                          epoch_year = date[0])
            self.climate_and_air_data[date]["extra_Gb(n)"] = extra_Gbn
        return None
    
    def compute_cummulative_time_integral_of_irradiances(self):
        # NOTE: Time integral of irradiances is given in Wh/m^2

        """
        Computes the cummulative time integral of the cols {"G(h)", "Gb(n)", "Gd(h)"}
        in *self.climate_and_air_data[date]*, for all dates. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it adds
        the columns {"int G(h)", "int Gb(n)", "int Gd(h)"} to each dataframe of the dict
        *self.climate_and_air_data*; containing the cummulative time integral of
        the columns {"G(h)", "Gb(n)" and "Gd(h)"}, respectively. Units of this new columns
        are Wh/m^2.

        Warns
        -----
        1) Warning :
            f"NaN/None values detected in self.climate_and_air_df[{date}][{col}].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 
        """
        
        for date, climate_and_air_df in self.climate_and_air_data.items():
            
            t_vals = np.array(climate_and_air_df["hms_float"])
            
            for col in ["G(h)", "Gb(n)", "Gd(h)"]:
                y_vals = np.array(climate_and_air_df[col])

                if any(pd.isnull(y_vals)):
                    msg = f"NaN/None values detected in self.climate_and_air_df[{date}][{col}]"
                    msg = f"{msg}. NaN input values will produce NaN output values" 
                    msg = f"{msg}. None input values will raise Exceptions." 
                    warnings.warn(msg)
                
                integral_of_y_vals = scipy.integrate.cumulative_trapezoid(y_vals, t_vals)
                integral_of_y_vals = np.insert(integral_of_y_vals, 0, 0)
                self.climate_and_air_data[date][f"int {col}"] = integral_of_y_vals
            
        return None


    # 4) --- SUN FUNCS ---

    def compute_sun_data(self, NaN_handling = "strict"):

        """
        Compute solar position data and related parameters for a specific location and time.

        Parameters
        ----------
        NaN_handling : {"strict", "loose", "null"}, optional
            How to handle NaN and None values when present in "SP" and "T2m" columns of the DataFrames stored in *self.climate_and_air_data*
            If "strict" an Exception is raised.
            If "loose", default values are used instead (see notes for more info).
            If "null", nothing is done about it and NaN/None values are directly passed onto the calculation, which may
            produce NaN results or raise another Exception. 
            Default is "strict".


        Returns
        --------
        None

        Produces
        --------
        Filled *self.sun_data* attribute. 

        See Also
        --------
        ``pvlib.solarposition.get_solarposition``

        Notes
        -----
        1) In case that NaN_handling is "loose", the default value of temperature used is 15°C and the default
        value of pressure is computed from altitude using the function ``pvlib.atmosphere.alt2pres``.

        """

        SUN_COLS = ["apzen", "zen", "apel", "el", "az", "i", "j", "k", "rel_airmass"]

        pressure = {date:np.array(climate_air_df["SP"]).astype(float) for 
                    date, climate_air_df in self.climate_and_air_data.items()}
        
        temperature = {date:np.array(climate_air_df["T2m"]).astype(float) for 
                       date, climate_air_df in self.climate_and_air_data.items()}

        sun_data = sun.compute_sun_data(latitude         = self.latitude,
                                        longitude        = self.longitude,
                                        altitude         = self.altitude,
                                        time_data        = self.simulation_time_data,
                                        pressure_data    = pressure,
                                        temperature_data = temperature,
                                        NaN_handling     = NaN_handling)
        

        for date, sun_data_df in sun_data.items():
            self.sun_data[date][SUN_COLS] = sun_data_df[SUN_COLS]

        return None
    

    # 5) --- AIR FUNCS ---

    def compute_ozone_column_using_van_Heuklon_model(self):

        """
        Computes the ozone column values (in atm-cm) for the site, using  
        van Heuklon's Ozone model. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it
        fills the "O3" column of all the DataFrames contained by the
        *self.climate_and_air_data* dict.

        """

        for date, DatetimeIndex_obj in self.simulation_time_data.items():
            self.climate_and_air_data[date]['O3'] =\
            oz.compute_van_Heuklon_ozone(latitude  = np.full(DatetimeIndex_obj.size, self.latitude),
                                         longitude = np.full(DatetimeIndex_obj.size, self.longitude),
                                         timestamp = DatetimeIndex_obj)
        return None


    def compute_ozone_column_using_satelite_data(self, path, percentile = 0.5, interp_method = "linear"):  

        """
        Computes the monthly year-wise average or 'percentile'-th percentile of atmospheric ozone column
        for the site. The raw data used for the calculation is extracted from the database referenced in [1].
        
        Parameters
        -----------
        path : path-str
            Path of the folder where the ozone column raw.ny and filled_NaNs.npy files 
            are stored. That is, the path to the local ozone column database.     

        percentile : float or None, optional
            If float, it computes the monthly year-wise 'percentile'-th percentile of ozone column. 
            If NONE,  it computes the monthly year-wise average of ozone column.
            Default is 0.5.
            
        interp_method : {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}, optional
            The method of interpolation to perform when computing the 
            *res["filled_nans_data_funcs"]*, *res["avg_data_funcs"]* and *res["percentile_data_funcs"]*
            dictionaries. Supported methods are the same as supported by scipy's 
            RegularGridInterpolator. Default is "linear".

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it
        fills the "O3" column of all the DataFrames contained by the
        *self.climate_and_air_data* dict.


        References
        ----------
        [1] Copernicus Climate Change Service, Climate Data Store, (2020):
        Ozone monthly gridded data from 1970 to present derived from satellite
        observations. Copernicus Climate Change Service (C3S) Climate Data Store 
        (CDS). DOI: 10.24381/cds.4ebfe4eb 
        
        """

        processed_ozone_column_data =\
        oz.process_CDS_ozone_column_data(path = path, percentile = percentile,
                                         interp_method = interp_method)
    
        if percentile is None:
            processed_ozone_column_data_funcs = processed_ozone_column_data["avg_data_funcs"]
        else:             
            processed_ozone_column_data_funcs = processed_ozone_column_data["percentile_data_funcs"]

        for date in self.simulation_time_data.keys():     
            month = date[1]
            self.climate_and_air_data[date]['O3'] =\
            float(processed_ozone_column_data_funcs[(month)]([self.latitude, self.longitude])[0])

        return None
    

    def compute_water_column_using_gueymard94_model(self):

        """
        Computes the Precipitable Water Column values (in atm-cm) for the site,
        using pvlib's implementation of the gueymard94 model. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. More specifically, it
        fills the "H2O" column of all the DataFrames contained by the
        *self.climate_and_air_data* dict.

        Notes
        -----
        1) Uses the "RH" and "T2m" columns in the dataframes stored by the
        *self.climate_and_air_data* attribute for the calculation.

        Warns
        -----
        1) Warning 
            f"NaN/None values detected in self.climate_and_air_df[{date}]['T2m'].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 

        2) Warning 
            f"NaN/None values detected in self.climate_and_air_df[{date}]['RH'].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 
        
        """
            
        for date, climate_and_air_df in self.climate_and_air_data.items():

            T2m = np.array(climate_and_air_df["T2m"]).astype(float)
            RH  = np.array(climate_and_air_df["RH"]).astype(float)
            
            if any(pd.isnull(T2m)):
                msg = f"NaN/None values detected in self.climate_and_air_df[{date}]['T2m']"
                msg = f"{msg}. NaN input values will produce NaN output values" 
                msg = f"{msg}. None input values will raise Exceptions." 
                warnings.warn(msg)
                
            if any(pd.isnull(RH)):
                msg = f"NaN/None values detected in self.climate_and_air_df[{date}]['RH']"
                msg = f"{msg}. NaN input values will produce NaN output values" 
                msg = f"{msg}. None input values will raise Exceptions." 
                warnings.warn(msg)
                
            
            self.climate_and_air_data[date]['H2O'] =\
            pv.atmosphere.gueymard94_pw(temp_air = T2m, relative_humidity = RH)
            
        return None
    

    def compute_water_column_using_satelite_data(self, path, percentile = 0.5, interp_method = "linear"):  

        """
        Computes the monthly year-wise average or 'percentile'-th percentile of atmospheric water column
        for the site. The raw data used for the calculation is extracted from the database referenced in [1].
        
        Parameters
        -----------
        path : path-str
            Path of the folder where the water column raw.ny and filled_NaNs.npy files 
            are stored. That is, the path to the local water column database.        

        percentile : float or None, optional
            If float, it computes the monthly year-wise 'percentile'-th percentile of water_column. 
            If NONE,  it computes the monthly year-wise average of water_column.
            Default is 0.5.
            
        interp_method : {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}, optional
            The method of interpolation to perform when computing the 
            *res["filled_nans_data_funcs"]*, *res["avg_data_funcs"]* and *res["percentile_data_funcs"]*
            dictionaries. Supported methods are the same as supported by scipy's 
            RegularGridInterpolator. Default is "linear".

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it
        fills the "H2O" column of all the DataFrames contained by the
        *self.climate_and_air_data* dict.

        References
        ----------
        [1] Preusker, R., El Kassar, R. (2022): Monthly total column water vapour 
        over land and ocean from 2002 to 2012 derived from satellite observation.
        Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
        DOI: 10.24381/cds.8e0e4724 
        
        """

        processed_water_column_data =\
        wat.process_CDS_water_column_data(path = path, percentile = percentile,
                                          interp_method = interp_method)
    
        if percentile is None:
            processed_water_column_data_funcs = processed_water_column_data["avg_data_funcs"]
        else:             
            processed_water_column_data_funcs = processed_water_column_data["percentile_data_funcs"]

        for date in self.simulation_time_data.keys():     
            month = date[1]
            self.climate_and_air_data[date]['H2O'] =\
            float(processed_water_column_data_funcs[(month)]([self.latitude, self.longitude])[0])

        return None
    

    def compute_angstrom_turbidity_exponent_500nm_using_SF_model(self):

        """
        Compute the Ansgtrom turbidity exponent at 500nm for the site using the
        Shettel and Fenn model, as detailed in [1].
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_an_air_data* attribute. Specifically, 
        it fills the "alpha_500nm" column of all DataFrames contained by 
        the *self.climate_and_air_data* dict.

        Warns
        -----
        1) Warning 
            f"NaN/None values detected in self.climate_and_air_df[{date}]['RH'].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 

        References
        ----------
        [1] Copernicus Climate Change Service, Climate Data Store, (2019): Aerosol 
        properties gridded data from 1995 to present derived from satellite observation. 
        Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.239d815c 
        
        """
            
        for date, climate_and_air_df in self.climate_and_air_data.items():
            
            RH = np.array(climate_and_air_df["RH"]).astype(float)

            if any(pd.isnull(RH)):
                msg = f"NaN/None values detected in self.climate_and_air_df[{date}]['RH']"
                msg = f"{msg}. NaN input values will produce NaN output values" 
                msg = f"{msg}. None input values will raise Exceptions." 
                warnings.warn(msg)                
            
            self.climate_and_air_data[date]['alpha_500nm'] =\
            angsexp.compute_angstrom_exponent_using_SF(RH = RH,
                                                       wavelength = 500,
                                                       model = self.SF_model)
            
        return None


    def compute_aod_500nm_using_satelite_data(self, path, percentile = 0.5, interp_method = "linear"):  

        """
        Computes the monthly year-wise average or 'percentile'-th percentile of Aerosol
        Optical Depth at 500nm (aod_500nm) for the site. The raw data used for the calculation 
        is extracted from the database referenced in [1].
        
        Parameters
        -----------
        path : path-str
            Path of the folder where the aod_500nm raw.ny and filled_NaNs.npy files 
            are stored. That is, the path to the local aod_550nm database.   
            
        percentile : float or None, optional
            If float, it computes the monthly year-wise 'percentile'-th percentile of aod_500nm. 
            If NONE,  it computes the monthly year-wise average of aod_500nm.
            Default is 0.5.
            
        interp_method : {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}, optional
            The method of interpolation to perform when computing the 
            *res["filled_nans_data_funcs"]*, *res["avg_data_funcs"]* and *res["percentile_data_funcs"]*
            dictionaries. Supported methods are the same as supported by scipy's 
            RegularGridInterpolator. Default is "linear".

        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_and_air_data* attribute. Specifically, it
        fills the "aod_500nm" column of all the DataFrames contained by the
        *self.climate_and_air_data* dict.

        Notes
        -----
        1) Uses the "alpha_500nm" column in the dataframes stored by the
        *self.climate_and_air_data* attribute for the calculation.

        Warns
        -----
        1) Warning :
            f"NaN/None values detected in self.climate_and_air_df[{date}]['alpha_500nm'].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 

        References
        ----------
        [1] Copernicus Climate Change Service, Climate Data Store, (2019): Aerosol 
        properties gridded data from 1995 to present derived from satellite observation. 
        Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.239d815c 
        
        """

        processed_aod_550nm_data =\
        aod550.process_CDS_aod_550nm_data(path = path, percentile = percentile,
                                          interp_method = interp_method)
        
        if percentile is None:
            processed_aod_550nm_data_funcs = processed_aod_550nm_data["avg_data_funcs"]
        else:             
            processed_aod_550nm_data_funcs = processed_aod_550nm_data["percentile_data_funcs"]


        for date, climate_and_air_df in self.climate_and_air_data.items():     
            month = date[1]

            alpha_500nm  = np.array(climate_and_air_df["alpha_500nm"]).astype(float)

            if any(pd.isnull(alpha_500nm)):
                msg = f"NaN/None values detected in self.climate_and_air_df[{date}]['alpha_500nm']"
                msg = f"{msg}. NaN input values will produce NaN output values" 
                msg = f"{msg}. None input values will raise Exceptions." 
                warnings.warn(msg)

            aod_550nm =\
            float(processed_aod_550nm_data_funcs[(month)]([self.latitude, self.longitude])[0]) 

            # We convert AOD_550nm to AOD_500nm using the pre-computed
            # values of Angstrom Turbidity Coefficient at 500nm (Note: 
            # according to the SF model, alpha becomes independent of
            # wavelength for wavelengths at or over 500nm).
            aod_500nm =\
            pv.atmosphere.angstrom_aod_at_lambda(aod0    = aod_550nm, 
                                                 lambda0 = 550, 
                                                 alpha   = alpha_500nm,
                                                 lambda1 = 500)

            self.climate_and_air_data[date]['aod_500nm'] = aod_500nm

        return None
    



    # 5) --- AEROSOL PROPERTIES FUNCS ---
        
    def compute_single_scattering_albedo_using_SF_model(self, interp_method = "linear"):

        """
        Compute the Single Scattering Albedo within the spectral range
        300 nm - 4000 nm, for the site, using the Ansgtrom Shettel and Fenn
        model, as detailed in [1].
        
        Parameters
        ----------            
        interp_method : {"linear", "nearest", "cubic"}
            Method of interpolation to use on the data. Default is "linear".
            
        Returns
        -------
        None
        
        Produces
        --------
        Filled *self.single_scattering_albedo* attribute. More specifically, it
        fills completely all the DataFrames contained by the 
        *self.single_scattering_albedo* dict.

        Notes
        -----
        1) Uses the "RH" column in the dataframes stored by the
        *self.climate_and_air_data* attribute for the calculation.

        Warns
        -----
        1) Warning :
            f"NaN/None values detected in self.climate_and_air_df[{date}]['RH'].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 

        References
        ----------
        [1] Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the 
        Lower Atmosphere and the Effects of Humidity Variations on their Optical 
        Properties. Environ. Res.. 94. 

        """

        wavelength = np.array(_SPECTRL2_WAVELENGTHS)
        
        for date, climate_and_air_df in self.climate_and_air_data.items():
            
            RH  = np.array(climate_and_air_df["RH"]).astype(float)
            
            if any(pd.isnull(RH)):
                msg = f"NaN/None values detected in self.climate_and_air_df[{date}]['RH']"
                msg = f"{msg}. NaN input values will produce NaN output values" 
                msg = f"{msg}. None input values will raise Exceptions." 
                warnings.warn(msg)
                
            Wavelengths, RHs = np.meshgrid(wavelength, RH) 
            Wavelengths = Wavelengths.flatten()
            RHs = RHs.flatten()
            
            single_scattering_albedo =\
            ssa.compute_single_scattering_albedo_using_SF(RH = RHs, 
                                                          wavelength    = Wavelengths, 
                                                          model         = self.SF_model, 
                                                          interp_method = interp_method)
            
            single_scattering_albedo =\
            single_scattering_albedo.reshape(len(RH), len(wavelength))
            
            self.single_scattering_albedo[date].iloc[:,1:] =\
            single_scattering_albedo
            
            
        return None
    

    def compute_aerosol_asymmetry_factor_using_SF_model(self, interp_method = "linear"): 
        """
        Compute the Aersol Asymmetry Factor within the spectral range
        300 nm - 4000 nm, for the site, using the Ansgtrom Shettel and Fenn
        model, as detailed in [1].
        
        Parameters
        ----------            
        interp_method : {"linear", "nearest", "cubic"}
            Method of interpolation to use on the data. Default is "linear".
            
        Returns
        -------
        None
        
        Produces
        --------
        Filled *self.aerosol_assymetry_factor* attribute. More specifically, it
        fills completely all the DataFrames contained by the 
        *self.aerosol_assymetry_factor* dict.

        Notes
        -----
        1) Uses the "RH" column in the dataframes stored by the
        *self.climate_and_air_data* attribute for the calculation.

        Warns
        -----
        1) Warning 
            f"NaN/None values detected in self.climate_and_air_df[{date}]['RH'].
            NaN input values will produce NaN output values.
            None input values will raise Exceptions." 

        References
        ----------
        [1] Shettle, Eric & Fenn, Robert. (1979). Models for the Aerosols of the 
        Lower Atmosphere and the Effects of Humidity Variations on their Optical 
        Properties. Environ. Res.. 94. 
        """

        wavelength = np.array(_SPECTRL2_WAVELENGTHS)
        
        
        for date, climate_and_air_df in self.climate_and_air_data.items():
            
            RH  = np.array(climate_and_air_df["RH"]).astype(float)
            
            if any(pd.isnull(RH)):
                msg = f"NaN/None values detected in self.climate_and_air_df[{date}]['RH']"
                msg = f"{msg}. NaN input values will produce NaN output values" 
                msg = f"{msg}. None input values will raise Exceptions." 
                warnings.warn(msg)
                
            Wavelengths, RHs = np.meshgrid(wavelength, RH) 
            Wavelengths = Wavelengths.flatten()
            RHs = RHs.flatten()
            
            aerosol_asymmetry_factor =\
            aaf.compute_aerosol_asymmetry_factor_using_SF(RH = RHs, 
                                                          wavelength = Wavelengths, 
                                                          model = self.SF_model, 
                                                          interp_method = interp_method)
            
            aerosol_asymmetry_factor =\
            aerosol_asymmetry_factor.reshape(len(RH), len(wavelength))
            
            self.aerosol_asymmetry_factor[date].iloc[:,1:] =\
            aerosol_asymmetry_factor
            
        return None
    

    def compute_spectrally_averaged_aerosol_asymmetry_factor(self, spectral_range = (300, 4000)):

        """
        Compute the spectral average of the aerosol asymmetry factor, 
        for the interval of wavelengths specified.
         
        We take the *self.aerosol_asymmetry_factor* attribute, loop over all
        the DataFrames stored in it and compute the row-wise mean of the
        values for the interval of wavelengths specified by *spectral range*. 
        We then use the the computed values to fill the "spectrally_averaged_aaf"
        column in all dataframes of the *self.climate_and_air_data* attribute.
        
        Parameters
        ----------
        spectral_range : 2-tuple of float
            Tuple containing the lower and upper bounds of wavelengths
            (in nm) that make up the spectral range meant for averaging the
            aerosol asymmetry factor.
            
        Returns
        -------
        None
        
        Produces
        --------
        Partially filled *self.climate_an_air_data* attribute. Specifically, 
        it fills the "spectrally_averaged_aaf" column of all DataFrames
        contained by the *self.climate_and_air_data* dict.

        Notes
        -----
        1) Uses the *self.aerosol_asymmtry_factor* attribute for the calculation.
        
        """
                
        if spectral_range[0] > spectral_range[1]:
            msg = "The upper bound of 'spectral_range' must be greater"
            msg = f"{msg} than the lower bound."
            raise Exception(msg)
        
        
        wavelength = np.array(_SPECTRL2_WAVELENGTHS)
        
        idx0 = (wavelength -  spectral_range[0])**2
        idx0 = list(idx0).index(idx0.min()) + 1 
        
        idx1 = (wavelength -  spectral_range[1])**2
        idx1 = list(idx1).index(idx1.min()) + 1


        for date, aaf_df in self.aerosol_asymmetry_factor.items():
            self.climate_and_air_data[date]["spectrally_averaged_aaf"] =\
            np.array(aaf_df.iloc[:, idx0:idx1+1].mean(axis=1)).astype(float)
            
        return None
        


    # 7) --- OTHER FUNCS ---


    def plot_data(self, col, years, months, days, hours, mode=2, interp_method = "linear", figsize = (16, 12)):

        """
        Plot site variable specified by *col*, for the period of time specified, 
        and using the mode selected.
        
        Parameters
        ----------
        col : str
            Name of the variable to be plotted. Must be one of the keys of 
            *self.variables_info["descriptions"]*. 
            
        years : list or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the years to plot. The first element of *years*
            would be the lower bound, while the second element would be the 
            upper bound. If None, the lower and upper bounds for the *years*
            variable are automatically selected by the program so that all 
            avialable years are included.
            
        months : list or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the months to plot. The first element of *months*
            would be the lower bound, while the second element would be the 
            upper bound. If None, the lower and upper bounds for the *months*
            variable are automatically selected by the program so that all 
            avialable months are included.
            
        days : list or None
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the days to plot. The first element of *days* 
            would be the lower bound, while the second element would be the 
            upper bound. If None, the lower and upper bounds for the 'days' 
            variable are automatically selected by the program so that all 
            avialable days are included.
            
        hours : list of hours
            If list, it must be a list of 2 elements, containing the lower and
            upper bounds of the hours to plot. The first element of *hours* 
            would be the lower bound, while the second element would be the 
            upper bound. 
            
        mode : int, optional
            Mode of plotting. There are 3 options:

                1) mode=1 : Plot all variable curves for all times.
                2) mode=2 : Plot all variable curves for all times + Plot average and 25th, 50th, 75th percentiles.
                3) mode=3 : Plot average and 25th, 50th, 75th percentiles.
    
            Default is mode=2.
    
            
        interp_method : {'linear', 'nearest', 'cubic'}, optional
            Method to use for the interpolation of data before plotting.
            The methods supported are the same ones as those supported by 
            ``scipy.interpolate.griddata`` function. Default is "linear".
            
        figsize : 2-tuple of int, optional
            Figure size of plot. Default is (16, 12).
        
        
        """
        
         # We accomodate the data inot a form which is more suitable for calulations.
        if not isinstance(years,  list):  years  = [years, years]
        if not isinstance(months, list):  months = [months, months]
        if not isinstance(days,   list):  days   = [days, days]       
        if not isinstance(hours,  list):  hours  = [hours, hours]
        
            
        if years[0]  is None : years[0]  = - np.inf
        if years[1]  is None : years[1]  =   np.inf
        if months[0] is None : months[0] = - np.inf
        if months[1] is None : months[1] =   np.inf
        if days[0]   is None : days[0]   = - np.inf
        if days[1]   is None : days[1]   =   np.inf
        if hours[0]  is None : hours[0]  = - np.inf
        if hours[1]  is None : hours[1]  =   np.inf
        
        
        years  = [min(years),  max(years)]
        months = [min(months), max(months)]
        days   = [min(days),   max(days)]
        hours  = [min(hours),  max(hours)]
        
        
        
        CLIMATE_AND_AIR_DATA_COLS =\
        ['hms_float', 'G(h)', 'Gb(n)', 'Gd(h)', 'T2m', 'SP', 'RH', 'O3', 'H2O',
         'AOD_500nm', 'alpha_500nm', 'spectrally_averaged_aaf', 'int G(h)', 
         'int Gb(n)', 'int Gd(h)', 'extra_Gb(n)']
        
        SUN_DATA_COLS =\
        ['hms_float', 'apzen', 'zen', 'apel', 'el', 'az', 'i', 'j', 'k',
         'rel_airmass']
        
        
        
        # We defined the iterator, depending on which variable we want to plot.
        if col in CLIMATE_AND_AIR_DATA_COLS:
            iterator = self.climate_and_air_data.items()
            
        elif col in SUN_DATA_COLS:
            iterator = self.sun_data.items()
            
        elif col == "single_scattering_albedo":
            iterator = self.single_scattering_albedo.items()
            
        elif col == "aerosol_asymmetry_factor":
            iterator = self.aerosol_asymmetry_factor.items()
            
        else:
            msg = f"{col} variable either doesn't exist or cannot be" 
            msg = f"{msg} plotted using this function."
            raise Exception(msg)
            
            

        # --- PLOT SITE OR SUN DATA VARIABLES ----

        if col in CLIMATE_AND_AIR_DATA_COLS  or  col in SUN_DATA_COLS:
            
            dates = []
            data_to_plot = []
            x_eval = np.linspace(hours[0], hours[1], 1000)
            
            for (year, month, day), df in iterator:
                
                # We plot the data which is inside the interval previously 
                # specified.
                if year  < years[0]  or year > years[1]:   continue
                if month < months[0] or month > months[1]: continue
                if day   < days[0]   or day > days[1]:     continue
            
                # As all data may not line-up, we gotta make it line up by
                # evaluating it at the same specified hours.
                x_interp = np.array(df["hms_float"])
                y_interp = np.array(df[col])
                interp_func = interp1d(x = x_interp,
                                       y = y_interp, 
                                       kind = interp_method)
                
                try:
                    y_eval = interp_func(x_eval)
                except ValueError as m:
                    msg = f"{m} Range of hours specified is not valid for"
                    msg = f"{msg} the timeframe selected. Some days"
                    msg = f"{msg} within timeframe selected do not"
                    msg = f"{msg} contain all the hours specified. Try"
                    msg = f"{msg} using a smaller or a different hour"
                    msg = f"{msg} range."
                    raise Exception(msg)
                    
                
                data_to_plot.append(y_eval)
                dates.append((year, month, day))
                
                
            # Compute percentiles and averages.        
            data_to_plot = np.stack(data_to_plot, axis=0)   
            p25 = np.percentile(data_to_plot, q = 25, axis=0)
            p50 = np.percentile(data_to_plot, q = 50, axis=0)
            p75 = np.percentile(data_to_plot, q = 75, axis=0) 
            avg = np.nanmean(data_to_plot, axis=0)
            
            
            _ = plt.figure(figsize=figsize)
            
            
            # We plot Variable vs. Time for all days specified.
            if mode == 0 or mode == 2:
                for i in range(data_to_plot.shape[0]):
                    plt.plot(x_eval, data_to_plot[i,:], color="gray", linestyle="-.")
                    
            if mode == 1 or mode == 2:
                plt.plot(x_eval, p25, color="black", linestyle="-.", label="p25")
                plt.plot(x_eval, p50, color="black", linestyle="-" , label="p50")
                plt.plot(x_eval, p75, color="black", linestyle="--", label="p75")
                plt.plot(x_eval, avg, color="black", linestyle=":",  label="avg")
            

            plt.grid()
            plt.legend(prop={'size': 12})
            plt.xlim(hours[0], hours[1])
            plt.xlabel("Hour [24h-format]", fontsize = 16)
            plt.ylabel(f"{col} {self.variables_info['units'][col]}", fontsize = 16)
            plt.suptitle(
            f"{self.name}: lat={self.latitude}° lon={self.longitude}°, alt={self.altitude} m",
            fontsize = 16)
            plt.title(
            f"{col} vs Time. From inital date: {dates[0]} to final date: {dates[-1]}.", 
            fontsize = 16)
            plt.show()
            
            
        # --- PLOT SINGLE_SCATTERING_ALBEDO, AEROSOL_ASYMMETRY_FACTOR OR GROUND_ALBEDO----
        else:
            
            dates = []
            data_to_plot = []
            
            x_eval = np.linspace(hours[0], hours[1], 24)
            y_eval = np.array(_SPECTRL2_WAVELENGTHS)
            Y_eval, X_eval = np.meshgrid(y_eval, x_eval)
            eval_pts = np.stack([X_eval.flatten(), Y_eval.flatten()], axis=1)
            
            for (year, month, day), df in iterator:
                
                # We plot the data which is inside the interval previously 
                # specified.
                if year  < years[0]  or year > years[1]:   continue
                if month < months[0] or month > months[1]: continue
                if day   < days[0]   or day > days[1]:     continue
            
                # As all data may not line-up, we gotta make it line up by
                # evaluating it at the same specified hours and wavelengths.    
                x_interp = np.array(df["hms_float"])
                y_interp = y_eval.copy()
                Y_interp, X_interp = np.meshgrid(y_interp, x_interp)
                interp_pts = np.stack([X_interp.flatten(), Y_interp.flatten()], axis=1)
                interp_vals = np.array(df.iloc[:,1:]).flatten()
                
                try:
                    evaluated_vals = scipy.interpolate.griddata(points = interp_pts, 
                                                                values = interp_vals,
                                                                xi = eval_pts)
                except ValueError as m:
                    msg = f"{m} Range of hours specified is not valid for"
                    msg = f"{msg} the timeframe selected. Some days"
                    msg = f"{msg} within timeframe selected do not"
                    msg = f"{msg} contain all the hours specified. Try"
                    msg = f"{msg} using a smaller or a different hour"
                    msg = f"{msg} range."
                    raise Exception(msg)
                
                evaluated_vals = evaluated_vals.reshape(len(x_eval),len(y_eval))
                
                data_to_plot.append(evaluated_vals)
                dates.append((year, month, day))
                
                
            # Compute percentiles and averages.
            data_to_plot = np.vstack(data_to_plot) 
            p25 = np.percentile(data_to_plot, q = 25, axis=0)
            p50 = np.percentile(data_to_plot, q = 50, axis=0)
            p75 = np.percentile(data_to_plot, q = 75, axis=0) 
            avg = np.nanmean(data_to_plot, axis=0)     
            
    
            _ = plt.figure(figsize=figsize)
            
            # We plot Variable vs. Wavelength for all dates specified.
            if mode == 0 or mode == 2:
                for i in range(data_to_plot.shape[0]):
                    plt.plot(y_eval, data_to_plot[i,:], color="gray", linestyle="-.")
                    
            if mode == 1 or mode == 2:
                plt.plot(y_eval, p25, color="black", linestyle="-.", label="p25")
                plt.plot(y_eval, p50, color="black", linestyle="-" , label="p50")
                plt.plot(y_eval, p75, color="black", linestyle="--", label="p75")
                plt.plot(y_eval, avg, color="black", linestyle=":",  label="avg")
            

            plt.grid()
            plt.legend(prop={'size': 12})
            plt.xlim(y_eval[0], y_eval[-1])
            plt.xlabel("Wavelengths [nm]", fontsize = 16)
            plt.ylabel(f"{col} {self.variables_info['units'][col]}", fontsize = 16)
            plt.suptitle(
            f"{self.name}: lat={self.latitude}° lon={self.longitude}°, alt={self.altitude} m",
            fontsize = 16)
            plt.title(
            f"{col} vs Wavelength. From inital date: {dates[0]} to final date: {dates[-1]}.", 
            fontsize = 16)
            plt.show()
            
        
        return None



    def time_interpolate_variable(self, col, year, month, day, new_hms_float, interp_method = "linear"):       

        """
        Interpolate a site variable across time.
        
        Parameters
        ----------
        col : str
            Name of the variable to be plotted. Must be one of the keys of 
            *self.variables_info["descriptions"]*.  
            
        year : int
            Year at which the variable is defined.
            
        month : int
            Month at which the variable is defined.
        
        day : int
            Day at which the variable is defined.
            
        new_hms_float : array-like of floats (npoints,)
            Fractional hours at which to evaluate the interpolated variable.
            The range of *new_hms_float* must be the same as the the variable's
            original hms_float.
            
        interp_method : {'linear', 'slinear', 'quadratic', 'cubic'}, optional
            Interpolation method. Any str supported by the ``scipy.interpolate.interp1d`` 
            function is accepted. Default is "linear".
        
            
        Returns
        -------
        interpd_y : numpy.array of floats (npoints,)
            Array of interpolated values for the variable specified by "col",
            at the times specified by "year", "month", "day".

        
        """
        
        if col in self.climate_and_air_data[(year, month, day)].columns:
            
            original_y = self.climate_and_air_data[(year, month, day)][col]
            original_t = self.climate_and_air_data[(year, month, day)]["hms_float"]
            interpd_y = interp1d(original_t, original_y, interp_method)(new_hms_float)
            
            
            
        elif col in self.sun_data[(year, month, day)].columns:
            
            original_y = self.sun_data[(year, month, day)][col]
            original_t = self.sun_data[(year, month, day)]["hms_float"]
            interpd_y = interp1d(original_t, original_y, interp_method)(new_hms_float)
            
            
            
        elif col == "single_scattering_albedo":
            
            interpd_y = np.zeros((len(new_hms_float), 122))
            original_t = self.single_scattering_albedo[(year, month, day)]["hms_float"]
            
            for i in range(122):
                original_y =\
                self.single_scattering_albedo[(year, month, day)].iloc[:,i+1]
                
                interpd_y[:,i] =\
                interp1d(original_t, original_y, interp_method)(new_hms_float)
                    


        elif col == "aerosol_asymmetry_factor":
            
            interpd_y = np.zeros((len(new_hms_float), 122))
            original_t = self.aerosol_asymmetry_factor[(year, month, day)]["hms_float"]
            
            for i in range(122):
                original_y =\
                self.aerosol_asymmetry_factor[(year, month, day)].iloc[:,i+1]
                
                interpd_y[:,i] =\
                interp1d(original_t, original_y, interp_method)(new_hms_float)
                
        else:
            msg = f"{col} variable either doesn't exist or cannot be" 
            msg = f"{msg} interpolated using this function."
            raise Exception(msg)
        
        return interpd_y



    
        
        
   

# %%
