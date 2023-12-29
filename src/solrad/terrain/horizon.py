#%%                 IMPORTATION OF LIBRARIES
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#%%                 DEFINITION OF PATH CONSTANTS

# PVGIS Api url, for extracting the horizon profile.
URL = 'https://re.jrc.ec.europa.eu/api/printhorizon'


#%%                 DEFINITION OF FUNCTIONS
def horizon_arrays_from_pvgis(latitude, longitude, timeout=30):
    """
    Get a site's horizon profile, computed by PVGIS, using its API
    Non-Interactive Service.

    Parameters
    ----------
    latitude : float
        Site's latitude in sexagesimal degrees. Must be a number between -90
        and 90.

    longitude : float
        Site's longitude in sexagesimal degrees. Must be a number between -180
        and 180.

    timeout : float
        Number of seconds after which the requests library will stop waiting
        for a response of the server. That is, if the requests library does not
        receive a response in the specified number of seconds, it will raise a
        Timeout error.

    Returns
    -------
    azimuth : numpy.array of floats
        Array of azimuth angle values in degrees.

    elevation : numpy.array of floats
        Array of elevation angle values in degrees.

    Note
    ----
    1) Horizon height is the angle between the local horizontal plane and the
    horizon. In other words, the Horizon height is equal to the horizon's
    elevation angle.

    Examples
    --------
    >>> from solrad.terrain.horizon import horizon_arrays_from_pvgis
    >>> azimuth, elevation = horizon_arrays_from_pvgis(latitude  =  6.25184,
    >>>                                                longitude = -75.56359)

    """


    # We request the horizon profile calculated by PVGIS from its API,
    # for given latitude and longitude.
    res = requests.get(URL, params={"lat":latitude, "lon":longitude}, timeout=timeout)


    # We check for any errors in the request.
    if not res.ok:
        try:
            err_msg = res.json()
        except Exception:
            res.raise_for_status()
        else:
            raise requests.HTTPError(err_msg['message'])

    # The values obtained from the request are given as a string. As such,
    # some procesing is necessary to extract the numerical values of the
    # horizon profile.
    lines = res.text.split("\n")[4:-9]
    horizon_df = pd.DataFrame(columns=["az", "H_hor"], index=range(len(lines)))

    for i, line in enumerate(lines):
        values = [float(value) for value in line.split("\t\t")]
        horizon_df.loc[i,"az"] = values[0]
        horizon_df.loc[i,"H_hor"] = values[1]

    horizon_df = horizon_df.astype(float)

    # The coordinate system used by PVGIS for the horizon profile is different
    # that the one used in this package. In particular, for PVGIS: N = ± 180°,
    # E = -90°, S = 0°, and W = 90°. While, for us: N = 0°, E = 90°, S = 180°
    # and W = 270°. Adding 180° to PVGIS's azimuth resolves the problem.

    horizon_df["az"] += 180

    azimuth, elevation = np.array(horizon_df["az"]), np.array(horizon_df["H_hor"])

    return azimuth, elevation


def _check_for_azimuth_array_compliance(azimuth):

    """
    Check if the azimuth array is monotonic-increasing and covers the full range from 0° to 360°.

    Parameters
    ----------
    azimuth : (npoints, ) array_like
        Array of azimuth angle values in degrees.

    Raises
    ------
    Exception
      The azimuth array must be monotonic increasing and encompass the whole range from 0° to 360°.
      Current range is: {az_min}° to {az_max}°. Monotonic increasing is {az_mono_increasing}.

    Returns
    -------
    None

    """

    az_min, az_max = azimuth.min(),   azimuth.max()
    az_mono_increasing = all(np.gradient(azimuth) >= 0)

    if az_min != 0 or az_max != 360 or not az_mono_increasing:
      msg =  "The azimuth array must be monotonic increasing and encompass the whole range from 0° to 360°"
      msg = f"{msg}. Current range is: {az_min}° to {az_max}°. Monotonic increasing is {az_mono_increasing}."
      raise Exception(msg)

    return None


def _check_for_elevation_array_compliance(elevation):

    """
    Check if the elevation array values lie between 0° and 90°.

    Parameters
    ----------
    elevation : (npoints, ) array_like
        Array of elevation angle values in degrees.

    Raises
    ------
    Exception
      Elevation values must lie in the range of 0° to 90°.

    Returns
    -------
    None

    """

    el_min, el_max = elevation.min(), elevation.max()
    if el_min < 0 or el_max > 90:
      msg = "Elevation values must lie in the range of 0° to 90°."
      raise Exception(msg)

    return None



def horizon_func_from_arrays(azimuth, elevation, interp_method = 'linear'):
  """
  Create horizon elevation function by interpolating provided data points.

  Parameters
  ----------
  azimuth : (npoints, ) array_like
      Array of azimuth angle values in degrees, from 0 to 360. Must be monotonic-increasing.

  elevation : (npoints, ) array_like
      Array of horizon elevation angle values in degrees. Elevation values must
      lie between 0 and 90.

  interp_method : {'linear', 'quadratic', 'cubic'}, optional
      Order of the spline interpolator to use. Default is 'linear'.

  Returns
  -------
  terrain_func : callable
      Interpolation function for azimuth, elevation values.

  Examples
  --------
  >>> import numpy as np
  >>> from solrad.terrain.horizon import horizon_func_from_arrays
  >>>
  >>> azimuth = np.linspace(0, 360, 100)
  >>> elevation = np.sin(azimuth*np.pi/180)**2
  >>> func = horizon_func_from_arrays(azimuth, elevation, interp_method="linear")
  >>> func([88, 89, 90])
  """

  _check_for_azimuth_array_compliance(azimuth)
  _check_for_elevation_array_compliance(elevation)
  horizon_func = interp1d(azimuth, elevation, kind=interp_method)

  return horizon_func




def plot_horizon(y, azimuth, config=None):

    """
    Plots the horizon profile based on the provided azimuth and elevation data.

    Parameters
    ----------
    y : array_like (npoints,) or callable
        Elevation data in degrees or a callable function that takes azimuth values and returns elevation values.

    azimuth : array_like
        Array of azimuth angle values in degrees.

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

    Notes
    -----
    The function utilizes the '_check_for_elevation_array_compliance' and '_check_for_azimuth_array_compliance' functions
    to ensure the compliance of elevation and azimuth data before plotting.

    Examples
    --------

    >>> from terrain.horizon import , plot_horizon
    >>>
    >>> # Plots the horizon profile based on the provided data.
    >>> azimuth_data   = np.linspace(0, 360, 100)
    >>> elevation_data = 10*np.sin(np.deg2rad(azimuth_data))**2
    >>> plot_horizon(elevation_data, azimuth_data)
    >>> # Plots the horizon profile based on the provided data.
    >>>
    >>> # Plots the horizon profile using a custom function and customized configuration.
    >>> azimuth_data   = np.linspace(0, 360, 100)
    >>> def custom_function(azimuth):
    >>>     return 30*np.sin(np.deg2rad(azimuth)) + 45
    >>> plot_horizon(custom_function, azimuth_data, config={"projection": "cartesian"})
    """

    try:
      elevation = y(azimuth)
    except TypeError:
      elevation = y

    _check_for_elevation_array_compliance(elevation)


    # Default plot settings.
    config_ = {"projection"           : "polar",
               "show_polar_elevation" : False,
               "title"                : "Horizon profile",
               "facecolor"            : "0.5",
               "figsize"              : (12,12)}


    # User settings overwrite default settings.
    if isinstance(config, dict):
      config_.update(config)


    # --- MAKE POLAR PLOT ---
    if config_["projection"] == "polar":
        _check_for_azimuth_array_compliance(azimuth)

        fig, ax =\
        plt.subplots(figsize = config_["figsize"], subplot_kw={"polar":True})

        ax.patch.set_facecolor(config_["facecolor"])
        rad, theta = 90 - elevation, np.deg2rad(azimuth)
        ax.plot(theta, rad, color='black', ls='-', linewidth=1)

        # The Horizon height lies between 0 and 90 degrees.
        ax.set_rlim(0, 90)
        ax.fill(theta, rad, 'w')

        # We hide or show the radius ticks based on user input.
        ax.set_yticklabels([])
        if config_["show_polar_elevation"]:
          ax.set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])

        # We count the angles clockwise.
        ax.set_theta_direction(-1)
        plt.suptitle(config_["title"])

        # We put the 0 angle at the top.
        ax.set_theta_zero_location("N")
        ax.set_title("(N = 0°, E = 90°, S = 180°, W = 270°)")

        plt.show()
        plt.draw()


    # --- MAKE CARTESIAN PLOT ---
    elif config_["projection"] == "cartesian":

        _ = plt.figure(figsize = config_["figsize"])
        plt.plot(azimuth, elevation, color="k")

        plt.grid()
        plt.xlim(azimuth.min(), azimuth.max())
        plt.title(config_["title"])
        plt.ylabel("Horizon Height [°]")
        plt.xlabel("Azimuth [°]  (N = 0°, E = 90°, S = 180°, W = 270°)")


    return None



def clear_horizon_func(azimuth):

    """
    Generates an array of elevation values filled with zeros based on the provided azimuth data.

    Parameters
    ----------
    azimuth : float or array_like (npoints,)
        Single value or array of azimuth angle values.

    Returns
    -------
    elevation : float or numpy.array (npoints,)
        Single value or array of elevation values, initialized with zeros.
    """

    try:
      elevation = np.zeros(np.array(azimuth).shape)
    except AttributeError:
      elevation = 0

    return elevation



def compute_fraction_of_unblocked_sky_patch_by_horizon(horizon_func, el0, el1, az0, az1, npts = 250):
    """
    Compute the fraction of a sky patch that is not blocked by the horizon.

    Parameters
    ----------
    horizon_func : callable
        A function that takes an array of azimuth angles and returns the corresponding elevation angles
        of the horizon.

    el0 : int or float
        Lower limit of elevation angle (in degrees) of the sky patch.
        
    el1 : int or float
        Upper limit of elevation angle (in degrees) of the sky patch.

    az0 : int or float
        Lower limit of azimuth angle (in degrees) of the sky patch.

    az1 : int or float
        Upper limit of azimuth angle (in degrees) of the sky patch.

    npts : int, optional
        The number of points used for discretization along azimuth and elevation, default is 250.

    Returns
    -------
    unblocked_fraction : float
        The fraction of area of the sky patch not blocked by the horizon.

    Raises
    ------
    1) Exception 
        az0 must be strictly greater than az1.

    2) Exception 
        el0 must be strictly greater than el1.

    Notes
    -----
    This function calculates the fraction of the sky patch that is unblocked by the horizon,
    given a function *horizon_func* that provides the elevation angles of the horizon for a given
    array of azimuth angles. The computation is performed by discretizing the azimuth and elevation
    angles, and then integrating over the unblocked region.

    
    """

    if az0 >= az1:
       raise Exception("az0 must be strictly greater than az1")
    
    if el0 >= el1:
       raise Exception("el0 must be strictly greater than el1")
    

    azs = np.linspace(az0, az1, npts)
    els = np.linspace(el0, el1, npts)
    horizon_els = horizon_func(azs)

    # If the sky patch is fully above the horizon, the it remains fully unblocked.
    if all(el0 > horizon_els):
        unblocked_fraction = 1
   
    # If the sky patch is fully below the horizon, the it remains fully blocked.
    elif all(el1 <= horizon_els):
        unblocked_fraction = 0

    # If the horizon line crosses the sky patch, it becomes partially blocked.
    else:
        azs = np.deg2rad(azs)
        els = np.deg2rad(els)
        horizon_els = np.deg2rad(horizon_els)

        # We sample a bunch of points on the sky patch and determine
        # wether or not they are blocked.
        Azs, Els   = np.meshgrid(azs, els)
        Azs, H_Els = np.meshgrid(azs, horizon_els)
        unblocked_pts = (Els > H_Els).astype(int)

        # Total area of the sky patch.
        Atot  = np.deg2rad(az1) - np.deg2rad(az0)
        Atot *= np.cos(np.deg2rad(90-el1)) - np.cos(np.deg2rad(90-el0))
        
        dAz   = np.deg2rad(az1 - az0)/(npts-1)
        dZen  = np.deg2rad(el1 - el0)/(npts-1)

        # We compute the unblocked area of the sky patch and get the fraction.
        integrand = unblocked_pts*np.sin(np.pi/2 - Els)
        integral = 0.25*(integrand[:-1, :-1] + integrand[:-1,  1:] +
                         integrand[1:,  :-1] + integrand[1:,  1:]).sum()*dZen*dAz
        
        unblocked_fraction = integral/Atot


    return unblocked_fraction







