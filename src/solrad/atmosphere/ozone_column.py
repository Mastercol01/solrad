#%%                MODULE DESCRIPTION AND/OR INFO
"""
This module contains all functions, methods and classes related to the
computation and manipulation of the atmospheric ozone column of a site.


**CLIMATE DATA STORE**

"The Copernicus - Climate Data Store (CDS) is an online open and free service
that allows users to browse and access a wide range of climate datasets via a 
searchable catalogue. It also allows users to build their own applications,
maps and graphs."

1) CDS webpage at:
https://cds.climate.copernicus.eu/cdsapp#!/home

2) More info about its API, at:
https://cds.climate.copernicus.eu/api-how-to

3) Useful tutorial on how to use the API, at
https://youtu.be/cVtiVTSVdlo
    

**NOTE:** As described by the links in 2) and 3), it is necessary to have a CDS 
account (and be currently logged in) account in order to be able to use
the API. Furtheremore, the user's key and the API website link should be 
stored in a place, recognisable by the system being used. 
"""

#%%                  IMPORTATION OF LIBRARIES

import os
import cdsapi
import zipfile
import warnings
import numpy as np
import netCDF4 as nc
from functools import wraps
from dateutil.parser import parse
from scipy.interpolate import RegularGridInterpolator



#%%


def compute_van_Heuklon_ozone(latitude, longitude, timestamp):
    """
    Returns the ozone contents in atm-cm for the given latitude/longitude and
    timestamp according to van Heuklon's Ozone model. The model is described in
    Van Heuklon, 'T. K. (1979). Estimating atmospheric ozone for solar radiation 
    models. Solar Energy, 22(1), 63-68'. This function uses numpy functions, 
    so you can pass arrays and it will return an array of results. The
    timestamp argument can be either an array/list or a single value. 
    If timestamp is a single value then this will be used for all lat/lon 
    values given. 
    
    Parameters
    ----------
    latitude : float or array-like of floats (npoints,)
        Site's latitude in degrees. Values must be between -90 and 90.
        
    longitude : float or array-like of floats (npoints,)
        Site's longitude in degrees. Values must be between -180 and 180.
    
    timestamp : pandas.Timestamp object or array-like of pandas.Timestamp objects float or array-like of floats (npoints,)
        The times for which the ozone is to be computed. It is strongly 
        recommend that the timestamp use an ISO 8601 format of yyyy-mm-dd.
        
    Returns
    -------
    result : float
        Ozone amount in atm-cm.

    Raises
    ------
    1) ValueError 
        "lan and lon arrays must be the same length"

    2) ValueError 
        "Timestamp must be the same length as lat and lon"
        
    Notes
    -----
    1) This function was directly taken from https://github.com/robintw/vanHOzone
       all credit goes to him. I copy-pasted the code rather than directly 
       downloading tthe pachage from pip as I wanted to add some very minor
       changes.
       
    2) The function supports array-like inputs for latitude, longitude
       and timestamp, as long as all 3 arguments are the same length. 
      
    3) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
      
    4) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.

    """

    lat, lon = latitude, longitude

    # Deal with scalar values
    try:
        lat_count = len(lat)
    except:
        lat = [lat]
        lat_count = 1

    try:
        lon_count = len(lon)
    except:
        lon = [lon]
        lon_count = 1

    if lat_count != lon_count:
        raise ValueError("lan and lon arrays must be the same length")

    lat = np.array(lat)
    lon = np.array(lon)

    # Set the Day of Year
    try:
        # Try and do list-based things with it
        # If it works then it is a list, so check length is correct
        # and process
        count = len(timestamp)
        if count == len(lat):
            try:
                E = [t.timetuple().tm_yday for t in timestamp]
                E = np.array(E)
            except:
                d = [parse(t) for t in timestamp]
                E = [dt.timetuple().tm_yday for dt in d]
                E = np.array(E)
        else:
            raise ValueError("Timestamp must be the same length as lat and lon")
    except:
        # It isn't a list, so just do it once
        try:
            # If this works then it is a datetime obj
            E = timestamp.timetuple().tm_yday
        except:
            # If not then a string, so parse it and set it
            d = parse(timestamp)
            E = d.timetuple().tm_yday

    # Set parameters which are the same for both
    # hemispheres
    D = 0.9865
    G = 20.0
    J = 235.0

    # Set to Northern Hemisphere values by default
    A = np.zeros(np.shape(lat)) + 150.0
    B = np.zeros(np.shape(lat)) + 1.28
    C = np.zeros(np.shape(lat)) + 40.0
    F = np.zeros(np.shape(lat)) - 30.0
    H = np.zeros(np.shape(lat)) + 3.0
    I = np.zeros(np.shape(lat))

    # Gives us a boolean array indicating
    # which indices are below the equator
    # which we can then use for indexing below
    southern = lat < 0

    A[southern] = 100.0
    B[southern] = 1.5
    C[southern] = 30.0
    F[southern] = 152.625
    H[southern] = 2.0
    I[southern] = -75.0

    # Set all northern I values to 20.0
    # (the northern indices are the inverse (~) of
    # the southern indices)
    I[~southern] = 20.0

    I[(~southern) & (lon <= 0)] = 0.0

    bracket = (A + (C * np.sin(np.radians(D * (E + F))) + G *
                    np.sin(np.radians(H * (lon + I)))))

    sine_bit = np.sin(np.radians(B * lat))
    sine_bit = sine_bit ** 2

    result = J + (bracket * sine_bit)
    
    # We convert from dobson to atm-cm.
    result /= 1000

    return result   



def get_CDS_ozone_column_data(path, year, month=None, file_format = "numpy"):
    
    """
    Connects to the Climate Data Store (CDS) through its API and downloads the 
    ozone-column data from the database in [1], for the requested time frame.
    A new folder (whose path is specified by the user) is then automatically
    created to store the downloaded files.
    
    Parameters
    ----------
    path : path-str
        Path of the folder where one wishes to store the downloaded files.
    
    year : list of str
        List of years for which the ozone-column data is to be retieved. 
        The years must be of type str rather than int. 
        Eg.: year = ["2019", "2020", "2021"].
        
    month  : None or list of str
        If is None (default), all months for the selected years are retrieved.
        If not None, it must be list of months for which to retrieve the data.
        Said months must be of type str rather than int. 
        Eg.: month = ["01", "02", "11", "12"].

    file_format : {'NetCDF4', 'numpy'}, optional
        Format in which the data is to be downloaded. 
        If 'NetCDF4', the data is downloaded in its original format and no changes are made to it.
        If 'numpy'  , the relevant water column data are downloaded, along with the latitude
        and longitude data, as numpy arrays. We also convert units from Dobson to atm-cm (see notes 
        for more info). Default is 'numpy'.
        

        
    Returns
    -------
    None

    
    References
    ----------
    [1] Copernicus Climate Change Service, Climate Data Store, (2020): 
    Ozone monthly gridded data from 1970 to present derived from satellite observations.
    Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.4ebfe4eb 
        
    Notes
    -----
    1) For this function to work, the user must have a Climate Data Store 
       account and be currently logged in. Furtheremore, the user's key and the
       API website link should be stored in a place, recognisable by the system being used. 
       See https://youtu.be/DIdgltyoIYg?si=q7Ylu2p0IDFT9UGm for a quick Youtube tutorial about it.
       See https://cds.climate.copernicus.eu/api-how-to for the official documentation.
       
    2) For more information on the specific databse used, see:
       https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-ozone-v1?tab=overview

    3) The downloaded data are .nc files holding the monthly-average of ozone-column data 
       (in m atm-cm, i.e, Dobson) of the whole globe, for the requested time frame.

    """
    
    # 1) --- DOWNLOAD CDS DATA ---

    zip_path = path + ".zip"
    
    
    if month is None:
        month_ = ["0" + str(i) for i in range(1,10)]
        month_ +=  [str(i) for i in range(10,13)]
        
    else: 
        month_ = month
    

    # The database being used here actually allows for the download of other 
    # variables. The settings used here are such as to only retrive ozone 
    # column data.
    
    c = cdsapi.Client()
    c.retrieve(
        'satellite-ozone-v1',
        {
            'processing_level'     : 'level_4',
            'variable'             : 'atmosphere_mole_content_of_ozone',
            'vertical_aggregation' : 'total_column',
            'sensor'               : 'msr',
            'year'                 : year,
            'month'                : month_,
            'version'              : 'v0024',
            'format'               : 'zip',
        }, 
        zip_path)
    
      
    with zipfile.ZipFile(zip_path, "r") as my_zip:
        my_zip.extractall(path = path)
        
    # Erase unextracted zip folder.
    os.remove(zip_path)

    
    # 2) --- CONVERT CDS DATA TO NUMPY ---

    if file_format == 'numpy':

        # Get all .nc filenames in specified dir.s
        ncfile_names = [i for i in os.listdir(path) if ".nc" in i]

        for i, ncfile_name in enumerate(ncfile_names):

            # Read .nc file.
            ncfile_path = os.path.join(path, ncfile_name)
            ncfile = nc.Dataset(ncfile_path) 

            #Save one copy of the latitude and longitude arrays.
            if i == 0:
                latitude_data  = np.array(ncfile.variables["latitude"])
                longitude_data = np.array(ncfile.variables["longitude"])
                latitude_data_path  = os.path.join(path, ncfile_name[7:].replace(".nc", "_latitude.npy"))
                longitude_data_path = os.path.join(path, ncfile_name[7:].replace(".nc", "_longitude.npy"))
                np.save(latitude_data_path,  latitude_data)
                np.save(longitude_data_path, longitude_data)

            # We convert the units from Dobson to atm-cm.  
            raw_data = np.array(ncfile.variables["total_ozone_column"][0,:,:])/1000 
            raw_data[raw_data < 0] = np.nan
            ncfile.close()

            # Save data as new .npy file.
            raw_data_path = os.path.join(path, ncfile_name.replace(".nc", "_raw.npy"))
            np.save(raw_data_path, raw_data)

            # Erase original .nc file.
            os.remove(ncfile_path)

    return None




    

def process_CDS_ozone_column_data(path, percentile = 0.5, interp_method = "linear"):

    """
    Process ozone data located in the local ozone database. This function reads the raw.npy 
    ozone-column files (files which were obtained via the :func:`~solrad.atmosphere.ozone_column.get_CDS_ozone_column_data` function)
    stored at the directory specified by *path* and then computes multiple useful quantities.
    
    Parameters
    -----------
    path : path-str
        Path of the folder where the water column raw.ny and filled_NaNs.npy files 
        are stored. That is, the path to the local water column database.        
        
    percentile : float
        Percentile for computing the 'percentile_data' and 'percentile_data_funcs'
        dictionaries. Must be a number between 0 and 1. Default is 0.5.
        
        
    interp_method : {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}
        The method of interpolation to perform when computing the 
        res['raw_data_funcs'], res['avg_data_funcs'] and res['percentile_data_funcs']
        dictionaries. Supported methods are the same as supported by scipy's 
        RegularGridInterpolator. Default is "linear".
        
    Returns
    -------
    res : dict
        Dictionary of computed quantities. It has the following key-value
        pairs:
            
            Key : Value
            ------------
            
            "latitude" : numpy.array of floats (npoints,)
                Array of latitude values (in degrees) used by the raw.npy and filled_NaNs.npy 
                files to specify the locations at which the water column data is 
                reported.
                
            "longitude" : numpy.array of floats (mpoints,)
                Array of longitude values (in degrees) used by the raw.npy and filled_NaNs.npy 
                files to specify the locations at which the water column data is 
                reported.
                
            "raw_data" : dict
                Dictionary containing the raw data of ozone-column values stored by the
                raw.npy files in the local water column database. It has the following 
                key-value pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : 2D numpy.array of floats (npoints, mpoints)
                        Where 'year' and 'month' are integers that specify the 
                        time period for the array of data (with units atm-cm).
                        
            "raw_data_funcs" : dict 
                Dictionary containing the interpolating functions of the raw 
                data of ozone-column values stored in the local ozone database.
                It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : scipy.interpolate.RegularGridInterpolator object
                        Where 'year' and 'month' are integers that specify the 
                        time period for the array of data (with units atm-cm)
                        that the function interpolates. It takes as input a value
                        of latitude and longitude and returns the ozone-column value 
                        expected at that location.
              
            "avg_data" : dict
                Dictionary containing the year-wise averaged data of ozone-column 
                values stored in the local ozone database, for each month. It has 
                the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats (npoints, mpoints)
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm).
                        
            "avg_data_funcs" : dict 
                Dictionary containing the interpolating functions of the year-wise 
                averaged data of ozone-column values stored in the local ozone database, 
                for each month. It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : scipy.interpolate.RegularGridInterpolator object
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm)
                        that the function interpolates. It takes as input a value
                        of latitude and longitude and returns the ozone-column value 
                        expected at that location.
                        
            "percentile_data" : dict
                Dictionary containing the year-wise 'percentile'-th percentile
                of the ozone-column data values stored in the local ozone
                database, for each month. It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats (npoints, mpoints)
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm).
                        
            "percentile_data_funcs" : dict 
                Dictionary containing the interpolating functions of the year-wise 
                'percentile'-th percentile of the ozone-column data values stored in
                the local ozone database, for each month. It has the following 
                key-value pair format:
                    
                    Key : Value
                    -----------
                    month : scipy.interpolate.RegularGridInterpolator object
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm)
                        that the function interpolates. It takes as input a value
                        of latitude and longitude and returns the ozone-column value 
                        expected at that location.
                   
    Raises
    ------
    1) Exception 
        "Local ozone column database is empty. No ozone column raw.npy nor filled_NaNs.npy files to retrieve were found."

    2) Exception 
        "Latitude data could not be recovered. No latitude.npy files are present in the database."

    3) Exception 
        "Longitude data could not be recovered. No longitude.npy files are present in the database."

    Warns
    -----
    1) Warning
       "WARNING: Local ozone column database lacks raw.npy data files for all 12 months of the year."
 
    Notes
    -----       
    1) res["raw_data"], res["raw_data_funcs"], res["avg_data"], res["avg_data_funcs"],
       res["precentile_data"] and res["precentile_data_funcs"] as many key-value pairs 
       as there are raw.npy files in the local ozone database.
    
    2) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
       
    3) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.
       
    
    """
    
    
    #  1) ---- FILENAMES RETRIEVAL ----

    # We get all the raw.npy filenames of stored at the local water column database.
    raw_data_filenames    = [i for i in os.listdir(path) if "raw.npy" in i]

    # We attempt to get latitude and longitude .npy files
    latitude_file_names  = [i for i in os.listdir(path) if "latitude.npy" in i]
    longitude_file_names = [i for i in os.listdir(path) if "longitude.npy" in i]
    

    # 2) ---- CHECK PRESENCE AND ABSENCE OF FILES ----

    # We check if the local ozone column database is empty of "raw.npy" files      
    if len(raw_data_filenames) == 0: 
        msg = "Local ozone column database is empty."
        msg = f"{msg}. No ozone column raw.npy files to retrieve were found."
        raise Exception(msg)


    # 3) ---- CHECK PRESENCE OR ABSENCE OF LATITUDE AND LONGITUDE DATA ----

    if len(latitude_file_names) > 0:
        latitude_path = os.path.join(path, latitude_file_names[0])
        lat = np.load(latitude_path)

    else:
        msg = "Latitude data could not be recovered"
        msg = f"{msg}. No latitude.npy files are present in the database."
        raise Exception(msg)
    

    if len(longitude_file_names) > 0:
        longitude_path = os.path.join(path, longitude_file_names[0])
        lon = np.load(longitude_path)

    else:
        msg = "Longitude data could not be recovered"
        msg = f"{msg}. No longitude.npy files are present in the database."
        raise Exception(msg)
    

    # 4) ---- RETRIEVE RAW DATA FROM RAW.NPY FILES AND GENERATE INTERPOLATING FUNCTIONS---

    years, months = set(), set()
    # We read all raw.npy files and store everything into a user-friendlier format.

    raw_data = {}
    raw_data_funcs = {}
    for raw_data_filename in raw_data_filenames:

        year_month = raw_data_filename.split("-")[0]
        year  = int(year_month[:4])
        month = int(year_month[4:])
        
        years.add(year)
        months.add(month)

        raw_data_file_path = os.path.join(path, raw_data_filename)
        raw_data[(year, month)] = np.load(raw_data_file_path)

        raw_data_funcs[(year, month)] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = raw_data[(year, month)],
                                method = interp_method)
                
        
    # If not all 12 months of the year are present in the local water column
    # database (with respect to the raw.npy files) we throw a warning.
    if len(months) < 12:
        message = "WARNING: Local ozone column database lacks raw.npy data files"
        message = f"{message} for all 12 months of the year."
        warnings.warn(message)
        

        
    #  5) ---- AVG DATA COMPUTATION AND INTERPOLATION ----    
        
    # We compute the year-averaged data and interpolating functions.
    avg_data = {}
    avg_data_funcs = {}
    for month in months:
        
        avg_data[month] = []
        for year in years:
            try: avg_data[month].append(raw_data[(year, month)])
            except KeyError: pass

        avg_data[month] = np.stack(avg_data[month], axis=2) 
        avg_data[month] = avg_data[month].mean(axis=2)
        
        avg_data_funcs[month] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = avg_data[month],
                                method = interp_method)
        
        
        
    #  6) ---- PERCENTILE DATA COMPUTATION AND INTERPOLATION ---- 
        
    # We compute the year-wise 'perecentile'-th percentile data and 
    # interpolating functions.      
    percentile_data = {}
    percentile_data_funcs = {}
    for month in months:
        
        percentile_data[month] = []
        for year in years:
            try: percentile_data[month].append(raw_data[(year, month)])
            except KeyError: pass
        
        percentile_data[month] = np.stack(percentile_data[month], axis=2) 
        percentile_data[month] = np.percentile(percentile_data[month], 
                                               q=percentile, 
                                               axis=2)
        
        percentile_data_funcs[month] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = percentile_data[month],
                                method = interp_method)
        
        
    #  ---- RESULTS ---- 
        
        
    # We store all the data and return it.
    res = {
    "latitude" : lat,
    "longitude" : lon,
    "raw_data" : raw_data,
    "raw_data_funcs" : raw_data_funcs,
    "avg_data" : avg_data,
    "avg_data_funcs" : avg_data_funcs,
    "percentile_data" : percentile_data,
    "percentile_data_funcs" : percentile_data_funcs }
        
        
        
    return res
        
        
        
        
    


# %%
