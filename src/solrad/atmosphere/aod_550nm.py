#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
computation and manipulation of the Aerosol Optical Depth at 550nm (AOD_550)
of a site.


                  ---- CLIMATE DATA STORE ----

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
    

NOTE: As described by the links in 2) and 3), it is necessary to have a CDS 
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
import solrad.auxiliary_funcs as aux
from scipy.interpolate import RegularGridInterpolator


#%%

def get_CDS_aod_550nm_data(path, year, month=None, file_format = "numpy"):

    """
    Connects to the Climate Data Store (CDS) through its API and downloads the 
    monthly-averaged aerosol optical depth at 550 nm (aod_550nm) data 
    from the database in [1], for the requested time frame. A new folder 
    (whose path is specified by the user) is then automatically created 
    to store the downloaded files.
    
    Parameters
    ----------
    path : path-str
        Path of the folder where one wishes to store the downloaded files in. 
        Said folder need not exist, as it will be automatically created.

    
    year : list of str
        List of years for which the water-column data is to be retieved. 
        The years must be of type str rather than int. 
        Eg.: year = ["2020", "2021", "2022"].
        
    month  : None or list of str, optional
        If is None (default), all months for the selected years are retrieved.
        If not None, it must be list of months for which to retrieve the data.
        Said months must be of type str rather than int. 
        Eg.: month = ["01", "02", "11", "12"].

    file_format : {'NetCDF4', 'numpy'}, optional
        Format in which the data is to be downloaded. 
        If 'NetCDF4', the data is downloaded in its original format and no changes are made.
        If 'numpy'  , the relevant aod_550nm data are downloaded, along with the latitude
        and longitude data, as numpy arrays. We also convert units from kg/m² to atm-cm (see notes 
        for more info). Default is 'numpy'.

        
    Returns
    -------
    None
    
        
    Notes
    -----
    1) For this function to work, the user must have a Climate Data Store 
       account and be currently logged in. Furtheremore, the user's key and the
       API website link should be stored in a place, recognisable by the system being used. 
       See https://youtu.be/DIdgltyoIYg?si=q7Ylu2p0IDFT9UGm for a quick Youtube tutorial.
       See https://cds.climate.copernicus.eu/api-how-to for the official documentation.
        
    2) For more information on the specific databse used, see:
       https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-total-column-water-vapour-land-ocean?tab=overview


    References
    ----------
    [1] Copernicus Climate Change Service, Climate Data Store, (2019): Aerosol
        properties gridded data from 1995 to present derived from satellite observation. 
        Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.239d815c 

    """
    
    # 1) --- DOWNLOAD CDS DATA ---       
    
    zip_path = path + ".zip"
    
    
    if month is None:
        month_ = ["0" + str(i) for i in range(1,10)]
        month_ +=  [str(i) for i in range(10,13)]
        
    else: 
        month_ = month
    

    # The database being used here actually allows for the download of other 
    # variables. The settings used here are such as to only retrive aod_550 
    # data.

    c = cdsapi.Client()
    c.retrieve(
        'satellite-aerosol-properties',
        {
            'time_aggregation': 'monthly_average',
            'variable': 'aerosol_optical_depth',
            'sensor_on_satellite': 'olci_on_sentinel_3a',
            'algorithm': 'ens',
            'year': year,
            'month': month_,
            'version': 'v1.1',
            'format': 'zip',
        },
        zip_path)
    
      
    with zipfile.ZipFile(zip_path, "r") as my_zip:
        my_zip.extractall(path = path)
        
    # Erase unextracted zip folder.
    os.remove(zip_path)

        
    # 2) --- CONVERT CDS DATA TO NUMPY ---

    if file_format == 'numpy':

        # Get all .nc filenames in specified dir.
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

            raw_data = np.array(ncfile.variables["AOD550"][:,:])
            raw_data[raw_data < 0] = np.nan
            ncfile.close()

            # Save data as new .npy file.
            raw_data_path = os.path.join(path, ncfile_name.replace(".nc", "_raw.npy"))
            np.save(raw_data_path, raw_data)

            # Erase original .nc file.
            os.remove(ncfile_path)

    return None


def fill_CDS_aod_550nm_data_nans(path, iterations = 20000, show_progress = False, replace = False):

    """
    Fill NaNs with suitable numeric aproximation of CDS aod_550nm raw data.


    The data retrieved from the Climate Data Store (CDS) database referenced
    in [1] has a considerable amount of missing or defective values in its files.
    This is inconvenient for later computations. As such, this function reads all
    'raw.npy' files in the directory specified by the 'path' variable (i.e, the local 
    aod_550nm database), that were obtained via the function 'get_CDS_water_column_data'; and 
    uses the function 'auxiliary_funcs.fill_CDS_globe_nans_using_laplace' to fill each file's 
    NaN values with a suitable numeric approximation and then saves each modified file back 
    to the same directory as a 'filled_NaNs.npy' file. 
    
    Parameters
    ----------
    path : path-str
        Path of the folder where the raw.npy files, containing the aerosol 
        optical depth at 550 nm (aod_550nm) information downloaded from the 
        aforementioned database, are stored. The resulting filled_NaNs.npy 
        files will also be stored in this same directory.
        
    iterations: int
        Number of iterations that the 'auxiliary_funcs.fill_CDS_globe_nans_using_laplace'
        function should use for computing the numerical approximation to the
        NaN values, before stopping (must be non-negative). The greater the 
        number of iterations, the greater the chance that convergence of the
        computed values has been reached. However, the time of computation also 
        increases. Default is 20000.
        
    show_progress : bool, optional
        If True, after processing each file, it prints how many files
        have been processed as of yet. If False, it prints nothing.
        Default is False.

    replace : bool, optional
        Whether to errase the original raw.npy files, after the new filled_NaNs.npy
        files (with have been created. This, as a way to save space in memory. 
        Default is False. 


    Returns
    -------
    None

    Raises
    ------
    1) Exception :
       "Local aod_550nm database is empty. No aod_550nm raw.npy files to retrieve were found."

    See Also
    --------
    1) auxiliary_funcs.fill_CDS_globe_nans_using_laplace 

    References
    ----------
    [1] Copernicus Climate Change Service, Climate Data Store, (2019): Aerosol
        properties gridded data from 1995 to present derived from satellite observation. 
        Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.239d815c 

    """

    # We get all the filenames of the raw.npy files stored at the local aod_550nm
    # database.
    raw_data_filenames = [i for i in os.listdir(path) if "raw.npy" in i]
    
    # If the local aod_550nm database is empty, we throw an error.    
    if len(raw_data_filenames) == 0:
        msg = "Local aod_550nm database is empty"
        msg = f"{msg}. No aod_550nm raw.npy files to retrieve were found."
        raise Exception(msg)

    
    # We read each raw.npy file and pass it to the 'auxilary_funcs.fill_CDS_globe_nans_using_laplace'
    # function in order to fill the missing values. We then save the modifed
    # arrays as a filled_NaNs.npy files to the same directory.
    
    for i, raw_data_filename in enumerate(raw_data_filenames):

        # Get raw data file.
        raw_data_path = os.path.join(path, raw_data_filename)
        raw_data = np.load(raw_data_path)
        
        # Fill NaNs.
        filled_nans_data = aux.fill_CDS_globe_nans_using_laplace(raw_data, iterations)
        
        # Save modified file to the same dir.
        filled_nans_data_filename = raw_data_filename.replace("raw.npy", "filled_NaNs.npy")
        filled_nans_data_path = os.path.join(path, filled_nans_data_filename)
        np.save(filled_nans_data_path, filled_nans_data)


        if replace:
            os.remove(raw_data_path)

        
        if show_progress:
            print(f"Processed files: {i+1}")
        

    return None




def process_CDS_aod_550nm_data(path, percentile = 0.5, interp_method = "linear"):

    """
    Process Aerosol Optical Depth at 550nm (aod_550) data located in the local aod_550 database. 
    This function reads all the raw.ny and filled_NaNs.npy aod_550nm files (files which
    were obtained via the 'get_CDS_aod_550nm_data' and 'fill_CDS_aod_550nm_data_nans' 
    functions) stored at the directory specified by 'path' and then computes multiple 
    useful quantities.
    
    Parameteres
    -----------
    path : path-str
        Path of the folder where the aud_550nm raw.ny and filled_NaNs.npy files 
        are stored. That is, the path to the local aod_550nm database.        
        
    percentile : float
        Percentile for computing the 'percentile_data' and 'percentile_data_funcs'
        dictionaries. Must be a number between 0 and 1. Default is 0.5.
        
    interp_method : {'linear', 'nearest', 'slinear', 'cubic', 'quintic'}
        The method of interpolation to perform when computing the 
        res['filled_nans_data_funcs'], res['avg_data_funcs'] and res['percentile_data_funcs']
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
                files to specify the locations at which the aod_550nm data is 
                reported.
                
            "longitude" : numpy.array of floats (mpoints,)
                Array of longitude values (in degrees) used by the raw.npy and filled_NaNs.npy 
                files to specify the locations at which the aod_550nm data is 
                reported.
                
            "raw_data" : dict
                Dictionary containing the raw data of aod_550nm values stored by the
                raw.npy files in the local aod_550nm database. It has the following 
                key-value pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : 2D numpy.array of floats (npoints, mpoints)
                        Where 'year' and 'month' are integers that specify the 
                        time period for the array of data (with units atm-cm).

            "filled_nan_data" : dict
                Dictionary containing the filled-NaN data of aod_550nm values stored by the
                filled_NaNs.npy files in the local aod_550nm database. It has the 
                following key-value pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : 2D numpy.array of floats (npoints, mpoints)
                        Where 'year' and 'month' are integers that specify the 
                        time period for the array of data (with units atm-cm).
                        
            "filled_nan_data_funcs" : dict 
                Dictionary containing the interpolating functions of the 
                filled-NaN data of aod_550nm values stored in the local 
                aod_550nm database. It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    (year, month) : scipy.interpolate.RegularGridInterpolator object
                        Where 'year' and 'month' are integers that specify the 
                        time period for the array of data (with units atm-cm)
                        that the function interpolates. It takes as input a value
                        of latitude and longitude and returns the aod_550nm value 
                        expected at that location.
              
                        
            "avg_data" : dict
                Dictionary containing the year-wise averaged data of filled-NaN
                aod_550nm values stored in the local aod_550nm database, for each
                month. It has the following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats (npoints, mpoints)
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm).
                        
            "avg_data_funcs" : dict 
                Dictionary containing the interpolating functions of the year-wise 
                averaged data of filled-NaN aod_550nm values stored in the local 
                aod_550nm database, for each month. It has the following key-value pair 
                format:
                    
                    Key : Value
                    -----------
                    month : scipy.interpolate.RegularGridInterpolator object
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm)
                        that the function interpolates. It takes as input a value
                        of latitude and longitude and returns the aod_550nm value 
                        expected at that location.
                        
            "percentile_data" : dict
                Dictionary containing the year-wise 'percentile'-th percentile
                of the filled-NaN aod_550nm data values stored in the local
                water database, for each month. It has the following key-value 
                pair format:
                    
                    Key : Value
                    -----------
                    month : 2D numpy.arrays of floats (npoints, mpoints)
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm).
                        
            "percentile_data_funcs" : dict 
                Dictionary containing the interpolating functions of the filled-NaN
                year-wise 'percentile'-th percentile data of aod_550nm
                values stored in the local aod_550nm database. It has the
                following key-value pair format:
                    
                    Key : Value
                    -----------
                    month : scipy.interpolate.RegularGridInterpolator object
                        Where 'month' is an integer that specifies the 
                        time period for the array of data (with units atm-cm)
                        that the function interpolates. It takes as input a value
                        of latitude and longitude and returns the aod_550nm value 
                        expected at that location.
                        
 
    Raises
    ------
    1) Exception :
       "Local aod_550nm database is empty. No aod_550nm raw.npy nor filled_NaNs.npy files to retrieve were found."

    2) Exception :
       "Latitude data could not be recovered. No latitude.npy files are present in the database."

    3) Exception :
       "Longitude data could not be recovered. No longitude.npy files are present in the database."

    Warns
    -----
    1) Warning:
       "Local aod_550nm database is empty of filled_NaNs.npy files. No column water filled_NaNs.npy files to retrieve were found."
    
    2) Warning:
       "Local aod_550nm database is empty of raw.npy files. No column water raw.npy files to retrieve were found."       

    3) Warning:
       "Not all raw.npy files have an associated filled_NaNs.npy file in the database, and viceversa. Therefore, not all information will be available for all files."

    4) Warning:
       "WARNING: Local aod_550nm database lacks raw.npy data files for all 12 months of the year."

    5) Warning:
       "WARNING: Local aod_550nm database lacks filled_NaNs.npy data files for all 12 months of the year."
                    
    Notes
    -----
    1) res["raw_data"] contains as many key-value pairs as there are raw.npy
      files in the local aod_550nm database.
       
    2) res["filled_nan_data"], res["filled_nan_data_funcs"], res["avg_data"],
       res["avg_data_funcs"], res["precentile_data"] and res["precentile_data_funcs"]
       as many key-value pairs as there are filled_NaNs.npy files in the local aod_550nm
       database.
    
    3) Latitude of -90° corresponds to the geographic South pole, while a 
       latitude of 90° corresponds to the geographic North Pole.
       
    4) A negative longitude correspondes to a point west of the greenwhich 
       meridian, while a positive longitude means it is east of the greenwhich 
       meridian.
       
    
    """
    
    #  1) ---- FILENAMES RETRIEVAL ----

    # We get all the raw.npy and filled_NaNs.npy filenames stored at the local aod_550nm database.
    raw_data_filenames    = [i for i in os.listdir(path) if "raw.npy" in i]
    filled_nans_filenames = [i for i in os.listdir(path) if "filled_NaNs.npy" in i]

    # We attempt to get latitude and longitude .npy files
    latitude_file_names  = [i for i in os.listdir(path) if "latitude.npy" in i]
    longitude_file_names = [i for i in os.listdir(path) if "longitude.npy" in i]
    

    # 2) ---- CHECK PRESENCE AND ABSENCE OF FILES ----

    # We check if the local aod_550nm database is empty of "raw.npy" files   
    raw_data_filenames_absent = False    
    if len(raw_data_filenames) == 0: raw_data_filenames_absent = True
    
    # We check if local aod_550nm database is empty of "filled_NaNs.npy" files.
    filled_nans_filenames_absent = False
    if len(filled_nans_filenames) == 0: filled_nans_filenames_absent = True

    # Depending on the case, we raise warning or an exception.
    if raw_data_filenames_absent and filled_nans_filenames_absent:
        msg = "Local aod_550nm database is empty."
        msg = f"{msg}. No aod_550nm raw.npy nor filled_NaNs.npy files to retrieve were found."
        raise Exception(msg)
    
    elif raw_data_filenames_absent:
        msg = "Local aod_550nm database is empty of filled_NaNs.npy files"
        msg = f"{msg}. No column water filled_NaNs.npy files to retrieve were found."
        warnings.warn(msg)
    
    elif filled_nans_filenames_absent:
        msg = "Local aod_550nm database is empty of raw.npy files."
        msg = f"{msg}. No column water raw.npy files to retrieve were found."
        warnings.warn(msg)


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
    

    # 4) ---- CHECK WHETHER EACH RAW.NPY FILE HAS AN ASSOCIATED FILLED_NANS.NPY FILE ---
    check_list_raw = [i.replace("_raw.npy", "") for i in raw_data_filenames]
    check_list_filled_nans = [i.replace("_filled_NaNs.npy", "") for i in filled_nans_filenames] 

    if set(check_list_raw) != set(check_list_filled_nans):
        msg = "Not all raw.npy files have an associated filled_NaNs.npy file in the database, and viceversa"
        msg = f"{msg}. Therefore, not all information will be available for all files."
        warnings.warn(msg)



    # 5) ---- RETRIEVE RAW DATA FROM RAW.NPY FILES ---

    years, months = set(), set()
    # We read all raw.npy files and store everything into a user-friendlier format.

    raw_data = {}
    for raw_data_filename in raw_data_filenames:
        
        year_month = raw_data_filename.split("-")[0]
        year  = int(year_month[:4])
        month = int(year_month[4:])
        
        years.add(year)
        months.add(month)

        raw_data_file_path = os.path.join(path, raw_data_filename)
        raw_data[(year, month)] = np.load(raw_data_file_path)
                
        
    # If not all 12 months of the year are present in the local aod_550nm
    # database (with respect to the raw.npy files) we throw a warning.
    if len(months) < 12 and not raw_data_filenames_absent:
        message = "WARNING: Local aod_550nm database lacks raw.npy data files"
        message = f"{message} for all 12 months of the year."
        warnings.warn(message)
        

    # 6) --- RETRIEVE FILLED_NANS.NPY DATA AND CREATE INTERPOLATION FUNCS ---
    
    years, months = set(), set()
    # We read all the filled_NaNs.npy files and store everything into a 
    # user-friendlier format. We also initialize the interpolating functions
    # and store them.
    
    filled_nans_data = {}
    filled_nans_data_funcs = {}
    for filled_nans_filename in filled_nans_filenames:
        
        year_month = filled_nans_filename.split("-")[0]
        year  = int(year_month[:4])
        month = int(year_month[4:])
        
        years.add(year)
        months.add(month)
        
        filled_nans_file_path = os.path.join(path, filled_nans_filename)
        
        # Actual reading of filled_NaNs.npy files
        filled_nans_data[(year, month)] = np.load(filled_nans_file_path) 
        
        filled_nans_data_funcs[(year, month)] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = filled_nans_data[(year, month)],
                                method = interp_method)
        
        
   

    # If not all 12 months of the year are present in the local aod_550nm
    # database (with respect to the filled_NaNs.npy files) we throw a warning.
    if len(months) < 12 and not filled_nans_filenames_absent:
        message = "WARNING: Local aod_550nm database lacks filled_NaNs.npy "
        message = f"{message} data files for all 12 months of the year."
        warnings.warn(message)
        
        
        
    #  7) ---- AVG DATA COMPUTATION AND INTERPOLATION ----    
        
    # We compute the year-averaged data and interpolating functions.
    avg_data = {}
    avg_data_funcs = {}
    for month in months:
        
        avg_data[month] = []
        for year in years:
            try: avg_data[month].append(filled_nans_data[(year, month)])
            except KeyError: pass

        avg_data[month] = np.stack(avg_data[month], axis=2) 
        avg_data[month] = avg_data[month].mean(axis=2)
        
        avg_data_funcs[month] =\
        RegularGridInterpolator(points = (lat, lon), 
                                values = avg_data[month],
                                method = interp_method)
        
        
        
    #  8) ---- PERCENTILE DATA COMPUTATION AND INTERPOLATION ---- 
        
    # We compute the year-wise 'perecentile'-th percentile data and 
    # interpolating functions.      
    percentile_data = {}
    percentile_data_funcs = {}
    for month in months:
        
        percentile_data[month] = []
        for year in years:
            try: percentile_data[month].append(filled_nans_data[(year, month)])
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
    "filled_nans_data" : filled_nans_data,
    "filled_nans_data_funcs" : filled_nans_data_funcs,
    "avg_data" : avg_data,
    "avg_data_funcs" : avg_data_funcs,
    "percentile_data" : percentile_data,
    "percentile_data_funcs" : percentile_data_funcs }
        
        
    return res








