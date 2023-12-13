#%%                MODULE DESCRIPTION AND/OR INFO

"""
Simple module for containing auxiliary functions that are commonly used in 
other modules of the folder Ambience_Modelling or even other folders.
  
"""
#%%             IMPORTATION OF LIBRARIES
import pickle
import warnings
import numpy as np

#%%             DEFINITION OF FUNCTIONS
 
def save_obj_with_pickle(class_obj, path):
    """
    Save any Class instance using pickle.

    Parameters
    ----------
    class_obj : Class obj.
        Class instance/object to save using pickle.
        
    path : path-str
        Path of the .pkl file corresponding to Class obj that is to be saved.

    Returns
    -------
    None.

    """
    
    with open(path, 'wb') as outp:
        pickle.dump(class_obj, outp, pickle.HIGHEST_PROTOCOL)  
        
    return None



def load_obj_with_pickle(path):
    
    """
    Load any Class instance saved with pickle.

    Parameters
    ----------        
    path : path-str
        Path of the .pkl file corresponding to Class obj that is to be loaded.

    Returns
    -------
    class_obj : Class obj.
        Class instance/object loaded using pickle.

    """
    
    with open(path, 'rb') as inp:
        class_obj = pickle.load(inp)
        
    return class_obj



def fill_nans_using_laplace_1D(arr, iterations=500):
    
    """
    Fill NaN values of a flat numpy.array using the average value of its
    non-NaN neighbours. This procedure is iterative and, as such, the
    function performs the averaging until the specified number of iterations 
    is reached.
    
    arr : numpy.array of floats
        Array of scalar data with NaN values sprinkled throughout.
    
    iterations : int
        Number of iterations that the code should perform before stopping (must
        be non-negative). The greater the number of iterations, the greater 
        the chance that convergence has been reached. However, the time of
        computation also increases. Default is 500.
        
    Returns
    -------
    filled_nans_arr : numpy.array of floats
        Array of scalar data with NaN values having been filled using the 
        the average of their non-NaN neighbours.
    
    
    Notes
    -----
    1) This procedure is very similar to discretely solving the 1D version of
       laplace's equation. In this case, the domain of solution are the 
       NaN filled elements, while the boundary conditions are given by the
       neighbouring non-NaN values. The boundary conditions here would be, more 
       or less, zero outward-flux at the endpoints. Although, to be honest, the 
       code takes inspiration from the laplace procedure and is not too
       mathematically rigorous.
    
    """
    
    # We store the indices of the NaN values of arr.
    len_arr = len(arr)
    filled_nans_arr = arr.copy()
    nan_idxs = np.arange(len_arr)[np.isnan(filled_nans_arr)]
    
    # We set all NaN values of the the array equal to zero.
    filled_nans_arr[nan_idxs] = 0
    
    # We define the indices for computing the neighbour averages of the 
    # Ex-NaN values.
    nan_idxs_left  = nan_idxs - 1
    nan_idxs_right = nan_idxs + 1
    
    # We take care of the end points.
    nan_idxs_left[nan_idxs_left < 0] = 0
    nan_idxs_right[nan_idxs_right > len_arr - 1] = len_arr - 1
    
    # We perform the procedure iteratively.
    for _ in range(iterations):
        filled_nans_arr[nan_idxs] =\
        0.5*(filled_nans_arr[nan_idxs_left] + filled_nans_arr[nan_idxs_right])
    

    return filled_nans_arr



def fill_CDS_globe_nans_using_laplace(data, iterations = 20000):
    
    """
    Depending on the specific database used, some files downloaded from the 
    Climate Data Store (CDS) may contain missing values. This is a problem
    as the NaN values complicate the computation of other relevant quantities,
    as well as the easy interpolation of the desired dataset via scipy's  
    RegularGridInterpolator function, which is something crucial for 
    later on procedures. This function attempts to solve this problem
    by filling the missing values with the average of their neighbours,
    iteratively. The process, broadly speaking, is:
        
        1) Store the indices of all nan values of the input array.
        2) Make a copy of the array.
        3) Fill all nan values with zeros in the copied array.
        4) Loop over all the elements of the copied array that used to be NaN
           values and set the value of each element equal to the average of the
           values of its neighbours. Do this as many times as required to
           achieve the desired level of convergence.
           
    Some specific, yet important, things to note are:
        
        1) Updated values of an element x during iteration i should not be
           accesible to other elements until iteration i is finished. That is, 
           the new/updated value of element x should not be used in the 
           computation of the new value of element y, until the next iteration;
           as such, during the same iteration, element y's new value is computed
           using the un-updated value of element x.
           
        2) This code is intended to only be used on arrays which store a scalar
           quantity over the whole globe. This is because the boundary
           conditions used here are that of the surface of a sphere (see
           the 'Notes' section.).

    
    Parameters
    ----------
    data : 2D numpy.array of floats
        Array of scalar data with NaN values sprinkled throughout. The array
        should contain values corresponding to the whole globe, with the
        axis 0, accounting for the variation of said values with respect to the
        latitude, while the axis 1 accounts for the variation with respect to
        the longitude. Axes 0 and 1 must have the same constant spacing, meaning
        that axis 1 should be twice the length of axis 0. That is, 'data' must
        be defined over an equally-spaced regular rectangular grid of 
        latitude vs longitude. Finally, regarding the coordinate system: let 
        'data' be a Nx2N numpy.array of floats. Then data[0,:] is the
        circle of constant latitude equal to -90° (i.e, the geographic south 
        pole), data[-1,:] is the circle of constant latitude equal to 90° 
        (i.e, the geographic north pole), data[:,0] is the arc of constant 
        longitude equal to -180° and data[:,-1] is the arc of constant
        longitude equal to 180°.
        
    
    iterations: int
        Number of iterations that the code should perform before stopping (must
        be non-negative). The greater the number of iterations, the greater 
        the chance that convergence has been reached. However, the time of
        computation also increases. Default is 20000.
        
        
    Returns   
    -------
    new_data : 2D numpy.array of floats
         Array of scalar data with the NaN values having been filled with 
         numerical values based on the average value of their non-NaN 
         neighbours, as per the procedure explained above.

    Warns
    -----
    1) Warning :
       "WARNING: Length of axis 1 is not equal to 2 times the length
        of axis 0. This function requires for the data to
        be equally spaced and encompass the whole earth.
        That is only possible if data.shape[1] = 2*data.shape[0]. 
        If these conditions are not satisfied, results may be incorrect or misleading."

        
    Notes
    -----
    
    1) This function is equivalent to discretely solving laplace's equation on 
       the surface of a sphere. In this case, the domain of solution are the 
       NaN filled elements, while the boundary conditions are given by all
       the non-NaN values.
      
    2) Since we are operating on the surface of a sphere, two additional
       particular boundary conditions should be satisfied:
           
           a) The values of the array should 'wrap' along the longitudinal 
              direction. That is, for an infinitely fine mesh: data[i,0] == 
              data[i,-1], for all i.
              
           b) The values of the array at the each geographic pole should be 
              the same for all longitudes. That is, for an infinitely fine mesh: 
              data[0,:] and data[-1,:] are constant arrays.
              
       However, for finitely fine meshes we implement these conditions slightly
       differently. The way it is done is on how we compute the averages. Namely:
       data[i,j] = 0.25*( data[i-1,j] + data[i+1,j] + data[i,j-1] +
       data[i,j+1] ), for most cases. But when:
           
           1) j =  0,  data[i,j-1] equals data[i, -1]
           2) j = -1,  data[i,j+1] equals data[i, 0]
           3) i = 0,   data[i-1,j] equals data[0, bcj]
           4) i = -1,  data[i+1,j] equals data[-1, bcj]
    
    Where bcj is and index such that lon[bcj] == lon[j] + 180,  if lon[j] < 0 and 
    lon[bcj] == lon[j] - 180,  if lon[j] >= 0. Where
    lon = numpy.linspace(-180, 180, data.shape[1])
        
    """

    # We identify wich elements of data are Nans.
    mask = np.isnan(data)

    # We make a copy of data and set all NaN values in this copy equal to 0.
    new_data = data.copy()
    new_data[mask] = 0

    # We compute the index positions of all NaN values in data.
    j_grid, i_grid =\
    np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

    nans_i = i_grid[mask]
    nans_j = j_grid[mask]
    
    
    # We check the shape of the array and warn the user about the limitations
    # of the function when dealing with unequally-spaced grids.    
    lat_len, lon_len = data.shape[0], data.shape[1]
    
    if lon_len != 2*lat_len:
        msg = "WARNING: Length of axis 1 is not equal to 2 times the length"
        msg = f"{msg} of axis 0. This function requires for the data to"
        msg = f"{msg} be equally spaced and encompass the whole earth"
        msg = f"{msg}. That is only possible if data.shape[1] ="
        msg = f"{msg} 2*data.shape[0]"
        msg = f"{msg}. If these conditions are not satisfied, results"
        msg = f"{msg} may be incorrect or misleading."
        warnings.warn(msg)
        
        
    
    # jp is a float when lon_len is even. As such, it is better to
    # average the values of data, specified by: int(bcj) and int(bcj) + 1.    
    if lon_len % 2 == 0:
        left_weight  = 0.5
        right_weight = 0.5
    else: 
        left_weight  = 1
        right_weight = 0
        
        
    # ---COMPUTATION OF BCJS ARRAY ---

    # We compute an array that gives the bcj of each j.
    lon = np.linspace(-180, 180, lon_len)
    
    CT = (lon_len - 1)/360
    bcjs = np.zeros(lon_len)
    bcjs[lon < 0]  = CT*(lon[lon < 0] + 360)
    bcjs[lon >= 0] = CT*lon[lon >= 0]
  
    bcjs_left =  bcjs.astype(int)
    bcjs_right = bcjs_left + 1
    bcjs_right[bcjs_right > lon_len - 1] = 0


    # ---COMPUTATION OF AVERAGING INDICES ARRAY ---
    # Here is where we also implement the particular boundary conditions 
    # related to a sphere.
    
    # Computation of indices for data[i, j-1], for all i in nan_i and j in nans_j.
    val_minus_j_idxs = [nans_i.copy(), nans_j.copy() - 1]
    
    
    # Computation of indices for data[i, j+1], for all i in nan_i and j in nans_j.
    val_plus_j_idxs  = [nans_i.copy(), nans_j.copy() + 1]
    val_plus_j_idxs[1][nans_j + 1 > lon_len - 1] = 0
    
    
    # Computation of indices for data[i-1, j], for all i in nan_i and j in nans_j.
    val_minus_i_idxs_left = [nans_i.copy() - 1, nans_j.copy()]
    val_minus_i_idxs_left[0][nans_i - 1 < 0] = 0
    val_minus_i_idxs_left[1][nans_i - 1 < 0] = bcjs_left[nans_j[nans_i - 1 < 0]]
    
    val_minus_i_idxs_right = [nans_i.copy() - 1, nans_j.copy()]
    val_minus_i_idxs_right[0][nans_i - 1 < 0] = 0
    val_minus_i_idxs_right[1][nans_i - 1 < 0] = bcjs_right[nans_j[nans_i - 1 < 0]]
    
    
    # Computation of indices for data[i+1, j], for all i in nan_i and j in nans_j.
    val_plus_i_idxs_left = [nans_i.copy() + 1, nans_j.copy()]
    val_plus_i_idxs_left[0][nans_i + 1 > lat_len - 1] = lat_len - 1 
    val_plus_i_idxs_left[1][nans_i + 1 > lat_len - 1] = bcjs_left[nans_j[nans_i + 1 > lat_len - 1]]
    
    val_plus_i_idxs_right = [nans_i.copy() + 1, nans_j.copy()]
    val_plus_i_idxs_right[0][nans_i + 1 > lat_len - 1] = lat_len - 1
    val_plus_i_idxs_right[1][nans_i + 1 > lat_len - 1] = bcjs_right[nans_j[nans_i + 1 > lat_len - 1]]
    
    
    

    # ----- COMPUTATION OF NEIGHBOUR AVERAGES FOR ALL NaN VALUES -----
    for k in range(iterations):
        
        val_minus_j = new_data[val_minus_j_idxs[0], 
                               val_minus_j_idxs[1]]
        
        val_plus_j  = new_data[val_plus_j_idxs[0], 
                               val_plus_j_idxs[1]]
        
        val_minus_i  = left_weight * new_data[val_minus_i_idxs_left[0], 
                                              val_minus_i_idxs_left[1]]
        
        val_minus_i += right_weight * new_data[val_minus_i_idxs_right[0],
                                               val_minus_i_idxs_right[1]]
    
        val_plus_i  = left_weight * new_data[val_plus_i_idxs_left[0], 
                                             val_plus_i_idxs_left[1]]
        
        val_plus_i += right_weight * new_data[val_plus_i_idxs_right[0],
                                              val_plus_i_idxs_right[1]]
    
        
        new_data[nans_i, nans_j] =\
        0.25*(val_minus_i + val_minus_j + val_plus_i + val_plus_j)
        
        
    return new_data
