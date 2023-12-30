#%%                MODULE DESCRIPTION AND/OR INFO

"""
This module contains all functions, methods and classes related to the
modelling of radiation coming from the sky.
"""


#%%                   IMPORTATION OF LIBRARIES

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import simpson
from solrad.auxiliary_funcs import load_obj_with_pickle
from scipy.interpolate import RegularGridInterpolator
from solrad.radiation.spectral_radiance_model import compute_spectral_radiance



#%%                     DEFINITION OF CLASSES


class Sky:

    # 1) --- INITIALIZATION FUNCTIONS ---   
    
    def __init__(self, Site_obj, num_divisions = 400):
        
        """
        Class for spectrally and spatially modelling the radiation coming
        from the sky.
        
        Parameters
        ----------
        Site_obj : Site object
            Instance of class 'Site' from module 'Site', whose 
            self.climate_and_air_data, self.sun_data, self.single_scattering_albedo
            and self.aerosol_asymmetry_factor have already been computed/filled with data.

        num_divisions : int, optional
            Number of patches into which the Sky Vault should be discretised.
            Default is 400.
                    
        """
        
        self.Site_obj = Site_obj
        self._check_Site_obj_for_NaNs_and_warn()
        self.discretise(num_divisions)

    
        # Compute vectorized versions of sky-patch localization
        # functions using coordinates.
        self.sky_points_to_zones_patches =\
        np.vectorize(self.sky_point_to_zone_patch)
        
        self.disk_points_to_zones_patches =\
        np.vectorize(self.disk_point_to_zone_patch)

        return None
    
    def _check_Site_obj_for_NaNs_and_warn(self):
        """
        Private helper function. It searches each relevant attribute 
        of Site_obj and warns wether it has NaN values.
        """
        sun_data = []
        climate_and_air_data = []
        single_scattering_albedo = []
        aerosol_asymmetry_factor = []
    
        for sun_data_df in self.Site_obj.sun_data.values():
            sun_data.append(pd.isnull(sun_data_df).any().any())

        for climate_and_air_data_df in self.Site_obj.climate_and_air_data.values():
            climate_and_air_data.append(pd.isnull(climate_and_air_data_df).any().any())

        for single_scattering_albedo_df in self.Site_obj.single_scattering_albedo.values():
            single_scattering_albedo.append(pd.isnull(single_scattering_albedo_df).any().any())

        for aerosol_asymmetry_factor_df in self.Site_obj.aerosol_asymmetry_factor.values():
            aerosol_asymmetry_factor.append(pd.isnull(aerosol_asymmetry_factor_df).any().any())

        list_of_vals = [(sun_data, "sun_data"), 
                        (climate_and_air_data, "climate_and_air_data"),
                        (single_scattering_albedo, "single_scattering_albedo"),
                        (aerosol_asymmetry_factor, "aerosol_asymmetry_factor")]
        
        for val, name in list_of_vals:
            if any(val):
                msg = f"NaN/None values were detected in Site_obj.{name}"
                msg = f"{msg}. This may raise exceptions or incur strange behaviour during subsequent computations."
                warnings.warn(msg)

        return None
        
        
    # 2) --- SKY DISCRETISATION RELATED FUNCTIONS --- 

    def discretise(self, num_divisions):
        
        """
        Discretise the Sky Vault into non-congruent square-like patches of
        similar area, according to the procedure proposed in the paper "A 
        general rule for disk and hemisphere partition into equal-area cells"
        (see reference [1]). 
        
        Parameters
        ----------
        num_divisions : int
            Number of patches into which the Sky Vault will be discretised.
            
        Returns
        -------
        None
        
        Produces
        --------
        self.zone_data : dict of dicts
            Dictionary containing information about the discretization zones. 
            Each key of *self.zone_data* corresponds to a unique zone number.
            The component dictionaries (stored at *self.zone_data[zone_num]*)
            have the following Key-Value pairs:
                
                Keys : Values
                -------------
                "num_patches" : int
                    Number of sky patches contained inside the sky zone.
                    
                "inf_zen" : float
                    Inferior zenith angle bound delimiting the sky zone,
                    in degrees.
                    
                "sup_zen" : float
                    Superior zenith angle bound delimiting the sky zone,
                    in degrees.
                    
                "inf_rad" : float
                    Inferior radius bound delimiting the sky zone's plane 
                    projection [adm].
                    
                "sup_rad" : float
                    Superior radius bound delimiting the sky zone's plane 
                    projection [adm].
                    
                "azimuths" : numpy.array of floats
                    Array containing the azimuth angle intervals delimiting
                    each sky patch inside the zone, in degrees.
                    
                "patch_area" : float
                    Solid angle/area, taken up by each sky patch inside the
                    sky zone, in steradians.
                    
                "zone_area" : float
                    Total solid angle/area of the whole sky zone, in steradians.
                    
                
        self.patch_data : dict of dicts
            Dictionary containing information about the discretization patches. 
            Each key of *self.patch_data* is a 2-tuple of ints corresponding to 
            the patch (zone number, local patch number). The component dictionaries
            (stored at *self.patch_data[(zone_num, local_patch_num)]*) have the
            following Key-Value pairs:
                
                Keys : Values
                -------------
                "inf_zen" : float
                    Inferior zenith angle bound delimiting the sky patch,
                    in degrees.
                    
                "sup_zen" : float
                    Superior zenith angle bound delimiting the sky patch,
                    in degrees.
                    
                "inf_az" : float
                    Inferior azimuth angle bound delimiting the sky patch,
                    in degrees.
                    
                "sup_az" : float
                    Superior azimuth angle bound delimiting the sky patch,
                    in degrees.
                    
                "patch_area" : float
                    Solid angle/area, taken up by the sky patch, in steradians.
                    
                "unit_vector" : np.array of floats with shape (3,)
                    Unit solid angle vector of the center of the sky patch.
                    It is basically a unit vector with tail at the origin and 
                    which points to the center position of the sky patch.
                    "unit_vector"[i], with i = 0,1,2; gives the unit vector's
                    x,y,z component respecitvely.

        Notes
        -----
        Doing calling this method after initialization will also errase all other radiation quantities
        computed up to that point.

        References
        ----------
        [1] Benoit Beckers, Pierre Beckers, A general rule for disk and hemisphere partition into equal-area cells,
        Computational Geometry, Volume 45, Issue 7, 2012, Pages 275-283, ISSN 0925-7721, https://doi.org/10.1016/j.comgeo.2012.01.011.
        (https://www.sciencedirect.com/science/article/pii/S0925772112000296)

                    
        """
        
        # ----- SKY VAULT DISCRETIZATION --------
        # Compute radius and zenith defining each sky zone, as well as the number
        # of elements present within each concentric circle/ring.

        rad = [1]
        zenith = [np.pi/2]
        num_elem_in_rad = [num_divisions]

        i = 0
        while num_elem_in_rad[-1]>1:

            zenith.append\
            (zenith[i] - np.sqrt(2)*np.sin(zenith[i]/2)*np.sqrt(np.pi/num_elem_in_rad[i]))

            rad.append\
            (np.sqrt(2)*np.sin(zenith[i+1]/2))

            num_elem_in_rad.append\
            (round(num_elem_in_rad[i]*(rad[i+1]/rad[i])**2))

            i += 1
        
        
        # If the algorithm tells us that there are zero elements within the
        # innermost ring, we have to adjust the data of the last element 
        # to reflect that. 
        if num_elem_in_rad[-1] == 0:
            rad[-1] = 0
            zenith[-1] = 0
            i -= 1
            
        # If the algorithm tells us that there is 1 element within the
        # innermost ring, we have to add the missing data.
        elif num_elem_in_rad[-1] == 1:
            rad += [0]
            zenith += [0]
            num_elem_in_rad += [0]
            
            
        # We sort from min to max (i.e, reverse the lists, in this case).
        rad = np.array(rad[::-1])
        zenith = np.array(zenith[::-1])
        num_elem_in_rad = np.array(num_elem_in_rad[::-1])
        
        
        # We initialize relevant attributes.
        self.zone_data  = {}
        self.patch_data = {}
        self.zone_max_key = i
        self.num_divisions = num_divisions
        
        
       # ------- CREATION OF DATABASE FOR ZONE AND PATCH INFO --------
       
        # We store all relevant information in 2 custom databases. one
        # for zone infor and another for patch info. 
        for zone_num in range(self.zone_max_key + 1):
            
            num_patches =\
            num_elem_in_rad[zone_num + 1] - num_elem_in_rad[zone_num]
            
            self.zone_data[zone_num] =\
            { "num_patches" : num_patches,
              "inf_zen"     : np.rad2deg(zenith[zone_num]),
              "sup_zen"     : np.rad2deg(zenith[zone_num + 1]),
              "inf_rad"     : rad[zone_num],
              "sup_rad"     : rad[zone_num + 1],
              "azimuths"    : np.rad2deg(np.linspace(0, 2*np.pi, num_patches + 1)),
              "patch_area"  : (2*np.pi/num_patches)*(np.cos(zenith[zone_num])-np.cos(zenith[zone_num + 1])),
              "zone_area"   : 2*np.pi*(np.cos(zenith[zone_num])-np.cos(zenith[zone_num + 1]))
              
            }
            
            # Compute mean zenith angle for each zone in radians.
            mid_theta  = zenith[zone_num]
            mid_theta += zenith[zone_num + 1]
            mid_theta  = mid_theta/2
            
            for local_patch_num in range(num_patches):
                
                # Compute mean azimuth angle for each sky patch in radians.
                mid_az  = self.zone_data[zone_num]["azimuths"][local_patch_num]
                mid_az += self.zone_data[zone_num]["azimuths"][local_patch_num + 1]
                mid_az  = np.deg2rad(mid_az/2)
                
                
                self.patch_data[(zone_num, local_patch_num)] =\
                { "inf_zen"     : np.rad2deg(zenith[zone_num]),
                  "sup_zen"     : np.rad2deg(zenith[zone_num + 1]),
                  "inf_az"      : self.zone_data[zone_num]["azimuths"][local_patch_num],
                  "sup_az"      : self.zone_data[zone_num]["azimuths"][local_patch_num + 1],
                  "patch_area"  : self.zone_data[zone_num]["patch_area"],
                  "unit_vector" : np.array([np.cos(mid_az)*np.sin(mid_theta),
                                            np.sin(mid_az)*np.sin(mid_theta),
                                            np.cos(mid_theta)])

                }
            
        self.radainces = None
        self.exposure_vectors = None
        self.time_integrated_spectral_radiance = None
        return None   
    
    
    
    
    # -- METHODS FOR ZONE-PATCH LOCALIZATION FROM COORDINATES: SPHERE --

    def sky_point_to_zone_patch(self, zen, az):
        
        """
        Bin sky point into the correct sky patch. That is, given a sky point
        represented by a tuple of (zenith, azimuth) values, return the sky patch, 
        represented by a tuple of (zone_num, local_patch_num), to which said
        sky point belongs.
        
        Parameters
        ----------
        zen : float
            Zenith of sky point in degrees. Must be between 0 and 90.
        
        az : float
            Azimuth of sky point in degrees. Must be between 0 and 360.
            
        Returns
        -------
        zone_num : int or str
            Sky zone (int) to which the sky point belongs, or "not found" if
            search failed.
        
        local_patch_num : int or str
            Sky patch (int) (identified by its local patch number in reference 
            to the sky zone) to which the sky point belongs, or "not found" if 
            search failed.
        
        
        """
        
        zone_num = self.zenith_to_zone(zen)
        local_patch_num = self.azimuth_to_patch(zone_num, az)
        return zone_num, local_patch_num



    def zenith_to_zone(self, zen, start=None, end=None):
        
        """
        Bin zenith value into the correct sky zone via binary search.
        
        Parameters
        ----------
        zen : float
            Zenith value in degrees. Must be between 0 and 90.
        
        start : int or None
            Lower search bound for zone. If None, it defaults to the lowest
            bound possible.
            
        end : int or None
            Upper search bound for zone. If None, it defaults to the highest
            bound possible.
            
        Returns
        -------
        zone_num : int or str
            Sky zone (int) to which the zenith coordinate belongs, or "not found"
            if search failed.
        
        """

        if(start is None): start = 0
        if(end is None): end = self.zone_max_key

        if(start > end):
            return "Not found"

        zone_num = int((start + end)/2)
        
        inf_zen = self.zone_data[zone_num]["inf_zen"]
        sup_zen = self.zone_data[zone_num]["sup_zen"]

        if(zen <= sup_zen):

            if(zen >= inf_zen):
                return zone_num

            else:
                return self.zenith_to_zone(zen, start, zone_num-1)

        else:
            return self.zenith_to_zone(zen, zone_num+1, end)
        
        
        
        
    def azimuth_to_patch(self, zone_num, az, start=None, end=None):
        
        """
        Bin azimuth value into the correct sky patch via binary search.
        
        Parameters
        ----------
        zone_num : int
            Sky zone to which the azimuth value "belongs".
        
        az : float
            Azimuth value in degrees. Must be between 0 and 360.
        
        start : int or None
            Lower search bound for patch. If None, it defaults to the lowest
            bound possible.
            
        end : int or None
            Upper search bound for patch. If None, it defaults to the highest
            bound possible.
            
        Returns
        -------
        local_patch_num : int
            Sky patch (int) to which the zenith coordinate belongs, or "not found"
            if search failed.
        
        """

        if(start is None): 
            start = 0
        
        if(end is None):
            end = self.zone_data[zone_num]["num_patches"] - 1
            

        if(start > end):
            return "Not found"

        local_patch_num = int((start + end)/2)
        inf_az = self.patch_data[(zone_num, local_patch_num)]["inf_az"]
        sup_az = self.patch_data[(zone_num, local_patch_num)]["sup_az"]
        
        if(az <= sup_az):

            if(az >= inf_az):
                return local_patch_num

            else:
                return self.azimuth_to_patch(zone_num, az, start, local_patch_num-1)

        else:
            return self.azimuth_to_patch(zone_num, az, local_patch_num+1, end)
        
        
        
    # -- METHODS FOR ZONE-PATH LOCALIZATION FROM COORDINATES: DISK --      
        
    def disk_point_to_zone_patch(self, rad, az):
        
        """
        Bin disk point into the correct sky patch. That is, given a disk point
        represented by a tuple of (radius, azimuth) values, return the sky patch, 
        represented by a tuple of (zone_num, local_patch_num), to which said
        disk point belongs.
        
        Parameters
        ----------
        rad : float
            Radius of disk point [adm]. Must be between 0 and 1.
        
        az : float
            Azimuth of disk point in degrees. Must be between 0 and 360.
            
        Returns
        -------
        zone_num : int
            Sky zone to which the disk point belongs.
        
        local_patch_num : int
            Sky patch (int) (identified by its local patch number in reference
            to the sky zone) to which the disk point belongs, or "not found"
            if search failed.
        
        
        """
        
        zone_num = self.rad_to_zone(rad)
        local_patch_num = self.azimuth_to_patch(zone_num, az)
        return zone_num, local_patch_num
        
        
    def rad_to_zone(self, rad, start=None, end=None):
        
        """
        Bin radius value into the correct sky zone via binary search.
        
        Parameters
        ----------
        rad : float
            radius value [adm]. Must be between 0 and 1.
        
        start : int or None
            Lower search bound for zone. If None, it defaults to the lowest
            bound possible.
            
        end : int or None
            Upper search bound for zone. If None, it defaults to the highest
            bound possible.
            
        Returns
        -------
        zone_num : int
            Sky zone (int) to which the radius coordinate belongs, or "not found"
            if search failed.
        
        """

        if(start is None): start = 0
        if(end is None): end = self.zone_max_key
        

        if(start > end):
            return "Not found"

        zone_num = int((start + end)/2)
        
        inf_rad = self.zone_data[zone_num]["inf_rad"]
        sup_rad = self.zone_data[zone_num]["sup_rad"]

        if(rad <= sup_rad):

            if(rad >= inf_rad):
                return zone_num

            else:
                return self.rad_to_zone(rad, start, zone_num-1)

        else:
            return self.rad_to_zone(rad, zone_num+1, end)
        
    
    # -- METHODS FOR VISUALIZATION OF DISCRETISED SKY VAULT: DISK --
    
    def plot_disk_patches(self, figsize=(12,12)):
        
        """
        Visualize discretized Sky Vault in 2D.
        
        Paramters
        ---------
        figsize : 2-tuple of int
            Size of figure.

        Notes
        -----
        1) This method requires that the sky-vault already be discretised
        to be calculated. Check out :meth:`~solrad.Sky.Sky.discretise`
    
        """

        _, ax = plt.subplots(figsize=figsize, 
        subplot_kw={'projection': 'polar'})
        kwargs = {"edgecolor":"k",  "facecolor":"white"}

        for zone_num, zone_dict in self.zone_data.items():
            
            r0 = zone_dict["inf_rad"]
            r1 = zone_dict["sup_rad"]

            for i in range(zone_dict["num_patches"]):

                theta0 = np.deg2rad(zone_dict["azimuths"][i])
                theta1 = np.deg2rad(zone_dict["azimuths"][i+1])

                ax.bar(x = 0.5*(theta0 + theta1),
                       height = r1 - r0, 
                       width = theta1 - theta0, 
                       bottom = r0, 
                       **kwargs)
                
                ax.set_rlim(0, 1)
                ax.grid(False)
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_title("Discretised Sky Vault : 2D Visualization | N = 0°, E = 90°, S = 180°, W = 270°")

        plt.show()

        return None
    
    # -- METHODS FOR VISUALIZATION OF DISCRETISED SKY VAULT: SPHERE -- 
    
    
    def _compute_sphere_patch_lines(self, phis, thetas, n):
        """
        Private helper function. Computes the borders of a sky patch for 
        plotting.

        Parameters
        ----------
        phis : array-like
            Array-like of length 2 containing the lower and upper limits of the 
            range of phi (zenith angle) values for the patch, in radians.
            
        thetas : array-like
            Array-like of length 2 containing the lower and upper limits of the
            range of theta values for the patch, in radians.
            
        n : int
            Number of points to use for drawing the border of the patch.
            

        Returns
        -------
        lines : dict
            The function returns a dictionary lines with four keys, each 
            representing one of the borders of the patch. The values for each 
            key are dictionaries containing the x, y, and z coordinates of the
            points defining that border. 

        """

        lines = {}
        phi_arr   = np.linspace(phis[0], phis[1], n)
        theta_arr = np.linspace(thetas[0], thetas[1], n)

        for i in [0, 1]:

            lines[i] = {"x" : np.cos(phis[i])*np.sin(theta_arr), 
                        "y" : np.sin(phis[i])*np.sin(theta_arr), 
                        "z" : np.cos(theta_arr)}

            lines[i+2] = {"x" : np.cos(phi_arr)*np.sin(thetas[i]), 
                          "y" : np.sin(phi_arr)*np.sin(thetas[i]), 
                          "z" : np.cos(thetas[i])}

        return lines
    
    
    def plot_sphere_patches(self, figsize=(12,12), axis_view=(25, 30)):
        
        """
        Visualize discretized Sky Vault in 3D.
        
        Paramters
        ---------
        figsize : 2-tuple of int
            Size of figure.
            
        axis_view = 2-tuple of int
            Plot's elevation, azimuth in degrees.

        Notes
        -----
        1) This method requires that the sky-vault already be discretised
        to be calculated. Check out :meth:`~solrad.Sky.Sky.discretise`
    
        """
            
        n = 10        
        _ = plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")
 
        el, az = axis_view
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(el, az)

        for zone_num, zone_dict in self.zone_data.items(): 
            
            thetas = np.deg2rad([zone_dict["inf_zen"], zone_dict["sup_zen"]])

            for i in range(zone_dict["num_patches"]):

                phis = np.deg2rad(zone_dict["azimuths"][i:i+2])

                lines = self._compute_sphere_patch_lines(phis, thetas, n)

                for line in lines.values():
                    ax.plot(line["x"], line["y"], line["z"], color="k")
                    
        ax.set_title("Discretised Sky Vault : 3D Visualization")     
        ax.set_xlabel("X (↑ == N, ↓ == S)")
        ax.set_ylabel("Y (↑ == E, ↓ == W)")
        plt.show()

        return None
    

     
    # 3) --- FUNCTIONS RELATED TO THE COMPUTATION OF RADIOMETRIC QUANTITIES --- 

    # 3.1) -- COMPUTE RADIANCES --
    def compute_spectral_radiance_for_a_date(self, year, month, day, nel = 46, naz = 181, num_iterations = 150, use_site_horizon = False):
        
        """
        Compute spectral radiance across time for a complete day on a specific
        date, using the data stored in the *self.Site_obj* attribute.
        
        Parameters
        ----------
        year : int
            Year for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj.simulation_times_data'.
            
        month : int
            Month for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj.simulation_times_data'.
            
        day : int
            Day for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj.simulation_times_data'.
            
        nel : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the elevation coordinate. Default is 46.
            
        naz : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the azimuth coordinate. Default is 181.
        
        num_iterations : int, optional
            Number of iterations to use when filling NaN data. Default is 150.

        use_site_horizon : bool
            Include horizon effects. Default is False.

            
        Returns
        -------
        None

        Produces
        -------
        self.radiances : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : float or numpy.array of floats with shape (nel, naz)
                    Grid of azimuth coordinates (in degrees) of the sky elements for which the 
                    spectral radiances was calculated. Its values should vary along axis 1.
                    In any case, all values should be between 0 and 360 (inclusive). 
                
                "El" : float or numpy.array of floats with shape (nel, naz)
                    Grid of elevation coordinates (in degrees) of the sky elements for which the
                    spectral radiances was calculated. Its values should vary along axis 0.
                    In any case, all values should be between 0 and 90 (inclusive). 
                
                "Siv" : numpy.array of floats with shape (nt,)   
                    Igawa's 'Sky Index' parameter across time.
                
                "Kc" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Clear Sky Index' parameter across time.
                    
                "Cle" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Cloudless Index' parameter across time.
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.
                    
                "DatetimeIndex_obj" : pandas.Series of pandas.Timestamp objects.
                    Series of Timestamp values detailing the times at which each of the
                    samples of the time-dependent variables were taken. We denote its 
                    length as nt.
                    
                "spectral_direct" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Direct component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
                    
                "spectral_diffuse" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Diffuse component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
        
                                        
        Notes
        -----
        1) Initial time and final time of simulation are taken to be 
        *self.Site_obj_simulation_time_data[(year, month, day)][0]* and 
        *self.Site_obj_simulation_time_data[(year, month, day)][-1]* (respectively).
           
        2) Angular resolution in the Elevation coordinate is equal to 90/(nel - 1).

        3) Angular resolution in the Azimuth coordinate is equal to 360/(naz - 1).

        4) The time resolution used is the same as that of
        *self.Site_obj.simulation_times_data*.

        """
        global mask

        # MESH SKY VAULT 
        dAz, dEl = 360/(naz-1), 90/(nel-1)    
        Az, El = np.meshgrid(np.linspace(0, 360, naz), np.linspace(0, 90, nel)) 

        # COMPUTE MASK FOR HORIZON EFFECTS
        if use_site_horizon:
            azimuths = Az[0,:]
            mask = np.zeros(Az.shape).astype(bool)
            horizon_elevations = self.Site_obj.horizon["func"](azimuths)

            for i, horizon_elevation in enumerate(horizon_elevations):
                mask[:,i] = El[:,i] <= horizon_elevation
            
            mask = np.swapaxes(np.swapaxes(np.array([mask]*122),0,2),0,1)


        # RETRIVE DATA REQUIRED FOR MODEL 
        DatetimeIndex_obj = self.Site_obj.simulation_time_data[(year, month, day)]    
        sun_apel  = np.array(self.Site_obj.sun_data[(year, month, day)]["apel"])
        sun_az    = np.array(self.Site_obj.sun_data[(year, month, day)]["az"])

        Gh          = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["G(h)"])
        extra_Gbn   = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["extra_Gb(n)"])
        Gbn         = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["Gb(n)"])
        Gdh         = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["Gd(h)"])

        SP          = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["SP"])
        rel_airmass = np.array(self.Site_obj.sun_data[(year, month, day)]["rel_airmass"])
        H2O         = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["H2O"])
        O3          = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["O3"])

        aod_500nm   = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["aod_500nm"])
        alpha_500nm = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["alpha_500nm"])

        spectrally_averaged_aaf = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["spectrally_averaged_aaf"])
        single_scattering_albedo = np.array(self.Site_obj.single_scattering_albedo[(year, month, day)].iloc[:,1:]).astype(float)

        # COMPUTE SPECTRAL RADIANCE 
        res =\
        compute_spectral_radiance(Az                        = Az,
                                  El                        = El,
                                  dAz                       = dAz,
                                  dEl                       = dEl, 
                                  DatetimeIndex_obj         = DatetimeIndex_obj,
                                  sun_apel                  = sun_apel,
                                  sun_az                    = sun_az,
                                  Gh                        = Gh,
                                  extra_Gbn                 = extra_Gbn,
                                  Gbn                       = Gbn, 
                                  Gdh                       = Gdh,
                                  SP                        = SP, 
                                  rel_airmass               = rel_airmass,
                                  H2O                       = H2O, 
                                  O3                        = O3, 
                                  aod_500nm                 = aod_500nm,
                                  alpha_500nm               = alpha_500nm, 
                                  spectrally_averaged_aaf   = spectrally_averaged_aaf, 
                                  single_scattering_albedo  = single_scattering_albedo,  
                                  ground_albedo             = 0, 
                                  mean_surface_tilt         = 0, 
                                  num_iterations            = num_iterations
                                  )
        

        
        res["Az"] = Az
        res["El"] = El
        res["DatetimeIndex_obj"] = DatetimeIndex_obj
        res["spectral_direct"]   = res.pop("direct")
        res["spectral_diffuse"]  = res.pop("diffuse")

        # APPLY HORIZON EFFECTS
        if use_site_horizon:
            for nt in range(len(DatetimeIndex_obj)):
                res["spectral_direct"][nt][mask] = 0
                res["spectral_diffuse"][nt][mask] = 0

        self.radiances = res
        
        return None
    
    
    
    def compute_radiance_for_a_date(self):
        """
        Compute the radiance for a given date by integrating the (already
        computed) spectral radiance over the wavelength axis.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        -------
        self.radiances : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : float or numpy.array of floats with shape (nel, naz)
                    Grid of azimuth coordinates (in degrees) of the sky elements for which the 
                    radiances was calculated. Its values should vary along axis 1.
                    In any case, all values should be between 0 and 360 (inclusive). 
                
                "El" : float or numpy.array of floats with shape (nel, naz)
                    Grid of elevation coordinates (in degrees) of the sky elements for which the
                    radiances was calculated. Its values should vary along axis 0.
                    In any case, all values should be between 0 and 90 (inclusive). 
                
                "Siv" : numpy.array of floats with shape (nt,)   
                    Igawa's 'Sky Index' parameter across time.
                
                "Kc" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Clear Sky Index' parameter across time.
                    
                "Cle" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Cloudless Index' parameter across time.
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.
                    
                "DatetimeIndex_obj" : pandas.Series of pandas.Timestamp objects.
                    Series of Timestamp values detailing the times at which each of the
                    samples of the time-dependent variables were taken. We denote its 
                    length as nt.

                "spectral_direct" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Direct component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
                    
                "spectral_diffuse" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Diffuse component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
                    
                "direct" : numpy.array of floats with shape (nel,naz,nt)
                    Direct component of radiance across time. It has units
                    of W/m^2/sr.
                    
                "diffuse" : numpy.array of floats with shape (nel,naz,nt)
                    Diffuse component of radiance across time. It has units
                    of W/m^2/sr.
                    
        Notes
        -----
        1) This method requires the attribute *self.radiances* to
           already be defined. For this, please check out 
           :meth:`~solrad.Sky.Sky.compute_spectral_radiance_for_a_date`.
        """
        
        # Get number of sample points across space and time.
        nel, naz = self.radiances["Az"].shape
        nt = len(self.radiances["DatetimeIndex_obj"])
        
        # Initialize arrays for storing the radiance across time for all
        # points in the sky.
        new_direct  = np.zeros((nel, naz, nt))
        new_diffuse = np.zeros((nel, naz, nt))
        
        # --- COMPUTE RADIANCE BY INTEGRATING SPECTRAL RADIANCE ACROSS THE WAVELENGTHS AXIS ---
        for t in range(nt):
            
            new_direct[:,:,t] =\
            simpson(
            y    = self.radiances["spectral_direct"][t], 
            x    = self.radiances["wavelengths"],
            axis = 2
            )
            
            new_diffuse[:,:,t] =\
            simpson(
            y    = self.radiances["spectral_diffuse"][t], 
            x    = self.radiances["wavelengths"],
            axis = 2
            )

        # Include the direct and diffuse radiances.
        self.radiances["direct"]  = new_direct 
        self.radiances["diffuse"] = new_diffuse
        
        return None
    


    def compute_radiances_for_a_date(self, year, month, day, nel = 46, naz = 181, num_iterations = 150, use_site_horizon=False):
        """
        Compute radiance and spectral radiance across time for a complete day on a specific
        date, using the data stored in the *self.Site_obj* attribute.
        
        Parameters
        ----------
        year : int
            Year for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj.simulation_times_data'.
            
        month : int
            Month for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj.simulation_times_data'.
            
        day : int
            Day for which the spectral radiance is to be computed. Must be
            present in 'self.Site_obj.simulation_times_data'.
            
        nel : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the elevation coordinate. Default is 46.
            
        naz : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the azimuth coordinate. Default is 181.
        
        num_iterations : int, optional
            Number of iterations to use when filling NaN data. Default is 150.

        use_site_horizon : bool
            Include horizon effects. Default is False.
        
        Returns
        -------
        None
        
        Produces
        -------
        self.radiances : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : float or numpy.array of floats with shape (nel, naz)
                    Grid of azimuth coordinates (in degrees) of the sky elements for which the 
                    radiances was calculated. Its values should vary along axis 1.
                    In any case, all values should be between 0 and 360 (inclusive). 
                
                "El" : float or numpy.array of floats with shape (nel, naz)
                    Grid of elevation coordinates (in degrees) of the sky elements for which the
                    radiances was calculated. Its values should vary along axis 0.
                    In any case, all values should be between 0 and 90 (inclusive). 
                
                "Siv" : numpy.array of floats with shape (nt,)   
                    Igawa's 'Sky Index' parameter across time.
                
                "Kc" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Clear Sky Index' parameter across time.
                    
                "Cle" : numpy.array of floats with shape (nt,) 
                    Igawa's 'Cloudless Index' parameter across time.
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.
                    
                "DatetimeIndex_obj" : pandas.Series of pandas.Timestamp objects.
                    Series of Timestamp values detailing the times at which each of the
                    samples of the time-dependent variables were taken. We denote its 
                    length as nt.

                "spectral_direct" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Direct component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
                    
                "spectral_diffuse" : List with length nt of numpy.arrays of floats with shape (nel,naz,122)
                    Diffuse component of spectral radiance across time. It has
                    units of W/m^2/sr/nm.
                    
                "direct" : numpy.array of floats with shape (nel,naz,nt)
                    Direct component of radiance across time. It has units
                    of W/m^2/sr.
                    
                "diffuse" : numpy.array of floats with shape (nel,naz,nt)
                    Diffuse component of radiance across time. It has units
                    of W/m^2/sr.
        """
        self.compute_spectral_radiance_for_a_date(year, month, day, nel, naz, num_iterations, use_site_horizon)
        self.compute_radiance_for_a_date()
        return None


    # -- PLOT RADIANCES --
    def plot_spectral_radiance_for_a_date(self, component, nt=None, config = None):

        """
        Plot spectral radiance for a specific component at a given time.

        Parameters
        ----------
        component : {'direct', 'diffuse}
            The component spectral radiance to plot.

        nt : int or None, optional
            The time index for which to plot the spectral radiance. If None, plots for all times.

        config : dict or None, optional
            Configuration parameters for the plot. If None (the default), it uses
            default parameters. If provided, it may contain the following
            key-value pairs:

            Keys-Values
            -----------
            'figsize': tuple, optional, default: (16, 12)
                Figure size.

            'wavelength_idxs': numpy.ndarray, optional
                2D array specifying the wavelength indices to plot.

        Returns
        -------
        None

        Notes
        -----
        1) This method requires the attribute *self.radiances* to
        already be defined. For this, please check out 
        :meth:`~solrad.Sky.Sky.compute_radiances_for_a_date`.

        2) The method generates polar plots of spectral radiance for the specified component at the given time.
        The plots show color-contoured radiance values at different azimuth and elevation angles.

        """

        config_ = {
        "figsize" : (16,12),
        "wavelength_idxs" : np.array([[15, 30, 45], [50, 65, 80]])}

        if config is not None:
            config_.update(config)

        if len(config_["wavelength_idxs"].shape) == 1:
            config_["wavelength_idxs"] =\
            config_["wavelength_idxs"].reshape((1, config_["wavelength_idxs"].shape[0]))
        

        nrows = config_["wavelength_idxs"].shape[0]
        ncols = config_["wavelength_idxs"].shape[1]
        DatetimeIndex_obj = self.radiances["DatetimeIndex_obj"]
        Az, El = self.radiances["Az"], self.radiances["El"]
        Phi = np.deg2rad(Az)

        iterable = [nt]
        if nt is None:
            iterable = range(len(DatetimeIndex_obj))

        for nt in iterable:
            fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
            subplot_kw=dict(projection='polar'))
            axs = axs.reshape((nrows, ncols))
            fig.set_figwidth(config_["figsize"][0])
            fig.set_figheight(config_["figsize"][1])
            

            title = f"{self.Site_obj.name}: Spectral {component} radiance at time {DatetimeIndex_obj[nt]}"
            plt.suptitle(f"{title}.\n (N = 0°, E = 90°, S = 180°, W = 270°)")

            for nr in range(nrows):
                for nc in range(ncols):
                    wavelength_idx = config_["wavelength_idxs"][nr, nc]
                    wavelength     = self.radiances["wavelengths"][wavelength_idx]
                    Color          = self.radiances[f"spectral_{component}"][nt][:,:,wavelength_idx]

                    max_ = np.max(Color)
                    if(max_ > 0):
                        Color_ = Color/max_
                    else:
                        Color_ = Color

                    axs[nr,nc].contourf(Phi, 90-El, Color_, levels = 25, cmap = plt.cm.hot)
                    axs[nr,nc].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])
                    axs[nr,nc].set_title(f"{wavelength} nm")
                    axs[nr,nc].tick_params(axis='y', colors='gray')
                    axs[nr,nc].set_theta_zero_location("N")
                    axs[nr,nc].set_theta_direction(-1)
                    axs[nr,nc].grid(False)

                    m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
                    m.set_array(Color)
                    cbar = plt.colorbar(m, ax=axs[nr,nc])
                    cbar.ax.set_title('W/m^2/sr/nm')
            plt.show()

        return None
    

    def plot_radiance_for_a_date(self, component, nt=None, projection = "disk", figsize=(16,12), view_init = (45,180)):

        """
        Plot radiance for a specific component at a given time.

        Parameters
        ----------
        component : str
            The radiance component to plot (e.g., "direct", "diffuse").

        nt : int or None, optional
            The time index for which to plot the radiance. If None, plots for all times.

        projection : {'disk', 'sphere'}, optional
            The type of projection for the plot. Options are "disk" (polar) or "sphere" (3D). Default is "disk".

        figsize : tuple, optional
            Figure size. Default is (16, 12).

        view_init : tuple, optional
            Elevation and azimuth of the axes in degrees. Default is (45, 180).

        Returns
        -------
        None

        Notes
        -----
        1) This method requires the attribute *self.radiances* to
        already be defined. For this, please check out 
        :meth:`~solrad.Sky.Sky.compute_radiances_for_a_date`.

        2) The method generates plots of radiance for the specified component at the given time.
        It supports two types of projections: "disk" (polar plot) and "sphere" (3D plot).


        """

        DatetimeIndex_obj = self.radiances["DatetimeIndex_obj"]
        Az, El = self.radiances["Az"], self.radiances["El"]
        Phi = np.deg2rad(Az)

        if nt is None:
            iterable = range(len(DatetimeIndex_obj))
        else:
            iterable = [nt]

        for nt in iterable:
            Color = self.radiances[component][:,:,nt]
            
            max_ = np.max(Color)
            if(max_ > 0):
                Color_ = Color/max_
            else:
                Color_ = Color
            
                
            title = f"{self.Site_obj.name}: {component} radiance at time: {DatetimeIndex_obj[nt]}"

            if projection == "disk":
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                ax.contourf(Phi, 90-El, Color_, levels = 25, cmap = plt.cm.hot)
                ax.set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])
                ax.set_title("(N = 0°, E = 90°, S = 180°, W = 270°)")
                ax.tick_params(axis='y', colors='gray')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                fig.set_figwidth(figsize[0])
                fig.set_figheight(figsize[1])
                ax.grid(False)
                plt.suptitle(title)

            elif projection == "sphere":
                Theta = np.deg2rad(90 - El).reshape(El.shape[0], El.shape[1]) 
                X = np.cos(Phi)*np.sin(Theta)
                Y = np.sin(Phi)*np.sin(Theta)
                Z = np.cos(Theta)

                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(view_init[0], view_init[1])
                ax.set_xlabel("X (↑ == N, ↓ == S)")
                ax.set_ylabel("Y (↑ == E, ↓ == W)")
                ax.set_title(title)
                ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))               

            
            m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
            m.set_array(Color)
            cbar = plt.colorbar(m, ax=ax)
            cbar.ax.set_title('W/m^2/sr')
            plt.show()

        return None
    



    # 3.2) -- COMPUTE SPECTRAL RADIANT EXPOSURE VECTORS --    
    

    def _compute_time_integrated_spectral_radiance_for_a_date_interval(self, start_date = None, end_date = None, nel = 46, naz = 181, num_iterations = 150, use_site_horizon = False):
        
        """
        Compute time integral of spectral radiance for a specified interval of
        dates, using the data stored in the 'self.Site_obj' attribute.
        
        Parameters
        ----------
        start_date : 3-tuple of int or None, optional
            Start date of computation. 
            If None, the first date in self.Site_obj.simulation_times_data.keys() is chosen.
            Otherwise, it must be a is 3-tuple of (year, month, day) indicating the start date.

        end_date : 3-tuple of int or None, optional
            End date of computation. 
            If None, the last date in self.Site_obj.simulation_times_data.keys() is chosen.
            Otherwise, it must be a is 3-tuple of (year, month, day) indicating the end date.
                
        nel : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the elevation coordinate. Default is 46.
            
        naz : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the azimuth coordinate. Default is 181.
        
        num_iterations : int, optional
            Number of iterations to use when filling NaN data. Default is 150.

        use_site_horizon : bool
            Include horizon effects. Default is False.

        Returns
        -------
        None

        Produces
        -------
        time_integrated_spectral_radiance : dict
            Dictionary containing result variables. It has the following Key-Value
            pairs:
                
                Keys : Values
                -------------
                "Az" : float or numpy.array of floats with shape (nel, naz)
                    Grid of azimuth coordinates (in degrees) of the sky elements for which the 
                    spectral radiance was calculated. Its values should vary along axis 1.
                    In any case, all values should be between 0 and 360 (inclusive). 
                
                "El" : float or numpy.array of floats with shape (nel, naz)
                    Grid of elevation coordinates (in degrees) of the sky elements for which the
                    spectral radiance was calculated. Its values should vary along axis 0.
                    In any case, all values should be between 0 and 90 (inclusive). 
                    
                "wavelengths" : numpy.array of floats with shape (122,)
                    Wavelengths in nanometers.

                "Siv_avg" : float
                    Igawa's 'Sky Index' parameter, averaged across time.
                
                "Kc_vg" : float
                    Igawa's 'Clear Sky Index' parameter, averaged across time.
                    
                "Cle_avg" : float 
                    Igawa's 'Cloudless Index' parameter, averaged across time.
                    
                "direct" : numpy.array of floats with shape (nel,naz,122)
                    Time integral of direct component of spectral radiance across time.
                    It has units of Wh/m^2/sr/nm.
                    
                "diffuse" : numpy.array of floats with shape (nel,naz,122)
                    Time integral of diffuse component of spectral radiance across time.
                    It has units of Wh/m^2/sr/nm.

                "start_date" : 3-tuple of int
                    Start date of computation.

                "end_date" : 3-tuple of int
                    End date of computation.
        
                                        
        Notes
        -----
        1) Initial time and final time of simulation for each day are taken to be 
           self.Site_obj_simulation_time_data[(year, month, day)][0] and 
           self.Site_obj_simulation_time_data[(year, month, day)][-1] (respectively).
           
         2) Angular resolution in the Elevation coordinate is equal to 90/(nel - 1).
    
         3) Angular resolution in the Azimuth coordinate is equal to 360/(naz - 1).
    
         4) The time resolution used is the same as that of
            self.Site_obj.simulation_times_data.

        """
        # COMPUTE START AND END DATES 
        if start_date is None:
            start_date_ts = min([pd.Timestamp(f"{date[0]}-{date[1]}-{date[2]}")
                                 for date in self.Site_obj.simulation_times_data.keys()])
        else:
            start_date_ts = pd.Timestamp(f"{start_date[0]}-{start_date[1]}-{start_date[2]}")

        if end_date is None:
            end_date_ts = max([pd.Timestamp(f"{date[0]}-{date[1]}-{date[2]}")
                               for date in self.Site_obj.simulation_times_data.keys()])
        else:
            end_date_ts   = pd.Timestamp(f"{end_date[0]}-{end_date[1]}-{end_date[2]}")
        
        if  end_date_ts < start_date_ts:
            msg = "Start date cannot be greater than end date. "
            raise Exception(msg)       
        

        # MESH SKY VAULT 
        dAz, dEl = 360/(naz-1), 90/(nel-1)    
        Az, El = np.meshgrid(np.linspace(0, 360, naz), np.linspace(0, 90, nel)) 

        # COMPUTE MASK FOR HORIZON EFFECTS
        if use_site_horizon:
            azimuths = Az[0,:]
            mask = np.zeros(Az.shape).astype(bool)
            horizon_elevations = self.Site_obj.horizon["func"](azimuths)

            for i, horizon_elevation in enumerate(horizon_elevations):
                mask[:,i] = El[:,i] <= horizon_elevation
            
            mask = np.swapaxes(np.swapaxes(np.array([mask]*122),0,2),0,1)
        

        # INITIALIZE RESULTS DICT
        time_integrated_spectral_radiance =\
        {"Siv_avg"    : [],
         "Kc_avg"     : [],
         "Cle_avg"    : [],
         "direct"     : np.zeros((nel, naz, 122)),
         "diffuse"    : np.zeros((nel, naz, 122)),
         "Az"         : Az,
         "El"         : El,
         "start_date" : (start_date_ts.year, start_date_ts.month, start_date_ts.day),
         "end_date"   : (end_date_ts.year, end_date_ts.month, end_date_ts.day)}
         
        # COMPUTE dt IN HOURS
        dt = pd.Timedelta(self.Site_obj.simulation_time_data_freq)
        dt = 24*dt.days + dt.seconds/3600

        # Initialize time delta
        one_day = pd.Timedelta("1D")

        current_date_ts = start_date_ts
        while current_date_ts <= end_date_ts:
            year  = current_date_ts.year
            month = current_date_ts.month
            day   = current_date_ts.day

            DatetimeIndex_obj = self.Site_obj.simulation_time_data[(year, month, day)]

            # COMPUTE SPECTRAL RADIANCE 
            res =\
            compute_spectral_radiance(Az                        = Az,
                                      El                        = El,
                                      dAz                       = dAz,
                                      dEl                       = dEl, 
                                      DatetimeIndex_obj         = DatetimeIndex_obj ,
                                      sun_apel                  = np.array(self.Site_obj.sun_data[(year, month, day)]["apel"]),
                                      sun_az                    = np.array(self.Site_obj.sun_data[(year, month, day)]["az"]),
                                      Gh                        = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["G(h)"]),
                                      extra_Gbn                 = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["extra_Gb(n)"]),
                                      Gbn                       = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["Gb(n)"]), 
                                      Gdh                       = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["Gd(h)"]),
                                      SP                        = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["SP"]), 
                                      rel_airmass               = np.array(self.Site_obj.sun_data[(year, month, day)]["rel_airmass"]),
                                      H2O                       = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["H2O"]), 
                                      O3                        = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["O3"]), 
                                      aod_500nm                 = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["aod_500nm"]),
                                      alpha_500nm               = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["alpha_500nm"]), 
                                      spectrally_averaged_aaf   = np.array(self.Site_obj.climate_and_air_data[(year, month, day)]["spectrally_averaged_aaf"]), 
                                      single_scattering_albedo  = np.array(self.Site_obj.single_scattering_albedo[(year, month, day)].iloc[:,1:]).astype(float),  
                                      ground_albedo             = 0, 
                                      mean_surface_tilt         = 0, 
                                      num_iterations            = num_iterations
                                      )
            
            # COMPUTE AVERAGE IGAWA PROPERTIES
            Siv_avg = res["Siv"]
            Siv_avg[0], Siv_avg[-1] = Siv_avg[0]/2, Siv_avg[-1]/2
            Siv_avg = np.mean(Siv_avg)

            Kc_avg = res["Kc"]
            Kc_avg[0], Kc_avg[-1] = Kc_avg[0]/2, Kc_avg[-1]/2
            Kc_avg = np.mean(Kc_avg)

            Cle_avg = res["Cle"]
            Cle_avg[0], Cle_avg[-1] = Cle_avg[0]/2, Cle_avg[-1]/2
            Cle_avg = np.mean(Cle_avg)

            # INTEGRATE SPECTRAL RADIANCE ACROSS TIME FOR THE CURRENT DATE
            # For this, we make use of the trapezoid rule.
            direct, diffuse = res["direct"], res["diffuse"]

            direct[0],  direct[-1]  =  direct[0]/2,  direct[-1]/2
            diffuse[0], diffuse[-1] =  diffuse[0]/2, diffuse[-1]/2
            
            time_integral_direct  = sum(direct)*dt
            time_integral_diffuse = sum(diffuse)*dt

            # ADD RESULTS TO TOTAL DATA
            time_integrated_spectral_radiance["Siv_avg"].append(Siv_avg)
            time_integrated_spectral_radiance["Kc_avg"].append(Kc_avg)
            time_integrated_spectral_radiance["Cle_avg"].append(Cle_avg)
            time_integrated_spectral_radiance["direct"]  += time_integral_direct
            time_integrated_spectral_radiance["diffuse"] += time_integral_diffuse

            # ITERATE TO NEXT DATE
            current_date_ts += one_day

        # PERFORM LAST COMPUTATIONS
        time_integrated_spectral_radiance["Siv_avg"] = np.mean(time_integrated_spectral_radiance["Siv_avg"])
        time_integrated_spectral_radiance["Kc_avg"]  = np.mean(time_integrated_spectral_radiance["Kc_avg"])
        time_integrated_spectral_radiance["Cle_avg"] = np.mean(time_integrated_spectral_radiance["Cle_avg"])
        time_integrated_spectral_radiance["wavelengths"] = res["wavelengths"]

        # APPLY HORIZON EFFECTS
        if use_site_horizon:
            time_integrated_spectral_radiance["direct"][mask]  = 0
            time_integrated_spectral_radiance["diffuse"][mask] = 0

        self.time_integrated_spectral_radiance = time_integrated_spectral_radiance
        return None


    def _plot_time_integrated_spectral_radiance_for_a_date_interval(self, component, config = None):

        """
        Plot time-integrated spectral radiance for a specific component at a given time.

        Parameters
        ----------
        component : {'direct', 'diffuse}
            The component spectral radiance to plot.

        config : dict or None, optional
            Configuration parameters for the plot. If None (the default), it uses
            default parameters. If provided, it may contain the following
            key-value pairs:

            Keys-Values
            -----------
            'figsize': tuple, optional, default: (16, 12)
                Figure size.

            'wavelength_idxs': numpy.ndarray, optional
                2D array specifying the wavelength indices to plot.

        Returns
        -------
        None

        Notes
        -----
        The method generates polar plots of spectral radiance for the specified component at the given time.
        The plots show color-contoured radiance values at different azimuth and elevation angles.

        """

        config_ = {
        "figsize" : (16,12),
        "wavelength_idxs" : np.array([[15, 30, 45], [50, 65, 80]])}

        if config is not None:
            config_.update(config)

        if len(config_["wavelength_idxs"].shape) == 1:
            config_["wavelength_idxs"] =\
            config_["wavelength_idxs"].reshape((1, config_["wavelength_idxs"].shape[0]))
        

        nrows = config_["wavelength_idxs"].shape[0]
        ncols = config_["wavelength_idxs"].shape[1]
        start_date = self.time_integrated_spectral_radiance["start_date"]
        end_date   = self.time_integrated_spectral_radiance["end_date"]
        Az = self.time_integrated_spectral_radiance["Az"]
        El = self.time_integrated_spectral_radiance["El"]
        Phi = np.deg2rad(Az)


        fig, axs = plt.subplots(nrows = nrows, ncols = ncols,
        subplot_kw=dict(projection='polar'))
        axs = axs.reshape((nrows, ncols))
        fig.set_figwidth(config_["figsize"][0])
        fig.set_figheight(config_["figsize"][1])
        

        title = f"{self.Site_obj.name}: Time-integrated Spectral radiance ({component}) from {start_date} to {end_date}."
        plt.suptitle(f"{title}.\n (N = 0°, E = 90°, S = 180°, W = 270°)")

        for nr in range(nrows):
            for nc in range(ncols):
                wavelength_idx = config_["wavelength_idxs"][nr, nc]
                wavelength     = self.time_integrated_spectral_radiance["wavelengths"][wavelength_idx]
                Color          = self.time_integrated_spectral_radiance[f"{component}"][:,:,wavelength_idx]

                max_ = np.max(Color)
                if(max_ > 0):
                    Color_ = Color/max_
                else:
                    Color_ = Color

                axs[nr,nc].contourf(Phi, 90-El, Color_, levels = 25, cmap = plt.cm.hot)
                axs[nr,nc].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])
                axs[nr,nc].set_title(f"{wavelength} nm")
                axs[nr,nc].tick_params(axis='y', colors='gray')
                axs[nr,nc].set_theta_zero_location("N")
                axs[nr,nc].set_theta_direction(-1)
                axs[nr,nc].grid(False)

                m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
                m.set_array(Color)
                cbar = plt.colorbar(m, ax=axs[nr,nc])
                cbar.ax.set_title('Wh/m^2/sr/nm')
        plt.show()

        return None
    
    
    def compute_exposure_vectors_for_a_date_interval(self, start_date = None, end_date = None, nel = 46, naz = 181, num_iterations = 150, use_site_horizon = False, int_nzen = 20, int_naz = 30):
        
        """
        Compute the radiant exposure and spectral radiant exposure vectors
        of each Sky patch. That is, the time-integrated spectral irradiance 
        over each Sky patch (for a given date interval) by integrating the
        time-integrated spectral radiance with respect to the solid angle, over
        each sky patch of the discretised Sky Vault.
        
        Parameters
        ----------
        start_date : 3-tuple of int or None, optional
            Start date of computation. 
            If None, the first date in self.Site_obj.simulation_times_data.keys() is chosen.
            Otherwise, it must be a is 3-tuple of (year, month, day) indicating the start date.

        end_date : 3-tuple of int or None, optional
            End date of computation. 
            If None, the last date in self.Site_obj.simulation_times_data.keys() is chosen.
            Otherwise, it must be a is 3-tuple of (year, month, day) indicating the end date.
                
        nel : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the elevation coordinate. Default is 46.
            
        naz : int, optional
            Number of samples for dicretizing the sky vault with regards to
            the azimuth coordinate. Default is 181.
        
        num_iterations : int, optional
            Number of iterations to use when filling NaN data. Default is 150.

        use_site_horizon : bool
            Include horizon effects. Default is False.

        int_nzen : int
            Number of samples for dicretizing each sky patch, with regards to
            the zenith coordinate, in order to compute the diffuse spectral
            irradiance via integration. Default is 20.
            
        int_naz : int
            Number of samples for dicretizing each sky patch, with regards to
            the zenith coordinate, in order to compute the diffuse spectral
            irradiance via integration. Default is 30.
            
        Returns
        -------
        None

        Produces
        --------
        self.patch_data[(zone_num, local_patch_num)]["exposure"] : dict of dicts
            Each sky patch recieves a new key in its database called 
            "exposure". This is a dict with keys: "direct", "diffuse", "global",
            "spectral_direct", "spectral_diffuse", "spectral_global"; and "wavelengths".
            Aside from "wavelengths", every other key holds another dictionary that stores 
            some relevant information about the direct, diffuse and global radiant and 
            spectral radiant exposure (respectively) related to that particular sky patch.

            For keys "spectral_direct", "spectral_diffuse", "spectral_global",
            each of these dicts contains the following key-value pairs:

                Keys : Values
                -------------
                "vector" : np.array of floats with shape (3,122)
                    spectral direct/spectral diffuse/spectral global
                    (depending on the case) radiant exposure vector. 
                    "vector"[0,:], "vector"[1,:] and "vector"[2,:], 
                    hold the x, y and z components of the 
                    spectral radiant exposure vector, respectively, 
                    for all wavelengths in key "wavelengths". 
                    Each component has units of Wh/m^2/nm.

                "magnitude" : np.array of floats with shape (122,)
                    Magnitude of the spectral direct/spectral diffuse/spectral global
                    (depending on the case) spectral radiant exposure vector.
                    It has units of Wh/m^2/nm.

                "spectrally_averaged_unit_vector" : np.array of floats with shape (3,)
                    Average position of irradiance within a sky patch. That is,
                    the unit vector version of the spectrally averageds
                    spectral direct/spectral diffuse/spectral global (depending 
                    on the case) radiant exposure vector. 
                    In the case, however, that said Spectral radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.

            Now, for keys "direct", "diffuse", "global",
            each of these dicts contains the following key-value pairs:

                Keys : Values
                -------------
                "vector" : np.array of floats with shape (3,)
                    direct/diffuse/global (depending on the case) 
                    radiant exposure vector. "vector"[0], "vector"[1]
                    and "vector"[2], hold the x, y and z components of the 
                    radiant exposure vector, respectively, for all 
                    wavelengths in key "wavelengths". Each component
                    has units of Wh/m^2.
                    
                "magnitude" : float
                    Magnitude of the direct/diffuse/global (depending on 
                    the case) spectral radiant exposure vector.
                    It has units of Wh/m^2.
                    
                "spectrally_averaged_unit_vector" : np.array of floats with shape (3,)
                    Average position of irradiance within a sky patch. That is,
                    the unit vector version of the spectrally averaged
                    direct/diffuse/global (depending on the case) Radiant exposure vector. 
                    In the case, however, that said Spectral radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.

            Finally, in the case of the key "wavelengths", it does not store any dicts
            but rather a numpy.array of float values:
            
            np.array of floats with shape (122,)
                Array of wavelengths over which the spectral irradiances are defined.
        
        self.exposure_vectors : dict of numpy.arrays
            Dict containing the same info as above, but in another format
            that is handier for other things. It has the following
            key-value pairs:

                Keys : Values
                -------------
                "spectral_direct" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                    Direct spectral radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2/nm.
    
                "spectral_direct_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                    Magnitude of the direct spectral radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2/nm.

                "spectral_direct_unit_savgd" : numpy.array of floats with shape (self.num_divisions, 3)
                    Average position of radiant exposure within a sky patch. That is,
                    unit vector version of the spectrally averaged spectral direct
                    (depending on the case) radiant exposure vector. 
                    In the case, however, that said Spectral radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.
                    
                "spectral_diffuse" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                    Diffuse spectral radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2/nm.
    
                "spectral_diffuse_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                    Magnitude of the diffuse spectral radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2/nm.

                "spectral_diffuse_unit_savgd" : numpy.array of floats with shape (self.num_divisions, 3)
                    Average position of radiant exposure within a sky patch. That is,
                    unit vector version of the spectrally averaged spectral diffuse
                    (depending on the case) radiant exposure vector. 
                    In the case, however, that said Spectral radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.
                    
                "spectral_global" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                    Global spectral radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2/nm.
    
                "spectral_global_global" : numpy.array of floats with shape (self.num_divisions, 122)
                    Magnitude of the global spectral radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2/nm.

                "spectral_global_unit_savgd" : numpy.array of floats with shape (self.num_divisions, 3)
                    Average position of radiant exposure within a sky patch. That is,
                    unit vector version of the spectrally averaged spectral diffuse
                    (depending on the case) radiant exposure vector. 
                    In the case, however, that said Spectral radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.

                "direct" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                    Direct radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2.
    
                "direct_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                    Magnitude of the direct radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2.

                "direct_unit" : numpy.array of floats with shape (self.num_divisions, 3)
                    Average position of radiant exposure within a sky patch. That is,
                    unit vector version of the spectrally averaged spectral direct
                    (depending on the case) radiant exposure vector. 
                    In the case, however, that said radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.

                "diffuse" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                    Diffuse radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2.
    
                "diffuse_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                    Magnitude of the diffuse radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2.

                "diffuse_unit" : numpy.array of floats with shape (self.num_divisions, 3)
                    Average position of radiant exposure within a sky patch. That is,
                    unit vector version of the spectrally averaged spectral diffuse
                    (depending on the case) radiant exposure vector. 
                    In the case, however, that said radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.
                    
                "global" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                    Global radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2.
    
                "global_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                    Magnitude of the global radiant exposure vector for each of the sky patches. 
                    It has units of Wh/m^2.

                "global_unit" : numpy.array of floats with shape (self.num_divisions, 3)
                    Average position of radiant exposure within a sky patch. That is,
                    unit vector version of the spectrally averaged spectral global
                    (depending on the case) radiant exposure vector. 
                    In the case, however, that said radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.

                "wavelengths" : np.array of floats with shape (122,)
                    Array of wavelengths over which the spectral irradiances 
                    are defined.
            
        """

        # --- COMPUTE TIME-INTEGRATED SPECTRAL RADIANCE --- 
        self._compute_time_integrated_spectral_radiance_for_a_date_interval(start_date, end_date, nel, naz, num_iterations, use_site_horizon)


        # --- INITIALIZE NEW ATTRIBUTE FOR STORING THE RESULTS IN ARRAY FORM ---
        self.exposure_vectors = \
        {
        "spectral_direct"       : np.zeros((self.num_divisions, 3, 122)),
        "spectral_diffuse"      : np.zeros((self.num_divisions, 3, 122)),
        "spectral_global"       : np.zeros((self.num_divisions, 3, 122)),

        "spectral_direct_mag"   : np.zeros((self.num_divisions,    122)),
        "spectral_diffuse_mag"  : np.zeros((self.num_divisions,    122)),
        "spectral_global_mag"   : np.zeros((self.num_divisions,    122)),

        "spectral_direct_unit_savgd"  : np.zeros((self.num_divisions, 3)),
        "spectral_diffuse_unit_savgd" : np.zeros((self.num_divisions, 3)),
        "spectral_global_unit_savgd"  : np.zeros((self.num_divisions, 3)),

        }

        
        
        # ---- RETREIVE/COMPUTE DOMAIN ARRAYS ----
        
        # Retrieve data related to the domain over which we shall compute the
        # spectral irradiance quantities.
    
        Az  = self.time_integrated_spectral_radiance["Az"] 
        Zen = 90 - self.time_integrated_spectral_radiance["El"] 
        wavelengths = self.time_integrated_spectral_radiance["wavelengths"] 
        self.exposure_vectors["wavelengths"] = wavelengths
        Phi, Theta = np.deg2rad(Az), np.deg2rad(Zen)
        
        
        # --- INTERPOLATE DIFFUSE RADIANCE, OVER THE WHOLE DOMAIN, AT EACH WAVELENGTH -----
        
        # We need to do this in order to later calculate the diffuse spectral exposure vector
        # (i.e, time-integrated diffuse irradiance vector).
        # Since the diffuse spectral radiance is taken to be a continuous function rather 
        # than a delta dirac (unlike the direct spectral radiance), we must 
        # compute its irradiance quantity via actual integration. Hence why we 
        # need this function.
        
        interp_diffuse = []
        for wv in range(len(wavelengths)):
            interp_diffuse.append(
            RegularGridInterpolator(
            points = (Zen[:,0], Az[0,:]),
            values = self.time_integrated_spectral_radiance["diffuse"][:,:,wv]))


        # ---- COMPUTE SOLID ANGLE UNIT VECTORS FOR ALL SKY POINTS ----
        
        # When computing the direct spectral irradiance vector, we need to 
        # mutliply each direct spectral radiance value by its corresponding 
        # unit solid angle vector and sum them all toguether. Having these 
        # vectors already precomputed to just retrieve them later, saves us 
        # computational power.
        
        solid_angle_unit_vecs = np.zeros(list(Az.shape)+[3])
        solid_angle_unit_vecs[:,:,0] = np.cos(Phi)*np.sin(Theta)
        solid_angle_unit_vecs[:,:,1] = np.sin(Phi)*np.sin(Theta)
        solid_angle_unit_vecs[:,:,2] = np.cos(Theta)
        
        
        #     --- GET SKY POINTS WITHIN SKY PATCH ---
        
        # We go over each sky patch, retrive the sky patch's limits. Then, in the case
        # of the direct spectral exposure (i.e, time-integrated direct spectral irradiance) 
        # we also compute a logic array for retrieving all sky points that lie within said sky patch.

        for c, ((zone_num, local_patch_num), patch_dict) in enumerate(self.patch_data.items()):
    
            # We retrieve the sky patch's limits.
            inf_az,  sup_az  = patch_dict["inf_az"],  patch_dict["sup_az"]
            inf_zen, sup_zen = patch_dict["inf_zen"], patch_dict["sup_zen"]
            
            # We compute a logic array for retrieving all sky points that
            # lie within the given sky patch. We use this for computing
            # the direct spectral radiance.
            if sup_az == 360:
                  logic_az = np.logical_and(inf_az <= Az, Az <= sup_az)
            else: logic_az = np.logical_and(inf_az <= Az, Az <  sup_az)
                
            if sup_zen == 90:
                  logic_zen = np.logical_and(inf_zen <= Zen, Zen <= sup_zen)
            else: logic_zen = np.logical_and(inf_zen <= Zen, Zen <  sup_zen)
                
            logic_patch = np.logical_and(logic_az, logic_zen) 
            
            
            
            # -------- (1) COMPUTE DIRECT COMPONENT QUANTITIES ------------


            #     --- (1.A) RETRIEVE LOCAL SOLID ANGLE UNIT VECTORS ---
            # Retrieve the unit vectors that indicate the position of each sky 
            # point, that lies within the current sky patch.
            
            num_pts = logic_patch.sum() 
            local_solid_angle_unit_vecs = np.zeros((num_pts, 3))            
            local_solid_angle_unit_vecs[:,0] = solid_angle_unit_vecs[:,:,0][logic_patch]
            local_solid_angle_unit_vecs[:,1] = solid_angle_unit_vecs[:,:,1][logic_patch]
            local_solid_angle_unit_vecs[:,2] = solid_angle_unit_vecs[:,:,2][logic_patch]
            
            
            #    --- (1.B) INTIALIZE LOCAL DIRECT SPECTRAL IRRADIANCE VECTOR --- 
            local_direct_spectral_exposure_vec = np.zeros((3, 122))
            # (i.e, the local_direct_time_integrated_spectral_irradiance_vec)
            
            
            #    --- (1.C) RETRIEVE DIRECT SPECTRAL RADIANCES --- 
            # Retrieve direct component of spectral radiance, for all
            # sky points that lie within the current sky patch.
            local_direct_time_integrated_spectral_radiance_vals =\
            self.time_integrated_spectral_radiance["direct"][logic_patch] 
            

            # --- (1.D) COMPUTE LOCAL DIRECT SPECTRAL IRRADIANCE VECTOR ---
            
            # We compute the local direct spectral exposure vector
            # (i.e, local direct spectral irradiance vector) component
            # by component. A general procedure of the description is as follows:
            # We gather the different direct spectral radiance values within the
            # current sky patch. Multiply each value one by its corresponding 
            # unit solid angle vector and then sum them all toguether, wavelength-wise.
            
            local_direct_spectral_exposure_vec[0,:] =\
            (local_solid_angle_unit_vecs[:,0].reshape(num_pts, 1)*
             local_direct_time_integrated_spectral_radiance_vals).sum(axis=0)
            
            local_direct_spectral_exposure_vec[1,:] =\
            (local_solid_angle_unit_vecs[:,1].reshape(num_pts, 1)*
             local_direct_time_integrated_spectral_radiance_vals).sum(axis=0)
            
            local_direct_spectral_exposure_vec[2,:] =\
            (local_solid_angle_unit_vecs[:,2].reshape(num_pts, 1)*
             local_direct_time_integrated_spectral_radiance_vals).sum(axis=0)
            
            
            # --- (1.E) COMPUTE SPECTRAL IRRADIANCE VECTOR MAGINTUDE FOR CURRENT SKY PATCH ---
            # We compute the magnitudes of the computed vector for each wavelength.
            
            local_direct_spectral_exposure_magnitude =\
            np.linalg.norm(local_direct_spectral_exposure_vec, axis = 0)
            #(i.e, local_direct_time_integrated_spectral_irradiance_magnitude)

            # --- (1.F) COMPUTE SPECTRALLY AVERAGED UNIT IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # We average the vector over its wavelengths by integrating it over 
            # its wavelength axis. We then normalize the resulting vector. This 
            # provides us with a vector that points from origin to the "center 
            # of radiation" (so to speak) of the radiation found within the
            # current sky patch.
            

            # (i.e, local_direct_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)
            local_direct_spectral_exposure_uvec_spectrally_avgd = \
            simpson(
            y    = local_direct_spectral_exposure_vec, 
            x    = wavelengths,
            axis = 1
            )
            
            local_direct_spectral_exposure_uvec_spectrally_avgd /=\
            np.linalg.norm(local_direct_spectral_exposure_uvec_spectrally_avgd)
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(local_direct_spectral_exposure_uvec_spectrally_avgd)):
                local_direct_spectral_exposure_uvec_spectrally_avgd = patch_dict["unit_vector"]
                
                
            
        
            # --------- (2) COMPUTE DIFFUSE COMPONENT QUANTITIES --------------
            
            
            #    --- (2.A) INTIALIZE LOCAL DIFFUSE SPECTRAL IRRADIANCE VECTOR --- 
            local_diffuse_spectral_exposure_vec = np.zeros((3,122))
            # (i.e, local_diffuse_time_integrated_spectral_irradiance_vec)
            
            
            #   --- (2.B) COMPUTE LOCAL DOMAIN VARIABLES FOR INTEGRATION --- 
            # We are going to compute the diffuse spectral irradiance for each
            # sky patch, via integration. As such, we define a new domain for
            # such a purpose. This is necessary, as it is possible that the overall
            # angular resolution of the data is not enough to compute a good
            # approximation of the integral. Therefore, we use another domain with
            # linearly interpolated data in order to gain angular resolution and,
            # therefore, accuracy in the integration procedure.
            
            int_dAz          = np.deg2rad(sup_az  -  inf_az)/(int_naz - 1)
            int_dZen         = np.deg2rad(sup_zen - inf_zen)/(int_nzen - 1)
            int_dAzZen       = int_dAz*int_dZen
            
            # New integration domain.
            int_Az, int_Zen = np.meshgrid(np.linspace(inf_az,  sup_az,  int_naz),
                                          np.linspace(sup_zen, inf_zen, int_nzen))
            
            # Evaluation points for the interpolation of diffuse spectral radiance.
            eval_pts = np.stack([int_Zen.flatten(), int_Az.flatten()], axis=1)
            
            # We compute part of the integrand ahead of time in order so save
            # computational resources.
            int_Phi, int_Theta = np.deg2rad(int_Az), np.deg2rad(int_Zen)
            x_integrand_add_on_term = np.cos(int_Phi)*np.sin(int_Theta)**2
            y_integrand_add_on_term = np.sin(int_Phi)*np.sin(int_Theta)**2
            z_integrand_add_on_term = np.cos(int_Theta)*np.sin(int_Theta)
            
            
            #  --- (2.C) COMPUTE LOCAL DIFFUSE SPECTRAL IRRADIANCE VECTOR VIA INTEGRATION, WALENGTH BY WAVELENGTH --- 
            for wv in range(len(wavelengths)):
                
                # COMPUTE INTERPOLATED VALUES OF DIFFUSE SPECTRAL RADIANCE FOR INTEGRATION
                local_diffuse_vals_at_wv =\
                interp_diffuse[wv](eval_pts).reshape(int_nzen, int_naz)
                
                # COMPUTE INTEGRAND FOR THE 3 COMPONENTS OF THE DIFFUSE SPECTRAL IRRADIANCE VECTOR
                diffuse_x_integral_at_wv =\
                local_diffuse_vals_at_wv*x_integrand_add_on_term
                
                diffuse_y_integral_at_wv =\
                local_diffuse_vals_at_wv*y_integrand_add_on_term
                
                diffuse_z_integral_at_wv =\
                local_diffuse_vals_at_wv*z_integrand_add_on_term
                
                # COMPUTE INTEGRALS FOR THE 3 COMPONENTS OF THE DIFFUSE SPECTRAL IRRADIANCE VECTOR VIA THE TRAPEZOIDAL RULE.
                diffuse_x_integral_at_wv = 0.25*\
                (diffuse_x_integral_at_wv[:-1, :-1] +
                 diffuse_x_integral_at_wv[:-1,  1:] +
                 diffuse_x_integral_at_wv[1:,  :-1] + 
                 diffuse_x_integral_at_wv[1:,  1:]).sum()*int_dAzZen
                                         
                diffuse_y_integral_at_wv = 0.25*\
                (diffuse_y_integral_at_wv[:-1, :-1] +
                 diffuse_y_integral_at_wv[:-1,  1:] +
                 diffuse_y_integral_at_wv[1:,  :-1] + 
                 diffuse_y_integral_at_wv[1:,  1:]).sum()*int_dAzZen
                
                diffuse_z_integral_at_wv = 0.25*\
                (diffuse_z_integral_at_wv[:-1, :-1] +
                 diffuse_z_integral_at_wv[:-1,  1:] +
                 diffuse_z_integral_at_wv[1:,  :-1] + 
                 diffuse_z_integral_at_wv[1:,  1:]).sum()*int_dAzZen
                
                # TRANSFER COMPUTE VALUES TO THE PREDEFINED-VARIABLE
                local_diffuse_spectral_exposure_vec[0, wv] = diffuse_x_integral_at_wv
                local_diffuse_spectral_exposure_vec[1, wv] = diffuse_y_integral_at_wv
                local_diffuse_spectral_exposure_vec[2, wv] = diffuse_z_integral_at_wv
                
                
            # --- (2.D) COMPUTE  SPECTRAL IRRADIANCE VECTOR MAGNITUDE FOR CURRENT SKY PATCH ---
            # We compute the magnitudes of the computed vector for each wavelength.
            local_diffuse_spectral_exposure_magnitude =\
            np.linalg.norm(local_diffuse_spectral_exposure_vec, axis = 0)
            # (i.e, local_diffuse_time_integrated_spectral_irradiance_magnitude)
            

            # --- (2.E) COMPUTE SPECTRALLY AVERAGED UNIT IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # We average the vector over its wavelengths by integrating it over 
            # its wavelength axis. We then normalize the resulting vector. This 
            # provides us with a vector that points from origin to the "center 
            # of radiation" (so to speak) of the radiation found within the
            # current sky patch.

            # (i.e, local_diffuse_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)
            local_diffuse_spectral_exposure_uvec_spectrally_avgd = \
            simpson(
            y    = local_diffuse_spectral_exposure_vec, 
            x    = wavelengths,
            axis = 1
            )
            
            local_diffuse_spectral_exposure_uvec_spectrally_avgd /=\
            np.linalg.norm(local_diffuse_spectral_exposure_uvec_spectrally_avgd)
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(local_diffuse_spectral_exposure_uvec_spectrally_avgd)):
                local_diffuse_spectral_exposure_uvec_spectrally_avgd = patch_dict["unit_vector"]
            
            
            
            # --------- (3) COMPUTE GLOBAL COMPONENT QUANTITIES --------------
            
            
            # --- (3.A) COMPUTE GLOBAL SPECTRAL IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # Global irradiance is just the sum of the direct and diffuse irradiances. As such, the 
            # global spectral irradiance vector is jsut the sum of the direct and diffuse 
            # spectral irradiance vectors, wavelength by wavelngth.
            
            # (i.e, local_global_time_integrated_spectral_irradiance_vec)
            local_global_spectral_exposure_vec =\
            local_direct_spectral_exposure_vec +\
            local_diffuse_spectral_exposure_vec
            
            # --- (3.B) COMPUTE  SPECTRAL IRRADIANCE VECTOR MAGINTUDE FOR CURRENT SKY PATCH ---

            # (i.e, local_global_time_integrated_spectral_irradiance_magnitude)
            local_global_spectral_exposure_magnitude =\
            np.linalg.norm(local_global_spectral_exposure_vec, axis = 0)
            
            # --- (3.C) COMPUTE SPECTRALLY AVERAGED UNIT IRRADIANCE VECTOR FOR CURRENT SKY PATCH ---
            # We average the vector over its wavelengths by integrating it over 
            # its wavelength axis. We then normalize the resulting vector. This 
            # provides us with a vector that points from origin to the "center 
            # of radiation" (so to speak) of the radiation found within the
            # current sky patch.

            # (i.e, local_global_time_integrated_spectral_irradiance_unit_vec_spectrally_avgd)
            local_global_spectral_exposure_uvec_spectrally_avgd = \
            simpson(
            y    = local_global_spectral_exposure_vec, 
            x    = wavelengths,
            axis = 1
            )
            
            local_global_spectral_exposure_uvec_spectrally_avgd /=\
            np.linalg.norm(local_global_spectral_exposure_uvec_spectrally_avgd)
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(local_global_spectral_exposure_uvec_spectrally_avgd)):
                local_global_spectral_exposure_uvec_spectrally_avgd = patch_dict["unit_vector"]
            
            
            
            # --------- (4) SAVE RESULTS --------------
            
            
            # Save results to each sky patch dict.
            self.patch_data[(zone_num, local_patch_num)]["exposure"] =\
            {
            "spectral_direct"  : {"vector"      : local_direct_spectral_exposure_vec,
                                  "magnitude"   : local_direct_spectral_exposure_magnitude,
                                  "spectrally_averaged_unit_vector" : local_direct_spectral_exposure_uvec_spectrally_avgd
                                 },   
                
            "spectral_diffuse" : {"vector"      : local_diffuse_spectral_exposure_vec,
                                  "magnitude"   : local_diffuse_spectral_exposure_magnitude,
                                  "spectrally_averaged_unit_vector" : local_diffuse_spectral_exposure_uvec_spectrally_avgd
                                 }, 
                
            "spectral_global" : {"vector"      :  local_global_spectral_exposure_vec,
                                 "magnitude"   :  local_global_spectral_exposure_magnitude,
                                 "spectrally_averaged_unit_vector" : local_global_spectral_exposure_uvec_spectrally_avgd
                                },

            "wavelengths" : wavelengths
            }

            
            
            # Save results to independent attribute.
            self.exposure_vectors["spectral_direct"][c,:,:]          = local_direct_spectral_exposure_vec
            self.exposure_vectors["spectral_direct_mag"][c,:]        = local_direct_spectral_exposure_magnitude
            self.exposure_vectors["spectral_direct_unit_savgd"][c,:] = local_direct_spectral_exposure_uvec_spectrally_avgd
            
            self.exposure_vectors["spectral_diffuse"][c,:,:]          = local_diffuse_spectral_exposure_vec
            self.exposure_vectors["spectral_diffuse_mag"][c,:]        = local_diffuse_spectral_exposure_magnitude
            self.exposure_vectors["spectral_diffuse_unit_savgd"][c,:] = local_diffuse_spectral_exposure_uvec_spectrally_avgd
            
            self.exposure_vectors["spectral_global"][c,:,:]           = local_global_spectral_exposure_vec
            self.exposure_vectors["spectral_global_mag"][c,:]         = local_global_spectral_exposure_magnitude
            self.exposure_vectors["spectral_global_unit_savgd"][c,:]  = local_global_spectral_exposure_uvec_spectrally_avgd

        # --- COMPUTE SPECTRALLY-INTEGRATED QUANTITIES --- 
        self._compute_regular_exposure_vectors_for_a_date_interval()
            
        return None        
                
                
            
            
    def _compute_regular_exposure_vectors_for_a_date_interval(self):
        
        """
        Private helper function. Compute the exposure vectors for a date
        interval (i.e, time-integrated irradiance) over each Sky patch (for
        a given date interval) by integrating the (already computed) 
        spectral radiant exposure (i.e, time-integrated spectral irradiance)
        over the wavelength axis.
        
        Parameters
        ----------
        None
            
        Returns
        -------
        None
        
        Produces
        --------
        self.patch_data[(zone_num, local_patch_num)]["exposure"] : dict of dicts
            Each sky patch recieves a new key in its database called 
            "exposure". This is a dict with keys: "direct", "diffuse", "global";
            each holding another dictionary that stores some relevant information
            about the direct, diffuse and global radiant and spectral radiant exposure
            (respectively) related to that particular sky patch.

            Now, for keys "direct", "diffuse", "global",
            each of these dicts contains the following key-value pairs:

                Keys : Values
                -------------
                "vector" : np.array of floats with shape (3,)
                    direct/diffuse/global (depending on the case) 
                    radiant exposure vector. 'vector'[0], 'vector'[1]
                    and 'vector'[2], hold the x, y and z components of the 
                    radiant exposure vector, respectively, for all 
                    wavelengths in key "wavelengths". Each component
                    has units of Wh/m^2.
                    
                "magnitude" : float
                    Magnitude of the direct/diffuse/global (depending on 
                    the case) spectral radiant exposure vector.
                    It has units of Wh/m^2.
                    
                "spectrally_averaged_unit_vector" : np.array of floats with shape (3,)
                    Average position of irradiance within a sky patch. That is,
                    the unit vector version of the spectrally averaged
                    direct/diffuse/global (depending on the case) Radiant exposure vector. 
                    In the case, however, that said Spectral radiant exposure
                    vector is zero, we default to using the unit vector
                    pointing to the center of the current sky patch. 
                    It is adimensional.

                    
       self.exposure_vectors : dict of numpy.arrays
                    Dict containing the same info as above, but in another format
                    that is handier for other things. It has the following
                    key-value pairs:
                        
                    Keys : Values
                    -------------
                    "direct" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                        Direct radiant exposure vector for each of the sky patches. 
                        It has units of Wh/m^2.
        
                    "direct_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                        Magnitude of the direct radiant exposure vector for each of the sky patches. 
                        It has units of Wh/m^2.

                    "direct_unit" : numpy.array of floats with shape (self.num_divisions, 3)
                        Average position of radiant exposure within a sky patch. That is,
                        unit vector version of the spectrally averaged spectral direct
                        (depending on the case) radiant exposure vector. 
                        In the case, however, that said radiant exposure
                        vector is zero, we default to using the unit vector
                        pointing to the center of the current sky patch. 
                        It is adimensional.

                        
                    "diffuse" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                        Diffuse radiant exposure vector for each of the sky patches. 
                        It has units of Wh/m^2.
        
                    "diffuse_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                        Magnitude of the diffuse radiant exposure vector for each of the sky patches. 
                        It has units of Wh/m^2.

                    "diffuse_unit" : numpy.array of floats with shape (self.num_divisions, 3)
                        Average position of radiant exposure within a sky patch. That is,
                        unit vector version of the spectrally averaged spectral diffuse
                        (depending on the case) radiant exposure vector. 
                        In the case, however, that said radiant exposure
                        vector is zero, we default to using the unit vector
                        pointing to the center of the current sky patch. 
                        It is adimensional.

                        
                    "global" : numpy.array of floats with shape (self.num_divisions, 3, 122)
                        Global radiant exposure vector for each of the sky patches. 
                        It has units of Wh/m^2.
        
                    "global_mag" : numpy.array of floats with shape (self.num_divisions, 122)
                        Magnitude of the global radiant exposure vector for each of the sky patches. 
                        It has units of Wh/m^2.

                    "global_unit" : numpy.array of floats with shape (self.num_divisions, 3)
                        Average position of radiant exposure within a sky patch. That is,
                        unit vector version of the spectrally averaged spectral global
                        (depending on the case) radiant exposure vector. 
                        In the case, however, that said radiant exposure
                        vector is zero, we default to using the unit vector
                        pointing to the center of the current sky patch. 
                        It is adimensional.


                    "wavelengths" : np.array of floats with shape (122,)
                        Array of wavelengths over which the spectral irradiances 
                        are defined.
                    
        """
         # ----- INITIALIZE VARIABLES -----
        
        wavelengths = self.exposure_vectors["wavelengths"]
        
        self.exposure_vectors.update(
        {
        "direct"       : np.zeros((self.num_divisions, 3)),
        "diffuse"      : np.zeros((self.num_divisions, 3)),
        "global"       : np.zeros((self.num_divisions, 3)),

        "direct_mag"   : np.zeros(self.num_divisions),
        "diffuse_mag"  : np.zeros(self.num_divisions),
        "global_mag"   : np.zeros(self.num_divisions),

        "direct_unit"  : np.zeros((self.num_divisions, 3)),
        "diffuse_unit" : np.zeros((self.num_divisions, 3)),
        "global_unit"  : np.zeros((self.num_divisions, 3))
        })
        
        
        
        # --- COMPUTE INTEGRAL OVER WAVELENGTHS OF EACH TYPE SPECTRAL IRRADIANCE FOR EACH SKY PATCH ---
        
        for c, ((zone_num, local_patch_num), patch_dict) in enumerate(self.patch_data.items()):
            
            new_direct_vector =\
            simpson(
            y    = patch_dict["exposure"]["spectral_direct"]["vector"], 
            x    = wavelengths,
            axis = 1
            )
            new_direct_magnitude   = np.linalg.norm(new_direct_vector)
            new_direct_unit_vector = new_direct_vector/new_direct_magnitude
            
            
            new_diffuse_vector =\
            simpson(
            y    = patch_dict["exposure"]["spectral_diffuse"]["vector"], 
            x    = wavelengths,
            axis = 1
            )
            new_diffuse_magnitude   = np.linalg.norm(new_diffuse_vector)
            new_diffuse_unit_vector = new_diffuse_vector/new_diffuse_magnitude
            
            
            new_global_vector =\
            simpson(
            y    = patch_dict["exposure"]["spectral_global"]["vector"], 
            x    = wavelengths,
            axis = 1
            )
            new_global_magnitude   = np.linalg.norm(new_global_vector)
            new_global_unit_vector = new_global_vector/new_global_magnitude
            
            
            # However, the normalization fails if there is no radation within 
            # said sky patch, in which case, we default to use a unit vector 
            # pointing to the center of the sky patch, as the "center of radiation".
            # At the end of the day this doesn't matter very much as that sky patch
            # holds no radiation.
            
            if any(np.isnan(new_direct_unit_vector)):
                new_direct_unit_vector = patch_dict["unit_vector"]
            
            if any(np.isnan(new_diffuse_unit_vector)):
                new_diffuse_unit_vector = patch_dict["unit_vector"]
                    
            if any(np.isnan(new_global_unit_vector)):
                new_global_unit_vector = patch_dict["unit_vector"]
            


            # --- SAVE THE RESULTS TO EACH SKY PATCH DATA ---
            self.patch_data[(zone_num, local_patch_num)]["exposure"].update(
            {  "direct"  : {"vector"       : new_direct_vector,
                            "magnitude"    : new_direct_magnitude,
                            "unit_vector"  : new_direct_unit_vector
                            },
             
               "diffuse" : {"vector"      : new_diffuse_vector,
                            "magnitude"   : new_diffuse_magnitude,
                            "unit_vector" : new_diffuse_unit_vector
                           },
              
               "global"  : {"vector"       : new_global_vector,
                            "magnitude"    : new_global_magnitude,
                            "unit_vector"  : new_global_unit_vector
                           }
            })
            
            
            # --- SAVE THE RESULTS TO A NEW ATTRIBUTE ---
            self.exposure_vectors["direct"][c,:]         = new_direct_vector
            self.exposure_vectors["direct_mag"][c]       = new_direct_magnitude
            self.exposure_vectors["direct_unit"][c]      = new_direct_unit_vector
            
            self.exposure_vectors["diffuse"][c,:]         = new_diffuse_vector
            self.exposure_vectors["diffuse_mag"][c]       = new_diffuse_magnitude
            self.exposure_vectors["diffuse_unit"][c]      = new_diffuse_unit_vector
            
            self.exposure_vectors["global"][c,:]          = new_global_vector
            self.exposure_vectors["global_mag"][c]        = new_global_magnitude
            self.exposure_vectors["global_unit"][c]       = new_global_unit_vector

        
        return None
    
    


    def _check_if_unit_vecs_are_within_sky_pacth_bounds(self):
        
        """
        Private helper method. This function goes over each of the defined sky
        pacthes and checks if the spectrally averaged unit vectors for the
        spectral irradiance (which are the same as those for irradiance) lie
        within the sky patch they belong to.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        Produces
        ---------
        self.unit_vecs_within_sky_patch_bounds_check : dict of lists
            Dictiornary whose keys are (zone_num, local_patch_num) for 
            all zone_nums, patch_nums containing lists of three bool
            elements. if the first element is true, that means that 
            the spectrally averaged unit vector of direct spectral irradiance
            does fall within the sky patch bounds for that sky patch. If not,
            that means it does not. The other 2 elements do the same but for
            the diffuse and global spectral irradiances.
            
            
        Notes
        -----
        1) This method requires that time-integrated spectral irradiances already 
           be calculated. Check the method 
           "self.compute_time_integrated_spectral_irradiances_for_a_date_interval"
           for more info.
            
        """
    
        self.unit_vecs_within_sky_patch_bounds_check = {}
        
        
        for (zone_num, local_patch_num), patch_dict in self.patch_data.items():
            self.unit_vecs_within_sky_patch_bounds_check[(zone_num, local_patch_num)] = []
            
            for name in ["direct", "diffuse", "global"]:
                unit_vec = patch_dict["exposure"]\
                                     [f"spectral_{name}"]["spectrally_averaged_unit_vector"]
                                     
                zenith  = np.rad2deg(np.arccos(unit_vec[2]))
                azimuth = np.rad2deg(np.arctan2(unit_vec[1], unit_vec[0]))
                
                if azimuth < 0:
                    azimuth += 360
                    
                if patch_dict["sup_zen"] == 90:
                    zenith_within_bounds =\
                    patch_dict["inf_zen"] <= zenith <= patch_dict["sup_zen"]
                    
                else: 
                    zenith_within_bounds =\
                    patch_dict["inf_zen"] <= zenith < patch_dict["sup_zen"]
                    
                                                 
                if patch_dict["sup_az"] == 360:
                    azimuth_within_bounds =\
                    patch_dict["inf_az"] <= azimuth <= patch_dict["sup_az"]
                    
                else: 
                    azimuth_within_bounds =\
                    patch_dict["inf_az"] <= azimuth < patch_dict["sup_az"]                
                    
                    
                self.unit_vecs_within_sky_patch_bounds_check[(zone_num, local_patch_num)].\
                append(zenith_within_bounds and azimuth_within_bounds)
            
        return None    
    







    def plot_exposures(self, config = None):
        
        """
        Plot radiant exposures.
        
        Parameters
        ----------
        config : dict or None
            Dict of plot configuration options. If None (the default), it uses
            the default confifuration plot options. If dict, it should include
            one or more of the following key-value pairs:
                
                Keys : Values
                -------------
                "projection" : {'disk', 'sphere'}, optional
                    Type of plot projection. Supported are: "disk" The
                    plots the radiant exposure in a 2D plot, 
                    while the second uses a 3D plot. Default is "disk".
                    
                "mode" : {'direct', 'diffuse', 'global'}, optional
                    Component of radiant exposure to plot. Default is 'global'.
                    
                "figsize" : 2-tuple of int
                    Figure size. Default is (13,13).
                
                "unit" : {"Wh/m^2", "kWh/m^2", "kJ/m^2", "MJ/m^2}
                    Units with which to display the radiant exposure.
                    In order, these mean: 'Watt-hours per meter squared',
                    'kilo Watt-hours per meter squared', 'kilo Joules per meter squared',
                    and 'Mega Joules per meter squared'. Default is "Wh/m^2".
                    
                "n" : int
                    Number of samples per axis to use for plot. A greater number 
                    means a more detailed plot (i.e, greater resolution) but it is 
                    resource intensive. Default is 1000.
                    
                "view" : 2-tuple of int
                    Elevation, azimuth of plot camara in degrees. It applies
                    only for "sphere" plot. Default is (45, 120).
                
        Returns
        -------
        None
        
        Produces
        --------
        None
        
        Notes
        -----
        1) This method requires the radiant exposure vectors
        to be calculated. Check out :meth:`~solrad.Sky.Sky.compute_exposure_vectors_for_a_date_interval`
           
        """
        
        config_ =\
        {"projection":"disk", "mode":"global", "figsize":(13,13),
         "unit":"Wh/m^2", "n":1000, "view":(45, 120)}
        
        # User-defined configuration overwrites default one.
        if(isinstance(config, dict)):
            for key, val in config.items():
                config_[key] = val
                
                
        # Sample points to plot in "disk".
        if config_["projection"] == "disk":
            Phi, R = np.meshgrid(np.linspace(0, 360, config_["n"]), 
                                 np.linspace(0,   1, config_["n"]))
            
            zone_nums, patch_nums =\
            self.disk_points_to_zones_patches(R.flatten(), Phi.flatten())
            
            Phi = np.deg2rad(Phi)
            
        # Sample points to plot in "sphere".   
        elif config_["projection"] == "sphere":
            Phi, Theta = np.meshgrid(np.linspace(0, 360, config_["n"]), 
                                     np.linspace(90,  0, config_["n"]))
            
            zone_nums, patch_nums =\
            self.sky_points_to_zones_patches(Theta.flatten(), Phi.flatten())
            
            Phi, Theta = np.deg2rad(Phi), np.deg2rad(Theta)
            
            X = np.cos(Phi)*np.sin(Theta)
            Y = np.sin(Phi)*np.sin(Theta)
            Z = np.cos(Theta)
            

            
        zone_nums  =  zone_nums.reshape(config_["n"],  config_["n"])
        patch_nums = patch_nums.reshape(config_["n"], config_["n"])
        
        
        # --- RETRIEVE DATA FOR EACH SAMPLE DATA ---
        
        Color = np.zeros((config_["n"], config_["n"]))
        
        for i in range(config_["n"]):
            for j in range(config_["n"]):
                zone_num, patch_num = zone_nums[i,j], patch_nums[i,j]
                
                Color[i,j] =\
                self.patch_data[(zone_num, patch_num)]\
                ["exposure"][config_["mode"]]["magnitude"]
                            
        
            
        # --- ACCOMODATE DATA TO SELECTED UNIT ---    
            
        if config_["unit"] == "kWh/m^2":
            Color /= 1000
            
        elif config_["unit"] == "kJ/m^2":
            Color *= 3.6 
            
        elif config_["unit"] == "MJ/m^2":
            Color *= 3.6/1000
            
        Color = Color.reshape(config_["n"], config_["n"])
        
        
        # --- COMPUTE TITLE ---  
        
        title = "Radiant exposure contributed by Sky-Patch "
        
        if config_["mode"]=="direct":
            cbar_title = f"Direct [{config_['unit']}]"
        
        elif config_["mode"]=="diffuse":
            cbar_title = f"Diffuse [{config_['unit']}]"
            
        elif config_["mode"]=="global":
            cbar_title = f"Global [{config_['unit']}]"
            
            
            
        # --- GET INITIAL AND FINAL DATE --- 
        start_date = self.time_integrated_spectral_radiance["start_date"]
        end_date = self.time_integrated_spectral_radiance["end_date"]
        title = f"{title} | From {start_date} to {end_date}."
        
            
        # --- PLOT DISK DATA --- s
        if config_["projection"] == "disk":
            
            fig, ax = plt.subplots(figsize=config_["figsize"], 
            subplot_kw={'projection': 'polar'})
            ax.grid(False)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.pcolormesh(Phi, R, Color, cmap="hot")
            ax.set_xlabel("N = 0°, E = 90°, S = 180°, W = 270°")

        

        # --- PLOT SPHERE DATA --- 
        elif config_["projection"] == "sphere":
            
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                   figsize=config_["figsize"])
            
            ax.view_init(config_["view"][0], config_["view"][1])
            ax.set_xlabel("X (↑ == N, ↓ == S)")
            ax.set_ylabel("Y (↑ == E, ↓ == W)")
            ax.set_title(title)
            
            if(np.max(Color)>1):
                Color_ = Color/np.max(Color)
            else:
                Color_ = Color
            
            ax.plot_surface(X, Y, Z, cmap="hot", facecolors = plt.cm.hot(Color_))
            
        m = plt.cm.ScalarMappable(cmap=plt.cm.hot)
        m.set_array(Color)
        cbar = plt.colorbar(m, ax=ax)
        cbar.ax.set_title(cbar_title)
        ax.set_title(title)
        plt.show()
        
        
        return None
            
            

    def compute_absorbed_energy_by_unit_plane(self, n_uvec, absorption_func = 1, component = "global"):

        """
        Compute the total energy absorbed by a unit plane from the sky for a specified component of radiation.

        Parameters
        ----------
        n_uvec : numpy.array of float with shape (3,)
            The normal unit vector of the unit plane.

        absorption_func : int, float, or callable, optional
            The absorption function. If int or float, a constant absorption coefficient is used.
            If callable, the function should take an array of arguments representing angle of incidence
            and wavelength and return the absorption coefficient. Default is 1.

        component : {"global", "direct", "diffuse"}, optional
            The spectral component to consider. Default is "global".

        Returns
        -------
        total_absorbed_incident_energy : float
            The total absorbed incident energy by the unit plane.

        Raises
        ------
        ValueError
            If the `absorption_func` is not a valid type.

        Notes
        -----
        This method computes the total absorbed energy by a unit plane for a specified spectral component.
        The computation considers the spectral exposure vectors and absorption function. The total absorbed
        energy is returned as a scalar value.

        """
        
        total_absorbed_incident_energy = 0
        wavelengths  = self.exposure_vectors["wavelengths"]

        for key in self.patch_data.keys():
            spectral_exposure_vec  = self.patch_data[key]["exposure"][f"spectral_{component}"]["vector"]
            spectral_exposure_uvec = self.patch_data[key]["exposure"][f"spectral_{component}"]["spectrally_averaged_unit_vector"]
            cos_aoi = np.dot(n_uvec, spectral_exposure_uvec)

            if cos_aoi > 0:
                incident_spectral_energy =\
                (1)*n_uvec[0]*spectral_exposure_vec[0, :] +\
                (1)*n_uvec[1]*spectral_exposure_vec[1, :] +\
                (1)*n_uvec[2]*spectral_exposure_vec[2, :] 
                # We multiply by one to remind us that we are ealing with a unit area plane. 

                if isinstance(absorption_func, int) or isinstance(absorption_func, float):
                    absorbed_incident_energy =\
                    simpson(y = absorption_func*incident_spectral_energy, x = wavelengths)

                elif callable(absorption_func):
                    aoi = np.rad2deg(np.arccos(cos_aoi))
                    aoi = np.full(len(wavelengths), aoi)
                    arguments = np.stack([aoi,wavelengths], axis=1)

                    absorbed_incident_energy =\
                    simpson(y = absorption_func(arguments)*incident_spectral_energy, x = wavelengths)
                
                else:
                    msg = "Invalid value for absorption_func."
                    raise ValueError(msg)
            else:
                absorbed_incident_energy = 0

                
            total_absorbed_incident_energy += absorbed_incident_energy

        return total_absorbed_incident_energy 
    

    def compute_optimal_plane_orientation(self, min_res = 0.5, naz = 13, nel = 4, absorption_func = 1, component = "global"):

        """
        Compute the optimal orientation of a plane for maximum absorbed energy.

        Parameters
        ----------
        min_res : float, optional
            Minimum angular resolution (in degrees) that wants to be achieved during
            the optimization process. Default is 0.5.

        naz : int, optional
            Number of azimuthal divisions for each iteration. Default is 13.

        nel : int, optional
            Number of elevation divisions. Default is 4.

        absorption_func : int, float, or callable, optional
            The absorption function. If int or float, a constant absorption coefficient is used.
            If callable, the function should take an array of arguments representing angle of incidence
            and wavelength and return the absorption coefficient. Default is 1.
            
        component : {"global", "direct", "diffuse"}, optional
            The spectral component to consider. Default is "global".

        Returns
        -------
        opti_vals : dict
            A dictionary containing the optimal orientation information:
                
                - "energy": Total absorbed incident energy.
                - "az": Optimal azimuth angle in degrees.
                - "zen": Optimal zenith angle in degrees.
                - "el": Optimal elevation angle in degrees.
                - "n_uvec": Normalized unit vector of the optimal orientation.
                - "az_res": Azimuth resolution in degrees.
                - "el_res": Elevation resolution in degrees.

        Notes
        -----
        This method iteratively searches for the optimal plane orientation by dividing the
        azimuth and zenith angles into divisions and computing the absorbed energy for each
        combination. The resolution is refined until it reaches the specified minimum.

        """

        az_lims  = [0, 2*np.pi]
        zen_lims = [0, np.pi/2]
        opti_vals = {"energy":-1, "az":None, "zen":None}

        az_res  = (az_lims[1]  - az_lims[0])/(naz-1)
        zen_res = (zen_lims[1] - zen_lims[0])/(nel-1)
        worst_res = max(az_res, zen_res)

        while worst_res > np.deg2rad(min_res):

            azimuths = np.linspace(az_lims[0],  az_lims[1],  naz)
            zeniths  = np.linspace(zen_lims[0], zen_lims[1], nel)

            for zen in zeniths:
                for az in azimuths:
                    n_uvec = [np.cos(az)*np.sin(zen),
                              np.sin(az)*np.sin(zen),
                              np.cos(zen)]
                
                    total_absorbed_incident_energy =\
                    self.compute_absorbed_energy_by_unit_plane(n_uvec, absorption_func, component)

                    if total_absorbed_incident_energy > opti_vals["energy"]:
                        opti_vals["energy"] = total_absorbed_incident_energy
                        opti_vals["zen"] = zen
                        opti_vals["az"] = az

            az_lims[0] = opti_vals["az"] - az_res
            az_lims[1] = opti_vals["az"] + az_res
            zen_lims[0] = opti_vals["zen"] - zen_res
            zen_lims[1] = opti_vals["zen"] + zen_res
            az_res  = (az_lims[1]  - az_lims[0])/(naz-1)
            zen_res = (zen_lims[1] - zen_lims[0])/(nel-1)
            worst_res = max(az_res, zen_res)


        opti_vals["n_uvec"] =\
        [np.cos(opti_vals["az"])*np.sin(opti_vals["zen"]),
         np.sin(opti_vals["az"])*np.sin(opti_vals["zen"]),
         np.cos(opti_vals["zen"])]
          
        opti_vals["az"]  = np.rad2deg(opti_vals["az"])
        opti_vals["zen"] = np.rad2deg(opti_vals["zen"])
        opti_vals["el"]  = 90 - opti_vals["zen"]
        opti_vals["az_res"]  = np.rad2deg(az_res)
        opti_vals["el_res"]  = np.rad2deg(zen_res)
     
        return opti_vals


        


