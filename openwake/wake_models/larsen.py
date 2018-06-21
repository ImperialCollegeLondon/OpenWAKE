from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from flow_field_model.flow import Flow
import numpy as np
from helpers import *

class Larsen(BaseWake):
    """
    Defines the Larsen wake model
    https://www.researchgate.net/profile/Jie_Zhang37/publication/265844008_Impact_of_Different_Wake_Models_On_the_Estimation_of_Wind_Farm_Power_Generation/links/561703e208ae40a7199a52ac/Impact-of-Different-Wake-Models-On-the-Estimation-of-Wind-Farm-Power-Generation.pdf
    """

    def __init__(self, turbine = BaseTurbine(), flow= Flow(), wake_decay = 0.3, ambient_intensity = 0.1):
        self.set_wake_decay(wake_decay)
        super(Larsen, self).__init__(turbine, flow)
        self.set_ambient_intensity(ambient_intensity)
        self.set_wake_decay(wake_decay)

    def calc_r_nb(self, turbine_diameter, ambient_intensity):
        """
        Calculates R_nb, an empirical expression related to the ambient turbulence
        """
        k1, k2, k3 = 1.08, 21.7, 0.05
        val1 = k1 * turbine_diameter
        val2 = val1 + ((k2 * turbine_diameter) * (ambient_intensity - k3))
        return np.max([val1, val2])
    
    def calc_r_9_point_5(self, turbine_diameter, ambient_intensity, hub_height):
        """
        Calculates the wake radius at a distance of 9.5 rotor diameters
        downstream of the turbine.
        The blockage effect of the ground is also taken into account in this
        wake model; once the wake width exceeds the hub height as it expands
        along the downstream direction, it starts interacting with the ground.
        """
        k1 = 0.5
        r_nb = self.calc_r_nb(turbine_diameter, ambient_intensity)
        return k1*(r_nb + min([hub_height, r_nb]))

    def calc_x0(self, thrust_coefficient, turbine_diameter, hub_height, ambient_intensity):
        """
        Calculates the position of the rotor w.r.t the applied
        coordinate system
        """
        diameter_9_point_5 = 2 * self.calc_r_9_point_5(turbine_diameter, ambient_intensity, hub_height)
        effective_diameter = self.calc_effective_diameter(thrust_coefficient, turbine_diameter)
        k1, k2 = 9.5, 3
        return (k1*turbine_diameter)/((diameter_9_point_5/effective_diameter)**k2 - 1)

    def calc_effective_diameter(self, thrust_coefficient, turbine_diameter):
        """
        Calculates effective rotor diameter
        """
        k1, k2 = 0.5, 2
        
        return turbine_diameter * ((1 + (1 - thrust_coefficient)**k1)/(k2 * (1 - thrust_coefficient)**k1))**k1
    
    def calc_prandtl_mixing(self, thrust_coefficient, turbine_diameter, rotor_disc_area, hub_height, ambient_intensity):
        """
        Calculates the prandtl non-dimensional mixing length, c1
        """
        effective_radius = self.calc_effective_diameter(thrust_coefficient, turbine_diameter)/2
        x0 = self.calc_x0(thrust_coefficient, turbine_diameter, hub_height, ambient_intensity)
        
        k1, k2, k3, k4 = 5/2, 105/2, -1/2, -5/6
        
        return (effective_radius**k1) * ((k2/np.pi)**k3) * ((thrust_coefficient * rotor_disc_area * x0)**k4)
 
    def get_wake_decay(self):
        return self.wake_decay

    def get_ambient_intensity(self):
        return self.ambient_intensity
    
    def set_wake_decay(self, wake_decay = 0.3):
        try:
            assert isinstance(wake_decay, float)
        except AssertionError:
            raise TypeError("'wake_decay' must be of type 'float'")
        else:
            self.wake_decay = wake_decay    

    def set_ambient_intensity(self, ambient_intensity = 0.1):
        try:
            assert isinstance(ambient_intensity, float)
        except AssertionError:
            raise TypeError("'ambient_intensity' must be of type 'float'")
        else:
            self.ambient_intensity = ambient_intensity

    def calc_distance_along_flow(self, pnt_coords):
        """
        Returns distance between turbine and point, parallel to the direction of
        flow at turbine
        """
        rel_pnt_coords = self.relative_position(pnt_coords)
        #hyp = np.linalg.norm(rel_pnt_coords,2) - TODO won't work because of z_coord
        #opp = self.calc_dist_from_wake_centre(rel_pnt_coords)

        # by Pythagoras' Thm
        #adj = (hyp**2 - opp**2)**0.5

        #return adj
        # return x_coordinate of relative position of point to turbine
        return rel_pnt_coords[0]

    def calc_dist_from_wake_centre(self, pnt_coords):
        """
        Returns the distance of point from the centerline of the wake generated
        by turbine
        """
        
        rel_pnt_coords = self.relative_position(pnt_coords)
        
        # distance from point to line formula in 3d
        # centerline can now be assumed to be colinear with target vector (1,0,0)
        # point vector, first line vector, second line vector
        #x0, x1, x2 = rel_pnt_coords, np.array([0,0,0]), np.array([1,0,0])
        #dist = np.linalg.norm(np.cross((x0-x1),(x0-x2)),2)/np.linalg.norm(x2-x1,2)
        return rel_pnt_coords[2]
    
    def calc_wake_radius(self, pnt_coords):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """

        turbine = self.get_turbine()
        turbine_diameter = 2 * turbine.get_radius()
        u_0 = self.get_flow_mag_at_turbine()
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        rotor_disc_area = turbine.calc_area()
        hub_height = turbine.get_coords()[2]
        ambient_intensity = self.get_ambient_intensity()
        
        x = self.calc_distance_along_flow(pnt_coords)
        prandtl_mixing = self.calc_prandtl_mixing(thrust_coefficient, turbine_diameter, rotor_disc_area, hub_height, ambient_intensity)
        x0 = self.calc_x0(thrust_coefficient, turbine_diameter, hub_height, ambient_intensity)

        k1, k2, k3, k4, k5 = 35/2, 1/5, 3, 1/3, 2
        
        return (k1 / np.pi)**k2 * (k3 * prandtl_mixing**k5)**k2 * (thrust_coefficient * rotor_disc_area * (x + x0))**k4

    def calc_vrf_at_point(self, pnt_coords):
        """
        Returns the individual velocity reduction factor
        """
        # check if point is in wake caused by turbine
        if self.is_in_wake(pnt_coords):
            #x = self.calc_distance_along_flow(pnt_coords) 
            #r = self.calc_dist_from_wake_centre(pnt_coords)
            
            rel_pnt_coords = self.relative_position(pnt_coords)
            rel_x_coord, rel_r_coord = rel_pnt_coords[0], rel_pnt_coords[2]
            
            turbine = self.get_turbine()
            hub_height = turbine.get_coords()[2]
            ambient_intensity = self.get_ambient_intensity()
            turbine_diameter = 2 * turbine.get_radius()
            u_0 = self.get_flow_mag_at_turbine()
            thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
            rotor_disc_area = turbine.calc_area()
            
            prandtl_mixing = self.calc_prandtl_mixing(thrust_coefficient, turbine_diameter, rotor_disc_area, hub_height, ambient_intensity)
            x0 = self.calc_x0(thrust_coefficient, turbine_diameter, hub_height, ambient_intensity)

            k1, k2, k3, k4, k5, k6, k7, k8, k9 = 1/3, 3/2, 3, -1/2, 3/10, -1/5, 17.5, 1/9, 2
            
            bracket1 = thrust_coefficient * rotor_disc_area / (rel_x_coord+x0)**k9
            bracket2 = k3 * prandtl_mixing**k9 * thrust_coefficient * rotor_disc_area * (rel_x_coord+x0)
            bracket3 = k7 / np.pi
            bracket4 = k3 * prandtl_mixing**k9
            with np.errstate(all="ignore"):
                velocity_deficit = u_0 * k8 * (bracket1**k1) * ((abs(rel_r_coord)**k2 * bracket2**k4) - (bracket3**k5 * bracket4**k6))**k9
    
            velocity_reduction_factor = velocity_deficit/u_0# if np.isnan(velocity_deficit) == False and np.isinf(velocity_deficit) == False else 0
            return velocity_reduction_factor
        else:
            return 0

    def calc_search_radius(self, recovery_loss=2.5):
        """
        Returns the search radius for an acceptable recovery loss
        """
        turbine = self.get_turbine()
        turbine_diameter = 2* turbine.get_radius()
        u_0 = self.get_flow_mag_at_turbine()
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        rotor_disc_area = turbine.calc_area()
        hub_height = turbine.get_coords()[2]
        ambient_intensity = self.get_ambient_intensity()
        prandtl_mixing = self.calc_prandtl_mixing(thrust_coefficient, turbine_diameter, rotor_disc_area, hub_height, ambient_intensity)
        
        k1, k2, k3, k4, k5, k6 = 100, 17.5, 3/10, -1/5, 9, 3
        # check if recovery_loss is zero
        if (recovery_loss < 1e-8):
            return np.inf
        else:
            recovery_loss /= k1
            alpha = thrust_coefficient * rotor_disc_area
            beta0 = -(k2/np.pi)**k3
            beta1 = k6 * (prandtl_mixing**2)**k4
            beta = (beta0 * beta1)**2
            return (alpha / ((k5 * (1-recovery_loss)) / beta)**k6)**0.5
        
