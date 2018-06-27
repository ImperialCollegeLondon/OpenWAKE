from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from wake_models.wake_field_model import WakeField
from flow_field_model.flow import FlowField
import numpy as np
from helpers import *

class Larsen(BaseWake):
    """
    Defines the Larsen wake model
    https://www.researchgate.net/profile/Jie_Zhang37/publication/265844008_Impact_of_Different_Wake_Models_On_the_Estimation_of_Wind_Farm_Power_Generation/links/561703e208ae40a7199a52ac/Impact-of-Different-Wake-Models-On-the-Estimation-of-Wind-Farm-Power-Generation.pdf
    """

    def __init__(self, turbine = BaseTurbine(), flow_field = FlowField(), wake_decay = 0.3, ambient_intensity = 0.1, wake_field = WakeField()):
        self.set_wake_decay(wake_decay)
        self.set_ambient_intensity(ambient_intensity)
        self.set_wake_decay(wake_decay)
        super(Larsen, self).__init__(turbine, flow_field, wake_field)

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

        effective_radius = self.calc_effective_diameter(thrust_coefficient, turbine_diameter) / 2
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
    
    def calc_wake_radius(self, pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """

        rotor_disc_area = np.pi * turbine_radius**2
        turbine_diameter = 2 * turbine_radius
        hub_height = turbine_coords[2]
        ambient_intensity = self.get_ambient_intensity()
        
        x_rel = relative_position(turbine_coords, pnt_coords, flow_field)[0]
        prandtl_mixing = self.calc_prandtl_mixing(thrust_coefficient, turbine_radius, rotor_disc_area, hub_height, ambient_intensity)
        x0 = self.calc_x0(thrust_coefficient, turbine_diameter, hub_height, ambient_intensity)

        k1, k2, k3, k4, k5 = 35/2, 1/5, 3, 1/3, 2
        
        return (k1 / np.pi)**k2 * (k3 * prandtl_mixing**k5)**k2 * (thrust_coefficient * rotor_disc_area * (x_rel + x0))**k4

    def calc_vrf_at_point(self, pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient, u_0):
        if self.is_in_wake(pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
            turbine_diameter = 2 * turbine_radius
            wake_decay = self.get_wake_decay()
            hub_height = turbine_coords[2]
            x_rel, y_rel, z_rel = relative_position(turbine_coords, pnt_coords, flow_field)
            
            ambient_intensity = self.get_ambient_intensity()
            rotor_disc_area = np.pi * turbine_radius**2
            
            prandtl_mixing = self.calc_prandtl_mixing(thrust_coefficient, turbine_diameter, rotor_disc_area, hub_height, ambient_intensity)
            x0 = self.calc_x0(thrust_coefficient, turbine_diameter, hub_height, ambient_intensity)

            k1, k2, k3, k4, k5, k6, k7, k8, k9 = 1/3, 3/2, 3, -1/2, 3/10, -1/5, 17.5, 1/9, 2
            
            bracket1 = thrust_coefficient * rotor_disc_area / (x_rel + x0)**k9
            bracket2 = k3 * prandtl_mixing**k9 * thrust_coefficient * rotor_disc_area * (x_rel + x0)
            bracket3 = k7 / np.pi
            bracket4 = k3 * prandtl_mixing**k9
            with np.errstate(all="ignore"):
                velocity_reduction_factor = k8 * (bracket1**k1) * ((abs(z_rel)**k2 * bracket2**k4) - (bracket3**k5 * bracket4**k6))**k9
            #velocity_reduction_factor = velocity_deficit/u_0# if np.isnan(velocity_deficit) == False and np.isinf(velocity_deficit) == False else 0
            return velocity_reduction_factor

        else:
            return 0

    def calc_search_radius(self, turbine_coords, flow_field, turbine_radius, thrust_coefficient, recovery_loss=2.5):
        """
        Returns the search radius for an acceptable recovery loss
        """
        turbine_diameter = 2  * turbine_radius
        rotor_disc_area = np.pi * turbine_radius**2
        hub_height = turbine_coords[2]
        ambient_intensity = self.get_ambient_intensity()
        prandtl_mixing = self.calc_prandtl_mixing(thrust_coefficient, turbine_diameter, rotor_disc_area, hub_height, ambient_intensity)
        
        k1, k2, k3, k4, k5, k6 = 100, 17.5, 3/10, -1/5, 9, 3
        # check if recovery_loss is zero
        if (recovery_loss < 1e-8):
            return np.inf
        else:
            recovery_loss /= k1
            alpha = thrust_coefficient * rotor_disc_area
            beta0 = -(k2 / np.pi)**k3
            beta1 = k6 * (prandtl_mixing**2)**k4
            beta = (beta0 * beta1)**2
            return (alpha / ((k5 * (1 - recovery_loss)) / beta)**k6)**0.5
        
