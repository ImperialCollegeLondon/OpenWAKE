from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from flow_field_model.flow import Flow
import numpy as np

class Larsen(BaseWake):
    """
    Defines the Larsen wake model
    """

    def __init__(self, turbine, flow, wake_decay, ambient_intensity):
        self.set_wake_decay(wake_decay)
        super(Larsen, self).__init__(turbine, flow)
        self.set_ambient_intensity(ambient_intensity)
        self.set_wake_decay(wake_decay)

    def calc_r_nb(self):
        """
        Calculates R_nb, an empirical expression related to the ambient turbulence
        """
        k1, k2, k3 = 1.08, 21.7, 0.05
        turbine_diameter = 2 * self.get_turbine().get_radius()
        ambient_intensity = self.get_ambient_intensity()
        val1 = k1 * turbine_diameter
        val2 = val1 + ((k2 * turbine_diameter) * (ambient_intensity - k3))
        return max([val1, val2])
    
    def calc_r_9_point_5(self):
        """
        Calculates the wake radius at a distance of 9.5 rotor diameters
        downstream of the turbine.
        The blockage effect of the ground is also taken into account in this
        wake model; once the wake width exceeds the hub height as it expands
        along the downstream direction, it starts interacting with the ground.
        """
        k1 = 0.5
        hub_height = self.get_turbine().get_coords()[2]
        r_nb = self.calc_r_nb()
        return k1*(r_nb + min([hub_height, r_nb]))

    def calc_x0(self):
        """
        Calculates the position of the rotor w.r.t the applied
        coordinate system
        """
        diameter_9_point_5 = 2 * self.calc_r_9_point_5()
        turbine_diameter = 2 * self.get_turbine().get_radius()
        effective_diameter = self.calc_effective_diameter()
        k1, k2 = 9.5, 3
        return ((k1*turbine_diameter)/
                ((diameter_9_point_5/effective_diameter)**k2 - 1))

    def calc_effective_diameter(self):
        """
        Calculates effective rotor diameter
        param u is hub-height axial flow speed
        """
        turbine = self.get_turbine()
        turbine_diameter = 2 * turbine.get_radius()
        u = self.get_flow().get_flow_at_point(turbine.get_coords())
        thrust_coefficient = turbine.calc_thrust_coefficient(u)
        return turbine_diameter * ((1+(1-thrust_coefficient)**0.5)/(2*(1-thrust_coefficient)**0.5))**0.5
    
    def calc_prandtl_mixing(self):
        """
        Calculates the prandtl mixing length, c1
        """
        turbine = self.get_turbine()
        effective_radius = self.calc_effective_diameter()/2
        turbine_radius = turbine.get_radius()
        rotor_disc_area = turbine.calc_area()

        turbine = self.get_turbine()
        flow_at_point = self.get_flow().get_flow_at_point(turbine.get_coords())
        thrust_coefficient = turbine.calc_thrust_coefficient(flow_at_point)
        
        x0 = self.calc_x0()
        k1, k2, k3, k4 = 5/2, 105, -1/2, -5/6
        return (effective_radius**k1) * ((k2/(2*np.pi))**k3) * ((thrust_coefficient * rotor_disc_area * x0)**k4)
 
    def get_wake_decay(self):
        return self.wake_decay

    def get_ambient_intensity(self):
        return self.ambient_intensity
    
    def set_wake_decay(self, wake_decay):
        default_wake_decay = 0.03
        if not (isinstance(wake_decay, float) or wake_decay == None):
            raise TypeError("'wake_decay' must be of type 'float'")
        else:
            self.wake_decay = wake_decay if wake_decay != None else default_wake_decay    

    def set_ambient_intensity(self, ambient_intensity):
        default_ambient_intensity = 0.1
        if not (isinstance(ambient_intensity, float) or ambient_intensity == None):
            raise TypeError("'ambient_intensity' must be of type 'float'")
        else:
            self.ambient_intensity = ambient_intensity if ambient_intensity != None else default_ambient_intensity

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
        x0, x1, x2 = rel_pnt_coords, np.array([0,0,0]), np.array([1,0,0])
        dist = np.linalg.norm(np.cross((x0-x1),(x0-x2)),2)/np.linalg.norm(x2-x1,2)
        return dist
    
    def calc_wake_radius(self, pnt_coords):
        """
        Returns the wake radius at point given that point is in the wake of
        turbine
        """
        dist = self.calc_distance_along_flow(pnt_coords)
        turbine = self.get_turbine()
        flow_at_point = self.get_flow().get_flow_at_point(pnt_coords)
        thrust_coefficient = turbine.calc_thrust_coefficient(flow_at_point)
        rotor_disc_area = turbine.calc_area()
        prandtl_mixing = self.calc_prandtl_mixing()
        x0 = self.calc_x0()
        k1,k2,k3,k4 = 35,(1/5),3,(1/3)
        return (k1/(2*np.pi))**k2 * (k3*(prandtl_mixing**2))**k2 * (thrust_coefficient * rotor_disc_area * (dist + x0))**k4

    def calc_vrf_at_point(self, undisturbed_flow_at_point, pnt_coords):
        """
        Returns the individual velocity reduction factor
        """
        x = self.calc_distance_along_flow(pnt_coords) 
        r = self.calc_dist_from_wake_centre(pnt_coords)
        xx0 = x + self.calc_x0()
        prandtl_mixing = self.calc_prandtl_mixing()
        rotor_disc_area = self.get_turbine().calc_area()
        
        turbine = self.get_turbine()
        flow_at_point = self.get_flow().get_flow_at_point(pnt_coords)
        thrust_coefficient = turbine.calc_thrust_coefficient(flow_at_point)

        k1, k2, k3, k4, k5, k6, k7, k8 = 1/3, 3/2, 3, -1/2, 3/10, -1/5, 17.5, 1/9
        
        bracket1 = thrust_coefficient * rotor_disc_area / xx0**2
        bracket2 = k3 * prandtl_mixing**2 * thrust_coefficient * rotor_disc_area * xx0
        bracket3 = k7 / np.pi
        bracket4 = k3 * prandtl_mixing**2

        vrf = k8 * (bracket1**k1) * (r**k2 * bracket2**k4 - (bracket3**k5)*(bracket4**k6))**2
        return vrf

    def calc_search_radius(self, recovery_loss=2.5):
        """
        Returns the search radius for an acceptable recovery loss
        """
        turbine = self.get_turbine()
        flow_at_point = self.get_flow().get_flow_at_point(turbine.get_coords())
        thrust_coefficient = turbine.calc_thrust_coefficient(flow_at_point)
        rotor_disc_area = self.get_turbine().calc_area()
        prandtl_mixing = self.calc_prandtl_mixing()
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
        
