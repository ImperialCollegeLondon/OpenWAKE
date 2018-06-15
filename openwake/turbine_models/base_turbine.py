"""
A base single turbine model from which others can be derived.
"""

import numpy as np

class BaseTurbine(object):
    """ Implements base turbine class."""

    def __init__(self, radius, coords, top_clearance, direction, thrust_coefficient_curve, power_coefficient_curve):
        """ 
        param radius is radius of turbine blades param hub_height is distance from floor to hub
        param coords is a list of x,y,z coordinates, where z is the distance from sea level in the
        case of an MCT, or None in the case of a wind-turbine
        param thrust_coefficient_curve is a 2D list of flow speeeds and corresponding
        thrust coefficient values
        param power_coefficient_curve = is a 2D list of flow speeds and corresponding power coefficient values param control is a
        boolean which indicates whether or not control is implemented in the turbine 
        """

        self.set_radius(radius)
        self.set_coords(coords)
        self.set_top_clearance(top_clearance)
        self.set_direction(direction)
        self.set_thrust_coefficient_curve(thrust_coefficient_curve)
        self.set_power_coefficient_curve(power_coefficient_curve)
        
    def get_radius(self):
        return self.radius
    
    def get_coords(self):
        return self.coords

    def get_top_clearance(self):
        return self.top_clearance

    def get_direction(self):
        return self.direction

    def get_thrust_coefficient_curve(self):
        return self.thrust_coefficient_curve

    def get_power_coefficient_curve(self):
        return self.power_coefficient_curve

    def calc_thrust_coefficient(self, flow_at_turbine):

        #turbine_direction = self.get_direction()
        turbine_coords = self.get_coords()
        #normalised_turbine_direction = turbine_direction/np.linalg.norm(turbine_direction,2)
        #u = np.dot(flow_at_turbine, normalised_turbine_direction)

        u = np.linalg.norm(flow_at_turbine,2)
        
        curve = self.get_thrust_coefficient_curve()
        # flow speed data points
        xp = curve[0]

        # thrust coefficient data points
        fp = curve[1]
        
        # check if xp is always increasing and lengths are equal
        if not (np.all(np.diff(xp) > 0) and len(xp) == len(fp)):
            raise ValueError("Values of fluid speed (first row of thrustCoefficient) should be in increasing order")
        else:
            return np.interp(u, xp, fp)

    def calc_power_coefficient(self, flow_at_turbine):
        turbine_direction = self.get_direction()
        turbine_coords = self.get_coords()

        #normalised_turbine_direction = turbine_direction/np.linalg.norm(turbine_direction,2)
        #u = np.dot(flow_at_turbine, normalised_turbine_direction)

        u = np.linalg.norm(flow_at_turbine,2)
        
        curve = self.get_power_coefficient_curve()
        # flow speed data points
        xp = curve[0]
        # thrust coefficient data points
        fp = curve[1]

        # check if xp is always increasing and lengths are equal
        if not (np.all(np.diff(xp) > 0) and len(xp) == len(fp)):
            raise ValueError("Values of fluid speed (first row of powerCoefficient) should be in increasing order")
        else:
            return np.interp(u, xp, fp)

    def calc_area(self):
        return np.pi * self.get_radius()*2
     
    def set_radius(self, radius):
        default_radius = 10
        if not (isinstance(radius, (float,int)) or radius == None):
            raise TypeError("'radius' must be of type 'float' or 'int'")
        else:
            self.radius = radius if radius != None else default_radius
 
    def set_coords(self, coords): 
        default_coords = [0,0,0]
        if not ((isinstance(coords, list) and len(coords) == 3 and all(isinstance(c, (float,int)) for c in coords) ) or coords == None): 
            raise TypeError("'coords' must be of type 'list' with three elements.") 
        else: 
            self.coords = np.array(coords) if coords != None else np.array(default_coords)

    def set_top_clearance(self, top_clearance):
        default_top_clearance = 'inf'
        if not (isinstance(top_clearance, (float,int)) or top_clearance == None):
            raise TypeError("'top_clearance' must be of type 'float' or 'int'")
        else:
            self.top_clearance = top_clearance if top_clearance != None else default_top_clearance

    def set_direction(self, direction): 
        default_direction = [-1,0, 0]
        if not ((isinstance(direction, list) and len(direction) == 3 and all(isinstance(d, (float,int)) for d in direction) ) or direction == None): 
            raise TypeError("'direction' must be of type 'list' with three elements.") 
        else: 
            self.direction = np.array(direction) if direction != None else np.array(default_direction)

    def set_thrust_coefficient_curve(self, thrust_coefficient_curve): 
        default_thrust_coefficient_curve = []
        if not ((isinstance(thrust_coefficient_curve, list) and len(thrust_coefficient_curve[0]) == len(thrust_coefficient_curve[1]) and all(isinstance(c, (float,int)) for c in thrust_coefficient_curve[0]) and all(isinstance(c, float) or isinstance(c, int) for c in thrust_coefficient_curve[1])) or thrust_coefficient_curve == None): 
            raise TypeError("'thrust_coefficient_curve' must be of type two-demensional 'list' where the first nested list are the flow speeds and the second nested list are the corresponding thrust coefficients, where both lists are the same length and all elements are integers or floats.") 
        else:
            self.thrust_coefficient_curve = np.array(thrust_coefficient_curve) if thrust_coefficient_curve != None else np.array(default_thrust_coefficient_curve)

    def set_power_coefficient_curve(self, power_coefficient_curve): 
        default_power_coefficient_curve = []
        if not ((isinstance(power_coefficient_curve, list) and len(power_coefficient_curve[0]) == len(power_coefficient_curve[1]) and all(isinstance(c, (float,int)) for c in power_coefficient_curve[0]) and all(isinstance(c, float) or isinstance(c, int) for c in power_coefficient_curve[1])) or power_coefficient_curve == None): 
            raise TypeError("'power_coefficient_curve' must be of type two-demensional 'list'  where the first nested list are the flow speeds and the second nested list are the corresponding power coefficients, where both lists are the same length and all elements are integers or floats.")
        else:
            self.power_coefficient_curve = np.array(power_coefficient_curve) if power_coefficient_curve != None else np.array(default_power_coefficient_curve)

    def calc_power_op(self, flow_at_turbine):
        k1, k2, rho = 1/2, 3, 1024
        power_coefficient = self.calc_power_coefficient(flow_at_turbine)
        u = np.linalg.norm(flow_at_turbine, 2)
        area = self.calc_area()
        power_extracted = k1 * rho * u**k2 * area
        return power_extracted

##    def calc_average_vrf(self, combined_vrf):
##        """
##        Returns average velocity reduction factor over the rotor plane
##        param combined_vrf meshgrid of velocity reduction factor at the coordinates
##        spanning turbine's rotor plane
##        as calculated by any wake combination model
##        """
##        #ave_vrf = (1/self.calc_area()) * np.sum(np.sum(combined_vrf))
