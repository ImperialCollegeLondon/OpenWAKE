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

    def calc_thrust_coefficient(self, flow_at_point):

        #turbine_direction = self.get_direction()
        turbine_coords = self.get_coords()
        #normalised_turbine_direction = turbine_direction/np.linalg.norm(turbine_direction,2)
        #u = np.dot(flow_at_point, normalised_turbine_direction)

        u = np.linalg.norm(flow_at_point,2)
        
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

    def calc_power_coefficient(self, flow_at_point):
        turbine_direction = self.get_direction()
        turbine_coords = self.get_coords()

        #normalised_turbine_direction = turbine_direction/np.linalg.norm(turbine_direction,2)
        #u = np.dot(flow_at_point, normalised_turbine_direction)

        u = np.linalg.norm(flow_at_point,2)
        
        curve = self.get_power_coefficient_curve()
        # flow speed data points
        xp = curve[0]
        # thrust coefficient data points
        fp = curve[1]

        # check if xp is always increasing and lengths are equal
        if not (np.all(np.diff(xp) > 0) and and len(xp) == len(fp)):
            raise ValueError("Values of fluid speed (first row of powerCoefficient) should be in increasing order")
        else:
            return np.interp(u, xp, fp)

    def calc_area(self):
        return np.pi * self.get_radius()*2
     
    def set_radius(self, radius):
        default_radius = 10
        if not (isinstance(radius, float) or isinstance(radius, int) or radius == None):
            raise TypeError("'radius' must be of type 'float' or 'int'")
        else:
            self.radius = radius if radius != None else default_radius
 
    def set_coords(self, coords): 
        default_coords = [0,0,0]
        if not ((isinstance(coords, list) and len(coords) == 3 and all(isinstance(c, float) or isinstance(c, int) for c in coords) ) or coords == None): 
            raise TypeError("'coords' must be of type 'list' with three elements.") 
        else: 
            self.coords = np.array(coords) if coords != None else np.array(default_coords)

    def set_top_clearance(self, top_clearance):
        default_top_clearance = 'inf'
        if not (isinstance(top_clearance, float) or isinstance(top_clearance, int) or top_clearance == None):
            raise TypeError("'top_clearance' must be of type 'float' or 'int'")
        else:
            self.top_clearance = top_clearance if top_clearance != None else default_top_clearance

    def set_direction(self, direction): 
        default_direction = [-1,0, 0]
        if not ((isinstance(direction, list) and len(direction) == 3 and all(isinstance(d, float) or isinstance(d, int) for d in direction) ) or direction == None): 
            raise TypeError("'direction' must be of type 'list' with three elements.") 
        else: 
            self.direction = np.array(direction) if direction != None else np.array(default_direction)

    def set_thrust_coefficient_curve(self, thrust_coefficient_curve): 
        default_thrust_coefficient_curve = []
        if not ((isinstance(thrust_coefficient_curve, list) and len(thrust_coefficient_curve[0]) == len(thrust_coefficient_curve[1]) and all(isinstance(c, float) or isinstance(c, int) for c in thrust_coefficient_curve[0]) and all(isinstance(c, float) or isinstance(c, int) for c in thrust_coefficient_curve[1])) or thrust_coefficient_curve == None): 
            raise TypeError("'thrust_coefficient_curve' must be of type two-demensional 'list' where the first nested list are the flow speeds and the second nested list are the corresponding thrust coefficients, where both lists are the same length and all elements are integers or floats.") 
        else:
            self.thrust_coefficient_curve = np.array(thrust_coefficient_curve) if thrust_coefficient_curve != None else np.array(default_thrust_coefficient_curve)

    def set_power_coefficient_curve(self, power_coefficient_curve): 
        default_power_coefficient_curve = []
        if not ((isinstance(power_coefficient_curve, list) and len(power_coefficient_curve[0]) == len(power_coefficient_curve[1]) and all(isinstance(c, float) or isinstance(c, int) for c in power_coefficient_curve[0]) and all(isinstance(c, float) or isinstance(c, int) for c in power_coefficient_curve[1])) or power_coefficient_curve == None): 
            raise TypeError("'power_coefficient_curve' must be of type two-demensional 'list'  where the first nested list are the flow speeds and the second nested list are the corresponding power coefficients, where both lists are the same length and all elements are integers or floats.")
        else:
            self.power_coefficient_curve = np.array(power_coefficient_curve) if power_coefficient_curve != None else np.array(default_power_coefficient_curve)
