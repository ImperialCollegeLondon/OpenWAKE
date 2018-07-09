"""
A base single turbine model from which others can be derived.
"""

import numpy as np
from helpers import *
from turbine_models.turbine_field import TurbineField

class BaseTurbine(object):
    """ Implements base turbine class."""

    def __init__(self, radius = 10, coords = [ 0,0,0 ], top_clearance = np.inf, direction = [ -1, 0, 0 ], thrust_coefficient_curve = [ [], [] ], power_coefficient_curve=[ [], [] ], turbine_field = TurbineField() ):
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
        self.set_turbine_field(turbine_field)
        
    def get_radius(self):
        """ getter for tubine radius """
        return self.radius
    
    def get_coords(self):
        """ getter for tubine coordinates """
        return self.coords

    def get_top_clearance(self):
        """ getter for tubine top clearance """
        return self.top_clearance

    def get_direction(self):
        """ getter for tubine direction """
        return self.direction

    def get_thrust_coefficient_curve(self):
        """ getter for tubine thrust coefficient curve """
        return self.thrust_coefficient_curve

    def get_power_coefficient_curve(self):
        """ getter for tubine power coefficient curve """
        return self.power_coefficient_curve

    def get_turbine_field(self):
        """ getter for tubine turbine_field object """
        return self.turbine_field

    def calc_thrust_coefficient(self, flow_mag_at_turbine):
        """ returns the thrust coefficient of the turbine for
            the given flow magnitude
            param flow_mag_at_turbine magnitude of flow incident to turbine surface area
        """
        turbine_coords = self.get_coords()        
        curve = self.get_thrust_coefficient_curve()
        
        # flow speed data points
        xp = curve[0]

        # thrust coefficient data points
        fp = curve[1]
        
        # check if xp is always increasing and lengths are equal
        try:
            assert np.all( np.diff( xp ) > 0 )
            assert len( xp ) == len( fp )
        except AssertionError:
            raise ValueError("Values of fluid speed (first row of thrustCoefficient) should be in increasing order")
        else:
            return np.interp( flow_mag_at_turbine, xp, fp )

    def calc_power_coefficient(self, flow_mag_at_turbine):
        """ returns the power coefficient of the turbine for
            the given flow magnitude
            param flow_mag_at_turbine magnitude of flow incident to turbine surface area
        """

        turbine_coords = self.get_coords()
        
        curve = self.get_power_coefficient_curve()
        
        # flow speed data points
        xp = curve[0]
        # thrust coefficient data points
        fp = curve[1]

        # check if xp is always increasing and lengths are equal
        try:
            assert np.all( np.diff( xp ) > 0 )
            assert len( xp ) == len( fp )
        except AssertionError:
            raise ValueError("Values of fluid speed (first row of powerCoefficient) should be in increasing order")
        else:
            return np.interp( flow_mag_at_turbine, xp, fp )

    def calc_area( self ):
        return np.pi * self.get_radius()*2
     
    def set_radius(self, radius):
        try:
            assert isinstance(radius, (float,int))
        except AssertionError:
            raise TypeError("'radius' must be of type 'float' or 'int'")
        else:
            self.radius = radius
 
    def set_coords(self, coords=[0,0,0]):
        try:
            assert isinstance(coords, list)
            assert len(coords) == 3
            assert all(isinstance(c, (float,int)) for c in coords)
        except AssertionError:
            raise TypeError("'coords' must be of type 'list' with three elements.") 
        else: 
            self.coords = np.array(coords)

    def set_top_clearance(self, top_clearance = np.inf):
        try:
            assert isinstance(top_clearance, (float,int))
        except AssertionError:
            raise TypeError("'top_clearance' must be of type 'float' or 'int'")
        else:
            self.top_clearance = top_clearance

    def set_direction(self, direction = [-1,0, 0]):
        try:
            assert isinstance(direction, (list, np.ndarray))
            assert len(direction) == 3
            assert all(isinstance(d, (float,int)) for d in direction)
        except AssertionError:
            raise TypeError("'direction' must be of type 'list' with three elements.") 
        else: 
            self.direction = np.array(direction)

    def set_thrust_coefficient_curve(self, thrust_coefficient_curve = [[],[]]): 
        try:
            assert isinstance(thrust_coefficient_curve, list)
            assert len(thrust_coefficient_curve[0]) == len(thrust_coefficient_curve[1])
            assert all(isinstance(c, (float,int)) for c in thrust_coefficient_curve[0])
            assert all(isinstance(c, float) or isinstance(c, int) for c in thrust_coefficient_curve[1])
        except AssertionError:
            raise TypeError("'thrust_coefficient_curve' must be of type two-demensional 'list' where the first nested list are the flow speeds and the second nested list are the corresponding thrust coefficients, where both lists are the same length and all elements are integers or floats.") 
        else:
            self.thrust_coefficient_curve = np.array(thrust_coefficient_curve)

    def set_power_coefficient_curve(self, power_coefficient_curve = [[],[]]): 
        try:
            assert isinstance(power_coefficient_curve, list)
            assert len(power_coefficient_curve[0]) == len(power_coefficient_curve[1])
            assert all(isinstance(c, (float,int)) for c in power_coefficient_curve[0])
            assert all(isinstance(c, float) or isinstance(c, int) for c in power_coefficient_curve[1])
        except AssertionError:
            raise TypeError("'power_coefficient_curve' must be of type two-demensional 'list'  where the first nested list are the flow speeds and the second nested list are the corresponding power coefficients, where both lists are the same length and all elements are integers or floats.")
        else:
            self.power_coefficient_curve = np.array(power_coefficient_curve)

    def calc_power_op(self, flow_mag_at_turbine):
        #TODO avaerage over turbine rotor area
        k1, k2, rho = 1/2, 3, 1024
        power_coefficient = self.calc_power_coefficient(flow_mag_at_turbine)
        area = self.calc_area()
        flow_mag_at_turbine = np.linalg.norm(flow_mag_at_turbine,2) if isinstance(flow_mag_at_turbine, (list, np.ndarray)) else flow_mag_at_turbine
        power_extracted = k1 * rho * flow_mag_at_turbine**k2 * area * power_coefficient
        return power_extracted

    def set_turbine_field(self, turbine_field = TurbineField()):
        try:
            assert isinstance(turbine_field, TurbineField)
        except AssertionError:
            raise TypeError("'turbine_field' must be of type 'TurbineField'")
        else:
            self.turbine_field = turbine_field
            if self not in turbine_field.get_turbines():
                self.turbine_field.add_turbines([self])

##    def calc_average_vrf(self, combined_vrf):
##        """
##        Returns average velocity reduction factor over the rotor plane
##        param combined_vrf meshgrid of velocity reduction factor at the coordinates
##        spanning turbine's rotor plane
##        as calculated by any wake combination model
##        """
##        #ave_vrf = (1/self.calc_area()) * np.sum(np.sum(combined_vrf))
