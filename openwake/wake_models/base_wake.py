"""
A BaseWake class from which other wake models may be derived.
"""
from base_field_model import BaseField
from turbine_models.base_turbine import BaseTurbine
from wake_models.wake_field_model import WakeField
from flow_field_model.flow import FlowField
import numpy as np
from helpers import *

class BaseWake( BaseField ):
    """A base class from which wake models may be derived."""

    def __init__( self, turbine, flow_field = FlowField(), wake_field = WakeField() ):
        super( BaseWake, self ).__init__( flow_field, wake_field )
        self.set_turbine( turbine )
        self.wake_field.add_wakes( [ self ] )
        self.calc_multiplier_grid( flow_field )

    def is_in_wake( self, rel_pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field ):

        """
        Returns True if point is in wake.
        param pnt_coords coordinates to check
        """
        
        x_rel, y_rel, z_rel = rel_pnt_coords
    
        if x_rel < 0.0 or x_rel > 10 * turbine_radius:
            return False
        else:
            
            wake_radius = self.calc_wake_radius( rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient )
            r_rel = np.linalg.norm( [ y_rel, z_rel ], 2 )
            return wake_radius > r_rel# and 5 * turbine_radius > r_rel

    def get_multiplier_grid( self ):
        try:
            self.multiplier_grid
        except AttributeError:
            self.calc_multiplier_grid()
            
        return self.multiplier_grid

    def set_multiplier_grid(self, multiplier_grid):
        self.multiplier_grid = multiplier_grid
        self.set_grid_outdated( False )

    def get_multiplier_at_point( self, pnt_coords, flow_field, turbine ):
        
        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        turbine_radius = turbine.get_radius()
        turbine_direction = turbine.get_direction()
        dx = dr = min( flow_field.get_diff() )
        num_coords = flow_field.get_coords().shape[1]
        u_0 = flow_field.get_undisturbed_flow_at_point( turbine_coords, True )
        thrust_coefficient = turbine.calc_thrust_coefficient( u_0 )
        
        rel_pnt_coords = relative_position( turbine_coords, pnt_coords, turbine_direction, True )
        
        if self.is_in_wake( rel_pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field ):
            x_index = int( rel_pnt_coords[0] / dx )
            r_index = int( np.linalg.norm( rel_pnt_coords[1:], 2) / dr )
            multiplier = self.get_multiplier_grid()[ x_index, r_index ]
        else:
            multiplier = 1
        
        return multiplier

    def calc_multiplier_grid( self, flow_field ):
        """
        Given the undisturbed flow grid, empty u, v, w arrays and x,y,z meshgrids,
        populates a vicinity relative to the turbine location
        Assumes that the turbine has rotated to face the flow
        Calculates the multipliers of the disturbed flow grid along the single longest radial line from turbine to farthest boundary
        """

        # the 0th dimension, x, corresponds to the axial distance along the flow at turbine vector,
        # from the centre of the turbine
        # the 1st dimension, r, corresponds to the radial distance from the flow at turbine vector

        # get variables
        turbine = self.get_turbine()
        turbine_radius = turbine.get_radius()
        flow_field = self.get_flow_field()
        dx = dr = min( flow_field.get_diff() )
        num_coords = flow_field.get_coords().shape[1]
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        # the dimensions of disturbed_flow_grid should allow for the maximum berth between the turbine and the domain boundaries
   
        # TODO mulitplier should have 2 or 3 elements: u,v,w
        # this is a 2D grid of the multiplier along the longest radial line for each value of x relative to flow direction at turbine
        # 0th (x) dimension should allow for maximum berth in absolute combination coordinate system
        # 1st (r) dimension should allow for maximum wake_radius at that value of x
        
        max_len_x = 10 * turbine_radius
        max_len_r = 5 * turbine_radius
        multiplier_grid = np.ones( ( max_len_x, max_len_r ) )
        
        #for i in range(start_x_index, end_x_index + 1):
        wake_radius = turbine_radius
        for c in range( max_len_x * max_len_r ):

            i, j = int( c / max_len_x ), int( c % max_len_x )
            
            x = i * dx
            r = j * dr

            wake_radius = self.calc_wake_radius([x, 0, 0], turbine_coords, flow_field, turbine_radius, thrust_coefficient)

            if r > wake_radius:
                continue
    
            # calculate the multiplier along the longest (positive or negative) radial y or z line (r) at this value of x along wake
            # then apply that multiplier to the reflected r coordinate and the corresonding non-r coordinate and its reflected coordinate
            rel_pnt_coords = [x, r, 0]

            multiplier = 1 - self.calc_vrf_at_point( rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient, u_0 )
            multiplier_grid[ i, j ] = multiplier
                    
        self.set_multiplier_grid( multiplier_grid )

    def get_turbine(self):
        return self.turbine

    def set_turbine(self, turbine):
        try:
            assert isinstance(turbine, BaseTurbine)
        except AssertionError:
            raise TypeError("'turbine' must be of type 'Turbine'")
        else:
            self.turbine = turbine
            # assume that turbine is controlled to face incoming undisturbed flow
            self.turbine.set_direction(self.get_flow_field().get_undisturbed_flow_at_point(self.turbine.get_coords(), False, True))
