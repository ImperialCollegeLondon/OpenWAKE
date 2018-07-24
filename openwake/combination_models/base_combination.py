""" A WakeCombination class from which other wake combination models may be derived.
"""
from base_field_model import BaseField
from helpers import *
from flow_field_model.flow import FlowField
from wake_models.wake_field_model import WakeField
import numpy as np

class BaseWakeCombination( BaseField ):
    
    """ A base class from which wake combination models may be derived."""
    def __init__( self, flow_field, wake_field ):
        
        # call BaseField super class to set flow and wake fields
        super(BaseWakeCombination, self).__init__(flow_field, wake_field)
        self.is_grid_outdated = False
        # set multiplier and disturbed flow grids with current flow and wake fields
        self.calc_combined_multiplier_grid( flow_field, wake_field )
        self.calc_disturbed_flow_grid( flow_field )
        
    def set_disturbed_flow_grid( self, disturbed_flow_grid ):
        
        """ Setter for disturbed_flow_grid
            param disturbed_flow_grid 4D np array of dimenstions
            len_x, len_y, len_z, 3 containing the disturbed flow at
            each point in the absolute coordinate system
        """
        self.disturbed_flow_grid = np.array( disturbed_flow_grid, dtype=np.float64 )
        self.is_grid_outdated = False 

    def get_disturbed_flow_grid( self, flow_field, wake_field ):
        
        """ Getter for disturbed_flow_grid. Calculates first if outdated.
            return up-to-date disturbed flow grid
        """
        if self.is_grid_outdated:
            self.calc_disturbed_flow_grid( flow_field )
            
        return self.disturbed_flow_grid
    
    def get_disturbed_flow_at_point( self, pnt_coords, flow_field, wake_field, mag = False, direction = False ):
        """ Getter for disturbed flow at a particular point in disturbed flow grid
            
        """
        if flow_field.is_in_flow_field( pnt_coords ):

            disturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point( pnt_coords ) * self.get_combined_multiplier_at_point( pnt_coords )

            if mag == True:
                disturbed_flow_at_point = np.linalg.norm( disturbed_flow_at_point, 2 )
            elif direction == True:
                try:
                    disturbed_flow_at_point = disturbed_flow_at_point / np.linalg.norm( disturbed_flow_at_point, 2 )
                except ZeroDivisionError:
                    disturbed_flow_at_point = np.array([0,0,0])

        else:
            disturbed_flow_at_point = np.array([0, 0, 0])

        return disturbed_flow_at_point

    def get_combined_multiplier_at_point( self, pnt_coords):
        diff = flow_field.get_diff()
        origin_point = np.array( [ 0, 0, 0 ] )
        rel_x_index, rel_y_index, rel_z_index = relative_index( origin_point, pnt_coords, diff )
        
        return self.get_multiplier_grid()[rel_x_index, rel_y_index, rel_z_index]

    def get_combined_multiplier_grid( self ):
        if self.is_grid_outdated:
            self.calc_combined_multiplier_grid( self.get_flow_field(), self.get_wake_field() )
        return self.combined_multiplier_grid

    def set_combined_multiplier_grid( self, combined_multiplier_grid ):
        self.combined_multiplier_grid = np.array( combined_multiplier_grid )

    def calc_combined_multiplier_grid( self, flow_field, wake_field ):
        
        """ Superimposes the relative multiplier grids of each wake
        onto the absolute grid. At points that are in >= 2 wakes
        the multipliers are combined.
        """
        wakes = wake_field.get_wakes()
        x_coords, y_coords, z_coords = coords = flow_field.get_coords()
        len_x, len_y, len_z = lengths = flow_field.get_lengths()
        v_combine = np.vectorize( self.calc_combined_multiplier_at_point, otypes=[ float ] )#, excluded = [ 'coords', 'lengths', 'wakes' ] )
        v_combine.excluded.add(1)
        v_combine.excluded.add(2)
        v_combine.excluded.add(3)
        v_combine.excluded.add(4)

        # TODO only need flow_grid at hub_heights
        combined_multiplier_grid = v_combine( np.arange( len_x * len_y * len_z ), flow_field, coords, lengths, wakes )
        combined_multiplier_grid = np.array( [ np.ones( 3 ) * x for x in combined_multiplier_grid ] ).reshape( ( len_x, len_y, len_z, 3 ) )
        self.set_combined_multiplier_grid( combined_multiplier_grid )
    
    def calc_disturbed_flow_grid( self, flow_field ):
        
        undisturbed_flow_grid = flow_field.get_flow()
        combined_multiplier_grid = self.get_combined_multiplier_grid()
        disturbed_flow_grid = np.multiply( undisturbed_flow_grid, combined_multiplier_grid )
        self.set_disturbed_flow_grid( disturbed_flow_grid )
