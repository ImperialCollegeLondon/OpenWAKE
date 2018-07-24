"""
Implements the Linear Superposition wake combination model.
"""

from combination_models.base_combination import BaseWakeCombination
from flow_field_model.flow import FlowField
from wake_models.wake_field_model import WakeField
import numpy as np

class LinearSuperposition(BaseWakeCombination):
    """
    Implements the Linear Superposition wake combination model.
    """
    def __init__(self, flow_field, wake_field):
        super(LinearSuperposition, self).__init__(flow_field, wake_field)

    def calc_combined_multiplier_at_point( self, pnt_index, flow_field, coords, lengths, wakes ):
        
        # get multipliers of all upstream turbines. TODO aargs being reduced to single element
        len_x, len_y, len_z = lengths
        
        i, j, k = int( pnt_index / ( len_z * len_y ) ), \
                  int( ( pnt_index % ( len_z * len_y ) ) / len_z ), \
                  int( pnt_index % len_z )

        x, y, z = coords[0][i], coords[1][j], coords[2][k]
        pnt_coords = np.array( [ x, y, z ] )
        wake_multipliers = np.array( [ w.get_multiplier_at_point( pnt_coords, flow_field, w.get_turbine() ) for w in wakes ] )
        
##        undisturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point(pnt_coords, False)
##        undisturbed_flow_mag_at_point = np.linalg.norm(undisturbed_flow_at_point, 2)
##        undisturbed_flow_dir_at_point = undisturbed_flow_at_point / undisturbed_flow_mag_at_point
        
        combined_multiplier = 1 - np.sum( 1 - wake_multipliers )
        if x == 30 and y == 25 and z == 15:
            print(combined_multiplier)
        #np.multiply( ( 1 - np.sum( 1 - wake_multipliers ) ), undisturbed_flow_dir_at_point )
        return combined_multiplier
