"""
Implements the Energy Balance wake combination model.
"""

from combination_models.base_combination import BaseWakeCombination
from flow_field_model.flow import FlowField
from wake_models.wake_field_model import WakeField
import numpy as np

class SumOfSquares(BaseWakeCombination):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self, flow_field, wake_field):
        super(SumOfSquares, self).__init__(flow_field, wake_field)

    def calc_combination_speed_at_point(self,  pnt_coords, flow_field, u_j, u_ij, mag):
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        See Renkema, D. [2007] section 4.8.1.

        Returns total velocity reduction factor at point i as calculated by sum of squares method
        """

        wake_freestream_velocity_ratio = self.calc_velocity_ratio(u_j, u_ij)

        undisturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point(pnt_coords, False)
        undisturbed_flow_mag_at_point = np.linalg.norm(undisturbed_flow_at_point, 2)
        undisturbed_flow_dir_at_point = undisturbed_flow_at_point / undisturbed_flow_mag_at_point

        # subtract ratio from one for each element corresponding to turbine j
        # and calculate the sum of their squares
        sum_of_squares = np.sum((1 - wake_freestream_velocity_ratio)**2, axis = 0)

        if mag == True:
            return (1 - sum_of_squares**0.5) * undisturbed_flow_mag_at_point
        else:
            return undisturbed_flow_dir_at_point * (1 - sum_of_squares**0.5) * undisturbed_flow_mag_at_point
