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

    def calc_combination_speed_at_point(self, pnt_coords, flow_field, u_j, u_ij, mag):
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        .. math:: \sum_j \left( 1 - \frac{u_{ij}}{u_j}\right)

        See Renkema, D. [2007] section 4.8.1.

        Returns total velocity reduction factor at point i as calculated by linear superposition method
        """
        
        wake_freestream_velocity_ratio = self.calc_velocity_ratio(u_j, u_ij)

        # subtract ratio from one for each element corresponding to turbine j,
        # and sum linearly
        linear_sum = np.sum((1 - wake_freestream_velocity_ratio), axis = 0)

        undisturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point(pnt_coords, False)
        undisturbed_flow_mag_at_point = np.linalg.norm(undisturbed_flow_at_point, 2)
        undisturbed_flow_dir_at_point = undisturbed_flow_at_point / undisturbed_flow_mag_at_point

        if mag == True:
            return (1 - linear_sum) * undisturbed_flow_mag_at_point
        else:
            return undisturbed_flow_dir_at_point * (1 - linear_sum) * undisturbed_flow_mag_at_point
