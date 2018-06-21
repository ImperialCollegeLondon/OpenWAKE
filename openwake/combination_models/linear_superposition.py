"""
Implements the Linear Superposition wake combination model.
"""
import numpy as np
from combination_models.base_combination import BaseWakeCombination

class LinearSuperposition(BaseWakeCombination):
    """
    Implements the Linear Superposition wake combination model.
    """
    def __init__(self, u_ij = [], u_j = []):
        super(LinearSuperposition, self).__init__(u_ij, u_j)


    def combine(self):
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        .. math:: \sum_j \left( 1 - \frac{u_{ij}}{u_j}\right)

        See Renkema, D. [2007] section 4.8.1.

        Returns total velocity reduction factor at point i as calculated by linear superposition method
        """
        
        wake_freestream_velocity_ratio = self.calc_velocity_ratio()

        # subtract ratio from one for each element corresponding to turbine j,
        # and sum linearly
        linear_sum = np.sum((1 - wake_freestream_velocity_ratio), axis=0)

        return linear_sum

    def calc_flow_at_point(self, undisturbed_flow_at_point, pnt_coords=None):
        return (1 - self.combine()) * np.linalg.norm(undisturbed_flow_at_point,2)
