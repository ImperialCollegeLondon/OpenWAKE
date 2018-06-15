"""
Implements the Energy Balance wake combination model.
"""
import numpy as np
from combination_models.base_combination import BaseWakeCombination

class SumOfSquares(BaseWakeCombination):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self, u_ij = [], u_j = []):
        super(SumOfSquares, self).__init__(u_ij, u_j)

    def combine(self):
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        See Renkema, D. [2007] section 4.8.1.

        Returns total velocity reduction factor at point i as calculated by sum of squares method
        """

        wake_freestream_velocity_ratio = self.calc_velocity_ratio()

        # subtract ratio from one for each element corresponding to turbine j
        # and calculate the sum of their squares
        sum_of_squares = np.sum((1 - wake_freestream_velocity_ratio)**2, axis=0)

        return sum_of_squares

    def calc_flow_at_point(self, freestream_velocity, pnt_coords=None):
        return (1 - self.combine()**0.5) * np.linalg.norm(freestream_velocity,2)
