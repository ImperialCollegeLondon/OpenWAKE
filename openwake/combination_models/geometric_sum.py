"""
Implements the Geometric Sum wake combination model.
"""
import numpy as np
from combination_models.base_combination import BaseWakeCombination

class GeometricSum(BaseWakeCombination):
    """
    Implements the Geometric Sum wake combination model.
    """
    def __init__(self, u_ij = [], u_j = []):
        super(GeometricSum, self).__init__(u_ij, u_j)


    def combine(self):
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        See Renkema, D. [2007] section 4.8.1
        Returns total velocity reduction factor at point i as calculated by
        geometric sum method
     
        """
        
        wake_freestream_velocity_ratio = self.calc_velocity_ratio()

        return np.prod(wake_freestream_velocity_ratio, axis=0)

    def calc_flow_at_point(self, freestream_velocity, pnt_coords=None):
        return self.combine() * np.linalg.norm(freestream_velocity,2)
