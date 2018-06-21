"""
Implements the Energy Balance wake combination model.
"""
import numpy as np
from combination_models.base_combination import BaseWakeCombination

class EnergyBalance(BaseWakeCombination):
    """
    Implements the Energy Balance wake combination model.
    """
    def __init__(self, u_ij = [], u_j = []):
        super(EnergyBalance, self).__init__(u_ij, u_j)
        
    def combine(self):
        """
        Combines a number of wakes to give a single flow speed at a turbine.

        See Renkema, D. [2007] section 4.8.1.

        Returns total velocity reduction factor at point i as calculated
        by energy balance method
        """

        u_j = self.get_u_j()
        u_ij = self.get_u_ij()

        # subtract ratio from one for each element corresponding to turbine j,
        # and sum energy balance
        energy_balance_sum = np.sum((u_j**2-u_ij**2), axis=0)

        return energy_balance_sum

    def calc_flow_at_point(self, undisturbed_flow_at_point, pnt_coords=None):
        #return (np.linalg.norm(undisturbed_flow_at_point,2)**2 - self.combine())**0.5
        undisturbed_flow_at_point = np.linalg.norm(undisturbed_flow_at_point,2) if isinstance(undisturbed_flow_at_point, (list, np.ndarray)) else undisturbed_flow_at_point
        return (undisturbed_flow_at_point**2 - self.combine())**0.5
