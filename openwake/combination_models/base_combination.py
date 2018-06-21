"""
A WakeCombination class from which other wake combination models may be derived.
"""
import numpy as np
from helpers import *

class BaseWakeCombination(object):
    """A base class from which wake combination models may be derived."""
    def __init__(self, u_ij = [], u_j = []):
        try:
            assert len(u_ij) == len(u_j)
        except AssertionError:
            raise ValueError("The lengths of 'u_ij' and 'u_j' should be equal.")
        else:
            # The speed at turbine i due to the wake of turbine j
            self.set_u_ij(u_ij)
            # The speed at turbine j (causing wake for turbine i)
            self.set_u_j(u_j)

    def get_u_ij(self):
        """
        Returns list of speeds at turbine i due to the wake of turbine j
        """
        return self.u_ij

    def get_u_j(self):
        """
        Returns list of speeds at turbine j
        """
        return self.u_j

    def set_u_ij(self, u_ij = []):
        """
        Sets the list speed at turbine i due to the wake of turbine j
        param u_ij list of float or int
        """
        try:
            assert isinstance(u_ij, list)
            assert all(isinstance(u, (float,int, np.int64, np.float64)) for u in np.array(u_ij).flatten() )
        except AssertionError:
            raise TypeError("'u_ij' must be of type 'int' or 'float'")
        else:
            self.u_ij = np.array(u_ij)

    def set_u_j(self, u_j = []):
        """
        Sets the speed at turbine j
        param u_j list of float or int
        """
        try:
            assert isinstance(u_j, list)
            assert all(isinstance(u, (float,int, np.int64, np.float64)) for u in np.array(u_j).flatten())
        except AssertionError:
            raise TypeError("'u_j' must be of type 'int' or 'float'")
        else:
            self.u_j = np.array(u_j)

    def add_to_speed_list(self, u_ij, u_j):
        """
        Adds speed_in_wake to the flow_speed_in_wake list.
        param u_ij speed at turbine i due to the wake of turbine j
        param u_j speed at turbine j
        """
        try:
            assert isinstance(u_ij, (list, tuple, np.ndarray))
            assert isinstance(u_j, (list, tuple, np.ndarray))
            assert (len(u_ij)==2)
            assert (len(u_j)==2)
        except AssertionError:
            raise TypeError("'speed_in_wake' must be a list, tuple or "
                            "np.ndarray of length 2")
        else:
            self.u_ij.append(u_ij)
            self.u_j.append(u_j)


    def calc_velocity_ratio(self):
        """
         Returns an array of the ratio of the wake velocity to the freestream velocity
         for each turbine j at a point i

        """

        # list of speeds at turbine i due to the wake of turbine j
        u_ij = self.get_u_ij()

        # list of speeds at turbine j
        u_j = self.get_u_j()

        u_ij = set_below_abs_tolerance_to_zero(u_ij)
        u_j = set_below_abs_tolerance_to_zero(u_j)

        # set error handling to known, 'ignore' state to execute
        # divisions without divide by zero error.
        with np.errstate(all="ignore"):
            wake_freestream_velocity_ratio = u_ij/u_j

        # Set all results from of zero division to zero.
        wake_freestream_velocity_ratio = set_nan_or_inf_to_zero(wake_freestream_velocity_ratio)

        return wake_freestream_velocity_ratio
