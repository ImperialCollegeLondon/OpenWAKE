from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from flow_field_model.flow import Flow
import numpy as np
from helpers import *

class Jensen(BaseWake):
    """Implements a Jensen wake model."""
    def __init__(self, turbine = BaseTurbine(), flow = Flow(), wake_decay = 0.03):
        self.set_wake_decay(wake_decay)
        super(Jensen, self).__init__(turbine, flow)

    def get_wake_decay(self):
        return self.wake_decay
    
    def set_wake_decay(self, wake_decay = 0.03):
        try:
            assert isinstance(wake_decay, float)
        except AssertionError:
            raise TypeError("'wake_decay' must be of type 'float'")
        else:
            self.wake_decay = wake_decay

    def calc_wake_radius(self, pnt_coords):
        """
        Returns the radius of the wake at a point
        given that point is in wake of turbine
        param pnt_coords point at which to calculate the radius of the wake
        """
        rel_pnt_coords = self.relative_position(pnt_coords)
        turbine_radius = self.get_turbine().get_radius()
        wake_decay = self.get_wake_decay()
        x_rel = rel_pnt_coords[0]
        
        return turbine_radius + (wake_decay * x_rel)

    def calc_vrf_at_point(self, pnt_coords):
        """
        Returns the a velocity reduction factor to scale the ambient flow by
        for another_turbine in wake of turbine, given that
        the flow and other point have been rotated such that the
        flow hits the turbine axially
        """

        #turbine_direction = turbine.get_direction()
        
        # check if point is in wake caused by turbine
        if self.is_in_wake(pnt_coords):
            turbine = self.get_turbine()
            u_0 = self.get_flow_mag_at_turbine()
            thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
            wake_decay = self.get_wake_decay()
            turbine_radius = turbine.get_radius()
            x_rel, y_rel, z_rel = self.relative_position(pnt_coords)
            # formula for velocity reduction factor
            return (1 - thrust_coefficient)**0.5/((1 + (wake_decay*x_rel/turbine_radius))**2)
        else:
            return 0
