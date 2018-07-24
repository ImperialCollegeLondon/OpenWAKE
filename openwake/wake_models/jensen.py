from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from wake_models.wake_field_model import WakeField
from flow_field_model.flow import FlowField
import numpy as np
from helpers import *

class Jensen(BaseWake):
    """Implements a Jensen wake model."""
    def __init__(self, turbine = BaseTurbine(), flow_field = FlowField(), wake_decay = 0.3, wake_field = WakeField()):
        self.set_wake_decay(wake_decay)
        super(Jensen, self).__init__(turbine, flow_field, wake_field)

    def get_wake_decay(self):
        return self.wake_decay
    
    def set_wake_decay(self, wake_decay = 0.3):
        try:
            assert isinstance(wake_decay, float)
        except AssertionError:
            raise TypeError("'wake_decay' must be of type 'float'")
        else:
            self.wake_decay = wake_decay

    def calc_wake_radius(self, rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient):
        """
        Returns the radius of the wake at a point
        given that point is in wake of turbine
        param pnt_coords point at which to calculate the radius of the wake
        """

        #rel_pnt_coords = relative_position(turbine_coords, pnt_coords, flow_field)
        wake_decay = self.get_wake_decay()
        x_rel = rel_pnt_coords[0]
        
        return turbine_radius + (wake_decay * x_rel)

    def calc_vrf_at_point(self, rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient, u_0):
        if self.is_in_wake(rel_pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
            wake_decay = self.get_wake_decay()
            x_rel = rel_pnt_coords[0]
            vrf = (1 - thrust_coefficient)**0.5/((1 + (wake_decay * x_rel / turbine_radius))**2)
        else:
            vrf =  0
        return vrf
