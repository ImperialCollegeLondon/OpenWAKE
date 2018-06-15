from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from flow_field_model.flow import Flow
import numpy as np

class Jensen(BaseWake):
    """Implements a Jensen wake model."""
    def __init__(self, turbine, flow, wake_decay):
        self.set_wake_decay(wake_decay)
        super(Jensen, self).__init__(turbine, flow)

    def get_wake_decay(self):
        return self.wake_decay
    
    def set_wake_decay(self, wake_decay):
        default_wake_decay = 0.03
        if not (isinstance(wake_decay, float) or wake_decay == None):
            raise TypeError("'wake_decay' must be of type 'float'")
        else:
            self.wake_decay = wake_decay if wake_decay != None else default_wake_decay

    def calc_wake_radius(self, pnt_coords):
        """
        Returns the radius of the wake at a point
        given that point is in wake of turbine
        """
        rel_pnt_coords = self.relative_position(pnt_coords)
        turbine_radius = self.get_turbine().get_radius()
        wake_decay = self.get_wake_decay()
        x = rel_pnt_coords[0]
        
        return turbine_radius + (wake_decay * x)

    def calc_vrf_at_point(self, undisturbed_flow_at_point, pnt_coords):
        """
        Returns the a velocity reduction factor to scale the ambient flow by
        for another_turbine in wake of turbine, given that
        the flow and other point have been rotated such that the
        flow hits the turbine axially
        """

        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        turbine_direction = turbine.get_direction()
        flow_at_point = self.get_flow().get_flow_at_point(turbine_coords)

        # check if point is in wake caused by turbine
        if self.is_in_wake(pnt_coords):
            wake_decay = self.get_wake_decay()
            thrust_coefficient = turbine.calc_thrust_coefficient(flow_at_point)
            turbine_radius = turbine.get_radius()
            x0, y0, z0 = self.relative_position(pnt_coords)
            
            # formula for velocity reduction factor
            return 1.0 - ((1.0 - thrust_coefficient)**0.5/((1.0 + wake_decay*x0/turbine_radius)**2))
        else:
            return 1.0
