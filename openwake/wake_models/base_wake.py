"""
A BaseWake class from which other wake models may be derived.
"""
from base_field_model import BaseField
from turbine_models.base_turbine import BaseTurbine
from wake_models.wake_field_model import WakeField
from flow_field_model.flow import FlowField
import numpy as np
from helpers import *

class BaseWake(BaseField):
    """A base class from which wake models may be derived."""

    def __init__(self, turbine = BaseTurbine(), flow_field = FlowField(), wake_field = WakeField()):
        super(BaseWake, self).__init__(flow_field, wake_field)
        self.set_turbine(turbine)
        self.wake_field.add_wakes([self])
        self.calc_disturbed_flow_grid(flow_field)
        self.set_grid_outdated(False)

    def is_in_wake(self, pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):

        """
        Returns True if point is in wake.
        param pnt_coords coordinates to check
        """
        
        x_rel, y_rel, z_rel = relative_position(turbine_coords, pnt_coords, flow_field)

        if (x_rel < 0.0):
            return False
        else:
            wake_radius = self.calc_wake_radius(pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient)
            return wake_radius > (abs(y_rel)**2 + abs(z_rel)**2)**0.5
            #return wake_radius > abs(y_rel) and wake_radius > abs(z_rel)

    def set_grid_outdated(self, grid_outdated):
        """
        Set to true every time wake parameters are changed, except for turbine location
        """
        self.is_grid_outdated = grid_outdated

    def get_disturbed_flow_at_point(self, pnt_coords, flow_field, mag = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """
        undisturbed_flow_grid = flow_field.get_flow()
        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        turbine_radius = turbine.get_radius()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        # check if disturbed flow grid needs updating
        if self.is_grid_outdated:
            self.calc_disturbed_flow_grid(flow_field)
            self.set_grid_outdated(False)

        disturbed_flow_grid = self.get_disturbed_flow_grid()
        origin_coords = np.array([0,0,0])
        rel_origin_indices = relative_index(origin_coords, pnt_coords, flow_field)
        rel_turbine_indices = relative_index(turbine_coords, pnt_coords, flow_field)
        x_index, y_index, z_index = abs(rel_turbine_indices[0]), 0, abs(rel_turbine_indices[2])
        wake_radius = self.calc_wake_radius(pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient)
        if self.is_in_wake(pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
            disturbed_flow_at_point = np.array(disturbed_flow_grid[x_index, y_index, z_index])
        else:
            disturbed_flow_at_point = np.array(undisturbed_flow_grid[rel_origin_indices[0], rel_origin_indices[1], rel_origin_indices[2]])
        if mag == True:
            disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
        return disturbed_flow_at_point
    
    def calc_disturbed_flow_grid(self, flow_field):
        """
        Given the undisturbed flow grid, empty u, v, w arrays and x,y,z meshgrids,
        populates a vicinity relative to the turbine location
        """
        undisturbed_flow_grid = flow_field.get_flow()
        # u, v and w represent the i, j and k components of the speeds at each point, respectively
        # x_grid, y_grid, z_grid are numpy arrays corresponding to the x, y and z components at each coordinate
        x_grid, y_grid, z_grid, u, v, w = self.generate_disturbed_flow_grid()

        # get variables
        turbine = self.get_turbine()
        turbine_radius = turbine.get_radius()
        flow_field = self.get_flow_field()
        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        start_x_index = find_index(x_coords, turbine_coords[0])
        len_x = x_coords.size
        end_x_index = len_x - 1

        start_y_index = find_index(y_coords, turbine_coords[1])
        len_y = y_coords.size

        start_z_index = find_index(z_coords, turbine_coords[2])
        len_z = z_coords.size

        # assumes that the turbines faces the flow orthogonally
        #undisturbed_flow_grid = np.linalg.norm([u,v,w], 2, 0)
        disturbed_flow_grid = np.array(undisturbed_flow_grid, dtype=np.float64)[start_x_index:len_x, start_y_index:len_y, start_z_index:len_z]
        
        for i in range(start_x_index, end_x_index + 1):
            wake_radius = self.calc_wake_radius([x_coords[i],turbine_coords[1],turbine_coords[2]], turbine_coords, flow_field, turbine_radius, thrust_coefficient)
            end_z_index = find_index(z_coords, turbine_coords[2] + wake_radius)
            for k in range(start_z_index, end_z_index + 1):
                # calculate the multiplier along a single radial z line at this value of x along wake
                pnt_coords = np.asarray([x_coords[i], y_coords[start_y_index], z_grid[k]])
                if not self.is_in_wake(pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
                    break
                else:
                # TODO mulitplier should have 2 or 3 elements: u,v,w
                    multiplier = 1 - self.calc_vrf_at_point(pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient, u_0)
                    disturbed_flow_grid[i-start_x_index, 0, k - start_z_index] = undisturbed_flow_grid[i, start_y_index, k] * multiplier
        self.set_disturbed_flow_grid(disturbed_flow_grid)

    def get_turbine(self):
        return self.turbine

    def set_turbine(self, turbine = BaseTurbine()):
        try:
            assert isinstance(turbine, BaseTurbine)
        except AssertionError:
            raise TypeError("'turbine' must be of type 'Turbine'")
        else:
            self.turbine = turbine
