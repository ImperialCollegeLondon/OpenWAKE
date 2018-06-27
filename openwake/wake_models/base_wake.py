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
        self.calc_disturbed_flow_grid()

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
            #return wake_radius > (abs(y_rel)**2 + abs(z_rel)**2)**0.5
            return wake_radius > abs(y_rel) and wake_radius > abs(z_rel)

    def get_disturbed_flow_at_point(self, pnt_coords, mag = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """
        flow_field = self.get_flow_field()
        undisturbed_flow_grid = flow_field.get_flow()
        disturbed_flow_grid = self.get_disturbed_flow_grid()
        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        turbine_radius = turbine.get_radius()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        origin_coords = np.array([0,0,0])
        rel_x_index_origin, rel_y_index_origin, rel_z_index_origin = relative_index(origin_coords, pnt_coords, flow_field)
        rel_x_index_turbine, rel_y_index_turbine, rel_z_index_turbine = relative_index(turbine_coords, pnt_coords, flow_field)
        x_index, y_index, z_index = abs(rel_x_index_turbine), 0, abs(rel_z_index_turbine)

        x_rel, y_rel, z_rel = relative_position(turbine_coords, pnt_coords, flow_field)
        #print(pnt_coords, [x_rel, y_rel, z_rel])
        wake_radius = self.calc_wake_radius(pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient)
        #print(wake_radius)
        #print((x_rel < 0.0), (wake_radius > abs(y_rel) and wake_radius > abs(z_rel)),'\n' )
        if self.is_in_wake(pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
            print(pnt_coords, [rel_x_index_turbine, rel_y_index_turbine, rel_z_index_turbine])#TODO y always 0 so rel_y always outside wake_radius!!
            disturbed_flow_at_point = disturbed_flow_grid[x_index, y_index, z_index]
        else:
            disturbed_flow_at_point = undisturbed_flow_grid[rel_x_index_origin, rel_y_index_origin, rel_z_index_origin]
        
        if mag == True:
            disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
        return disturbed_flow_at_point
    
    def calc_disturbed_flow_grid(self):
        """
        Given the undisturbed flow grid, empty u, v, w arrays and x,y,z meshgrids,
        populates a vicinity relative to the turbine location
        """
        
        undisturbed_flow_grid = self.get_flow_field().get_flow()
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
        disturbed_flow_grid = np.zeros((len_x - start_x_index, len_y - start_y_index, len_z - start_z_index, 3))
        
        for i in range(start_x_index, end_x_index + 1):
            wake_radius = self.calc_wake_radius([x_grid[i],0,0], turbine_coords, flow_field, turbine_radius, thrust_coefficient)
            end_z_index = find_index(z_coords, turbine_coords[2] + wake_radius)
            for k in range(start_z_index, end_z_index + 1):
                # calculate the multiplier along a single radial z line at this value of x along wake
                pnt_coords = np.array([x_grid[i], y_grid[start_y_index], z_grid[k]])
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
