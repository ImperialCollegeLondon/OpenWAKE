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
        self.calc_multiplier_grid(flow_field)

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

    def set_grid_outdated(self, grid_outdated):
        """
        Set to true every time wake parameters are changed, except for turbine location
        """
        self.is_grid_outdated = grid_outdated

    def get_multiplier_grid(self):
        return self.multiplier_grid

    def set_multiplier_grid(self, multiplier_grid):
        self.multiplier_grid = multiplier_grid
        self.set_grid_outdated(False)

    def get_disturbed_flow_at_point(self, pnt_coords, flow_field, mag = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """
        if flow_field.is_in_flow_field(pnt_coords):
            undisturbed_flow_grid = flow_field.get_flow()
            turbine = self.get_turbine()
            turbine_coords = turbine.get_coords()
            turbine_radius = turbine.get_radius()
            u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
            thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

            # check if disturbed flow grid needs updating
            if self.is_grid_outdated:
                self.calc_multiplier_grid(flow_field)

            multiplier_grid = self.get_multiplier_grid()
            origin_coords = np.array([0,0,0])
            rel_origin_indices = relative_index(origin_coords, pnt_coords, flow_field)
            rel_turbine_position = relative_position(turbine_coords, pnt_coords, flow_field)
            dx = flow_field.get_dx()
            r_coords = self.get_r_coords()
            dr = self.get_dr()
            x_index, r_index = abs(int((pnt_coords[0] - turbine_coords[0]) / dx)), abs(int(np.linalg.norm(pnt_coords[0:] - turbine_coords[0:], 2) / dr))
            
            if self.is_in_wake(pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
                #TODO x_index out of multiplier_grid bounds
                disturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point(pnt_coords, False) * multiplier_grid[x_index, r_index]
            else:
                disturbed_flow_at_point = np.array(undisturbed_flow_grid[rel_origin_indices[0], rel_origin_indices[1], rel_origin_indices[2]])
            if mag == True:
                disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
            return disturbed_flow_at_point
        else:
            disturbed_flow_at_point = np.array([0, 0, 0])
            if mag == True:
                disturbed_flow_at_point = 0

        return disturbed_flow_at_point

    def set_r_coords(self, r_coords):
        self.r_coords = r_coords

    def get_r_coords(self):
        return self.r_coords

    def set_dr(self, dr):
        self.dr = dr

    def get_dr(self):
        return self.dr
    
    def calc_multiplier_grid(self, flow_field):
        """
        Given the undisturbed flow grid, empty u, v, w arrays and x,y,z meshgrids,
        populates a vicinity relative to the turbine location
        Assumes that the turbine has rotated to face the flow
        Calculates the multipliers of the disturbed flow grid along the single longest radial line from turbine to farthest boundary
        """

        # get variables
        turbine = self.get_turbine()
        turbine_radius = turbine.get_radius()
        flow_field = self.get_flow_field()
        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        # the dimensions of disturbed_flow_grid should allow for the maximum berth between the turbine and the domain boundaries
        start_x_index = find_index(x_coords, turbine_coords[0])
        len_x = x_coords.size
        end_x_index = len_x - 1

        start_y_index = find_index(y_coords, turbine_coords[1])
        len_y = y_coords.size
        end_y_index = len_y - 1

        start_z_index = find_index(z_coords, turbine_coords[2])
        len_z = z_coords.size
        end_z_index = len_z - 1

        max_domain_berth_index = np.argmax([end_y_index - start_y_index, start_y_index, end_z_index - start_z_index, start_z_index])

        #end_r_index = [end_y_index, 0, end_z_index, 0][max_domain_berth_index]
        
        # r_increment == 1 if upper y or z limit is widest berth, == -1 if lower y or z limit is widest berth
        r_increment = (-1)**max_domain_berth_index
        reference_coord_index = int(max_domain_berth_index / 2) % 2
        start_r_index = [start_y_index, start_z_index][reference_coord_index]
        r_coords = [y_coords, z_coords][reference_coord_index]
        len_r = len(r_coords)

        self.set_r_coords(r_coords)
        self.set_dr(abs(r_coords[1] - r_coords[0]))
        
        # TODO mulitplier should have 2 or 3 elements: u,v,w
        # this is a 2D grid of the multiplier along the longest radial line for each value of x relative to flow direction at turbine
        multiplier_grid = np.ones((len_x - start_x_index, len_r))
        
        for i in range(start_x_index, end_x_index + 1):
            wake_radius = self.calc_wake_radius([x_coords[i], turbine_coords[1], turbine_coords[2]], turbine_coords, flow_field, turbine_radius, thrust_coefficient)

            # get turbine coordinate corresponding to reference radial line r, and add or subtract wake_radius from it depending on
            # wheter that reference line extends further above or below turbine in domain
            end_r_index = find_index(r_coords, turbine_coords[reference_coord_index + 1] + (r_increment * wake_radius))
            
            for j in range(start_r_index, end_r_index + 1, r_increment):
                # calculate the multiplier along the longest (positive or negative) radial y or z line (r) at this value of x along wake
                # then apply that multiplier to the reflected r coordinate and the corresonding non-r coordinate and its reflected coordinate
                
                pnt_coords = np.array([x_coords[i], y_coords[j], z_coords[start_z_index]]) if reference_coord_index == 0 else np.array([x_coords[i], y_coords[start_y_index], z_coords[j]])

                if self.is_in_wake(pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
                    # TODO multiplier diminishes too quickly with rel_x
                    multiplier = 1 - self.calc_vrf_at_point(pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient, u_0)
                    multiplier_grid[i - start_x_index, j - start_r_index] = multiplier
                    
        self.set_multiplier_grid(multiplier_grid)
        print(multiplier_grid)

    def get_turbine(self):
        return self.turbine

    def set_turbine(self, turbine = BaseTurbine()):
        try:
            assert isinstance(turbine, BaseTurbine)
        except AssertionError:
            raise TypeError("'turbine' must be of type 'Turbine'")
        else:
            self.turbine = turbine
