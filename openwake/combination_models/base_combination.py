"""
A WakeCombination class from which other wake combination models may be derived.
"""
from base_field_model import BaseField
from helpers import *
from flow_field_model.flow import FlowField
from wake_models.wake_field_model import WakeField
import numpy as np

class BaseWakeCombination(BaseField):
    """A base class from which wake combination models may be derived."""
    def __init__(self, flow_field, wake_field):
        super(BaseWakeCombination, self).__init__(flow_field, wake_field)
        self.u_ij = np.array([])
        self.u_j = np.array([])
        #self.set_coarse_disturbed_flow_grid([])
        #self.set_fine_disturbed_flow_grid([])
        #self.calc_disturbed_flow_grid()

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

    def grid_outdated(self):
        wake_field = self.get_wake_field()
        wakes = wake_field.get_wakes()
        num_wakes = wake_field.get_num_wakes()
        u_ij_size = self.get_u_ij().size
        if u_ij_size < num_wakes:
            return True
        else:
            return False
    
    def set_u_ij(self, pnt_coords):
        """
        Sets the list speed at turbine i (pnt_coords) due to the wake of turbine j
        param u_ij list of float or int
        """
        flow_field = self.get_flow_field()
        wake_field = self.get_wake_field()
        wakes = wake_field.get_wakes()
        num_wakes = wake_field.get_num_wakes()
        u_ij_size = self.u_ij.size
        if self.grid_outdated():
            np.append(self.u_ij, [w.get_disturbed_flow_at_point(pnt_coords, True) for w in wakes[u_ij_size:num_wakes:1]])

    def set_u_j(self):
        """
        Sets the speed at turbine j TODO undisturbed or disturbed???
        param u_j list of float or int
        """
        flow_field = self.get_flow_field()
        wake_field = self.get_wake_field()
        undisturbed_flow = flow_field.get_flow()
        wakes = wake_field.get_wakes()
        num_wakes = wake_field.get_num_wakes()
        u_j_size = self.u_j.size
        if self.grid_outdated():
            np.append(self.u_j, [flow_field.get_undisturbed_flow_at_point(w.get_turbine().get_coords(), True) for w in wakes[u_j_size:num_wakes:1]])

    def set_disturbed_flow_grid(self, disturbed_flow_grid, fine_mesh):
        if fine_mesh == True:
            self.fine_disturbed_flow_grid = np.array(disturbed_flow_grid)
        else:
            self.coarse_disturbed_flow_grid = np.array(disturbed_flow_grid)

    def get_disturbed_flow_grid(self, fine_mesh):
        if fine_mesh == True:
            try:
                self.fine_disturbed_flow_grid
            except AttributeError:
                self.calc_disturbed_flow_grid(fine_mesh = True)
            return self.fine_disturbed_flow_grid
        else:
            try:
                self.coarse_disturbed_flow_grid
            except AttributeError:
                self.calc_disturbed_flow_grid(fine_mesh = False)
            return self.coarse_disturbed_flow_grid
    
    def get_disturbed_flow_at_point(self, pnt_coords, mag = False, fine_mesh = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """
        flow_field = self.get_flow_field()

        # check if disturbed flow grid needs updating
        if self.grid_outdated() or (fine_mesh == True and not hasattr(self, 'fine_disturbed_flow_grid')) or (fine_mesh == False and not hasattr(self, 'coarse_disturbed_flow_grid')):
            self.calc_disturbed_flow_grid(fine_mesh)
        
        disturbed_flow_grid = self.get_disturbed_flow_grid(fine_mesh)
        
        origin_coords = np.array([0,0,0])
        rel_x_index, rel_y_index, rel_z_index = relative_index(origin_coords, pnt_coords, flow_field)
        disturbed_flow_at_point = disturbed_flow_grid[rel_x_index, rel_y_index, rel_z_index]

        if mag == True:
            disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
        return disturbed_flow_at_point
    
    def calc_disturbed_flow_grid(self, fine_mesh = False):

        self.set_u_j()
        u_j = self.get_u_j()
        flow_field = self.get_flow_field()
        undisturbed_flow_grid = flow_field.get_flow()
        
        # u, v and w represent the i, j and k components of the speeds at each point, respectively
        # x_grid, y_grid, z_grid are meshgrids corresponding to the x, y and z components at each coordinate
        #x_coords, y_coords, z_coords, u, v, w = self.generate_disturbed_flow_grid()

        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        len_x, len_y, len_z = len(x_coords), len(y_coords), len(z_coords)
        disturbed_flow_grid = np.zeros((len_x, len_y, len_z, 3))

        origin_coords = np.array([0,0,0])

        if fine_mesh == False:
            wakes = self.get_wake_field().get_wakes()
            for w in wakes:
                turbine_coords = w.get_turbine().get_coords()
                self.set_u_ij(turbine_coords)
                u_ij = self.get_u_ij()
                # assumes that u is orthogonal with turbine surface
                x_index, y_index, z_index = relative_index(origin_coords, turbine_coords, flow_field)
                disturbed_flow_grid[x_index, y_index, z_index] = self.calc_combination_speed_at_point(turbine_coords, flow_field, u_j, u_ij, False)

        else:
            for i in range(len_x):
                for j in range(len_y):
                    for k in range(len_z):
                        pnt_coords = np.array([x_coords[i], y_coords[j], z_coords[k]])
                        self.set_u_ij(pnt_coords)
                        u_ij = self.get_u_ij()
                        x_index, y_index, z_index = relative_index(origin_coords, pnt_coords, flow_field)
                        disturbed_flow_grid[x_index, y_index, z_index] = self.calc_combination_speed_at_point(pnt_coords, flow_field, u_j, u_ij, False)

        self.set_disturbed_flow_grid(disturbed_flow_grid, fine_mesh)
        
    def calc_velocity_ratio(self, u_j, u_ij):
        """
         Returns an array of the ratio of the wake velocity to the freestream velocity
         for each turbine j at a point i

        """

        u_ij = set_below_abs_tolerance_to_zero(u_ij)
        u_j = set_below_abs_tolerance_to_zero(u_j)

        # set error handling to known, 'ignore' state to execute
        # divisions without divide by zero error.
        with np.errstate(all="ignore"):
            wake_freestream_velocity_ratio = u_ij/u_j

        # Set all results from of zero division to zero.
        wake_freestream_velocity_ratio = set_nan_or_inf_to_zero(wake_freestream_velocity_ratio)

        return wake_freestream_velocity_ratio
