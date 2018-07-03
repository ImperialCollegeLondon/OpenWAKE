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
        self.calc_disturbed_flow_grid(flow_field, wake_field, True)
        self.set_grid_outdated(False)

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

##    def is_grid_outdated(self):
##        wake_field = self.get_wake_field()
##        wakes = wake_field.get_wakes()
##        num_wakes = wake_field.get_num_wakes()
##        u_j_size = self.get_u_j().size
##        if u_j_size < num_wakes:
##            return True
##        else:
##            return False
    
    def set_u_ij(self, pnt_coords, flow_field, wake_field):
        """
        Sets the list of speed magnitudes at turbine i (pnt_coords) due to the wake of turbine j
        param u_ij list of float or int
        """
        wakes = wake_field.get_wakes()
        self.u_ij = np.array([w.get_disturbed_flow_at_point(pnt_coords, flow_field, False) for w in wakes])

    def set_u_j(self, flow_field, wake_field):
        """
        Sets the list of speed magnitudes at turbine j TODO undisturbed or disturbed???
        param u_j list of float or int
        """
        wakes = wake_field.get_wakes()
        self.u_j = np.array([flow_field.get_undisturbed_flow_at_point(w.get_turbine().get_coords(), False) for w in wakes])

    def set_disturbed_flow_grid(self, disturbed_flow_grid, fine_mesh):
        if fine_mesh == True:
            self.fine_disturbed_flow_grid = np.array(disturbed_flow_grid, dtype=np.float64)
        else:
            self.coarse_disturbed_flow_grid = np.array(disturbed_flow_grid, dtype=np.float64)
        self.set_grid_outdated(False)

    def get_disturbed_flow_grid(self, flow_field, wake_field, fine_mesh):
        if fine_mesh == True:
            try:
                self.fine_disturbed_flow_grid
            except AttributeError:
                self.calc_disturbed_flow_grid(flow_field, wake_field, fine_mesh = True)
            return self.fine_disturbed_flow_grid
        else:
            try:
                self.coarse_disturbed_flow_grid
            except AttributeError:
                self.calc_disturbed_flow_grid(flow_field, wake_field, fine_mesh = False)
            return self.coarse_disturbed_flow_grid
    
    def get_disturbed_flow_at_point(self, pnt_coords, flow_field, wake_field, mag = False, fine_mesh = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """

        # check if disturbed flow grid needs updating
        if self.is_grid_outdated or (fine_mesh == True and not hasattr(self, 'fine_disturbed_flow_grid')) or (fine_mesh == False and not hasattr(self, 'coarse_disturbed_flow_grid')):
            self.calc_disturbed_flow_grid(flow_field, wake_field, fine_mesh)
        
        disturbed_flow_grid = self.get_disturbed_flow_grid(flow_field, wake_field, fine_mesh)
        
        origin_coords = np.array([0,0,0])
        rel_x_index, rel_y_index, rel_z_index = relative_index(origin_coords, pnt_coords, flow_field)
        disturbed_flow_at_point = np.array(disturbed_flow_grid[rel_x_index, rel_y_index, rel_z_index], dtype=np.float64)

        if mag == True:
            disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point

        return disturbed_flow_at_point

    def set_grid_outdated(self, grid_outdated):
        """
        Set to true every time wake parameters are changed, except for turbine location
        """
        self.is_grid_outdated = grid_outdated
        
    def calc_disturbed_flow_grid(self, flow_field, wake_field, fine_mesh = True):
        
        undisturbed_flow_grid = flow_field.get_flow()
        self.set_u_j(flow_field, wake_field)
        u_j = self.get_u_j()
        
        # u, v and w represent the i, j and k components of the speeds at each point, respectively
        # x_grid, y_grid, z_grid are meshgrids corresponding to the x, y and z components at each coordinate

        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        len_x, len_y, len_z = undisturbed_flow_grid.shape[0:3]
        disturbed_flow_grid = np.array(undisturbed_flow_grid, dtype = np.float64)

        origin_coords = np.array([0,0,0])

        checked_coords = []
        
        end_y_reached, end_y_dash_reached, end_z_reached, end_z_dash_reached = False, False, False, False
        
        wakes = self.get_wake_field().get_wakes()
        
        for w in wakes:
            turbine = w.get_turbine()
            turbine_coords = turbine.get_coords()
            turbine_radius = turbine.get_radius()
            flow_mag_at_turbine = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
            thrust_coefficient = turbine.calc_thrust_coefficient(flow_mag_at_turbine)
            
            if fine_mesh == False:
                self.set_u_ij(turbine_coords, flow_field, wake_field)
                u_ij = self.get_u_ij()
                x_index, y_index, z_index = relative_index(origin_coords, turbine_coords, flow_field)
                disturbed_flow_grid[x_index, y_index, z_index] = self.calc_combination_speed_at_point(turbine_coords, flow_field, u_j, u_ij, False)
            else:
                # start from each turbine location and iterate over all coordinates in its wake
                start_x_index, start_y_index, start_z_index = relative_index(origin_coords, turbine_coords, flow_field)
                end_x_index = len_x - 1
                for i in range(start_x_index, end_x_index + 1):
                    x = x_coords[i]
                    wake_radius = w.calc_wake_radius([x, turbine_coords[1], turbine_coords[2]], turbine_coords, flow_field, turbine_radius, thrust_coefficient)
                    
                    # if the coordinate at the wake radius exceeds the y or z boundary, set end index to boundary
                    # if symmetrical (dash) indices at wake radius coordinate exceed the y or z boundary, set end index to boundary
                    max_y_berth_index, max_z_berth_index = np.argmax([len_y - 1 - start_y_index, start_y_index]), np.argmax([len_z - 1 - start_z_index, start_z_index])
                    y_increment, z_increment = (-1)**max_y_berth_index, (-1)**max_z_berth_index
                    end_y_index, end_z_index = find_index(y_coords, turbine_coords[1] + (y_increment * wake_radius)),\
                                               find_index(z_coords, turbine_coords[2] + (z_increment * wake_radius))
                        
                    for j in range(start_y_index, end_y_index + 1, y_increment):
                        
                        y = y_coords[j]
    
                        j_dash = (2 * start_y_index) - (y_increment * j)
                        if j_dash >= 0 and j_dash < len_y:
                            y_dash = y_coords[j_dash]
                        else:
                            y_dash = -1
                            end_y_dash_reached = True
                        
                        # if y has already been found not to be outside wake, break to next x value
                        for k in range(start_z_index, end_z_index + 1, z_increment):

                            z = z_coords[k]

                            k_dash = (2 * start_z_index) - (z_increment * k)
                            if k_dash >= 0 and k_dash < len_z:
                                z_dash = z_coords[k_dash]
                            else:
                                z_dash = -1
                                end_z_dash_reached = True
                            
                            # if this coordinate is not in wake of turbine, skip coordinate and remainder of coordinates in current loop
                            pnt_coords = np.array([x, y, z])
                            if not (end_y_dash_reached or end_z_dash_reached):
                                pnt_coords_dash = np.array([x, y_dash, z_dash])
                            
                            # if pnt_coords has already had combination speed calculated, continue to next iteration 
                            if (x, y, z) not in checked_coords:
                                self.set_u_ij(pnt_coords, flow_field, wake_field)
                                u_ij = self.get_u_ij()
                                disturbed_flow_grid[i, j, k] = self.calc_combination_speed_at_point(pnt_coords, flow_field, u_j, u_ij, False)
                                checked_coords.append((x,y,z))
                                #print("1", disturbed_flow_grid[i,j,k], [i,j,k])

                            if not (end_y_dash_reached or end_z_dash_reached) and (x, y_dash, z_dash) not in checked_coords:
                                self.set_u_ij(pnt_coords_dash, flow_field, wake_field)
                                u_ij = self.get_u_ij()
                                disturbed_flow_grid[i, j_dash, k_dash] = self.calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)
                                checked_coords.append((x, y_dash, z_dash))
                                #print("2", disturbed_flow_grid[i,j_dash,k_dash], [i,j_dash,k_dash])

                            if end_z_dash_reached:
                                end_z_dash_reached = False
                                break
                            
                        if end_y_dash_reached:
                            end_y_dash_reached = False
                            break

        self.set_disturbed_flow_grid(disturbed_flow_grid, fine_mesh)
        #print(disturbed_flow_grid[15:, 10, 20])
        #print(disturbed_flow_grid[20:, 5, 15])
        #print(disturbed_flow_grid[25:, 7, 17])
        
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
        #print(wake_freestream_velocity_ratio, u_ij, u_j)
        return wake_freestream_velocity_ratio
