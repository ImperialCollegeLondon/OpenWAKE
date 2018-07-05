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
    
    def set_u_ij(self, pnt_coords, flow_field, wake_field):
        """
        Sets the list of speed magnitudes at turbine i (pnt_coords) due to the wake of turbine j
        param u_ij list of float or int
        """
        wakes = wake_field.get_wakes()
        self.u_ij = np.array([w.get_disturbed_flow_at_point(pnt_coords, flow_field, False ) for w in wakes ] )
        self.u_ij = np.array( self.u_ij )

    def set_u_j(self, flow_field, wake_field):
        """
        Sets the list of speed magnitudes at turbine j TODO undisturbed or disturbed???
        param u_j list of float or int
        """
        wakes = wake_field.get_wakes()
        self.u_j = np.array( [ ( flow_field.get_undisturbed_flow_at_point(w.get_turbine().get_coords(), False ) ) for w in wakes ] )

    def set_disturbed_flow_grid(self, disturbed_flow_grid, fine_mesh):
        if fine_mesh == True:
            self.fine_disturbed_flow_grid = np.array(disturbed_flow_grid, dtype=np.float64)
        else:
            self.coarse_disturbed_flow_grid = np.array(disturbed_flow_grid, dtype=np.float64)
        self.set_grid_outdated(False)

    def get_disturbed_flow_grid(self, flow_field, wake_field, fine_mesh):
        if self.is_grid_outdated:
            self.calc_disturbed_flow_grid(flow_field, wake_field, fine_mesh)
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
    
    def get_disturbed_flow_at_point(self, pnt_coords, flow_field, wake_field, mag = False, direction = False, fine_mesh = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """
        wakes = wake_field.get_wakes()
        # check if disturbed flow grid needs updating
        if self.is_grid_outdated or (fine_mesh == True and not hasattr(self, 'fine_disturbed_flow_grid')) or (fine_mesh == False and not hasattr(self, 'coarse_disturbed_flow_grid')):
            self.calc_disturbed_flow_grid(flow_field, wake_field, fine_mesh)
        
        disturbed_flow_grid = self.get_disturbed_flow_grid(flow_field, wake_field, fine_mesh)
        
        dx, dy, dz = flow_field.get_dx(), flow_field.get_dy(), flow_field.get_dz()
        rel_x_index, rel_y_index, rel_z_index = int( pnt_coords[0] / dx ), int( pnt_coords[1] / dy ), int( pnt_coords[2] / dz )
        disturbed_flow_at_point = np.array(disturbed_flow_grid[rel_x_index, rel_y_index, rel_z_index], dtype=np.float64)

        if mag == True:
            disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
        elif direction == True:
            try:
                disturbed_flow_at_point = disturbed_flow_at_point / np.linalg.norm(disturbed_flow_at_point, 2)
            except ZeroDivisionError:
                disturbed_flow_at_point = np.array([0,0,0])

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
        
        end_y_dash_reached, end_z_dash_reached = False, False
        
        wakes = self.get_wake_field().get_wakes()

        coords_flattened = np.array([x_coords, y_coords, z_coords]).flatten()
        coords_len = len( coords_flattened )
        dy, dz = flow_field.get_dy(), flow_field.get_dz()
        
        for w in wakes:
            turbine = w.get_turbine()
            turbine_coords = turbine.get_coords()
            turbine_radius = turbine.get_radius()
            flow_mag_at_turbine = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
            thrust_coefficient = turbine.calc_thrust_coefficient(flow_mag_at_turbine)
            start_x_index, start_y_index, start_z_index = relative_index(origin_coords, turbine_coords, flow_field)
            
            if fine_mesh == False:
                self.set_u_ij(turbine_coords, flow_field, wake_field)
                u_ij = self.get_u_ij()
                disturbed_flow_grid[start_x_index, start_y_index, start_z_index] = self.calc_combination_speed_at_point(turbine_coords, flow_field, u_j, u_ij, False)
            else:

##                for c in range(len_x * len_y * len_z):
##                    # (len_z * len_y) is the number of flattened elements in each x row
##                    # (len_z) is the number of flattened elements in each y column of each x row
##                      x_index, y_index, z_index = int( c / ( len_z * len_y ) ), \
##                                    int( ( c % ( len_z * len_y ) ) / len_z), \
##                                    int( c % len_y )
##                    if x_index >= start_x_index and y_index >= start_y_index and z_index >= start_z_index:
##                        x = x_coords[x_index]
##                        y = y_coords[y_index]
##                        z = z_coords[z_index]
                
                # start from each turbine location and iterate over all coordinates in its wake
                end_x_index = len_x - 1
                for i in range(start_x_index, end_x_index + 1):
                    x = x_coords[i]
                    wake_radius = w.calc_wake_radius([x - turbine_coords[0], turbine_coords[1], turbine_coords[2]], turbine_coords, flow_field, turbine_radius, thrust_coefficient)
                    
                    # if the coordinate at the wake radius exceeds the y or z boundary, set end index to boundary
                    # if symmetrical (dash) indices at wake radius coordinate exceed the y or z boundary, set end index to boundary
                    max_y_berth_index, max_z_berth_index = np.argmax([len_y - 1 - start_y_index, start_y_index]), np.argmax([len_z - 1 - start_z_index, start_z_index])
                    y_increment, z_increment = (-1)**max_y_berth_index, (-1)**max_z_berth_index
                    # project direction of wake_radius onto absolute axis
                    turbine = w.get_turbine()
                    turbine_direction = turbine.get_direction()
                    turbine_coords = turbine.get_coords()
                    turbine_y_angle = np.arcsin(turbine_direction[1] / turbine_direction[0] )
                    turbine_z_angle = np.arcsin(turbine_direction[2] / turbine_direction[0] )
                    end_y_index, end_z_index = find_index( y_coords, turbine_coords[1] + ( y_increment * wake_radius * np.cos(turbine_y_angle) )),\
                                               find_index( z_coords, turbine_coords[2] + ( z_increment * wake_radius * np.cos(turbine_z_angle) ))

                    end_y_index = -1 if end_y_index == 0 else end_y_index
                    end_z_index = -1 if end_z_index == 0 else end_z_index

                    #TODO interpolate flow for higher resolution
                    end_y_dash_reached = False

                    #if y_index <= end_y_index and y_index >= 0:

                    for j in range(start_y_index, end_y_index + 1, y_increment):
                        
                        y = y_coords[j]

                        if not end_y_dash_reached:
                            # reflect over turbine direction axis
                            j_dash = int( round( j - ( 2 * ( j - start_y_index ) * np.cos( turbine_y_angle )**2 ) ) )
                            ij_dash = int( round( i + ( 2 * ( j - start_y_index ) * np.cos( turbine_y_angle ) * np.sin( turbine_y_angle ) ) ) )
                            if ij_dash >= 0 and ij_dash < len_x and j_dash >= 0 and j_dash < len_y:
                                xy_dash = [x_coords[ij_dash], y_coords[j_dash]]
                            else:
                                end_y_dash_reached = True
                        
                        # if y has already been found not to be outside wake, break to next x value
                        end_z_dash_reached = False
                        for k in range(start_z_index, end_z_index + 1, z_increment):
                            z = z_coords[k]

                            if not end_z_dash_reached:
                                # reflect over turbine direction axis
                                k_dash = int( round( k - ( 2 * ( k - start_z_index ) * np.cos( turbine_z_angle )**2 ) ) )
                                ik_dash = int( round( i + ( 2 * ( k - start_z_index ) * np.cos( turbine_z_angle ) * np.sin( turbine_z_angle ) ) ) )
                                if ik_dash >= 0 and ik_dash < len_x and k_dash >= 0 and k_dash < len_z:
                                    xz_dash = [x_coords[ik_dash], z_coords[k_dash]]
                                else:
                                    end_z_dash_reached = True
                            
                            # if this coordinate is not in wake of turbine, skip coordinate and remainder of coordinates in current loop
                            pnt_coords = np.array([x, y, z])
                            # if pnt_coords has already had combination speed calculated, continue to next iteration 
                            if (x, y, z) not in checked_coords:
                                self.set_u_ij(pnt_coords, flow_field, wake_field)
                                u_ij = self.get_u_ij()
                                disturbed_flow_grid[i, j, k] = self.calc_combination_speed_at_point(pnt_coords, flow_field, u_j, u_ij, False)
                                checked_coords.append((x,y,z))
                                
                            if not end_y_dash_reached:
                                pnt_coords_dash = np.array([xy_dash[0], xy_dash[1], z])
                                if (xy_dash[0], xy_dash[1], z) not in checked_coords:
                                    self.set_u_ij(pnt_coords_dash, flow_field, wake_field)
                                    u_ij = self.get_u_ij()
                                    disturbed_flow_grid[ij_dash, j_dash, k] = self.calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)
                                    checked_coords.append((xy_dash[0], xy_dash[1], z))

                            if not end_z_dash_reached:
                                pnt_coords_dash = np.array([xz_dash[0], y, xz_dash[1]])
                                if (xz_dash[0], y, xz_dash[1]) not in checked_coords:
                                    self.set_u_ij(pnt_coords_dash, flow_field, wake_field)
                                    u_ij = self.get_u_ij()
                                    disturbed_flow_grid[ik_dash, j, k_dash] = self.calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)
                                    checked_coords.append((xz_dash[0], y, xz_dash[1]))

                            if not (end_y_dash_reached or end_z_dash_reached):
                                ijz_dash = round( i +  ij_dash + ik_dash )
                                if ijz_dash >= 0 and ijz_dash < len_x:
                                    xyz_dash = [x_coords[ijz_dash], y_coords[j_dash], z_coords[k_dash]]
                                    pnt_coords_dash = np.array(xyz_dash)
                                    if (xyz_dash[0], xyz_dash[1], xyz_dash[2]) not in checked_coords:
                                        self.set_u_ij(pnt_coords_dash, flow_field, wake_field)
                                        u_ij = self.get_u_ij()
                                        disturbed_flow_grid[ijz_dash, j_dash, k_dash] = self.calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)
                                        checked_coords.append((xyz_dash[0], xyz_dash[1], xyz_dash[2]))

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
