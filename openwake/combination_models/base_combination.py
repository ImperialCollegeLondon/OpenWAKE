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
        self.set_u_j( flow_field, wake_field )
        self.calc_disturbed_flow_grid(flow_field, wake_field, True)

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
        TODO only takes undisturbed point into account
        """
        wakes = wake_field.get_wakes()
        self.u_ij = np.array( [ w.get_disturbed_flow_at_point(pnt_coords, flow_field, False ) for w in wakes ] )
        return self.u_ij

    def set_u_j(self, flow_field, wake_field):
        """
        Sets the list of speed magnitudes at turbine j
        param u_j list of float or int

        Start with undisturbed flow. Place each new turbine.
        For each other (old) turbine location, recalculate new flow (u_j) as a result.
        
        """
        
        wakes = wake_field.get_wakes()
        num_wakes = wake_field.get_num_wakes()
        undisturbed_flow_grid = flow_field.get_flow()
        calc_combination_speed_at_point = self.calc_combination_speed_at_point
        u_j = np.array( [ ( flow_field.get_undisturbed_flow_at_point( w.get_turbine().get_coords(), False ) ) for w in wakes ] )
        u_ij = np.array( u_j )
        w = 1

        # TODO could do recursively, until tolerance?
        # place each turbine at a 'new' turbine location
        for w in range( num_wakes ):
            
            # loop through each other turbine location and update their u_j
            for ww in range( num_wakes ):
                if ww != w:
                    turbine_coords = wakes[ ww ].get_turbine().get_coords()
                    # flow at turbine ww due to turbine w
                    u_ij[ w ] = wakes[w].get_disturbed_flow_at_point( turbine_coords, flow_field, False, False )
                    # flow at turbine ww due to existing u_j and u_ij
                    u_j[ ww ] = calc_combination_speed_at_point( turbine_coords, flow_field, u_j, u_ij, False )
                    
        self.u_j = u_j                              
            
        return self.u_j
    
    def set_disturbed_flow_grid(self, disturbed_flow_grid, fine_mesh):
        if fine_mesh == True:
            self.fine_disturbed_flow_grid = np.array(disturbed_flow_grid, dtype=np.float64)
        else:
            self.coarse_disturbed_flow_grid = np.array(disturbed_flow_grid, dtype=np.float64)
        self.set_grid_outdated( False )

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
        if flow_field.is_in_flow_field( pnt_coords ):
            wakes = wake_field.get_wakes()
            # check if disturbed flow grid needs updating
            if self.is_grid_outdated or (fine_mesh == True and not hasattr(self, 'fine_disturbed_flow_grid')) or (fine_mesh == False and not hasattr(self, 'coarse_disturbed_flow_grid')):
                self.calc_disturbed_flow_grid( flow_field, wake_field, fine_mesh )
            
            disturbed_flow_grid = self.get_disturbed_flow_grid(flow_field, wake_field, fine_mesh)
            
            dx, dy, dz = flow_field.get_diff()
            rel_x_index, rel_y_index, rel_z_index = relative_index([0,0,0], pnt_coords, [dx, dy, dz])
            disturbed_flow_at_point = np.array(disturbed_flow_grid[rel_x_index, rel_y_index, rel_z_index], dtype=np.float64)

            if mag == True:
                disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
            elif direction == True:
                try:
                    disturbed_flow_at_point = disturbed_flow_at_point / np.linalg.norm(disturbed_flow_at_point, 2)
                except ZeroDivisionError:
                    disturbed_flow_at_point = np.array([0,0,0])

        else:
            disturbed_flow_at_point = np.array([0, 0, 0])

        return disturbed_flow_at_point
    
    def calc_disturbed_flow_grid(self, flow_field, wake_field, fine_mesh = True):
        undisturbed_flow_grid = flow_field.get_flow()
        self.set_u_j(flow_field, wake_field)
        u_j = self.get_u_j()
        
        # u, v and w represent the i, j and k components of the speeds at each point, respectively
        # x_grid, y_grid, z_grid are meshgrids corresponding to the x, y and z components at each coordinate

        x_coords, y_coords, z_coords = flow_field.get_coords()
        len_x, len_y, len_z = undisturbed_flow_grid.shape[0:3]
        disturbed_flow_grid = np.array(undisturbed_flow_grid, dtype = np.float64)

        origin_coords = np.array([0,0,0])

        checked_indices = np.zeros( ( len_x, len_y, len_z ) )
        
        wakes = self.get_wake_field().get_wakes()

        coords_flattened = np.array([x_coords, y_coords, z_coords]).flatten()
        coords_len = len( coords_flattened )
        diff = flow_field.get_diff()
        dx, dy, dz = diff
        
##              TODO try optimizing by vectorizing calc_combination_speed_at_point and applying to disturbed flow grid 
        calc_combination_speed_at_point = self.calc_combination_speed_at_point
 #       calc_combination_speed_vectorized = np.vectorize( calc_combination_speed_at_point )
        set_u_ij = self.set_u_ij
        
##        for w in wakes:
##            turbine = w.get_turbine()
##            turbine_direction = turbine.get_direction()
##            turbine_coords = turbine.get_coords()
##            turbine_radius = turbine.get_radius()
##            turbine_y_angle = np.arctan(turbine_direction[1] / turbine_direction[0])
##            turbine_z_angle = np.arcsin(turbine_direction[2] / turbine_direction[0])
##            flow_mag_at_turbine = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
##            thrust_coefficient = turbine.calc_thrust_coefficient(flow_mag_at_turbine)
##            start_x_index, start_y_index, start_z_index = relative_index(origin_coords, turbine_coords, [dx, dy, dz])
##            
        if fine_mesh == False:
            u_ij = set_u_ij(turbine_coords, flow_field, wake_field)
            disturbed_flow_grid[start_x_index, start_y_index, start_z_index] = calc_combination_speed_at_point(turbine_coords, flow_field, u_j, u_ij, False)
        else:
##            initial_wake_radius = w.calc_wake_radius( [0, 0, 0], turbine_coords, flow_field, turbine_radius, thrust_coefficient )
##            max_angle = max( turbine_y_angle, turbine_z_angle )
##            x_increment = 1 if max_angle <= np.pi / 2 else -1
##            start_x = ( turbine_coords[0] - ( initial_wake_radius * np.sin( max_angle ) ) )
##            start_x_index = int( round( start_x ) )
##            end_x_index = len_x - 1 if x_increment == 1 else -1
                                   
            for c in range(len_x * len_y * len_z):
                # (len_z * len_y) is the number of flattened elements in each x row
                # (len_z) is the number of flattened elements in each y column of each x row
                i, j, k = int( c / ( len_z * len_y ) ), \
                                            int( ( c % ( len_z * len_y ) ) / len_z), \
                                            int( c % len_y )

                x, y, z = x_coords[i], y_coords[j], z_coords[k]
                pnt_coords = np.array([x, y, z])
                
                # for each x from intitial wake to boundary
##                if i not in range( start_x_index, end_x_index, x_increment ):
##                    continue

                flow_magnitude = np.linalg.norm( undisturbed_flow_grid[i,j,k], 2 )
                flow_direction = undisturbed_flow_grid[i,j,k] / flow_magnitude 
                if not any( [ w.is_in_wake( relative_position( w.get_turbine().get_coords(), pnt_coords, flow_direction ),
                                            w.get_turbine().get_coords(), w.get_turbine().get_radius(), \
                                            w.get_turbine().calc_thrust_coefficient( flow_magnitude ), flow_field ) \
                              for w in wakes ] ):
                    continue
                
                
                #x_rel = int( round( ( x - start_x ) * np.cos( turbine_y_angle ) * np.cos( turbine_z_angle ) / dx ) )
                #wake_radius = w.calc_wake_radius([x_rel, turbine_coords[1], turbine_coords[2]], turbine_coords, flow_field, turbine_radius, thrust_coefficient)

##                if j == 0:
##                    end_y_reached, end_y_dash_reached = False, False
##                
##                if k == 0:
##                    end_z_reached, end_z_dash_reached = False, False


##                    end_y_index = int( start_y_index + abs( round( wake_radius * np.cos( turbine_y_angle ) / dy ) ) )
##                    end_z_index = int( start_z_index + abs( round( wake_radius * np.cos( turbine_z_angle ) / dz ) ) )

##                    y_increment = 1 if turbine_y_angle <= np.pi / 2 or turbine_y_angle >= (3 / 2) * np.pi else -1
##                    z_increment = 1 if turbine_z_angle <= np.pi / 2 or turbine_z_angle >= (3 / 2) * np.pi else -1

                #if y_index in range( start_y_index, end_y_increment, y_increment ) and z_index in range( start_z_index, end_z_increment, z_increment ):
                # if y and z indices are within wake radius at this value of x
##                
##                if not end_y_reached:
##                    if j < len_y:
##                        y = y_coords[j]
##                    else:
##                        end_y_reached = True
##
##                if not end_z_reached:
##                    if k < len_z:
##                        z = z_coords[k]
##                    else:
##                        end_z_reached = True
##
##                if not end_y_dash_reached:
##                # reflect over turbine direction axis
##                    j_dash = int( round( j - ( 2 * ( j - start_y_index ) * np.cos( turbine_y_angle )**2 ) ) )
##                    ij_dash = int( round( i + ( 2 * ( j - start_y_index ) * np.cos( turbine_y_angle ) * np.sin( turbine_y_angle ) ) ) )
##                    if ij_dash < len_x and j_dash < len_y :
##                        xy_dash = [x_coords[ij_dash], y_coords[j_dash]]
##                    else:
##                        end_y_dash_reached = True
##
##                if not end_z_dash_reached:
##                    # reflect over turbine direction axis
##                    k_dash = int( round( k - ( 2 * ( k - start_z_index ) * np.cos( turbine_z_angle )**2 ) ) )
##                    ik_dash = int( round( i + ( 2 * ( k - start_z_index ) * np.cos( turbine_z_angle ) * np.sin( turbine_z_angle ) ) ) )
##                    if ik_dash in range( len_x ) and k_dash in range( len_z ):
##                        xz_dash = [x_coords[ik_dash], z_coords[k_dash]]
##                    else:
##                        end_z_dash_reached = True

                if not checked_indices[i, j, k]:
                    checked_indices[i, j, k] = 1
                    # if this coordinate is not in wake of turbine, skip coordinate and remainder of coordinates in current loop
                    #pnt_coords = np.array([x, y, z])
                    # if pnt_coords has already had combination speed calculated, continue to next iteration 
                    u_ij = set_u_ij(pnt_coords, flow_field, wake_field)
                    disturbed_flow_grid[i, j, k] = calc_combination_speed_at_point(pnt_coords, flow_field, u_j, u_ij, False)
                    
##                if not end_y_dash_reached and not checked_indices[ij_dash, j_dash, k ]:
##                    checked_indices[ij_dash, j_dash, k ]  = 1
##                    pnt_coords_dash = np.array([xy_dash[0], xy_dash[1], z])
##                    u_ij = set_u_ij(pnt_coords_dash, flow_field, wake_field)
##                    disturbed_flow_grid[ij_dash, j_dash, k] = calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)
##
##                if not end_z_dash_reached and not checked_indices[ ik_dash, j, k_dash ]:
##                    checked_indices[ ik_dash, j, k_dash ] = 1
##                    pnt_coords_dash = np.array([xz_dash[0], y, xz_dash[1]])
##                    u_ij = set_u_ij(pnt_coords_dash, flow_field, wake_field)
##                    disturbed_flow_grid[ik_dash, j, k_dash] = calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)
##
##                ijz_dash = int( round( i +  ij_dash + ik_dash ) )
##                if ijz_dash < len_x:
##                    if not (end_y_dash_reached or end_z_dash_reached) and checked_indices[ ijz_dash, j_dash, k_dash ]:
##                        checked_indices[ ijz_dash, j_dash, k_dash ] = 1
##                        xyz_dash = [x_coords[ijz_dash], y_coords[j_dash], z_coords[k_dash]]
##                        pnt_coords_dash = np.array(xyz_dash)
##                        u_ij = set_u_ij(pnt_coords_dash, flow_field, wake_field)
##                        disturbed_flow_grid[ijz_dash, j_dash, k_dash] = calc_combination_speed_at_point(pnt_coords_dash, flow_field, u_j, u_ij, False)

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
