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

    def __init__(self, turbine, flow_field = FlowField(), wake_field = WakeField()):
        super(BaseWake, self).__init__(flow_field, wake_field)
        self.set_turbine(turbine)
        self.wake_field.add_wakes([self])
        self.calc_multiplier_grid(flow_field)

    def is_in_wake(self, rel_pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):

        """
        Returns True if point is in wake.
        param pnt_coords coordinates to check
        """
        
        #x_rel, y_rel, z_rel = relative_position(turbine_coords, pnt_coords, flow_field)
        x_rel, y_rel, z_rel = rel_pnt_coords
    
        if (x_rel < 0.0):
            return False
        else:
            
            wake_radius = self.calc_wake_radius(rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient)
            return wake_radius > np.linalg.norm([y_rel, z_rel], 2)

    def set_grid_outdated(self, grid_outdated):
        """
        Set to true every time wake parameters are changed, except for turbine location
        """
        self.is_grid_outdated = grid_outdated

    def get_multiplier_grid(self):
        try:
            self.multiplier_grid
        except AttributeError:
            self.calc_multiplier_grid()
            
        return self.multiplier_grid

    def set_multiplier_grid(self, multiplier_grid):
        self.multiplier_grid = multiplier_grid
        self.set_grid_outdated(False)

    def get_disturbed_flow_at_point(self, pnt_coords, flow_field, mag = False, direction = False):
        """
        function that gets the created disturbed flow mesh of this wake combination, and accesses a particular
        point from that array.
        """

        # TODO adapt to take turbine orientation into account. take in absolute pnt_coords, and rotate to get relative pnt_coords
        # multiplier_grid will be relative grid system
        
        if flow_field.is_in_flow_field(pnt_coords):
            turbine = self.get_turbine()
            turbine_coords = turbine.get_coords()
            turbine_radius = turbine.get_radius()
            turbine_direction = turbine.get_direction()
            u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
            thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

            # check if disturbed flow grid needs updating
            if self.is_grid_outdated:
                self.calc_multiplier_grid(flow_field)

            multiplier_grid = self.get_multiplier_grid()
            origin_coords = np.array([0,0,0])
            x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
            #r_list = self.calc_r(turbine_coords, x_coords, y_coords, z_coords)
            #r_coords = r_list[0]
            #len_r = r_list[-2]
            #max_len,dx, dr = flow_field.get_dx(), r_list[-1]
            rel_pnt_coords = relative_position(turbine_coords, pnt_coords, turbine_direction, True)
            #x_index, r_index = abs(int((pnt_coords[0] - turbine_coords[0]) / dx)), min(abs(int(np.linalg.norm(pnt_coords[1:] - turbine_coords[1:], 2) / dr)), multiplier_grid.shape[1] - 1)
            #x_index, r_index = min( abs( int( rel_coords[0] / dx ) ), multiplier_grid.shape[0] - 1 ), min( abs( int( np.linalg.norm( rel_coords[1:], 2 ) / dr ) ), multiplier_grid.shape[1] - 1 )

            if self.is_in_wake(rel_pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
                max_len, dx, dr = self.calc_rel_frame( flow_field )
                x_index, r_index = int( round( rel_pnt_coords[0] / dx ) ), int( round( np.linalg.norm(rel_pnt_coords[1:], 2) / dr ) )
                disturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point( pnt_coords, False ) * multiplier_grid[x_index, r_index]
            else:
                disturbed_flow_at_point = flow_field.get_undisturbed_flow_at_point( pnt_coords, False )

        else:
            disturbed_flow_at_point = np.array([0, 0, 0])

        if mag == True:
            disturbed_flow_at_point = np.linalg.norm(disturbed_flow_at_point, 2) if isinstance(disturbed_flow_at_point, (list, np.ndarray)) else disturbed_flow_at_point
        elif direction == True:
            try:
                disturbed_flow_at_point = disturbed_flow_at_point / np.linalg.norm(disturbed_flow_at_point, 2)
            except ZeroDivisionError:
                disturbed_flow_at_point = np.array([0,0,0])

        return disturbed_flow_at_point

##    def calc_indices(self, turbine_coords, x_coords, y_coords, z_coords):
##
##        start_x_index = find_index(x_coords, turbine_coords[0])
##        len_x = x_coords.size
##        end_x_index = len_x - 1
##        
##        start_y_index = find_index(y_coords, turbine_coords[1])
##        len_y = y_coords.size
##        end_y_index = len_y - 1
##
##        start_z_index = find_index(z_coords, turbine_coords[2])
##        len_z = z_coords.size
##        end_z_index = len_z - 1
##        return [start_x_index, end_x_index, start_y_index, end_y_index, start_z_index, end_z_index]

    def calc_rel_frame(self, flow_field):
        
        len_x, len_y, len_z = flow_field.get_x_coords().size, flow_field.get_y_coords().size, flow_field.get_z_coords().size
        # maximum possible length is where turbine is located at corner and facing opposite corner
        length = int( ( 2 * max([len_x, len_y, len_z])**2 )**0.5 )
        # dx and dr should allow for finest difference in absolute coordinate system
        dx, dy, dz = flow_field.get_dx(), flow_field.get_dy(), flow_field.get_dz()
        diff = min([dx, dy, dz])
        dx = dr = diff
        return length, dx, dr

##    def calc_r(self, turbine_coords, x_coords, y_coords, z_coords):
##
##        start_x_index, end_x_index, start_y_index, end_y_index, start_z_index, end_z_index = self.calc_indices(turbine_coords, x_coords, y_coords, z_coords)
##
##        max_domain_berth_index = np.argmax([end_y_index - start_y_index, start_y_index, end_z_index - start_z_index, start_z_index])
##
##        # r_increment == 1 if upper y or z limit is widest berth, == -1 if lower y or z limit is widest berth
##        r_increment = (-1)**max_domain_berth_index
##        r_coord_index = (int(max_domain_berth_index / 2) % 2) + 1
##        start_r_index = [start_y_index, start_z_index][r_coord_index - 1]
##        r_coords = [y_coords, z_coords][r_coord_index - 1]
##        len_r = len(r_coords)
##        self.r_coords = r_coords
##        self.r_coord_index = r_coord_index
##        self.start_r_index = start_r_index
##        self.r_increment = r_increment
##        self.len_r = len_r
##        self.dr = abs(r_coords[1] - r_coords[0])
##        return [self.r_coords, self.r_coord_index, self.start_r_index, self.r_increment, self.len_r, self.dr]
        
    def calc_multiplier_grid(self, flow_field):
        """
        Given the undisturbed flow grid, empty u, v, w arrays and x,y,z meshgrids,
        populates a vicinity relative to the turbine location
        Assumes that the turbine has rotated to face the flow
        Calculates the multipliers of the disturbed flow grid along the single longest radial line from turbine to farthest boundary
        """

        # the 0th dimension, x, corresponds to the axial distance along the flow at turbine vector,
        # from the centre of the turbine
        # the 1st dimension, r, corresponds to the radial distance from the flow at turbine vector

        # get variables
        turbine = self.get_turbine()
        turbine_radius = turbine.get_radius()
        flow_field = self.get_flow_field()
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        # the dimensions of disturbed_flow_grid should allow for the maximum berth between the turbine and the domain boundaries
   
        # TODO mulitplier should have 2 or 3 elements: u,v,w
        # this is a 2D grid of the multiplier along the longest radial line for each value of x relative to flow direction at turbine
        # 0th (x) dimension should allow for maximum berth in absolute combination coordinate system
        # 1st (r) dimension should allow for maximum wake_radius at that value of x
        
        max_len, dx, dr = self.calc_rel_frame(flow_field)
        multiplier_grid = np.ones((max_len, max_len))
        
        #for i in range(start_x_index, end_x_index + 1):
        for i in range(max_len):
            x = i * dx
            rel_pnt_coords = [x, 0, 0]
            #wake_radius = self.calc_wake_radius([x, turbine_coords[1], turbine_coords[2]], turbine_coords, flow_field, turbine_radius, thrust_coefficient)
            wake_radius = self.calc_wake_radius(rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient)
            
            # get turbine coordinate corresponding to reference radial line r, and add or subtract wake_radius from it depending on
            # wheter that reference line extends further above or below turbine in domain
            #end_r_index = find_index(r_coords, turbine_coords[r_coord_index] + (r_increment * wake_radius))
            
            #for j in range(start_r_index, end_r_index + 1, r_increment):
            for j in range( min( int( ( wake_radius + ( wake_radius % dr ) ) / dr ), max_len ) ):
                r = j * dr
                # calculate the multiplier along the longest (positive or negative) radial y or z line (r) at this value of x along wake
                # then apply that multiplier to the reflected r coordinate and the corresonding non-r coordinate and its reflected coordinate
                rel_pnt_coords = [x, r, 0]
                #pnt_coords = np.array([x_coords[i], y_coords[j], z_coords[start_z_index]]) if r_coord_index == 1 else np.array([x_coords[i], y_coords[start_y_index], z_coords[j]])
                if self.is_in_wake(rel_pnt_coords, turbine_coords, turbine_radius, thrust_coefficient, flow_field):
                    multiplier = 1 - self.calc_vrf_at_point(rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient, u_0)
                    #multiplier_grid[i - start_x_index, j - start_r_index] = multiplier
                    multiplier_grid[ i, j ] = multiplier
                    
        self.set_multiplier_grid(multiplier_grid)

    def get_turbine(self):
        return self.turbine

    def set_turbine(self, turbine):
        try:
            assert isinstance(turbine, BaseTurbine)
        except AssertionError:
            raise TypeError("'turbine' must be of type 'Turbine'")
        else:
            self.turbine = turbine
            # assume that turbine is controlled to face incoming undisturbed flow
            self.turbine.set_direction(self.get_flow_field().get_undisturbed_flow_at_point(self.turbine.get_coords(), direction = True))
