"""
Implementation of ainslie wake model
"""

from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from wake_models.wake_field_model import WakeField
from flow_field_model.flow import FlowField
import numpy as np
from helpers import *

class Ainslie(BaseWake):
    """
    Implements an Ainslie wake model
    """

    def __init__(self, turbine = BaseTurbine(), flow_field = FlowField(), ambient_intensity = 0.1, wake_field = WakeField()):
        """
        param ambient_intensity (decimal)
        """
        self.set_ambient_intensity(ambient_intensity)
        self.set_centreline_vdf()
        super(Ainslie, self).__init__(turbine, flow_field, wake_field)

    def get_ambient_intensity(self):
        return self.ambient_intensity

    def set_ambient_intensity(self, ambient_intensity = 0.1):
        try:
            assert isinstance(ambient_intensity, float)
        except AssertionError:
            raise TypeError("'ambient_intensity' must be of type 'float'")
        else:
            self.ambient_intensity = ambient_intensity

    def calc_wake_radius(self, rel_pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient):
        """
        Returns the radius of the wake at a point
        given that point is in wake of turbine
        param pnt_coords point at which to calculate the radius of the wake
        """

        turbine_diameter = 2 * turbine_radius
        D_m = self.get_centreline_vdf(turbine_coords, rel_pnt_coords, flow_field)
        b = self.calc_wake_parameter(thrust_coefficient, D_m)
        
        return (turbine_diameter + self.calc_wake_width(b) * turbine_diameter) / 2

    def calc_eddy_viscosity(self, F, K1, b, D_m, K, ambient_intensity, u_i):
        #todo b and d_m can be inf,nan,0, probably due to d_m
        e_wake = F * K1 * b * D_m * u_i
        e_ambient = F * K**2 * ambient_intensity
        ee = e_wake + e_ambient
        return ee

    def calc_filter_function(self, x_rel, turbine_diameter, turbine_coords, flow_field):
        k1, k2, k3, k4, k5 = 4.5, 5.5, 23.32, 1/3, 0.65
        #x_rel = relative_position(turbine_coords, [x, 0, 0], flow_field)[0]
        if x_rel > k2:
            F = 1
        elif x > k1:
            F = k5 + (((x_rel/turbine_diameter) - k1) / k3)**k4
        else:
            F = k5 - (-((x_rel/turbine_diameter)-k1) / k3)**k4
        return F

    def calc_initial_centreline_vdf(self, thrust_coefficient, ambient_intensity):
        k1, k2, k3, k4 = 0.05, 16, 0.5, 10
        D_m = thrust_coefficient - k1 - (((k2 * thrust_coefficient) - k3) * (ambient_intensity / k4))
        return D_m

    def calc_off_centre_vdf(self, r_rel, b, D_m, turbine_diameter, turbine_coords, flow_field):
        #z_rel = relative_position(turbine_coords, [0,0,z], flow_field)[2]
        k1 = -3.56
        with np.errstate(all="ignore"):
            D_mr = D_m * np.exp(k1 * ((r_rel / turbine_diameter) / b)**2)
        return D_mr# if np.isinf(D_mr) == False and np.isnan(D_mr) == False else 1

    def calc_wake_parameter(self, thrust_coefficient, D_m):
        k1, k2, k3 = 3.56, 8, 0.5
        #TODO check this
        with np.errstate(all="ignore"):
            b = ((k1 * thrust_coefficient) / (k2 * D_m * (1 - (k3 * D_m))))**k3
        return b# if np.isinf(b) == False and np.isnan(b) == False else 0

    def calc_wake_width(self, b):
        k1, k2 = 0.5, 3.56
        W = b * (k1 / k2)**k1
        return W

    def set_centreline_vdf(self):
        self.centreline_vdf = np.array([])

    def add_centreline_vdf(self, D_m):
        np.append(self.centreline_vdf, D_m)

    def get_centreline_vdf(self, turbine_coords, rel_pnt_coords, flow_field):
        #rel_x_index, rel_y_index, rel_z_index = relative_index(turbine_coords, pnt_coords, flow_field)
        dx = self.calc_rel_frame(flow_field)[1]
        rel_x_index = int( rel_pnt_coords[0] / dx )
        return self.centreline_vdf[rel_x_index]

    def get_disturbed_flow_grid(self):
        try:
            self.disturbed_flow_grid
        except AttributeError:
            self.calc_disturbed_flow_grid()
        else:
            pass

        return self.disturbed_flow_grid

    def calc_multiplier_grid(self, flow_field):
        """
        The ainslie model is symetrical along the turbine axis - so we shall
        compute one side and reflect it along the turbine axis to find the complete
        wake. Our wake is currently dimensionless - i.e. turbine_diameter = 1.
        Returns the velocity deficit factor.
        """

        ## TODO should this be normed to remain consistent with other models
        undisturbed_flow_grid = flow_field.get_flow()
        # u, v and w represent the i, j and k components of the speeds at each point, respectively
        # x_grid, y_grid, z_grid are meshgrids corresponding to the x, y and z components at each coordinate

        # get variables
        turbine = self.get_turbine()
        turbine_radius = turbine.get_radius()
        turbine_diameter = 2 * turbine_radius
        flow_field = self.get_flow_field()
        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        v_0 = 0
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        ambient_intensity = self.get_ambient_intensity()
        
        max_len, dx, dr = self.calc_rel_frame(flow_field)
        multiplier_grid = np.ones((max_len, max_len, 2))
        
        # Dimensionless constant
        K1 = 0.015

        # Von Karman constant
        K = 0.4

        # Initial filter function (filter at x = 2D)
        start_x = 2 * turbine_diameter
        start_x_index = int( start_x / dx )
        F = self.calc_filter_function( start_x, turbine_diameter, turbine_coords, flow_field )
        
        # Initial centreline velocity deficit
        #self.set_centreline_vdf()
        D_m = self.calc_initial_centreline_vdf( thrust_coefficient, ambient_intensity )
        self.centreline_vdf = np.zeros( max_len )
        self.centreline_vdf[0] = D_m

        # Initial width of the wake
        b = self.calc_wake_parameter( thrust_coefficient, D_m )

        # Eddy-viscosity (epsilon), sum of ambient eddy viscosity and eddy viscosity generated by flow shear in the wake
        ee = self.calc_eddy_viscosity( F, K1, b, D_m, K, ambient_intensity, u_0 )

        # Set the initial u. We keep 0 as an initial guess for v and update it later
        #for j in range(start_r_index, len_r):
        for j in range(max_len):
            # Assuming to the Gaussian profile, the velocity deficit a distance ‘r’ from the wake centerline
            #D_mr = self.calc_off_centre_vdf(r_coords[j], b, D_m, turbine_diameter, turbine_coords, flow_field)
            #multiplier_grid[start_x_index - start_x_index_turbine, j - start_r_index, 0] = (1 - D_mr)
            r = j * dr
            D_mr = self.calc_off_centre_vdf( r, b, D_m, turbine_diameter, turbine_coords, flow_field )
            multiplier_grid[start_x_index, j, 0] = ( 1 - D_mr )
            
        # Initialise some iteration counters
        warm_up = 10
        i = 1
        
        # Begin looping in the downstream direction, as far as len_x and starting from start_x_inx (2 diameters downstream normally)
        #while i < len_x - start_x_index - 1:
        while i < max_len - start_x_index - 1:
            # i iterates in downstream (x) direction
            # Shift index to reflect the shifted start position
            I = start_x_index + i - 1
            x = I * dx
            # Calculate Filter Function for present index, starting from 2 diameters downstream
            F = self.calc_filter_function(x, turbine_diameter, turbine_coords, flow_field)

            # Calculate the eddy viscocity for present index
            ee = self.calc_eddy_viscosity(F, K1, b, D_m, K, ambient_intensity, u_0)

            # Initialise first element of vectors a, b, c and r
            d_ur = multiplier_grid[I, 1, 0] - multiplier_grid[I, 0, 0]
            a_vec = []
            b_vec = []
            c_vec = []
            r_vec = []
            a_vec.append(0)
            b_vec.append(2 * ((u_0 * multiplier_grid[I, 0, 0] * dr**2) + (ee * dx)))
            c_vec.append(-2 * ee * dx)
            r_vec.append(2 * ee * dx * (d_ur + (u_0 * multiplier_grid[I, 0, 0] * dr)**2))
            
            # Compute remainder of vectors a, b, c
            for j in range( 1, max_len ):
                #r = r_grid[I, j]#TODO J sometimes out of r_grids range here, maybe when cut-off due to large radius?
                r = j * dr
                u_r = u_0 * multiplier_grid[I, j, 0]
                v_r = v_0 * multiplier_grid[I , j, 1]
                bracket1 = v_r * r * dr
                bracket2 = 2 * ee * r
                bracket3 = ee * dr
                bracket4 = ee * dx
                a_vec.append((bracket3 - bracket1 - bracket2) * dx)
                b_vec.append(4 * r * ((u_r * dr**2) + bracket4))
                c_vec.append((bracket1 - bracket2 - bracket3) * dx)

            # Compute r
            #for j in range(start_r_index + 1, len_r - 1):
            for j in range( 1, max_len - 1 ):
                #r = r_grid[I, j]
                r = j * dr
                u_r = u_0 * multiplier_grid[I, j, 0]
                v_r = v_0 * multiplier_grid[I, j, 1]
                d_ur = u_0 * ( multiplier_grid[I, j + 1, 0] - multiplier_grid[I, j - 1, 0] )
                dd_ur = u_0 * ( multiplier_grid[I, j + 1, 0] - ( 2 * multiplier_grid[I, j, 0]) + multiplier_grid[I, j - 1, 0] )
                d_product = dr * dx * d_ur
                r = (ee * d_product) \
                    + (2 * r * ee * dx * dd_ur) \
                    - (r * v_r * d_product) \
                    + (4 * r * (u_r * dr)**2)
                r_vec.append(r)

            # Final value of z is computed differently
            #r = r_grid[I, -1]
            r = ( max_len - 1 ) * dr
            u_r = u_0 * multiplier_grid[I, -1, 0]
            v_r = v_0 * multiplier_grid[I, -1, 1]
            d_ur = u_0 * multiplier_grid[I, -1, 0] - multiplier_grid[I, -2, 0]
            dd_ur = u_0 * (1 - (2 * multiplier_grid[I, -1, 0]) + multiplier_grid[I - start_x_index, -2, 0])
            d_product = dr * dx * d_ur
            r = (ee * d_product)\
                + (2 * r * dx * ee * dd_ur) \
                - (r * v_r * dr * dx * (u_0 * (1 - multiplier_grid[I - start_x_index, -2, 0])) \
                + (4 * r * (dr * u_r)**2)) \
                - (c_vec[-1] * u_r)
            r_vec.append(r)

            # Construct the tri-diagonal matrix
            A_mat = np.diag(a_vec[1:], -1) + np.diag(b_vec[:], 0) + np.diag(c_vec[:-1], 1)
            # Solve the system
            multiplier_grid[I + 1, :, 0] = np.linalg.solve(A_mat, np.array(r_vec).conj().transpose()) / u_0
            # Update v on centreline
            multiplier_grid[I - start_x_index, 0, 1] = 0
            #for j in range(start_r_index + 1, len_r):
            for j in range( 1, max_len ):
                #r = r_grid[I, j]
                r = j * dr
                d_ux = u_0 * (multiplier_grid[I + 1, j, 0] - multiplier_grid[I, j, 0])
                multiplier_grid[I + 1, j, 1] = ( r / ( r + dr ) ) * ( multiplier_grid[I + 1, j - 1, 1] - ( ( dr / dx ) * d_ux ) )

            # Check if we're still warming up - in which case decrement j so
            # we can have another go around
            if i == 1 and warm_up > 0:
                warm_up -= 1
                # replace the initial guess with what we just computed
                multiplier_grid[start_x_index, :, 1] = multiplier_grid[start_x_index + 1, :, 1]
                i -= 1

            # Increment j
            i += 1

            # Update centreline velocity at present x coordinate.
            D_m = 1 - multiplier_grid[I, 0, 0]
            self.centreline_vdf[I] = D_m

            # Update width parameter at present x_coordinate
            b = self.calc_wake_parameter(thrust_coefficient, D_m)

        # fill in the gaps up to the start_x index with the value at start_x_inx + 1
        #for i in range(start_x_index - start_x_index_turbine):
        for i in range( start_x_index ):
            multiplier_grid[i, :, 0] = multiplier_grid[start_x_index + 1, :, 0]
            multiplier_grid[i, :, 1] = multiplier_grid[start_x_index + 1, :, 1]
            self.centreline_vdf[i] = self.centreline_vdf[start_x_index + 1]

        # set velocities over full r range at last x coordinate equal to those of second last coordinate
        multiplier_grid[-1, :, 0] = multiplier_grid[-2, :, 0]
        multiplier_grid[-1, :, 1] = multiplier_grid[-2, :, 1]

        # exclude v component for now
        multiplier_grid = np.abs(multiplier_grid[:,:,0])
        self.set_multiplier_grid(multiplier_grid)
