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

    def __init__(self, turbine=BaseTurbine(), flow_field=FlowField(), ambient_intensity=0.1, wake_field = WakeField()):
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

    def calc_wake_radius(self, pnt_coords, turbine_coords, flow_field, turbine_radius, thrust_coefficient):
        """
        Returns the radius of the wake at a point
        given that point is in wake of turbine
        param pnt_coords point at which to calculate the radius of the wake
        """

        turbine_diameter = 2 * turbine_radius
        D_m = self.get_centreline_vdf(turbine_coords, pnt_coords, flow_field)
        b = self.calc_wake_parameter(thrust_coefficient, D_m)
        
        return (turbine_diameter + self.calc_wake_width(b) * turbine_diameter) / 2

    def calc_eddy_viscosity(self, F, K1, b, D_m, K, ambient_intensity, u_i):
        e_wake = F * K1 * b * D_m * u_i
        e_ambient = F * K**2 * ambient_intensity
        ee = e_wake + e_ambient
        return ee

    def calc_filter_function(self, x, turbine_diameter, turbine_coords, flow_field):
        k1, k2, k3, k4, k5 = 4.5, 5.5, 23.32, 1/3, 0.65
        x_rel = relative_position(turbine_coords, [x, 0, 0], flow_field)[0]
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

    def calc_off_centre_vdf(self, z, b, D_m, turbine_diameter, turbine_coords, flow_field):
        z_rel = relative_position(turbine_coords, [0,0,z], flow_field)[2]
        k1 = -3.56
        with np.errstate(all="ignore"):
            D_mr = D_m * np.exp(k1 * ((z_rel / turbine_diameter) / b)**2)
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

    def get_centreline_vdf(self, turbine_coords, pnt_coords, flow_field):
        rel_x_index, rel_y_index, rel_z_index = relative_index(turbine_coords, pnt_coords, flow_field)
        return self.centreline_vdf[rel_x_index]

    def get_disturbed_flow_grid(self):
        try:
            self.disturbed_flow_grid
        except AttributeError:
            self.calc_disturbed_flow_grid()
        else:
            pass

        return self.disturbed_flow_grid

    def calc_disturbed_flow_grid(self, flow_field):
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
        x_grid, y_grid, z_grid, u, v, w = self.generate_disturbed_flow_grid()
        x_grid, y_grid, z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # get variables
        turbine = self.get_turbine()
        turbine_radius = turbine.get_radius()
        turbine_diameter = 2 * turbine_radius
        flow_field = self.get_flow_field()
        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        turbine_coords = turbine.get_coords()
        u_0 = flow_field.get_undisturbed_flow_at_point(turbine_coords, True)
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        ambient_intensity = self.get_ambient_intensity()

        # TODO fix/warn case where there is not enough room for start_x
        start_x = turbine_coords[0] + (2 * turbine_diameter) if turbine_coords[0] + (2 * turbine_diameter) <= x_coords[-2] else x_coords[-2]
        start_x_index_turbine = find_index(x_coords, turbine_coords[0])
        start_x_index = find_index(x_coords, start_x)
        len_x = x_coords.size

        start_y = turbine_coords[1]
        start_y_index = find_index(y_coords, start_y)
        len_y = y_coords.size
        
        start_z = turbine_coords[2]
        start_z_index = find_index(z_coords, start_z)
        len_z = z_coords.size

        dx, dy, dz = abs(x_coords[0]-x_coords[1]), abs(y_coords[0] - y_coords[1]), abs(z_coords[0] - z_coords[1]),

        #u = np.zeros((len_x - start_x_index_turbine, len_y - start_y_index, len_z - start_z_index))
        disturbed_flow_grid = np.array(undisturbed_flow_grid, dtype=np.float32)[start_x_index_turbine:len_x, start_y_index:len_y, start_z_index:len_z]

        # Dimensionless constant
        K1 = 0.015

        # Von Karman constant
        K = 0.4

        # Initial filter function (filter at x = 2D)
        F = self.calc_filter_function(start_x, turbine_diameter, turbine_coords, flow_field)
        
        # Initial centreline velocity deficit
        #self.set_centreline_vdf()
        D_m = self.calc_initial_centreline_vdf(thrust_coefficient, ambient_intensity)
        self.centreline_vdf = np.zeros((len_x))
        self.centreline_vdf[0] = D_m

        # Initial width of the wake
        b = self.calc_wake_parameter(thrust_coefficient, D_m)

        # Eddy-viscosity (epsilon), sum of ambient eddy viscosity and eddy viscosity generated by flow shear in the wake
        ee = self.calc_eddy_viscosity(F, K1, b, D_m, K, ambient_intensity, u_0)

        # Set the initial u. We keep 0 as an initial guess for v and update it later
        for k in range(start_z_index, len_z):
            # Assuming to the Gaussian profile, the velocity deficit a distance ‘r’ from the wake centerline
            D_mr = self.calc_off_centre_vdf(z_coords[k], b, D_m, turbine_diameter, turbine_coords, flow_field)
            disturbed_flow_grid[start_x_index - start_x_index_turbine, 0, k - start_z_index, 0] = u_0 * (1 - D_mr)
            
        # Initialise some iteration counters
        warm_up = 10
        i = 1
        
        # Begin looping in the downstream direction, as far as len_x and starting from start_x_inx (2 diameters downstream normally)
        while i < len_x - start_x_index - 1:
            # j iterates in downstream (x) direction
            # Shift index to reflect the shifted start position
            I = start_x_index + i - 1

            # Calculate Filter Function for present index, starting from 2 diameters downstream
            F = self.calc_filter_function(x_coords[I], turbine_diameter, turbine_coords, flow_field)

            # Calculate the eddy viscocity for present index
            ee = self.calc_eddy_viscosity(F, K1, b, D_m, K, ambient_intensity, u_0)

            # Initialise first element of vectors a, b, c and r
            d_uz = disturbed_flow_grid[I - start_x_index_turbine, 0, 1, 0] - disturbed_flow_grid[I - start_x_index, 0, 0, 0]
            a_vec = []
            b_vec = []
            c_vec = []
            z_vec = []
            a_vec.append(0)
            b_vec.append(2 * ((u[I - start_x_index_turbine, 0, 0] * dz**2) + (ee * dx)))
            c_vec.append(-2 * ee * dx)
            z_vec.append(2 * ee * dx * (d_uz + (u[I - start_x_index_turbine, 0, 0] * dz)**2))
            
            # Compute remainder of vectors a, b, c
            for k in range(start_z_index + 1, len_z):
                z = z_grid[I, start_y_index, k]#TODO J sometimes out of r_grids range here, maybe when cut-off due to large radius?
                u_z = disturbed_flow_grid[I - start_x_index_turbine, 0, k - start_z_index, 0]
                v_z = v[I - start_x_index_turbine, 0, k - start_z_index]
                bracket1 = v_z * z * dz
                bracket2 = 2 * ee * z
                bracket3 = ee * dz
                bracket4 = ee * dx
                a_vec.append((bracket3 - bracket1 - bracket2) * dx)
                b_vec.append(4 * z * ((u_z * dz**2) + bracket4))
                c_vec.append((bracket1 - bracket2 - bracket3) * dx)

            # Compute z
            for k in range(start_z_index + 1, len_z-1):
                z = z_grid[I, start_y_index, k]
                u_z = disturbed_flow_grid[I - start_x_index_turbine, 0, k - start_z_index, 0]
                v_z = disturbed_flow_grid[I - start_x_index_turbine, 0, k - start_z_index, 1]
                d_uz = disturbed_flow_grid[I - start_x_index_turbine, 0, k + 1 - start_z_index, 0] - disturbed_flow_grid[I - start_x_index_turbine, 0, k - 1 - start_z_index, 0]
                dd_uz = disturbed_flow_grid[I - start_x_index_turbine, 0, k + 1 - start_z_index, 0] - (2 * disturbed_flow_grid[I - start_x_index_turbine, 0, k - start_z_index, 0]) + disturbed_flow_grid[I - start_x_index, 0, k - 1 - start_z_index, 0]
                d_product = dz * dx * d_uz
                z = (ee * d_product) \
                    + (2 * z * ee * dx * dd_uz) \
                    - (z * v_z * d_product) \
                    + (4 * z * (u_z * dz)**2)
                z_vec.append(z)

            # Final value of z is computed differently
            z = z_grid[I, start_y_index, -1]
            u_z = disturbed_flow_grid[I - start_x_index_turbine, 0, -1, 0]
            v_z = v[I - start_x_index_turbine, 0, -1]
            d_uz = disturbed_flow_grid[I - start_x_index_turbine, 0, -1, 0] - disturbed_flow_grid[I - start_x_index, 0, -2, 0]
            dd_uz = u_0 - (2 * disturbed_flow_grid[I - start_x_index_turbine, 0, -1, 0]) + disturbed_flow_grid[I - start_x_index, 0, -2, 0]
            d_product = dz * dx * d_uz
            z = (ee * d_product)\
                + (2 * z * dx * ee * dd_uz) \
                - (z * v_z * dz * dx * (u_0 - disturbed_flow_grid[I - start_x_index, 0, -2, 0]) \
                + (4 * z * (dz * u_z)**2)) \
                - (c_vec[-1] * u_z)
            z_vec.append(z)

            # Construct the tri-diagonal matrix
            A_mat = np.diag(a_vec[1:], -1) + np.diag(b_vec[:], 0) + np.diag(c_vec[:-1], 1)

            # Solve the system TODO fix this
            disturbed_flow_grid[I + 1 - start_x_index_turbine, 0, 0:, 0] = np.linalg.solve(A_mat, np.array(z_vec).conj().transpose())

            # Update v on centreline
            disturbed_flow_grid[I - start_x_index, 0, 0, 1] = 0
            for k in range(start_z_index + 1, len_z):
                z = z_grid[I, start_y_index, k]
                d_ux = disturbed_flow_grid[I + 1 - start_x_index_turbine, 0, k - start_z_index, 0] - disturbed_flow_grid[I - start_x_index_turbine, 0, k - start_z_index, 0]
                disturbed_flow_grid[I + 1 - start_x_index_turbine, 0, k - start_z_index, 1] = (z / (z + dz)) * (disturbed_flow_grid[I + 1 - start_x_index_turbine, 0, k - 1 - start_z_index, 1] - ((dz / dx) * d_ux))

            # Check if we're still warming up - in which case decrement j so
            # we can have another go around
            if i == 1 and warm_up > 0:
                warm_up -= 1
                # replace the initial guess with what we just computed
                disturbed_flow_grid[start_x_index - start_x_index_turbine, 0, :, 1] = disturbed_flow_grid[start_x_index - start_x_index_turbine + 1, 0, :, 1]
                i -= 1

            # Increment j
            i += 1

            # Update centreline velocity at present x coordinate. TODO check logic
            D_m = 1 - (disturbed_flow_grid[I - start_x_index_turbine, 0, 0, 0] / u_0)
            self.centreline_vdf[I - start_x_index_turbine] = D_m

            # Update width parameter at present x_coordinate
            b = self.calc_wake_parameter(thrust_coefficient, D_m)

        # fill in the gaps up to the start_x index with the value at start_x_inx + 1
        for i in range(start_x_index - start_x_index_turbine):
            disturbed_flow_grid[i, 0, :, 0] = disturbed_flow_grid[start_x_index - start_x_index_turbine + 1, 0, :, 0]
            disturbed_flow_grid[i, 0, :, 1] = disturbed_flow_grid[start_x_index - start_x_index_turbine + 1, 0, :, 1]

        # set velocities over full r range at last x coordinate equal to those of second last coordinate
        disturbed_flow_grid[-1, 0, :, 0] = disturbed_flow_grid[-2, 0, :, 0]
        disturbed_flow_grid[-1, 0, :, 1] = disturbed_flow_grid[-2, 0, :, 1]

        
        disturbed_flow_grid = np.abs(disturbed_flow_grid)
        self.set_disturbed_flow_grid(disturbed_flow_grid)
