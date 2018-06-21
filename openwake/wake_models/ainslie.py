"""
Implementation of ainslie wake model
"""

from wake_models.base_wake import BaseWake
from turbine_models.base_turbine import BaseTurbine
from flow_field_model.flow import Flow
import numpy as np
from helpers import *

class Ainslie(BaseWake):
    """
    Implements an Ainslie wake model
    """

    def __init__(self, turbine, flow, ambient_intensity):
        """
        param ambient_intensity (decimal)
        """
        self.set_ambient_intensity(ambient_intensity)
        super(Ainslie, self).__init__(turbine, flow)

    def get_vrf_grid(self):
        return self.vrf_grid

    def set_vrf_grid(self, vrf_grid):
        self.vrf_grid = np.array(vrf_grid)

    def get_ambient_intensity(self):
        return self.ambient_intensity

    def set_ambient_intensity(self, ambient_intensity = 0.1):
        try:
            assert isinstance(ambient_intensity, float)
        except AssertionError:
            raise TypeError("'ambient_intensity' must be of type 'float'")
        else:
            self.ambient_intensity = ambient_intensity

    def calc_wake_radius(self, pnt_coords):
        """
        Returns the radius of the wake at a point
        given that point is in wake of turbine
        param pnt_coords point at which to calculate the radius of the wake
        """
        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        turbine_diameter = 2 * turbine.get_radius()
        flow = self.get_flow()
        u_0 = self.get_flow_mag_at_turbine()
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        vrf_grid = self.get_vrf_grid()
        rel_pnt_inx = self.relative_inx(pnt_coords)
        rel_x_inx, rel_z_inx = rel_pnt_inx[0], abs(rel_pnt_inx[2])
        
        D_m = vrf_grid[rel_x_inx, rel_z_inx]
        b = self.calc_wake_parameter(thrust_coefficient, D_m)
        
        return (turbine_diameter + self.calc_wake_width(b) * turbine_diameter) / 2

    def calc_vrf_at_point(self, pnt_coords):
        if self.is_in_wake(pnt_coords): 
            flow = self.get_flow()
            vrf_grid = self.get_vrf_grid()
            #origin_z_inx = int(np.floor(vrf_grid.shape[1]/2))
            #rel_x_inx, rel_z_inx = int(rel_pnt_coords[0]/dx), int((origin_z_inx + rel_pnt_coords[2])/dz)
            rel_pnt_inx = self.relative_inx(pnt_coords)
            rel_x_inx, rel_z_inx = rel_pnt_inx[0], abs(rel_pnt_inx[2])
            return vrf_grid[rel_x_inx, rel_z_inx]
        else:
            return 0

    def calc_eddy_viscosity(self, F, K1, b, D_m, K, ambient_intensity, u_i):
        e_wake = F * K1 * b * D_m * u_i
        e_ambient = F * K**2 * ambient_intensity
        ee = e_wake + e_ambient
        return ee

    def calc_filter_function(self, x, turbine_diameter):
        k1, k2, k3, k4, k5 = 4.5, 5.5, 23.32, 1/3, 0.65
        if x > k2:
            F = 1
        elif x > k1:
            F = k5 + (((x/turbine_diameter) - k1) / k3)**k4
        else:
            F = k5 - (-((x/turbine_diameter)-k1) / k3)**k4
        return F

    def calc_initial_centreline_vdf(self, thrust_coefficient, ambient_intensity):
        k1, k2, k3, k4 = 0.05, 16, 0.5, 10
        D_m = thrust_coefficient - k1 - (((k2 * thrust_coefficient) - k3) * (ambient_intensity / k4))
        return D_m

    def calc_off_centre_vdf(self, r, b, D_m, turbine_diameter):
        k1 = -3.56
        with np.errstate(all="ignore"):
            D_mr = D_m * np.exp(k1 * ((r/turbine_diameter)/b)**2)
        return D_mr# if np.isinf(D_mr) == False and np.isnan(D_mr) == False else 1

    def calc_wake_parameter(self, thrust_coefficient, D_m):
        k1, k2, k3 = 3.56, 8, 0.5
        #TODO check this
        with np.errstate(all="ignore"):
            b = ((k1 * thrust_coefficient) / (k2 * D_m * (1 - (k3 * D_m))))**k3
        return b# if np.isinf(b) == False and np.isnan(b) == False else 0

    def calc_wake_width(self, b):
        W = b * (0.5/3.56)**(1/2)
        return W
        
    def calc_vrf_grid(self):
        """
        The ainslie model is symetrical along the turbine axis - so we shall
        compute one side and reflect it along the turbine axis to find the complete
        wake. Our wake is currently dimensionless - i.e. turbine_diameter = 1.
        Returns the velocity deficit factor.
        param x_grid
        param r_grid
        """

        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        turbine_diameter = 2 * turbine.get_radius()
        flow = self.get_flow()
        u_0 = self.get_flow_mag_at_turbine()
        full_x_coords, full_r_coords = flow.get_x_coords(), flow.get_z_coords()
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        ambient_intensity = self.get_ambient_intensity()

        # Get domain geometry from the meshgrid
        # we will start the calc downstream from the turbine (by 2D) and
        # interpolate back up to it
        
        full_x_coords_arr = np.array(full_x_coords)
        start_x = turbine_coords[0]
        start_x_inx = find_index(full_x_coords, start_x)
        end_x_inx = full_x_coords_arr.argmax()
        #start_inx_dx = int(start_x/dx)
        x_coords = full_x_coords_arr[start_x_inx:end_x_inx+1] - full_x_coords_arr[start_x_inx]
        start_x = 2 * turbine_diameter if (2 * turbine_diameter) <= x_coords[-2] else x_coords[-2]
        start_x_inx = find_index(x_coords, start_x)
        len_x = x_coords.size
        
        full_r_coords_arr = np.array(full_r_coords)
        start_r_inx = find_index(full_r_coords, turbine_coords[2])
        # half_wake will be flipped, so should fit smallest available berth between turbine z coord and grid boundary
        #end_r = turbine_coords[2] + np.amin([abs(np.amax(full_r_coords_arr) - turbine_coords[2]),  abs(np.amin(full_r_coords_arr) - turbine_coords[2])])
        #end_r_inx = find_index(full_r_coords, end_r)
        end_r_inx = full_r_coords_arr.argmax()
        #full_r_coords_arr.argmax() if abs(np.amax(full_r_coords_arr) - turbine_coords[2]) <= abs(np.amin(full_r_coords_arr) - turbine_coords[2]) else turbine_coords[2] + full_r_coords_arr.argmin()
        r_coords = full_r_coords_arr[start_r_inx:end_r_inx+1] - full_r_coords[start_r_inx]
        len_r = r_coords.size

        full_len_x, full_len_r = len(full_x_coords), len(full_r_coords)
        
        x_grid, r_grid = np.meshgrid(x_coords, r_coords, indexing='ij')
        
        dx, dr = abs(full_x_coords[0]-full_x_coords[1]), abs(full_r_coords[0] - full_r_coords[1])
        
        # initialise the solution matrices, holding direction data for flow in wake
        u = np.zeros((len_x, len_r))
        v = np.zeros((len_x, len_r))

        # Dimensionless constant
        K1 = 0.015

        # Von Karman constant
        K = 0.4

        # Initial filter function (filter at x = 2D)
        
        F = self.calc_filter_function(start_x, turbine_diameter)
        
        # Initial centreline velocity deficit
        D_m = self.calc_initial_centreline_vdf(thrust_coefficient, ambient_intensity)

        # Initial width of the wake
        b = self.calc_wake_parameter(thrust_coefficient, D_m)

        # Eddy-viscosity (epsilon), sum of ambient eddy viscosity and eddy viscosity generated by flow shear in the wake

        ee = self.calc_eddy_viscosity(F, K1, b, D_m, K, ambient_intensity, u_0)

        # Set the initial u. We keep 0 as an initial guess for v and update it later
        for i in range(len_r):
            # Assuming to the Gaussian profile, the velocity deficit a distance ‘r’ from the wake centerline
            D_mr = self.calc_off_centre_vdf(r_coords[i], b, D_m, turbine_diameter)
            u[start_x_inx, i] = u_0 * (1 - D_mr)

        # Initialise some iteration counters
        warm_up = 10
        j = 1
        
        # Begin looping in the downstream direction, as far as len_x and starting from start_x_inx (2 diameters downstream normally)
        while j < len_x - start_x_inx - 1:
            # j iterates in downstream (x) direction
            # Shift index to reflect the shifted start position
            J = start_x_inx + j - 1

            # Calculate Filter Function for present index, starting from 2 diameters downstream
            F = self.calc_filter_function(x_coords[J], turbine_diameter)

            # Calculate the eddy viscocity for present index
            ee = self.calc_eddy_viscosity(F, K1, b, D_m, K, ambient_intensity, u_0)

            # Initialise first element of vectors a, b, c and r
            d_ur = u[J,1] - u[J,0]
            a_vec = [0]
            b_vec = [2 * ((u[J,0] * dr**2) + (ee * dx))]
            c_vec = [-2 * ee * dx]
            r_vec = [2 * ee * dx * (d_ur + (u[J,0] * dr)**2)]
        
            # Compute remainder of vectors a, b, c

            for i in range(1, len_r):
                r = r_grid[J, i]#TODO J sometimes out of r_grids range here, maybe when cut-off due to large radius?
                u_r = u[J, i]
                v_r = v[J, i]
                bracket1 = v_r * r * dr
                bracket2 = 2 * ee * r
                bracket3 = ee * dr
                bracket4 = ee * dx
                a_vec.append((bracket3 - bracket1 - bracket2) * dx)
                b_vec.append(4 * r * ((u_r * dr**2) + bracket4))
                c_vec.append((bracket1 - bracket2 - bracket3) * dx)

            # Compute r
            for i in range(1, len_r-1):
                r = r_grid[J, i]
                u_r = u[J, i]
                v_r = v[J, i]
                d_ur = u[J,i+1] - u[J,i-1]
                dd_ur = u[J, i+1] - 2*u[J, i] + u[J, i-1]
                d_product = dr * dx * d_ur
                r = (ee * d_product) \
                    + (2 * r * ee * dx * dd_ur) \
                    - (r * v_r * d_product) \
                    + (4 * r * (u_r * dr)**2)
                r_vec.append(r)

            # Final value of r is computed differently
            r = r_grid[J,-1]
            u_r = u[J,-1]
            v_r = v[J,-1]
            d_ur = u[J,-1] - u[J,-2]
            dd_ur = u_0 - (2 * u[J,-1]) + u[J,-2]
            d_product = dr * dx * d_ur
            r = (ee * d_product)\
                + (2 * r * dx * ee * dd_ur) \
                - (r * v_r * dr * dx * (u_0 - u[J,-2]) \
                + (4 * r * (dr * u_r)**2)) \
                - (c_vec[-1] * u_r)
            r_vec.append(r)

            # Construct the tri-diagonal matrix
            A_mat = np.diag(a_vec[1:], -1) + np.diag(b_vec[:], 0) + np.diag(c_vec[:-1], 1)

            # Solve the system
            u[J+1,:] = np.linalg.solve(A_mat, np.array(r_vec).conj().transpose())

            # Update v on centreline
            v[J, 0] = 0
            for i in range(1, len_r):
                r = r_grid[J, i]
                d_ux = u[J+1, i] - u[J, i]
                v[J+1, i] = (r / (r + dr)) * (v[J+1, i-1] - ((dr / dx) * d_ux))

            # Check if we're still warming up - in which case decrement j so
            # we can have another go around
            if j == 1 and warm_up > 0:
                warm_up -= 1
                # replace the initial guess with what we just computed
                v[0,:] = v[1,:]
                j -= 1

            # Increment j
            j += 1

            # Update centreline velocity at present x coordinate. TODO check logic
            D_m = 1 - (u[J,0]/u_0)

            # Update width parameter at present x_coordinate
            b = self.calc_wake_parameter(thrust_coefficient, D_m)

        # fill in the gaps up to the start_x index with the value at start_x_inx + 1
        for j in range(start_x_inx):
            u[j,:] = u[start_x_inx + 1,:]
            v[j,:] = v[start_x_inx + 1,:]

        # set velocities over full r range at last x coordinate equal to those of second last coordinate
        u[-1,:] = u[-2,:]
        v[-1,:] = v[-2,:]

        # Convert velocity to velocity reduction factor
        velocity_reduction_factor = 1 - (u / u_0)

        # Compute the half wake (split along turbine axis) TODO check this
        upper_half_wake = np.abs(velocity_reduction_factor)

        # Flip along turbine axis to get other half
        #lower_half_wake = np.flipud(upper_half_wake)
        velocity_reduction_factor_grid = upper_half_wake
        # Stitch them together
        #velocity_reduction_factor_grid = np.vstack((lower_half_wake, upper_half_wake))
        #full_velocity_reduction_factor_grid
        self.set_vrf_grid(velocity_reduction_factor_grid)
