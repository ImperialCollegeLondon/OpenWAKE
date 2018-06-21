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
        flow = self.get_flow()
        u_0 = self.get_flow_mag_at_turbine()
        x_coords, z_coords = flow.get_x_coords(), flow.get_z_coords()
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)

        rel_pnt_coords = self.relative_position(pnt_coords)
        x_rel, z_rel = rel_pnt_coords[0], rel_pnt_coords[0]
        dx, dz = x_coords[1] - x_coords[0], z_coords[1] - z_coords[0]
        rel_x_inx, rel_z_inx = int(rel_pnt_coords[0]/dx), int(rel_pnt_coords[2]/dz)
        vrf_grid = self.get_vrf_grid()
        
        D_M = vrf_grid[rel_x_inx, rel_z_inx] * u_0
        b = self.calc_wake_parameter(thrust_coefficient, D_M, u_0)
        
        return self.calc_wake_width(b) / 2

    def calc_vrf_at_point(self, pnt_coords):
        if self.is_in_wake(pnt_coords): 
            flow = self.get_flow()
            vrf_grid = self.get_vrf_grid()
            x_coords, z_coords = flow.get_x_coords(), flow.get_z_coords()
            rel_pnt_coords = self.relative_position(pnt_coords)
            dx, dz = x_coords[1] - x_coords[0], z_coords[1] - z_coords[0]
            rel_x_inx, rel_z_inx = int(rel_pnt_coords[0]/dx), int(rel_pnt_coords[2]/dz)

            return vrf_grid[rel_x_inx, rel_z_inx]
        else:
            return 0

    def calc_wake_parameter(self, thrust_coefficient, D_M, u_0):
        alpha, beta, gamma = 3.56, 8, 0.5
        #TODO check this
        with np.errstate(all="ignore"):
            b = ((alpha*thrust_coefficient)/(beta*(D_M/u_0)*(1-(gamma*(D_M/u_0)))))**(1/2)
        return b if np.isinf(b) == False and np.isnan(b) == False else 0

    def calc_wake_width(self, b):
        W = b * (0.5/3.56)**(1/2)
        return W

    def calc_eddy_viscosity(self, F, K1, b, D_M, K, ambient_intensity):
        e_wake = F * K1 * b * D_M
        e_ambient = F * K**2 * ambient_intensity
        ee = e_wake + e_ambient
        return ee

    def calc_filter_function(self, x, turbine_diameter):
        if x > 5.5 * turbine_diameter:
            F = 1
        elif x > 4.5 * turbine_diameter:
            F = 0.65 + ((x-(4.5*turbine_diameter))/23.32)**(1/3)
        else:
            F = 0.65 - (-(x-(4.5*turbine_diameter))/23.32)**(1/3)
        return F
        
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
        x_coords, z_coords = flow.get_x_coords(), flow.get_z_coords()
        thrust_coefficient = turbine.calc_thrust_coefficient(u_0)
        ambient_intensity = self.get_ambient_intensity()

        # Get domain geometry from the meshgrid
        # we will start the calc downstream from the turbine (by 2D) and
        # interpolate back up to it
        flow = self.get_flow()
        full_x_coords, full_r_coords = flow.get_x_coords(), flow.get_z_coords()
        full_len_x, full_len_r = len(full_x_coords), len(full_r_coords)

        full_x_coords_arr = np.array(full_x_coords)
        start_x = turbine_coords[0]
        start_x_inx = find_index(full_x_coords, start_x)
        end_x_inx = full_x_coords_arr.argmax()
        #start_inx_dx = int(start_x/dx)
        x_coords = full_x_coords_arr[start_x_inx:end_x_inx+1] - full_x_coords_arr[start_x_inx]
        start_x = 2 * turbine_diameter if (2 * turbine_diameter) < x_coords[-2] else x_coords[-2]
        start_x_inx = find_index(x_coords, start_x)
        len_x = x_coords.size
            
        full_r_coords_arr = np.array(full_r_coords)
        start_r_inx = find_index(full_r_coords, turbine_coords[2])
        # half_wake will be flipped, so should fit smallest available berth between turbine z coord and grid boundary
        end_r_inx = full_r_coords_arr.argmax() if abs(np.amax(full_r_coords_arr) - turbine_coords[2]) <= abs(np.amin(full_r_coords_arr) - turbine_coords[2]) else turbine_coords[2] + full_r_coords_arr.argmin()
        r_coords = full_r_coords_arr[start_r_inx:end_r_inx+1] - full_r_coords[start_r_inx]
        len_r = r_coords.size
        
        x_grid, r_grid = np.meshgrid(x_coords, r_coords)
        
        dx, dr = abs(x_coords[0]-x_coords[1]), abs(r_coords[0] - r_coords[1])
        
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
        alpha, beta, gamma = 0.05, 16, 0.5
        D_M = (thrust_coefficient - alpha - (((beta * thrust_coefficient) - gamma) * (ambient_intensity / 10))) * u_0

        # Initial width of the wake
        b = self.calc_wake_parameter(thrust_coefficient, D_M, u_0)

        # Eddy-viscosity (epsilon), sum of ambient eddy viscosity and eddy viscosity generated by flow shear in the wake

        ee = self.calc_eddy_viscosity(F, K1, b, D_M, K, ambient_intensity)

        # Set the initial u. We keep 0 as an initial guess for v and update it later
        for i in range(len_r):
            velocity_reduction_factor =  D_M * np.exp(-3.56 * (r_coords[i]/b)**2)/u_0
            u[start_x_inx, i] = u_0 * (1 - velocity_reduction_factor)

        # Initialise some iteration counters
        warm_up = 10
        j = 1
        # Begin looping in the downstream direction, as far as
        while j < len_x - start_x_inx - 1:
            # j iterates in downstream (x) direction
            # Shift index to reflect the shifted start position
            J = start_x_inx + j - 1

            # Filter
            F = self.calc_filter_function(x_coords[J], turbine_diameter)

            # Calculate the eddy viscocity
            ee = self.calc_eddy_viscosity(F, K1, b, D_M, K, ambient_intensity)

            # Initialise first element of vectors a, b, c and r
            a_vec = [0]
            b_vec = [2 * ((u[J,0] * dr**2) + (ee * dx))]
            c_vec = [-2 * ee * dx]
            r_vec = [2 * ee * dx * ((u[J,1]) - (u[J,0])) + (u[J,0]**2 * dr**2)]
        
            # Compute remainder of vectors a, b, c

            for i in range(1, len_r):
                a_vec.append(dx * ((ee * dr) - (r_grid[J, i] * v[J, i] * dr) - (2 * r_grid[J, i] * ee)))
                b_vec.append(4 * r_grid[J, i] * (dr**2 * u[J, i] + dx * ee))
                c_vec.append(dx * (r_grid[J, i] * dr * v[J, i] - 2 * r_grid[J, i] * ee - dr * ee))

            # Compute r
            for i in range(1, len_r-1):
                r = dr * dx * ee * (u[J, i+1] - u[J, i-1]) \
                    + 2 * r_grid[J, i] * dx * ee * (u[J, i+1] - 2*u[J, i] + u[J, i-1]) \
                    - r_grid[J, i] * dr * dx * v[J, i] * (u[J, i+1] - u[J, i-1]) \
                    + 4 * r_grid[J, i] * (dr**2) * (u[J, i]**2)
                r_vec.append(r)

            # Final value of r is computed differently
            r = dr * dx * ee * (u[J,-1] - u[J,-2]) \
                + 2 * r_grid[J,-1] * dx *ee * (u_0 - 2 * u[J,-1] + u[J,-2]) \
                - r_grid[J,-1] * dr * dx * v[J,-1] * (u_0 - u[J,-2]) \
                + 4 * r_grid[J,-1] * (dr**2) * (u[J,-1]**2) \
                - c_vec[-1] * u[J,-1]
            r_vec.append(r)

            # Construct the tri-diagonal matrix
            A_mat = np.diag(a_vec[1:], -1) + np.diag(b_vec[:], 0) + np.diag(c_vec[:-1], 1)

            # Solve the system
            u[J+1,:] = np.linalg.solve(A_mat, np.array(r_vec).conj().transpose())

            # Update v on centreline
            v[J, 0] = 0
            for i in range(1, len_r):
                v[J+1, i] = (r_grid[J, i] / (r_grid[J, i] + dr)) * (v[J+1, i-1] - (dr / dx) * (u[J+1, i] - u[J, i]))

            # Check if we're still warming up - in which case decrement j so
            # we can have another go around
            if j == 1 and warm_up > 0:
                warm_up -= 1
                # replace the initial guess with what we just computed
                v[0,:] = v[1,:]
                j -= 1

            # Increment j
            j += 1

            # Update centreline velocity deficit 2 diameters downstream. TODO check logic
            D_M = u_0 - u[J,0]

            b = self.calc_wake_parameter(thrust_coefficient, D_M, u_0)

        # fill in the gaps up to the start_x index with the value at start_x_inx + 1
        for j in range(start_x_inx):
            u[j,:] = u[start_x_inx + 1,:]
            v[j,:] = v[start_x_inx + 1,:]

        u[-1,:] = u[-2,:]
        v[-1,:] = v[-2,:]

        # Convert velocity to velocity reduction factor
        velocity_reduction_factor = 1 - (u / u_0)

        # Compute the half wake (split along turbine axis) TODO check this
        upper_half_wake = np.abs(velocity_reduction_factor)

        # Flip along turbine axis to get other half
        lower_half_wake = np.flipud(upper_half_wake)
        
        # Stitch them together
        velocity_reduction_factor_grid = np.vstack((lower_half_wake, upper_half_wake))
        #full_velocity_reduction_factor_grid
        self.set_vrf_grid(velocity_reduction_factor_grid)
