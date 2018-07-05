"""
A Base FlowField class from which a flow field can be instantiated and the flow
at any given point (u,v) found.
"""

import numpy as np
from helpers import *

class FlowField(object):
    """
    A base class for defining the flow field.
    param x_coords is a list of x-coordinates
    param y_coords is a list of y-coordinates
    aram z_coords is a list of z-coordinates
    param flow is a 3D list of the flow velocity vector at the hub-height of each location
    """
    
    def __init__(self, x_coords = [], y_coords = [], z_coords = [], flow = []):
        # check if there is a y-coord for every x-coord, that the number of rows in each flow list correspond to a y-coord, and that the number of columns in each flow list correspond to an x-coord

        try:
            assert np.array(x_coords).size == np.array(y_coords).size == np.array(z_coords).size
            assert isinstance(x_coords, (list, np.ndarray))
            assert isinstance(y_coords, (list, np.ndarray))
            assert isinstance(z_coords, (list, np.ndarray))
            assert isinstance(flow, (list, np.ndarray))
            if np.array(flow).size > 0:
                assert np.array(flow).shape == ( (np.array(x_coords).size, np.array(y_coords).size, np.array(z_coords).size, 3 ) )
            
        except AssertionError:
            raise ValueError("'x_coords must be the same length as 'y_coords' and 'z_coords'. The shape of 'flow' should be (len(x_coords), len(y_coords), len(z_coords), 3)")
        else:
            self.set_x_coords(x_coords)
            self.set_y_coords(y_coords)
            self.set_z_coords(z_coords)
            self.set_flow(flow)

    def get_x_coords(self):
        return self.x_coords

    def get_y_coords(self):
        return self.y_coords

    def get_z_coords(self):
        return self.z_coords

    def get_flow(self):
        return self.flow

    def set_dx(self, dx):
        self.dx = dx

    def set_dy(self, dy):
        self.dy = dy

    def set_dz(self, dz):
        self.dz = dz

    def get_dx(self):
        return self.dx

    def get_dy(self):
        return self.dy

    def get_dz(self):
        return self.dz

    def is_in_flow_field(self, pnt_coords):
        return any(np.isin(self.get_x_coords(), pnt_coords[0])) \
               and any(np.isin(self.get_y_coords(), pnt_coords[1])) \
               and any(np.isin(self.get_z_coords(), pnt_coords[2])) 
    
    def set_x_coords(self, x_coords = []):
        try:
            assert isinstance(x_coords, (list, np.ndarray))
            assert all(isinstance(c, (float,int)) for c in x_coords)
        except AssertionError:
            raise TypeError("'x_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            if len(x_coords) > 1:
                dx_arr = np.diff(x_coords)
                dx = dx_arr.min()
                self.set_dx(dx)
                # if dx is not always equal and monotonoically increasing, interpolate in order
                if not all(d == dx_arr[0] for d in dx_arr) or any(dx_arr < 0):
                    x_coords = np.arange(min(x_coords), max(x_coords) + dy, dx).tolist()
            else:
                self.set_dx(1)
            
            self.x_coords =  np.array(x_coords, dtype=np.float64)

    def set_y_coords(self, y_coords = []):
        try:
            assert isinstance(y_coords, (list, np.ndarray))
            assert all(isinstance(c, (float,int)) for c in y_coords)
        except AssertionError:
            raise TypeError("'y_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            if len(y_coords) > 1:
                dy_arr = np.diff(y_coords)
                dy = dy_arr.min() if len(dy_arr) > 1 else 0
                self.set_dy(dy)
                # if dy is not always equal and monotonoically increasing, interpolate in order
                if not all(d == dy_arr[0] for d in dy_arr) or any(dy_arr < 0):
                    y_coords = np.arange(min(y_coords), max(y_coords) + dy, dy).tolist()
            else:
                self.set_dy(1)
            
            self.y_coords =  np.array(y_coords, dtype=np.float64)

    def set_z_coords(self, z_coords = []):
        default_z_coords = []
        try:
            assert isinstance(z_coords, (list, np.ndarray))
            assert all(isinstance(c, (float,int)) for c in z_coords)
        except AssertionError:
            raise TypeError("'z_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            if len(z_coords) > 1:
                dz_arr = np.diff(z_coords)
                dz = dz_arr.min() if len(dz_arr) > 1 else 0
                self.set_dz(dz)
                # if dz is not always equal and monotonoically increasing, interpolate in order
                if not all(d == dz_arr[0] for d in dz_arr) or any(dz_arr < 0):
                    z_coords = np.arange(min(z_coords), max(z_coords) + dz, dz).tolist()
            else:
                self.set_dz(1)
                
            self.z_coords =  np.array(z_coords, dtype=np.float64)

    def set_flow(self, flow = [0,0,0]):
        #flow_mg = np.meshgrid(flow)
        flow_arr = np.array(flow, dtype=np.float64)
        flow_arr_flatten = flow_arr.flatten()

        try:
            assert isinstance(flow, (list, np.ndarray))
            assert all(isinstance(f, (float,int, np.int64, np.float64)) for f in flow_arr_flatten) 
        except AssertionError:
            raise TypeError("'flow' must be of type three-dimensional 'list', where the first dimension represents a row in space, the second a column in space, and the third a list of the flow components (x,y,z) at that point in space with three elements of type 'int' or 'float'")
        else:
            self.flow = flow_arr

    def calc_flow_gradient_at_point(self, pnt_coords, delta):
        dx = dy = dz = delta
        gradient = []
        # partial derivative wrt x
        gradient[0] = (self.get_flow_at_point(pnt_coords[0]+delta/2) - self.get_flow_at_point(pnt_coords[0]-delta/2)) / delta
        # partial derivative wrt y
        gradient[1] = (self.get_flow_at_point(pnt_coords[1]+delta/2) - self.get_flow_at_point(pnt_coords[1]-delta/2)) / delta
        # partial derivative wrt z
        gradient[2] = (self.get_flow_at_point(pnt_coords[2]+delta/2) - self.get_flow_at_point(pnt_coords[2]-delta/2)) / delta

        return np.array(gradient)

    def get_undisturbed_flow_at_point(self, pnt_coords, mag = False, direction = False):
        """
        Returns the flow at a point in the flow-field, given the point coordinates and the combined wake at that point.
        param pnt_Coords list of [x,y,z] coordinates
        param vrf_func combined or single function returning net velocity
        given the undisturbed flow and the point
        """
        if self.is_in_flow_field(pnt_coords):
            
            flow = self.get_flow()
            x_coords, y_coords, z_coords = self.get_x_coords(), self.get_y_coords(), self.get_z_coords()
            x_coord, y_coord, z_coord = pnt_coords

            # Find index of nearest value in an array
            x_coord_index, y_coord_index, z_coord_index = find_index(x_coords, x_coord),\
                                                          find_index(y_coords, y_coord),\
                                                          find_index(z_coords, z_coord)

            # calculate undisturbed flow at point
            undisturbed_flow_at_point = np.array(flow[x_coord_index, y_coord_index, z_coord_index], dtype=np.float64)
            if mag == True:
                undisturbed_flow_at_point = np.linalg.norm(undisturbed_flow_at_point, 2) if isinstance(undisturbed_flow_at_point, (list, np.ndarray)) else undisturbed_flow_at_point
            elif direction == True:
                try:
                    undisturbed_flow_at_point = undisturbed_flow_at_point / np.linalg.norm(undisturbed_flow_at_point, 2)
                except ZeroDivisionError:
                    undisturbed_flow_at_point = np.array([0,0,0])
        else:
            undisturbed_flow_at_point = np.array([0, 0, 0])
        
        return undisturbed_flow_at_point


##     def compute_flow_field(self, ambient_flow_field, turbine_locations,
##                           power_curve=None):
##        """ Superimposes the turbine wakes onto the ambient flow field in order
##        to determine the resultant flow field. Also calculates the power
##        generated by the array.
##
##        ambient_flow_field should be a list of x points, y points and velocities e.g. a
##        mesh grid and corresponding array of velocities, with 3 elements in first level
##        corresponding to x_dom, r_dom and flow_field,
##        where x_dom, r_dom and flow_field ar of type numpy.array
##        of dimensions 1, 1 and 2.
##        turbine_locations should be a list of (x,y) tuples
##        """
##        # I got in a terrible pickle with indices ... this is the easy / lazy
##        # temporary fix until I can face sorting it out...
##        for loc in turbine_locations:
##            loc[0], loc[1] = loc[1], loc[0]
##
##
##        D = self.parameters.turbine_radius * 2
##        x_dom, r_dom, flow_field = ambient_flow_field
##        x_dom_points, r_dom_points = x_dom[0,:], r_dom[:,0]
##
        
##        # Shift the domain to a zero datum for convenience
##        x_shift, r_shift = min(xCoords), min(zCoords)
##        x_dom -= x_shift
##        r_dom -= r_shift
##
##        from copy import deepcopy #TODO this shouldn't be necessary
##        x_dom2 = deepcopy(x_dom)
##        r_dom2 = deepcopy(r_dom)
##
##        # Get dx and dr
##        dx = np.abs(x_dom_points[0] - x_dom_points[1])
##        dr = np.abs(r_dom_points[0] - r_dom_points[1])
##
##        # Nudge turbines onto the flow_field grid
##        for i in turbine_locations:
##            i[0] = find_nearest(x_dom_points, i[0]-x_shift)
##            i[1] = find_nearest(r_dom_points, i[1]-r_shift)
##        turbine_locations.sort()
##
##        for i in turbine_locations:
##            x_dom1, r_dom1 = np.meshgrid(x_dom_points, r_dom_points)
##
##            # Extract the ambient velocity at turbine, u_0, and calc power
##            wake_length = max(x_dom_points)-i[1]
##            wake_length /= D
##
##            u_0 = flow_field[find_index(x_dom_points, i[0]),
##                                  find_index(r_dom_points,i[1])]
##            u_0_tip1 = flow_field[find_index(x_dom_points, i[0]-D/2),
##                                  find_index(r_dom_points,i[1])]
##            u_0_tip2 = flow_field[find_index(x_dom_points, i[0]+D/2),
##                                  find_index(r_dom_points,i[1])]
##            u_0_avg = (u_0 + u_0_tip1 + u_0_tip2) / 3.
##
##            idx=find_index(x_dom1[0], i[0])
##            idr=find_index(r_dom1[:,0], i[1])
##            idx_lo = find_index(x_dom1[0], i[0]-2*D)
##            idx_hi = find_index(x_dom1[0], i[0]+2*D)
##            idr_hi = find_index(x_dom1[0], i[1]+wake_length*D)
##            print('flow speed at turbine: ', u_0, u_0_avg)
##            
##            if power_curve:
##                power = self.compute_power(u_0_avg)
##            else:
##                power = u_0**2 * self.parameters.rho \
##                        * self.parameters.turbine_radius **2 * np.pi
##            self.array_power += power
##
##            deficit = np.zeros(x_dom1.shape)
##            deficit[idx_lo:idx_hi, idr:idr_hi] = \
##                    self.scale_wake(x_dom1[idx_lo:idx_hi, idr:idr_hi],
##                            r_dom1[idx_lo:idx_hi, idr:idr_hi], u_0_avg, wake_length)
##
##            plt.imshow(deficit)
##            plt.show()
##
##            multiplier = 1-deficit
##            flow_field *= multiplier
##
##            self.plot_wake(x_dom2, r_dom2, flow_field)
##            print(power, self.array_power)
##            plt.imshow(flow_field)
##            plt.show()
##
## 
