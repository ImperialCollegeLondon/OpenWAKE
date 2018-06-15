"""
A Base FlowField class from which a flow field can be instantiated and the flow
at any given point (u,v) found.
"""

import numpy as np

class Flow(object):
    """
    A base class for defining the flow field.
    param x_coords is a list of x-coordinates
    param y_coords is a list of y-coordinates
    aram z_coords is a list of z-coordinates
    param flow is a 3D list of the flow velocity vector at the hub-height of each location
    """
    
    def __init__(self, x_coords = [], y_coords = [], z_coords = [], flow = [[[[]]]]):
        # check if there is a y-coord for every x-coord, that the number of rows in each flow list correspond to a y-coord, and that the number of columns in each flow list correspond to an x-coord
        
        if not ( (len(x_coords) == len(y_coords) and len(y_coords) == len(z_coords)
           and np.array(flow).shape == ( (len(x_coords), len(y_coords), len(z_coords), 3 ) ))
           or x_coords == None or y_coords == None or z_coords == None or flow == None ):
            raise ValueError("'x_coords must be the same length as 'y_coords' and 'z_coords'. The shape of 'flow' should be (len(x_coords), len(y_coords), len(z_coords))")
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

    def set_x_coords(self, x_coords = []):
        default_x_coords = []
        if not ((isinstance(x_coords, list) and all(isinstance(c, (float,int)) for c in x_coords)) or x_coords == None):
            raise TypeError("'x_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            self.x_coords =  np.array(x_coords) if x_coords != None else np.array(default_x_coords)

    def set_y_coords(self, y_coords = []):
        default_y_coords = []
        if not ((isinstance(y_coords, list) and all(isinstance(c, (float,int)) for c in y_coords)) or y_coords == None):
            raise TypeError("'y_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            self.y_coords =  np.array(y_coords) if y_coords != None else np.array(default_y_coords)

    def set_z_coords(self, z_coords = []):
        default_z_coords = []
        if not ((isinstance(z_coords, list) and all(isinstance(c, (float,int)) for c in z_coords)) or z_coords == None):
            raise TypeError("'z_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            self.z_coords =  np.array(z_coords) if z_coords != None else np.array(default_z_coords)

    def set_flow(self, flow = [[[[]]]]):
        default_flow = [[[[]]]]
        #flow_mg = np.meshgrid(flow)
        flow_arr = np.array(flow)
        if not ( ( isinstance(flow, list) and all(isinstance(f, (float,int)) for f in flow_arr.flatten() ) ) or flow == None):
            raise TypeError("'flow' must be of type three-dimensional 'list', where the first dimension represents a row in space, the second a column in space, and the third a list of the flow components (x,y,z) at that point in space with three elements of type 'int' or 'float'")
        else:
            self.flow = flow_arr if flow != None else np.arrat(default_flow)

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
        
    def get_flow_at_point(self, pnt_coords, vrf_func = None):
        """
        Returns the flow at a point in the flow-field, given the point coordinates and the combined wake at that point.
        param pnt_Coords list of [x,y,z] coordinates
        param vrf_func combined or single function returning net velocity
        given the undisturbed flow and the point
        """

        flow = self.get_flow()
        x_coords = self.get_x_coords()
        y_coords = self.get_y_coords()
        z_coords = self.get_z_coords()
        x_coord = pnt_coords[0]
        y_coord = pnt_coords[1]
        z_coord = pnt_coords[2]

        # Find index of nearest value in an array
        x_coord_index = (np.abs(x_coords-x_coord)).argmin()
        y_coord_index = (np.abs(y_coords-y_coord)).argmin()
        z_coord_index = (np.abs(z_coords-z_coord)).argmin()

        # calculate flow at point, given combined wakes
        undisturbed_flow_at_point = flow[x_coord_index][y_coord_index][z_coord_index]
        
        # return np array representing the 3D flow at the point (or nearest point) required
        if vrf_func == None:
            return undisturbed_flow_at_point
        else:
            disturbed_flow_at_point = vrf_func(undisturbed_flow_at_point, pnt_coords)
            return disturbed_flow_at_point
        
 
