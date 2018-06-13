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
    
    def __init__(self, x_coords, y_coords, z_coords, flow):
        # check if there is a y-coord for every x-coord, that the number of rows in each flow list correspond to a y-coord, and that the number of columns in each flow list correspond to an x-coord
        if not ( ( len(x_coords) == len(y_coords) and len(y_coords) == len(z_coords)
           and len(np.array(flow).flatten()) == (len(x_coords) * len(y_coords) * len(z_coords) * 3) )
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

    def set_x_coords(self, x_coords):
        default_x_coords = []
        if not ((isinstance(x_coords, list) and all(isinstance(c, float) or isinstance(c, int) for c in x_coords)) or x_coords == None):
            raise TypeError("'x_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            self.x_coords =  np.array(x_coords) if x_coords != None else np.array(default_x_coords)

    def set_y_coords(self, y_coords):
        default_y_coords = []
        if not ((isinstance(y_coords, list) and all(isinstance(c, float) or isinstance(c, int) for c in y_coords)) or y_coords == None):
            raise TypeError("'y_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            self.y_coords =  np.array(y_coords) if y_coords != None else np.array(default_y_coords)

    def set_z_coords(self, z_coords):
        default_z_coords = []
        if not ((isinstance(z_coords, list) and all(isinstance(c, float) or isinstance(c, int) for c in z_coords)) or z_coords == None):
            raise TypeError("'z_coords' must be of type 'list', where each element is of type 'int' or 'float'")
        else:
            self.z_coords =  np.array(z_coords) if z_coords != None else np.array(default_z_coords)

    def set_flow(self, flow):
        default_flow = [[[]]]
        np_flow = np.meshgrid(flow)
        if not ((isinstance(flow, list) and np_flow.shape[2] == 3 and all(isinstance(f, float) or isinstance(f, int) for f in np_flow.flatten())) or flow == None):
            raise TypeError("'flow' must be of type three-dimensional 'list', where the first dimension represents a row in space, the second a column in space, and the third a list of the flow components (x,y,z) at that point in space with three elements of type 'int' or 'float'")
        else:
            self.flow = np_flow if flow != None else np.meshgrid(default_flow)
    
    def calc_wake_flow_at_point(self, pnt_coords, vel_red_factor_func):
        """
        Calculate the flow at a point pnt_coords,
        taking the wake due to a single turbine at turbine_coords into consideration,
        given that flow hits turbine axially, as relative_position will enforce
        for the purpose of wake ca
        """
        ## TODO Check logic, change to 3d, assume that flow is axial to turbine because of relative_position?
        # select the x and y components of flow, calculate norm
        #flowArr = self.get_flow()
        #flowArrShape = flowArr.shape
        #twoDFlow = np.array([flowArr[r][c][0:2] for r in range(flowArrShape[0]) for c in range(flowArrShape[1])]).reshape(flowArrShape[0], flowArrShape[1], 2)
        #twoDFlowNorm = np.linalg.norm(twoDFlow, 2, 2)
        new_flow_at_point = self.get_flow_at_point(pnt_coords) * (1 - vel_red_factor_func(pnt_coords))
        return new_flow_at_point

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
        
    def get_flow_at_point(self, pnt_coords):
        flow = self.get_flow()
        x_coords = self.get_x_coords()
        y_coords = self.get_y_coords()
        z_coords = self.get_z_coords()
        x_coord = pnt_coords[0]
        y_coord = pnt_coords[1]
        z_coord = pnt_coords[2]
        x_coord_index = np.nonzero(x_coords == x_coord)
        y_coord_index = np.nonzero(y_coords == y_coord)
        z_coord_index = np.nonzero(z_coords == z_coord)
        
        # if no such coordinates exist, set coordinates to min/max value if
        # given coordinate was outside the flow_field range
        # otherwise set coordinatess to truncated match TODO TEST AND FIX
        if len(x_coord_index) == 0:
            x_coord_max = np.amax(x_coords)
            x_coord_min = np.amin(x_coords)
            if x_coord > x_coord_max:
                x_coord_index = np.nonzero(x_coords == x_coord_max)
            elif x_coord < x_coord_min:
               x_coord_index = np.nonzero(x_coords == x_coord_min)
            else:
               x_coord_index = np.nonzero(np.trunc(x_coords) == np.trunc(x_coord))
            raise ValueError("Exact x coordinate not found, replaced with " + str(x_coords[x_coord_index]))
        else:
            x_coord_index = x_coord_index[0][0]

        if len(y_coord_index) == 0:
            y_coord_max = np.amax(y_coords)
            y_coord_min = np.amin(y_coords)
            if y_coord > y_coord_max:
                y_coord_index = np.nonzero(y_coords == y_coord_max)
            elif y_coord < y_coord_min:
               y_coord_index = np.nonzero(y_coords == y_coord_min)
            else:
               y_coord_index = np.nonzero(np.trunc(y_coords) == np.trunc(y_coord))
            raise ValueError("Exact y coordinate not found, replaced with " + str(y_coords[y_coord_index]))
        else:
            y_coord_index = y_coord_index[0][0]

        if len(z_coord_index) == 0:
            z_coord_max = np.amax(z_coords)
            z_coord_min = np.amin(z_coords)
            if z_coord > z_coord_max:
                z_coord_index = np.nonzero(z_coords == z_coord_max)
            elif z_coord < z_coord_min:
               z_coord_index = np.nonzero(z_coords == z_coord_min)
            else:
               z_coord_index = np.nonzero(np.trunc(z_coords) == np.trunc(z_coord))
            raise ValueError("Exact z coordinate not found, replaced with " + str(z_coords[z_coord_index]))
        else:
            z_coord_index = z_coord_index[0][0]
            
        # return np array representing the 3D flow at the point (or nearest point) required
        return flow[x_coord_index][y_coord_index][z_coord_index]
