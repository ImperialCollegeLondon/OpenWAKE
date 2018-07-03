"""
A base Field class which defines methods, setters and getters used by
wake and wake combination classes
"""

import numpy as np
from helpers import *
from flow_field_model.flow import FlowField
from wake_models.wake_field_model import WakeField

class BaseField(object):

    def __init__(self, flow_field, wake_field):
        self.set_flow_field(flow_field)
        self.set_wake_field(wake_field)
    
    def get_flow_field(self):
        return self.flow_field

    def set_flow_field(self, flow_field = FlowField()):
        try:
            assert isinstance(flow_field, FlowField)
        except AssertionError:
            raise TypeError("'flow_field' must be of type 'FlowField'")
        else:
            self.flow_field = flow_field

    def get_wake_field(self):
        return self.wake_field

    def set_wake_field(self, wake_field=WakeField()):
        try:
            assert isinstance(wake_field, WakeField)
        except AssertionError:
            raise TypeError("'wake_field' must be of type 'WakeField'")
        else:
            self.wake_field = wake_field

    def generate_disturbed_flow_grid(self):
        """
        Generate empty u, v and w arrays corresponding to the
        x, y and z components of the disturbed flow speed at
        each x, y and z coordinate due to a single wake or
        combination of wakes.
        """
        
        flow_field = self.get_flow_field()
        x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
        len_x, len_y, len_z = x_coords.size, y_coords.size, z_coords.size
        x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

        # initialise the solution matrices, holding direction data for flow in wake
        # assumes that the flow is orthogonal to the turbine area
        #u = np.linalg.norm(flow_field.get_flow(), 2, 3).flatten().reshape((len_x, len_y, len_z))
        u = flow_field.get_flow().flatten()[0::3].reshape((len_x, len_y, len_z))
        v = flow_field.get_flow().flatten()[1::3].reshape((len_x, len_y, len_z))
        w = flow_field.get_flow().flatten()[2::3].reshape((len_x, len_y, len_z))

        #return x_grid, y_grid, z_grid, u, v, w
        return  x_coords, y_coords, z_coords, u, v, w
