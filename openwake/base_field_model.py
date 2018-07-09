""" A Base Field class which defines methods, setters and getters used by both
    wake and wake combination classes
"""

import numpy as np
from helpers import *
from flow_field_model.flow import FlowField
from wake_models.wake_field_model import WakeField

class BaseField( object ):

    def __init__( self, flow_field, wake_field ):
        self.set_flow_field( flow_field )
        self.set_wake_field( wake_field )

    def set_grid_outdated( self, grid_outdated ):
        """ setter for is_grid_outdated attribute
            should be set to true every time wake or flow parameters are changed
            except for turbine location
        """
        self.is_grid_outdated = grid_outdated
        
    def get_flow_field( self ):
        """ getter for flow_field object
        """
        return self.flow_field

    def set_flow_field(self, flow_field = FlowField() ):
        """ setter for flow_field object
            param flow_field flow_field object to set
        """
        try:
            assert isinstance( flow_field, FlowField )
        except AssertionError:
            raise TypeError( "'flow_field' must be of type 'FlowField'" )
        else:
            self.flow_field = flow_field

    def get_wake_field( self ):
        """ getter for wake_field object
        """
        return self.wake_field

    def set_wake_field(self, wake_field = WakeField() ):
        """ setter for wake_field object
            param wake_field wake_field object to set
        """
        try:
            assert isinstance(wake_field, WakeField)
        except AssertionError:
            raise TypeError( "'wake_field' must be of type 'WakeField'" )
        else:
            self.wake_field = wake_field
