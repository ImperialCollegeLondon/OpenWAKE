""" TurbineField Class that defines a group of turbines and their coordinates. """

import numpy as np

class TurbineField( object ):
    
    def __init__( self, turbine_list = [] ):
        self.turbines = []
        self.coords = [ [],[],[] ]
        self.add_turbines( turbine_list )

    def add_turbines( self, turbine_list = [] ):
        """ add given turbine objects to turbines list
            and their coordinates to coords list
            param turbine_list list of turbine objects to add
        """
        self.turbines = self.turbines + [ t for t in turbine_list ]

        num_dimensions = 3
        for i in range( num_dimensions ):
            self.coords[ i ] = self.coords[i] + [t.get_coords()[ i ] for t in turbine_list ]

    def get_turbines( self ):
        """ getter for turbines array
        """
        return np.array( self.turbines )

    def get_num_turbines( self ):
        """ returns number of turbines in turbine array """
        return self.get_turbines().size

    def get_coords( self ):
        """ returns coordinates of turbines in turbine array """
        return np.array( self.coords )
