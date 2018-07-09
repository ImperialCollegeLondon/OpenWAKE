""" Class of helper functions. """

import numpy as np

def relative_index( origin_pnt_coords, pnt_coords, diff ):
    """ return the indices of pnt_coords relative to origin_pnt coords
        param origin_pnt_coords 3-element list or array, usually [0, 0, 0] for absolute coordinate system
                                or turbine_coords for relative coordinate system
        param pnt_coords 3-element list or array, point of which index must be found
        param diff 3-element list or array of dx, dy and dz in this coordinate system        
    """
    rel_pnt_coords = np.array( pnt_coords - origin_pnt_coords )
    dx, dy, dz = diff
    rel_x_inx, rel_y_inx, rel_z_inx = int( rel_pnt_coords[0] / dx ), int( rel_pnt_coords[1] / dy ), int( rel_pnt_coords[2] / dz )
    return np.array( [rel_x_inx, rel_y_inx, rel_z_inx] )

def rotation_matrix( axis, theta ):
        """ return the 3D rotation matrix which will rotate a point counterclockwise around axis by theta
        param axis array vector
        param theta angle in radians
        """

        # if the axis is only a point, return the identity matrix
        if np.linalg.norm( axis, 2 ) == 0:
            return np.matrix( np.diag([1,1,1] ) )
        # else normalise the axis vector, build and return the 3D rotation matrix
        else:
            axis = axis / np.linalg.norm( axis, 2 )
            a = np.cos( theta / 2 )
            b, c, d = -axis * np.sin( theta / 2 )
            aa, bb, cc, dd = a**2, b**2, c**2, d**2
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            return np.matrix( [ [aa + bb - cc - dd, 2 * ( bc + ad ), 2 * ( bd - ac ) ],
                             [ 2 * ( bc - ad ), aa + cc - bb - dd, 2 * ( cd + ab ) ],
                             [ 2 * ( bd + ac ), 2 * ( cd - ab ), aa + dd - bb- cc ] ] )
        
def relative_position(origin_pnt_coords, pnt_coords, flow_dir_at_origin_pnt, clockwise=True):
    """ return the relative position of pnt_coords to the flow direction vector passing through origin_pnt_coords
    param origin_pnt_coords 3-element list or array, usually turbine_coords for coordinate system relative to wake
    param pnt_coords 3-element list or array of which relative position to direction vector must be found
    param flow_dir_at_origin_pnt direction of flow and thus direction of turbine at the origin point
    param clockwise boolean indicating whether rotation should be clockwise or not.
    """    
    # rotate flow_dir_at_origin_pnt from origin_pnt_coords to pnt_coords such that
    # the flow_dir_at_origin_pnt is parallel to the x-axis (the 'target' vector).
    # then rotate pnt_coords by the same amount around origin_pnt_coords

    target = np.array( [1.0, 0.0, 0.0] )    
    axis = np.cross( target, flow_dir_at_origin_pnt )
    theta = np.arccos( np.linalg.norm( np.dot( target, flow_dir_at_origin_pnt.reshape( ( -1,1 ) ) ),2 ) )
    
    if clockwise == False:
        theta = -theta
    
    rotation_matrix_coords = rotation_matrix(axis, theta)
    return np.array((pnt_coords - origin_pnt_coords) * rotation_matrix_coords).flatten()

def find_nearest( array, value ):
    """ return the value in array closest in value to value
    param array list or array to check
    param value value to search for
    """
    index = ( np.abs( array - value ) ).argmin()
    
    return array[ index ]

def find_index( array, value ):
    """ return the index of the value in array closest in value to value
    param array array or list to check
    param value value to search for
    """
    return ( np.abs( array - value ) ).argmin()

def set_nan_or_inf_to_zero( array ):
    """ returns given array with any NaN or Inf values set to zero
        useful when wishing to avoid zero division errors
    param array array or list to change
    """
    array[ np.isinf( array ) + np.isnan( array ) ] = 0
    
    return array

def set_below_abs_tolerance_to_zero(array, tolerance=1e-2):
    """ returns given array with values below given tolearance set to 0
    param array list of values to change
    param tolerance value below which array elements should be reassigned to zero
    """
    array[ abs( array ) < tolerance] = 0
    return array

