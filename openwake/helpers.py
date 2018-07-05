import numpy as np

def relative_index(origin_pnt_coords, pnt_coords, flow_field):
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    rel_pnt_coords = np.array(pnt_coords - origin_pnt_coords)
    dx, dy, dz = flow_field.get_dx(), flow_field.get_dy(), flow_field.get_dz()
    rel_x_inx, rel_y_inx, rel_z_inx = int(rel_pnt_coords[0] / dx), int(rel_pnt_coords[1] / dy), int(rel_pnt_coords[2] / dz)
    return np.array([rel_x_inx, rel_y_inx, rel_z_inx])

def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        if np.linalg.norm(axis, 2) == 0:
            return np.matrix(np.diag([1,1,1]))
        else:
            axis = axis / np.linalg.norm(axis, 2)
            a = np.cos(theta/2.0)
            b, c, d = -axis*np.sin(theta/2.0)
            aa, bb, cc, dd = a*a, b*b, c*c, d*d
            bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
            return np.matrix([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                             [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                             [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
        
def relative_position(origin_pnt_coords, pnt_coords, flow_dir_at_origin_pnt, clockwise=True):
    """
    Returns the relative position of pnt_coords to turbine, rotated by angle from flow at turbine to turbine.
    TODO adjust to incorporate direction of turbine, will this work when combining wakes and updating flow?
    In the coordinate system where the flow vector at turbine is parallel to
    the x-axis and turbine is at the origin we wish to get x- and
    y-component of pnt_coords.
    """    
    # We aim to rotate the vector from turbine to pnt_coords such that the flow
    # vector at turbine is parallel to the x-axis (the 'target' vector).
    target = np.array([1.0, 0.0, 0.0])
    
    axis_coords = np.cross(target, flow_dir_at_origin_pnt)
    
    theta_coords = np.arccos(np.linalg.norm(np.dot(target, flow_dir_at_origin_pnt.reshape((-1,1)) ),2))
    
    if clockwise == False:
        theta_coords = -theta_coords
    
    rotation_matrix_coords = rotation_matrix(axis_coords, theta_coords)
    return np.array((pnt_coords - origin_pnt_coords) * rotation_matrix_coords).flatten()

def find_nearest(array, value):
    """
    Find the nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def find_index(array, value):
    """
    Find index of nearest value in an array
    """
    return (np.abs(array-value)).argmin()

def set_nan_or_inf_to_zero(array):
    """
    Sets NaN or Inf values to zero.

    This is useful when wishing to avoid zero division errors when using the
    'reduce' method.

    Returns altered array
    param array list of values to change
    """
    array[np.isinf(array) + np.isnan(array)] = 0
    return array

def set_below_abs_tolerance_to_zero(array, tolerance=1e-2):
    """
    Sets values below the given tolerance to zero.

    This is useful when wishing to avoid zero division errors when using the
    'reduce' method.

    Returns altered array
    param array list of values to change
    param tolerance value below which array elements should be reassigned to zero
    """
    array[abs(array) < tolerance] = 0
    return array

