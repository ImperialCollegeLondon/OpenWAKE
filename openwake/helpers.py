import numpy as np

def relative_index(origin_pnt_coords, pnt_coords, flow_field):
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    rel_pnt_coords = np.array(pnt_coords - origin_pnt_coords)
    # TODO ensure on input that there is common dx, dy, dz
    dx, dy, dz = abs(x_coords[1] - x_coords[0]), abs(y_coords[1] - y_coords[0]), abs(z_coords[1] - z_coords[0])
    rel_x_inx, rel_y_inx, rel_z_inx = int(rel_pnt_coords[0]/dx), int(rel_pnt_coords[1]/dy), int(rel_pnt_coords[2]/dz)
    return np.array([rel_x_inx, rel_y_inx, rel_z_inx])

def point_axis_reflection(origin_pnt_coords, pnt_coords, flow_field):
    rel_pnt_coords = relative_position(origin_pnt_coords, pnt_coords, flow_field)
    reflected_pnt_coords = pnt_coords - (2 * rel_pnt_coords)
    return np.array(reflected_pnt_coords)

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
        
def relative_position(origin_pnt_coords, pnt_coords, flow_field):
    """
    Returns the relative position of pnt_coords to turbine, rotated by angle from flow at turbine to turbine.
    TODO adjust to incorporate direction of turbine, will this work when combining wakes and updating flow?
    In the coordinate system where the flow vector at turbine is parallel to
    the x-axis and turbine is at the origin we wish to get x- and
    y-component of pnt_coords.
    """

    #TODO fix, rotate turbine direction to face flow, then assume that centreline
    # is colinear with flow
    
    # We aim to rotate the vector from turbine to pnt_coords such that the flow
    # vector at turbine is parallel to the x-axis (the 'target' vector).
    #turbine_dir = turbine.get_direction()
    target = np.array([1.0, 0.0, 0.0]) # TODO should this instead be turbine_direction?
    flow_at_origin_pnt = flow_field.get_undisturbed_flow_at_point(origin_pnt_coords, False)

    # unit vector of flow at turbine.
    normalised_flow_at_origin_pnt = (flow_at_origin_pnt/np.linalg.norm(flow_at_origin_pnt, 2))
    #normalised_turbine_dir = (turbine_dir/np.linalg.norm(turbine_dir, 2))

    # TODO is this necessary if we assume that turbine turns to face flow???
    axis_coords = np.cross(target, normalised_flow_at_origin_pnt)
    #axis_dir = np.cross(target, normalised_turbine_dir)

    theta_coords = np.arccos(np.linalg.norm(np.dot(target, normalised_flow_at_origin_pnt.reshape((-1,1)) ),2))
    #theta_dir = np.arccos(np.linalg.norm(np.dot(target, normalised_turbine_dir.reshape((-1,1)) ),2))

    rotation_matrix_coords = rotation_matrix(axis_coords, theta_coords)
    #rotation_matrix_dir = rotation_matrix(axis_dir, theta_dir)
            
    #return np.array((pnt_coords-turbine_coords)*rotation_matrix_coords*rotation_matrix_dir).flatten()
    return np.array((pnt_coords - origin_pnt_coords) * rotation_matrix_coords).flatten()
    #return np.array(pnt_coords - origin_pnt_coords)

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

