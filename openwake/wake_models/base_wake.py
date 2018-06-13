"""
A BaseWakeModel class from which other wake models may be derived.
"""
import numpy as np
import math
from turbine_models.base_turbine import BaseTurbine
from wake_models.flow import Flow

class BaseWakeModel(object):
    """A base class from which wake models may be derived."""

    def __init__(self, turbine, flow):
        self.turbine = turbine
        self.flow = flow

    def get_turbine(self):
        return self.turbine

    def set_turbine(self, turbine):
        default_turbine = BaseTurbine(None,None,None,None,None)
        if not (isinstance(turbine, BaseTurbine) or turbine == None):
            raise TypeError("'turbine' must be of type 'Turbine'")
        else:
            self.turbine = turbine if turbine != None else default_turbine

    def get_flow(self):
        return self.flow

    def set_flow(self, flow):
        default_flow  = Flow(None, None, None)
        if not (isinstance(flow, Flow) or flow == None):
            raise TypeError("'flow' must be of type 'Flow'")
        else:
            self.flow = flow if flow != None else default_flow

    def is_in_wake(self, pnt_coords):

        """
        Returns True if point is in wake.
        """

        rel_pnt_coords = self.relative_position(pnt_coords)
        x0, y0, z0 = rel_pnt_coords

        if (x0 < 0.0):
            return False
        else:
            wake_radius = self.calc_wake_radius(x0)
            return wake_radius > abs(y0)

    def relative_position(self, pnt_coords):
        """
        Returns the relative position of pnt_coords to turbine, rotated by angle from flow at turbine to turbine.
        TODO adjust to incorporate direction of turbine, will this work when combining wakes and updating flow?
        In the coordinate system where the flow vector at turbine is parallel to
        the x-axis and turbine is at the origin we wish to get x- and
        y-component of pnt_coords.
        """
        # We aim to rotate the vector from turbine to pnt_coords such that the flow
        # vector at turbine is parallel to the x-axis (the 'target' vector).
        target = np.array([1.0, 0.0, 0.0]) # should this instead be turbine_direction?
        turbine = self.get_turbine()
        turbine_coords = turbine.get_coords()
        #turbine_dir = turbine.get_direction()
        flow_at_turbine = self.get_flow().get_flow_at_point(self.get_turbine().get_coords())

        # unit vector of flow at turbine.
        normalised_flow_at_turbine = (flow_at_turbine/np.linalg.norm(flow_at_turbine, 2))
        #normalised_turbine_dir = (turbine_dir/np.linalg.norm(turbine_dir, 2))

        def rotation_matrix(axis, theta):
            """
            Return the rotation matrix associated with counterclockwise rotation about
            the given axis by theta radians.
            """
            axis = axis/np.linalg.norm(axis, 2)
            a = np.cos(theta/2.0)
            b, c, d = -axis*np.sin(theta/2.0)
            aa, bb, cc, dd = a*a, b*b, c*c, d*d
            bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
            return np.matrix([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                             [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                             [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


        # TODO IS IT RIGHT TO PROJECT FOR DIRECTION TOO
        axis_coords = np.cross(target, normalised_flow_at_turbine)
        axis_dir = np.cross(target, normalised_turbine_dir)

        theta_coords = np.arccos(np.linalg.norm(np.dot(target, normalised_flow_at_turbine.reshape((-1,1)) ),2))
        #theta_dir = np.arccos(np.linalg.norm(np.dot(target, normalised_turbine_dir.reshape((-1,1)) ),2))

        rotation_matrix_coords = rotation_matrix(axis_coords, theta_coords)
        #rotation_matrix_dir = rotation_matrix(axis_dir, theta_dir)
                
        #return np.array((pnt_coords-turbine_coords)*rotation_matrix_coords*rotation_matrix_dir).flatten()
        return np.array((pnt_coords-turbine_coords)*rotation_matrix_coords).flatten() 
