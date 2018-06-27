import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonLinearConstraint, minimize as minimize


def minimise_disturbed_flow_deficit():
    # define function

    # define linear equality constraints

    # define linear bound constraints
    lower_bounds []
    upper_bounds = []
    linear_bounds = Bounds(lower_bounds, upper_bounds)

    # define non-linear equality constraints

    # define non-linear bound constraints

    pass

def maximise_power_output(flow_field, wake_field, var_list='xy'):

    # define variables
    wakes = wake_field.get_wakes()
    flow = flow_field.get_flow()
    x_coords, y_coords, z_coords = flow.get_x_coords(), flow.get_y_coords(), flow.get_z_coords()
    len_x = x_coords.size
    num_turbines = wake_field.get_num_wakes()
    min_x, max_x, min_y, max_y, min_z, max_z = x_coords.min(), x_coords.max(). y_coords.min(), y_coords.max(), z_coords.min(), z_coords.max()
    turbine_coords_arr = np.zeros((3, len_x))
    turbine_radius_arr = np.zeros(np.math.factorial(num_variables))

    lower_bounds_arr = np.zeros((3, len_x))
    upper_bounds_arr = np.zeros((3, len_x))

    for w in wakes:
        turbine = w.get_turbine()
        turbine_coords = turbine.get_coords()
        if 'x' in var_list:
            np.append(turbine_coords_arr[0], turbine_coords[0])
            np.append(lower_bounds_arr[0], min_x)
            np.append(upper_bounds_arr[0], max_x)
        if 'y' in var_list:
            np.append(turbine_coords_arr[1], turbine_coords[1])
            np.append(lower_bounds_arr[1], min_y)
            np.append(upper_bounds_arr[1], max_y)
        if 'z' in var_list:
            np.append(turbine_coords_arr[2], turbine_coords[2])
            np.append(lower_bounds_arr[2], min_z)
            np.append(upper_bounds_arr[2], max_z)

    variables = turbine_coords_arr.flatten().tolist()
    num_variables = variables.size
    
    # define function

    # define linear equality constraints

    # define bound constraints. all x, y and z turbine coordinates must be within the domain
    lower_bounds = lower_bounds_arr.flatten().tolist()
    upper_bounds = upper_bounds_arr.flatten().tolist()

    linear_bounds = Bounds(lower_bounds, upper_bounds)

    # define linear constraints
    
    #linear_constraint = LinearConstraint(coefficients, lower, upper)

    # define nonlinear constraints. all turbine coordinates must be at least the sum of its radius and each other turbines radius away from each other turbine
    # ri + rj <= ((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**0.5 for all i and j
    def non_linear_constraint_func(variables):
        non_linear_constraints_arr = np.zeros((np.math.factorial(num_variables)))
##        for w in wakes:
##            for v in wakes:
##                if w != v:

        for v in range(variables[0:num_turbines]):
            xi, yi, zi = variables[v], variables[v + num_turbines], variables[v + (2 * num_turbines)]
            for w in range(variables[v:num_turbines]):
                xi, yi, zi = variables[v], variables[v + num_turbines], variables[v + (2 * num_turbines)]
                xj, yj, zj = variables[w], variables[w + num_turbines], variables[w + (2 * num_turbines)]
##                    xi, yi, zi = turbine_i.get_coords()
##                    xj, yj, zj = turbine_j.get_coords()
                
                turbine_i_radius = wakes[v].get_turbine().get_radius()
                turbine_j_radius = wakes[w].get_turbine().get_radius()
                turbine_radius_arr[v + w] = turbine_i_radius + turbine_j_radius
                    
                non_linear_constraints_arr[v + w] = ((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**0.5)

        return non_linear_constraints_arr.tolist()
                    
    def non_linear_constraint_jacobian(variables):
        non_linear_constraints_jacobian_arr = np.zeros((np.math.factorial(num_variables), num_variables))
        for v in range(variables[0:num_turbines]):
            for w in range(variables[v:num_turbines]):
                xi, yi, zi = variables[v], variables[v + num_turbines], variables[v + (2 * num_turbines)]
                xj, yj, zj = variables[w], variables[w + num_turbines], variables[w + (2 * num_turbines)]
                non_linear_constraints_jacobian_arr[v + w, v] = 2 * (xi - xj) * xi
                non_linear_constraints_jacobian_arr[v + w, w] = -2 * (xi - xj) * xj
                non_linear_constraints_jacobian_arr[v + w, v + num_turbines] = 2 * (yi - yj) * yi
                non_linear_constraints_jacobian_arr[v + w, w + num_turbines] = -2 * (yi - yj) * yj
                non_linear_constraints_jacobian_arr[v + w, v + (2 * num_turbines)] = 2 * (zi - zj) * zi
                non_linear_constraints_jacobian_arr[v + w, w + (2 * num_turbines)] = -2 * (zi - zj) * zj
        return non_linear_constraints_arr.tolist()
 
    non_linear_constraint = NonlinearConstraint(non_linear_constraint_func, )

    # define non-linear bound constraints
    pass
