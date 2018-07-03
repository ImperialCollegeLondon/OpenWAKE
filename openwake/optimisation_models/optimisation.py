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

def maximise_power_output(flow_field, wake_field, turbine_field, var_list='xy'):

    # define variables
    wakes = wake_field.get_wakes()
    flow = flow_field.get_flow()
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), fflow_fieldlow.get_z_coords()
    len_x = x_coords.size
    num_turbines = turbine_field.get_num_turbines()
    min_x, max_x, min_y, max_y, min_z, max_z = x_coords.min(), x_coords.max(). y_coords.min(), y_coords.max(), z_coords.min(), z_coords.max()
    turbine_coords_arr = np.zeros((3, len_x))
    turbine_radius_arr = np.zeros(np.math.factorial(num_variables))

    lower_bounds_arr = np.zeros((3, len_x))
    upper_bounds_arr = np.zeros((3, len_x))

    for t in turbines:
        turbine_coords = t.get_coords()
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

    initial_variables = turbine_coords_arr.flatten().tolist()
    num_variables = variables.size

    # define bound constraints. all x, y and z turbine coordinates must be within the domain
    lower_bounds = lower_bounds_arr.flatten().tolist()
    upper_bounds = upper_bounds_arr.flatten().tolist()

    linear_bounds = Bounds(lower_bounds, upper_bounds)

    # define nonlinear constraints. all turbine coordinates must be at least the sum of its radius and each other turbines radius away from each other turbine
    # ri + rj <= ((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**0.5 for all i and j
    def non_linear_constraint_func(x):
        cons_f_arr = np.zeros((np.math.factorial(num_variables)))

        for v in range(0:num_turbines):
            xi, yi, zi = x[v], x[v + num_turbines], x[v + (2 * num_turbines)]
            for w in range(x[v:num_turbines]):
                xi, yi, zi = x[v], x[v + num_turbines], x[v + (2 * num_turbines)]
                xj, yj, zj = x[w], x[w + num_turbines], x[w + (2 * num_turbines)]
                
                turbine_i_radius = wakes[v].get_turbine().get_radius()
                turbine_j_radius = wakes[w].get_turbine().get_radius()
                turbine_radius_arr[v + w] = turbine_i_radius + turbine_j_radius
                    
                cons_f_arr[v + w] = ((xi - xj)**2 + (yi - yj)**2 + (zi - zj)**2)**0.5)

        return cons_f_arr.tolist()
                    
    def cons_j(x):
        cons_j_arr = np.zeros((np.math.factorial(num_variables), num_variables))
        for v in range(0:num_turbines-1):
            for w in range(v+1:num_turbines):
                xi, yi, zi = x[v], x[v + num_turbines], x[v + (2 * num_turbines)]
                xj, yj, zj = x[w], x[w + num_turbines], x[w + (2 * num_turbines)]
                cons_j_arr[v + w - 1, v] = 2 * (xi - xj)
                cons_j_arr[v + w - 1, w] = -2 * (xi - xj)
                cons_j_arr[v + w - 1, v + num_turbines] = 2 * (yi - yj)
                cons_j_arr[v + w - 1, w + num_turbines] = -2 * (yi - yj)
                cons_j_arr[v + w - 1, v + (2 * num_turbines)] = 2 * (zi - zj)
                cons_j_arr[v + w - 1, w + (2 * num_turbines)] = -2 * (zi - zj)
        return cons_j_arr.tolist()

    def cons_h(x, v):
        """
        linear combination of hessians
        """
        cons_h_arr = np.zeros((np.math.factorial(num_variables), num_variables))
        cons_h_arr[i, i], cons_h_arr[i, j] = 2, -2, 0
        for v in range(0:num_turbines-1):
            for w in range(v+1:num_turbines):
                xi, yi, zi = x[v], x[v + num_turbines], x[v + (2 * num_turbines)]
                xj, yj, zj = x[w], x[w + num_turbines], x[w + (2 * num_turbines)]
                cons_h_arr[v + w - 1, v] = 2
                cons_h_arr[v + w - 1, w] = -2
                cons_h_arr[v + w - 1, v + num_turbines] = 2
                cons_h_arr[v + w - 1, w + num_turbines] = -2
                cons_h_arr[v + w - 1, v + (2 * num_turbines)] = 2
                cons_h_arr[v + w - 1, w + (2 * num_turbines)] = -2
        return np.sum(v * cons_h_arr).tolist()
 
    non_linear_constraint = NonlinearConstraint(non_linear_constraint_func, turbine_radius_arr, np.inf, \
                                                jac = cons_j, hess = cons_h)

    # define objective function, jacobian, hessian
    def obj_j(x):
        obj_j_arr = np.zeros(())
        for t in range(num_turbines):
            #turbines[t].set_coords([x[t], x[t + num_turbines], x[t + (2 * num_turbines)]])
        wake_combination.calc_disturbed_flow_grid(False)
        flow_mag_at_turbine = wake_combination.get_disturbed_flow_at_point(turbine_coords, True, False)
        output_power = np.sum(t.calc_power_output(flow_mag_at_turbine))

    def obj_h(x):
        
    def obj_func()

    # find solution                                               
    result = minimize(obj_func, initial_variables, method='trust-constr', jac=non_linear_constraint_jacobian, hess='2-point',\
                      constraints=[non_linear_constraint], options={'verbose':1}, bounds=linear_bounds)
