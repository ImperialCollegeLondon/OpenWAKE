"""
Copyright 2017 NREL

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import numpy as np
from scipy.optimize import minimize, basinhopping
import warnings
from imperial_floris.turbine_map import TurbineMap
from imperial_floris.floris import Floris
from imperial_floris.coordinate import Coordinate
import copy
import matplotlib.pyplot as plt
from imperial_floris.visualization_manager import VisualizationManager
import csv
import time

#warnings.simplefilter('ignore', RuntimeWarning)

def print_output(floris, data, new_params, init_power, new_power, variables, num_turbines, is_opt, plot_wakes=True, name=None, time_diff=None):
    
    num_params = len(new_params)
    msgs = []
    
    if is_opt:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('Total Power Output (MW)')
        ax.plot(data[1], data[2])

        file_path = '../examples/results/{}/{}_{}'.format(name, name,'power')
        if not os.path.exists(os.path.dirname(file_path)):
            try:
                os.makedirs(os.path.dirname(file_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        plt.savefig(file_path)
    
    state = 'Optimised' if is_opt else 'Intermediate'
    print(state, ' Parameters:')
    for p, param in enumerate(new_params):
        msgs.append('Turbine {} parameter {} = {}'.format(p % num_turbines, variables[int(p / num_turbines)], param))
    
    msgs.append('Initial Power Output = {} MW'.format(init_power/10**6))
    msgs.append('{}, Power Output = {} MW'.format(state, new_power/10**6))
    msgs.append('Power increased by {}%'.format(100 * (new_power - init_power)/init_power))
    msgs.append('Time to optimise wake = {} s'.format(time_diff))
    
    for _,msg in enumerate(msgs):
        if is_opt:
            data.append(msg)
        print(msg)
    
    floris_viz = set_iteration_data(new_params, copy.deepcopy(floris), variables)
    flow_field_viz = floris_viz.farm.flow_field
    
    visualization_manager = VisualizationManager(flow_field_viz, name, plot_wakes)
    visualization_manager.plot_z_planes([0.5])
    visualization_manager.plot_x_planes([0.5])

    with open('../examples/results/{}/{}_{}.csv'.format(name, name, 'solution_data'), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def set_turbine_attr(t, turbine, v, variable, x, num_turbines):
    if 'layout' not in variable:
        value  = x[t + (v * num_turbines)]
        setattr(turbine, variable, value)

set_turbine_attr_vec = np.vectorize(set_turbine_attr)
set_turbine_attr_vec.excluded.add(4)        

def set_iteration_data(x, floris, variables):

    turbines = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    num_turbines, num_variables = len(turbines), len(variables)
    
    set_turbine_attr_vec(np.repeat(range(num_turbines), num_variables), np.repeat(turbines, num_variables), \
                         np.repeat(range(num_variables), num_turbines, axis=0), np.repeat(variables, num_turbines, axis=0), \
                         x, num_turbines)
    
    try:
        v  = variables.index('layout_x')
        
        new_turbine_dict = {Coordinate(x[t + (v * num_turbines)], x[t + ((v + 1) * num_turbines)]):\
                            turbines[t] \
                            for t in range(num_turbines)}
        
        floris.farm.layout_x, floris.farm.layout_y = x[(v * num_turbines):((v + 1) * num_turbines)],\
                                                     x[((v + 1) * num_turbines):((v + 2) * num_turbines)]

        floris.farm.turbine_map = floris.farm.flow_field.turbine_map = TurbineMap(new_turbine_dict)
        
    except ValueError:
        pass
    
    return floris
    
def calc_power(floris):
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    return np.sum([turbine.power for turbine in turbines])

def power_objective_func(x, floris, variables, data):
    
    floris = set_iteration_data(x, floris, variables)
    
    floris.farm.flow_field.calculate_wake()
    power = -calc_power(floris)
    
    # data[0] is a row containing headers, data[1][i] contains iteration number i, data[2][i] contains power output at iteration i
    # data[3][i] contains solution x at iteration i, data[4] on contain print strings 
    data[1].append(data[1][-1] + 1)
    data[2].append(-power/(10**6))
    data[3].append(x)
    
    return power

def optimise_func(floris, variables, minimum_values, maximum_values, name, case, global_search=True):
    """
    variables=['yaw_angle', 'tilt_angle', 'tsr','blade_pitch', 'blade_count',\
                                     'rotor_diameter', 'hub_height', 'layout_x', 'layout_y']
    """
    t1 = time.time()
    
    turbines = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    init_power = calc_power(floris)
    num_turbines, num_variables =  len(turbines), len(variables)
    x0, bnds = [], []
    data = [['Iteration Number', 'Power Output'],[0],[init_power/(10**6)]]
    for v, variable in enumerate(variables):
        bnds = bnds + [(minimum_values[v], maximum_values[v]) for t in range(num_turbines)]
        data[0].append(variable)
        if variable == 'layout_x':
            x0 = x0 + floris.farm.layout_x + floris.farm.layout_y
        elif variable == 'layout_y':
            continue
        else:
            x0 = x0 + [getattr(turbine, variable) for turbine in turbines]
    data.append([x0])
            
    # define nonlinear constraints. all turbine coordinates must be at least the sum of its radius and each other turbines
    #radius away from each other turbine
    # ((xi - xj)**2 + (yi - yj)**2)**0.5 - (ri + rj) >= 0 for all i and j

    def nonlin_constr_func(x):
        constr_f_arr = np.zeros((np.math.factorial(num_variables * num_turbines)))
        # apply only to location variables
        try:
            v  = variables.index('layout_x')
            for t in range(num_turbines):
                for tt in range(t, num_turbines):
                    i, ii = t + (v * num_turbines), tt + (v * num_turbines)
                    j, jj = t  + ((v + 1) * num_turbines), tt  + ((v + 1) * num_turbines)
                    xi, yi = x[i], x[j]
                    xj, yj = x[ii], x[jj]

                    t_rad = turbines[t + (v * num_turbines)].rotor_diameter / 2.0
                    tt_rad = turbines[tt + (v * num_turbines)].rotor_diameter / 2.0

                    constr_f_arr[i + ii] = np.hypot(xi - xj, yi - yj) - (t_rad + tt_rad)
                    
        except ValueError:
            pass
        
        return constr_f_arr
    
    def constr_j(x):
        constr_j_arr = np.zeros((np.math.factorial(num_variables*num_turbines), num_variables*num_turbines))
        # apply only to location variables
        try:
            v  = variables.index('layout_x')
            for t in range(num_turbines - 1):
                for tt in range(t + 1, num_turbines):
                    i, ii = t + (v * num_turbines), tt + (v * num_turbines)
                    j, jj = t  + ((v + 1) * num_turbines), tt  + ((v + 1) * num_turbines)
                    xi, yi = x[i], x[j]
                    xj, yj = x[ii], x[jj]
                    constr_j_arr[i + ii - 1, i] = 2 * (xi - xj)
                    constr_j_arr[i + ii - 1, ii] = -2 * (xi - xj)
                    constr_j_arr[i + ii - 1, j] = 2 * (yi - yj)
                    constr_j_arr[i + ii - 1, jj] = -2 * (yi - yj)
        
        except ValueError:
            pass
        
        return constr_j_arr
    
    ineq_constr = {'type': 'ineq', 
                   'fun' : lambda x: nonlin_constr_func(x),
                   'jac' : lambda x: constr_j(x)}

    print('=====================================================================')
    print('Optimizing...')
    print('Number of parameters to optimize = ', len(x0))
    print('=====================================================================')
    
    minimizer_kwargs = {'args':(floris, variables, data), 'method': 'SLSQP',\
                        'bounds':bnds, \
                        'options':{'disp':True}}#'constraints':ineq_constr, unneccessary
    
    # TODO could also 'basin-hop' intelligently by choosing areas of high flow
    if global_search:
        residual_plant = basinhopping(power_objective_func, x0, minimizer_kwargs=minimizer_kwargs, disp=True)
    else:
        residual_plant = minimize(power_objective_func, x0, args=(floris, variables, data), \
                                  method='SLSQP', bounds=bnds)#constraints=ineq_constr
    
    if np.sum(residual_plant.x) == 0:
        print('No change in controls suggested for this inflow condition...')

    opt_params, opt_power = residual_plant.x, -residual_plant.fun
    
    t2 = time.time()
    time_diff = t2 - t1
    
    print_output(floris, data, opt_params, init_power, opt_power, variables, num_turbines, is_opt=True, plot_wakes=True, name=name, time_diff=time_diff)
    
    return opt_params, opt_power, data

