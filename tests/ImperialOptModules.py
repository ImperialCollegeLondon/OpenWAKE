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

import sys
sys.path.append('../floris')
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
import warnings
from floris.turbine_map import TurbineMap
from floris.coordinate import Coordinate
import copy
import matplotlib.pyplot as plt
from visualization_manager import VisualizationManager
import csv

#warnings.simplefilter('ignore', RuntimeWarning)

def print_output(floris, data, new_params, init_power, new_power, variables, num_turbines, is_opt, name):
    
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Total Power Output (MW)')
    ax.plot(data[1], data[2])
    path = 'results/{}_{}'.format(name,'power')
    plt.savefig(path)
    
    state = 'Optimised' if is_opt else 'Intermediate'
    print(state, ' Parameters:')
    for p, param in enumerate(new_params):
        print('Turbine ', p % num_turbines, ' parameter ', variables[int(p / num_turbines)], ' = ', param)
    
    data.append('Initial Power Output = {} MW'.format(init_power/10**6))
    data.append('{}, Power Output = {} MW'.format(state, new_power/10**6))
    data.append('Power increased by {}%'.format(100 * (new_power - init_power)/init_power))
    print(data[4])
    print(data[5])
    print(data[6])
    
    ff_viz = floris.farm.flow_field
    visualization_manager = VisualizationManager(ff_viz, name)
    visualization_manager.plot_z_planes([0.5])
    visualization_manager.plot_x_planes([0.5])
    
    with open('results/{}.csv'.format(name), 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def set_turbine_attr(t, turbine, v, variable, x, num_turbines):
    if 'layout' not in variable:
        value  = x[t + (v * num_turbines)]
        setattr(turbine, variable, value)
    
set_turbine_attr_vec = np.vectorize(set_turbine_attr)
set_turbine_attr_vec.excluded.add(4)

def calc_power(floris):
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    return np.sum([turbine.power for turbine in turbines])

def power_objective_func(x, floris, variables, data):
    turbines    = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
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
        
        #print('1', x)
        # data[v + 2] contains updating turbine_map
        floris.farm.turbine_map = floris.farm.flow_field.turbine_map = copy.deepcopy(TurbineMap(new_turbine_dict))
        #data[v + 2].append(copy.deepcopy(floris.farm.turbine_map))
        
    except ValueError:
        pass

    floris.farm.flow_field.calculate_wake()
    power = -calc_power(floris)
    # data[0] contains iteration number, data[1] contains power output
    data[1].append(data[1][-1] + 1)
    data[2].append(-power/(10**6))
    data[3].append(x)
    return power

def optimise_func(floris, variables, minimum_values, maximum_values, name, global_search=True):
    """
    variables=['yaw_angle', 'tilt_angle', 'tsr','blade_pitch', 'blade_count',\
                                     'rotor_diameter', 'hub_height', 'layout_x', 'layout_y']
    """
    turbines = [turbine for _, turbine in floris.farm.flow_field.turbine_map.items()]
    init_power = calc_power(floris)
    num_turbines, num_variables = len(turbines), len(variables)
    x0, bnds = [], []
    data = [[],[0],[init_power/(10**6)]]
    for v, variable in enumerate(variables):
        bnds = bnds + [(minimum_values[v], maximum_values[v]) for t in range(num_turbines)]
        if variable == 'layout_x':
            data[0].append(variable)
            x0 = x0 + floris.farm.layout_x + floris.farm.layout_y
            #data.append([floris.farm.turbine_map])
        elif variable == 'layout_y':
            continue
        else:
            x0 = x0 + [getattr(turbine, variable) for turbine in turbines]
            #data.append([getattr(turbine, variable) for turbine in turbines])
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
                        'options':{'disp':True}}#'constraints':ineq_constr,
    # 'ftol':1e-06, 'maxiter':100, 'eps':1e-06
    
    # TODO could also 'basin-hop' intelligently by choosing areas of high flow
    # basin-hop
    # BEWARE PLOTTING ERROR: occurds if temperature T is set too high, should be comparable to 
    # difference in objective function value between local minima
    # smaller value results in shorter processing time
    # niter=10, niter_success=2, T=100 works
    if global_search:
        residual_plant = basinhopping(power_objective_func, x0, minimizer_kwargs=minimizer_kwargs, disp=True )
        #                              
        #niter=100, niter_success=10), T=10^3, disp=True)
        #residual_plant = differential_evolution(power_objective_func, bounds=bnds, args=(floris, variables, data), disp=True)
    else:
        residual_plant = minimize(power_objective_func, x0, args=(floris, variables, data), \
                                  method='SLSQP', constraints=ineq_constr, bounds=bnds)
    
    # Plot Power vs Number of Iterations
    """
    fig, ax = plt.subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()

    for i in range(0,100):
        ax.clear()
        ax.plot(matrix[i,:])
        fig.canvas.draw()
        ax.plot(data[0], data[1])
        plt.show()
    """

    if np.sum(residual_plant.x) == 0:
        print('No change in controls suggested for this inflow condition...')

    opt_params, opt_power = residual_plant.x, -residual_plant.fun
    print_output(floris, data, opt_params, init_power, opt_power, variables, num_turbines, True, name)
    
    return opt_params, opt_power