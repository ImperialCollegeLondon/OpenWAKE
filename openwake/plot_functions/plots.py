import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from helpers import *

def plot_flow_field(flow_field):
    # Make the grid. coordinates of the arrow locations
    x, y, z = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    xMin, xMax, yMin, yMax, zMin, zMax = np.amin(x), np.amax(x), np.amin(y), np.amax(y), np.amin(z), np.amax(z)
    x, y, z = np.meshgrid(x, y, z)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    flow = flow_field.get_flow()

    # Make the direction data for the arrows. x,y,z components of the arrow vectors
    flowFlatten = np.array(flow_field).flatten()
    flowLength = flowFlatten.size
    u, v, w = flowFlatten[0:flowLength:3], flowFlatten[1:flowLength:3], flowFlatten[2:flowLength:3]
   
    flowFig = plt.figure()
    flowAx = flowFig.gca(projection='3d')
    flowAx.quiver(x, y, z, u, v, w, label='Flow Vector', length=0.3)
    flowFig.suptitle(r'Flow Field', fontsize=20)
    flowAx.legend(loc='best')
    flowAx.set_xlabel('x (m)')
    flowAx.set_ylabel('y (m)')
    flowAx.set_zlabel('z (m)')
    flowAx.set_xlim3d(xMin, xMax)
    flowAx.set_ylim3d(yMin, yMax)
    flowAx.set_zlim3d(zMin, zMax)

    plt.show()

def plot_turbine_coefficients(turbine):

    coeffPlt, coeffAx = plt.subplots()
    thrustCoefficient = turbine.get_thrust_coefficient_curve()
    powerCoefficient = turbine.get_power_coefficient_curve()
    coeffAx.plot(thrustCoefficient[0], thrustCoefficient[1], 'r', label='Thrust Coefficient')
    coeffAx.plot(powerCoefficient[0], powerCoefficient[1], 'b', label='Power Coefficient')
    coeffPlt.suptitle(r'Turbine Coefficients', fontsize=20)
    coeffAx.legend(loc='best')
    coeffAx.set_xlabel('speed (m/s)')
    coeffAx.set_ylabel('coefficient')
    plt.show()

def plot_turbine_location(turbine_field):
    locPlt, locAx = plt.subplots()
    locPlt.suptitle(r'Turbine Array Grid', fontsize=20)
    locAx.set_xlabel('x (m)')
    locAx.set_ylabel('y (m)')
    #locAx.set_xlim(0,5)
    #locAx.set_ylim(0,5)
    turbines = turbine_field.get_turbines()
    for t in turbines:
        x, y, z = t.get_coords()
        locAx.plot(x, y, 'o', label=', '.join([str(x), str(y), str(z)]))

    locAx.legend(loc='best')
    plt.show()

def plot_wakes(wake_field, wake_combination, turbine_field, flow_field, plane='xz'):
    """
    param plane which plane to view wakes in. 'xz' produces a side-view, 'xy' produces a birdseye view
    """
    flow = flow_field.get_flow()
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    len_x, len_y, len_z = flow.shape[0:3]

    if plane == 'xz':
        len_r = len_z
    elif plane == 'xy':
        len_r = len_y
        
    undisturbed_flow_contour = np.zeros((len_x, len_r, 2))
    disturbed_flow_contour = np.zeros((len_x, len_r, 2))

    wakes = wake_field.get_wakes()
    turbines = turbine_field.get_turbines()
    
    # x and y or z components of quiver coordinates
    x_vec, r_vec = [], []

    # set turbine coords of minimum x coordinate as origin
    origin_coords = np.array([0,0,0])
    for i in range(len_x):
        x = x_coords[i]
        for j in range(len_y):
            if plane == 'xz':
                # select x, z plane parallel to y coordinate of turbine, then dispense with y-component of flow, assuming that turbine will turn to face it
                start, end, increment = 0, 3, 2
            y = y_coords[j]#TODO set this or z to the turbine_coordinate of each wake
            for k in range(len_z):
                if plane == 'xy':
                    # select x, y plane parallel to z coordinate of turbine, then dispense with z-component of flow, assuming that turbine will turn to face it
                    start, end, increment = 0, 2, 1
                z = z_coords[k]
                
                rel_pos = relative_position(origin_coords, np.array([x, y, z]), flow_field)
                undisturbed_flow_contour[i, k] = flow_field.get_undisturbed_flow_at_point([x, y, z], False)[start : end : increment]
                for w in wakes[0:1]:
                    disturbed_flow_contour[i, k] = w.get_disturbed_flow_at_point([x, y, z], False)[start : end : increment]                   

                # append y or z coordinate to r_vec, but only once
                if i == 0:
                    if plane == 'xz' and j == 0:
                        r_vec.append(rel_pos[2])
                    elif plane == 'xy':
                        r_vec.append(rel_pos[1])
    
        # append x coordinate to x_vec, but only once
        x_vec.append(rel_pos[0])

    # x and z components of quiver directions
    # in multi-dimensional flow array, first dim = x, second dim = y, third dim = z
    # but in meshgrid, these axis are swapped visually (different 'rows' correspond to r axis)
    # therefore, flow contours are transposed for contour plots
    # and meshgrids are transposed for quiver plots
    u, v = np.meshgrid(x_vec, r_vec)
    x_min = np.amin(x_vec); x_max = np.amax(x_vec)
    r_min = np.amin(r_vec); r_max = np.amax(r_vec)
    
    wake_fig, wake_axes = plt.subplots(1, 2, constrained_layout=True)
    #wake_fig.tight_layout(pad=0.4, w_pad=6.0, h_pad=1.0)
    for i in range(2):
        wake_axes[i].set_xlabel('Longitudinal Distance, r (m)')
        wake_axes[i].set_ylabel('Transverse Distance, x (m)')
        wake_axes[i].set_xlim(x_min, x_max)
        wake_axes[i].set_ylim(r_min, r_max)

        for t in turbines:
            turbine_radius = t.get_radius()
            turbine_coords = relative_position(origin_coords, t.get_coords(), flow_field)
            x  = turbine_coords[0]
            if plane == 'xz':
                r = turbine_coords[2]
            elif plane == 'xy':
                r = turbine_coords[1]
            turbine_rect = Rectangle((x, r - turbine_radius), 1, 2 * turbine_radius, hatch = '//')
            wake_axes[i].add_patch(turbine_rect)
        
    wake_axes[0].set_title('Undisturbed Flow Contour')
    wake_axes[0].contourf(u, v, np.transpose(np.linalg.norm(undisturbed_flow_contour,2,2)), cmap='bwr')
        
    wake_axes[1].set_title('Disturbed Flow Contour')
    wake_axes[1].contourf(u, v, np.transpose(np.linalg.norm(disturbed_flow_contour,2,2)), cmap='bwr')
    
    x_vec, r_vec = np.meshgrid(x_vec, r_vec, indexing='ij')

    undisturbed_flow_flattened = np.array(undisturbed_flow_contour).flatten()
    u = [i for i in undisturbed_flow_flattened][0:undisturbed_flow_flattened.size:2]
    v = [i for i in undisturbed_flow_flattened][1:undisturbed_flow_flattened.size:2]
    
    flow_fig, flow_axes = plt.subplots(1, 2, constrained_layout=True)
    #flow_fig.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    
    for i in range(2):
        flow_axes[i].set_xlabel('Longitudinal Distance, r (m)')
        flow_axes[i].set_ylabel('Transverse Distance, x (m)')
        flow_axes[i].set_xlim(x_min, x_max)
        flow_axes[i].set_ylim(r_min, r_max)

        for t in turbines:
            turbine_radius = t.get_radius()
            turbine_coords = relative_position(origin_coords, t.get_coords(), flow_field)
            x  = turbine_coords[0]
            if plane == 'xz':
                r = turbine_coords[2]
            elif plane == 'xy':
                r = turbine_coords[1]                
                
            turbine_rect = Rectangle((x, r-turbine_radius), 1, 2 * turbine_radius, hatch = '//')
            flow_axes[i].add_patch(turbine_rect)

    flow_axes[0].set_title('Undisturbed Flow Quiver')
    flow_axes[0].quiver(x_vec, r_vec, u, v, scale=200, alpha = 0.9)

    disturbed_flow_flattened = np.array(disturbed_flow_contour).flatten()
    u = [i for i in disturbed_flow_flattened][0:disturbed_flow_flattened.size:2]
    w = [i for i in disturbed_flow_flattened][1:disturbed_flow_flattened.size:2]

    flow_axes[1].set_title('Disturbed Flow Quiver')
    flow_axes[1].quiver(x_vec, r_vec, u, v, scale=200, alpha = 0.9)
    
    plt.show()

def plot_power_vs_flow_dir(wake_field, wake_combination, turbine_field, flow_field, flow_mag):
    turbines = turbine_field.get_turbines()
    alpha_arr = range(0, 2*pi, 0.1)
    power_arr = []
    for alpha in alpha_arr:
        power = 0
        undisturbed_flow_at_turbine = np.array([np.cos(alpha), np.sin(alpha), 0]) * flow_mag
        for t in turbines:
            turbine_coords = t.get_coords()
            disturbed_flow_mag_at_turbine = wake_combination.get_disturbed_flow_at_point(turbine_coords, True)
            power += t.calc_power_op(disturbed_flow_mag_at_turbine)
        power_arr.append(power)

    power_fig, power_axes = plt.subplots(constrained_layout = True)
    power_axes.set_title('Power Output vs Flow Direction')
    power_axes.set_xlabel('Total Power Output, P (W)')
    power_axes.set_ylabel('XY Flow Direction, alpha (rad)')
    power_axes.plot(alpha_arr, power_arr)
    
