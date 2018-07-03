import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from helpers import *
from flow_field_model.flow import FlowField

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
    undisturbed_flow_grid = flow_field.get_flow()
    disturbed_flow_grid = wake_combination.get_disturbed_flow_grid(flow_field, wake_field, True)
    
    undisturbed_flow_flatten = undisturbed_flow_grid.flatten()
    disturbed_flow_flatten = disturbed_flow_grid.flatten()
    flow_size = undisturbed_flow_flatten.size
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    len_x, len_y, len_z = undisturbed_flow_grid.shape[0:3]

    wakes = wake_field.get_wakes()
    turbines = turbine_field.get_turbines()
    turbines_coords = turbine_field.get_coords()
    num_turbines = turbine_field.get_num_turbines()
    
    if plane == 'xz':
        # select x, z plane parallel to y coordinate of turbine, then dispense with y-component of flow, assuming that turbine will turn to face it
        # layers of 2d plot
        start, end, increment = 0, 3, 2
        l_vec, r_vec = y_coords, z_coords
        l_index, r_index = 1, 2
        
    else:
        # select x, y plane parallel to z coordinate of turbine, then dispense with z-component of flow, assuming that turbine will turn to face it                   
        # layers of 2d plot
        start, end, increment = 0, 2, 1
        l_vec, r_vec = z_coords, y_coords
        l_index, r_index = 2, 1
        
    len_l, len_r = len(l_vec), len(r_vec)
    dl = abs(l_vec[1] - l_vec[0])

    x_vec = x_coords

    # for an xz plane, select only the x and z flow components. for an xy plane, select only the x and y flow components.
    # included are all undisturbed flows in the plane of choice at each turbine coordinate not in the plane of choice
    # Reshape in a multi-dimensional array of shape x, r (be it z or y) and the number of layers (each corresponding to the y or z coordinates of the turbines)

##    undisturbed_flow_contour = np.array([\
##                                         undisturbed_flow_flatten[s : s + end : increment] \
##                                         for s in range(start, flow_size, 3)\
##                                         if any(np.isin(turbines_coords[l_index], l_vec[int((s - start) / 3) % len_l]))\
##                                         ]).flatten().reshape((len_x, len_r, num_turbines, 2))
##    print(undisturbed_flow_contour)
##
####    disturbed_flow_contour = np.array([\
####                                         disturbed_flow_flatten[s : s + end : increment] \
####                                         for s in range(start, flow_size, 3)\
####                                         if any(np.isin(turbines_coords[l_index], l_vec[int((s - start) / 3) % len_l]))\
####                                         ]).flatten().reshape((len_x, len_r, num_turbines, 2))
##    disturbed_flow_contour = np.array(undisturbed_flow_contour)
    
    for s in range(start, flow_size, 3):
        # (3 * len_z * len_y) is the number of flattened elements in each x row
        # (3 * len_z) is the number of flattened elements in each y column of each x row
        # (3) is the number of flattened elements in each z column of each y column of each x row
        x_contour_index = int((s - start) / (3 * len_z * len_y))
        y_contour_index = int(((s - start) % (3 * len_z * len_y)) / (3 * len_z))
        z_contour_index = int(((s - start) % (3 * len_y)) / 3)
        
        l_contour_index = np.argwhere(turbines_coords[l_index] == l_vec[y_contour_index]) if plane == 'xz' else \
                          np.argwhere(turbines_coords[l_index] == l_vec[z_contour_index])
        
        r_contour_index = z_contour_index if plane == 'xz' else y_contour_index

        if len(l_contour_index) != 0:
            undisturbed_flow_contour[x_contour_index, r_contour_index, l_contour_index[0]] = \
                                                      undisturbed_flow_flatten[s : s + end : increment]
            disturbed_flow_contour[x_contour_index, r_contour_index, l_contour_index[0]] = \
                                                    disturbed_flow_flatten[s : s + end : increment]
    

    # set turbine coords of minimum x coordinate as origin
    origin_coords = np.array([0,0,0])
    
##    for i in range(len_x):
##        x = x_coords[i]
##        for j in range(len_r):
##            # if xz plane, r is current z value, else r is current y value
##            if plane == 'xz':
##                z = z_coords[j]
##            else:
##                y = y_coords[j]
##                
##            for k in range(num_turbines):
##                # if xz plane, layer l is current turbine y value, else l is current turbine z value
##                if plane == 'xz':
##                    y = turbines[k].get_coords()[l_index]
##                else:
##                    z = turbines[k].get_coords()[l_index]
##                
##                #undisturbed_flow_contour[i, j, k] = flow_field.get_undisturbed_flow_at_point([x, y, z], False)[start : end : increment]
##                
##                # TODO same irrespective of any single wake or combination of wakes
##                # combination unchanging for different values of relative z at each turbine, or negative values of relative z
##                disturbed_flow_contour[i, j, k] = wake_combination.get_disturbed_flow_at_point([x, y, z], flow_field, wake_field, False, True)[start : end : increment]

                #print([i,j,k], [x, y, z], disturbed_flow_contour[i, j, k, 0])

##                # append y or z coordinate to r_vec, but only once
##                if i == 0 and j == 0:
##                    rel_pos = relative_position(origin_coords, np.array([x, y, z]), flow_field)
##                    if plane == 'xz':
##                        np.append(r_vec, rel_pos[2])
##                    elif plane == 'xy':
##                        np.append(r_vec, rel_pos[1])
    
##        # append x coordinate to x_vec, but only once
##        np.append(x_vec, rel_pos[0])

    ## PLOT UNDISTURBED AND DISTURBED FLOW CONTOURS
    # u and v are x and y or z components of quiver coordinates.
    x_vec, r_vec = np.meshgrid(x_vec, r_vec, indexing='ij')
    x_min = np.amin(x_vec); x_max = np.amax(x_vec)
    r_min = np.amin(r_vec); r_max = np.amax(r_vec)

    wake_fig, wake_axes = plt.subplots(1, 2, constrained_layout = True)
    flow_contours = [undisturbed_flow_contour, disturbed_flow_contour]
    num_flows = len(flow_contours)

    for i in range(num_flows):
        wake_axes[i].set_xlabel('Longitudinal Distance, r (m)')
        wake_axes[i].set_ylabel('Transverse Distance, x (m)')
        wake_axes[i].set_xlim(x_min, x_max)
        wake_axes[i].set_ylim(r_min, r_max)

        wake_axes[i].set_title('Undisturbed Flow Contour') if i == 0 else wake_axes[1].set_title('Disturbed Flow Contour')

        for t in range(num_turbines):
            turbine_radius = turbines[t].get_radius()
            turbine_coords = turbines[t].get_coords()
            rel_index = relative_index(origin_coords, turbine_coords, flow_field)
            x, r  = rel_index[0], rel_index[r_index]
            turbine_rect = Rectangle((x, r - turbine_radius), 1, 2 * turbine_radius, hatch = '//')
            wake_axes[i].add_patch(turbine_rect)
            wake_axes[i].contourf(x_vec, r_vec, np.linalg.norm(flow_contours[i][:,:,t], 2, 2), cmap='bwr')           
            #wake_axes[i].contourf(u, v, np.transpose(np.linalg.norm(flow_contours[i][:,:,t], 2, 2)), cmap='bwr')

    undisturbed_flow_flattened = np.array(undisturbed_flow_contour).flatten()
    disturbed_flow_flattened = np.array(disturbed_flow_contour).flatten()
    flow_contours_flattened = [undisturbed_flow_flattened, disturbed_flow_flattened]
    
    flow_fig, flow_axes = plt.subplots(1, 2, constrained_layout=True)
    
    for i in range(num_flows):
        flow_axes[i].set_xlabel('Longitudinal Distance, r (m)')
        flow_axes[i].set_ylabel('Transverse Distance, x (m)')
        flow_axes[i].set_xlim(x_min, x_max)
        flow_axes[i].set_ylim(r_min, r_max)

        flow_axes[i].set_title('Undisturbed Flow Quiver') if i == 0 else flow_axes[1].set_title('Disturbed Flow Quiver')

        u = [[f for f in flow_contours_flattened[i]][0:flow_size:2]]
        v = [[f for f in flow_contours_flattened[i]][1:flow_size:2]]
        
        flow_axes[i].quiver(x_vec, r_vec, u, v, scale = 200, alpha = 0.9)
        
        for t in turbines:
            turbine_radius = t.get_radius()
            turbine_coords = relative_position(origin_coords, t.get_coords(), flow_field)
            rel_index = relative_index(origin_coords, turbine_coords, flow_field)
            x, r  = rel_index[0], rel_index[r_index]
            turbine_rect = Rectangle((x, r - turbine_radius), 1, 2 * turbine_radius, hatch = '//')
            flow_axes[i].add_patch(turbine_rect)

    plt.show()

def plot_power_vs_flow_dir(wake_field, wake_combination, turbine_field, flow_field):
    turbines = turbine_field.get_turbines()
    num_turbines = turbine_field.get_num_turbines()
    alpha_arr = np.linspace(0, 2 * np.pi, 50)
    power_arr = []
    flow_mag_grid = np.linalg.norm(flow_field.get_flow(), 2, 2)
    flow_mag_mean = np.mean(flow_mag_grid)
    flow_shape = flow_mag_grid.shape
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    x_turbine_coords, y_turbine_coords, z_turbine_coords = [t.get_coords()[0] for t in turbines], [t.get_coords()[1] for t in turbines], [t.get_coords()[2] for t in turbines]
    origin_coords = np.array([0, 0, 0])
    
    for alpha in alpha_arr:
        power = 0
        
        flow_alpha = np.zeros(flow_shape).tolist()

        for x in range(x_turbine_coords):
            for y in range(y_turbine_coords):
                for z in range(z_turbine_coords):
                    i, j, k = relative_index(origin_coords, [x, y, z], flow_field)
                    flow_alpha[i, j, k] = [np.cos(alpha), np.sin(alpha), 0] * flow_mag_mean

        flow_field_alpha = FlowField(x_coords, y_coords, z_coords, flow_alpha)
        for t in turbines:
            turbine_coords = t.get_coords()
            wake_combination.is_grid_outdated = True
            disturbed_flow_mag_at_turbine = wake_combination.get_disturbed_flow_at_point(turbine_coords, flow_field_alpha, wake_field, True, False)
            power += t.calc_power_op(disturbed_flow_mag_at_turbine)
        power_arr.append(power)

    power_fig, power_axes = plt.subplots(constrained_layout = True)
    power_axes.set_title('Power Output vs Flow Direction')
    power_axes.set_xlabel('Total Power Output, P (W)')
    power_axes.set_ylabel('XY Flow Direction, alpha (rad)')
    power_axes.plot(alpha_arr, power_arr)

def plot_wakes_vs_flow_dir(wake_field, wake_combination, turbine_field, flow_field):
    turbines = turbine_field.get_turbines()
    num_turbines = turbine_field.get_num_turbines()
    alpha_arr = np.linspace(0, 2 * np.pi, 50)
    power_arr = []
    flow_mag_grid = np.linalg.norm(flow_field.get_flow(), 2, 2)
    flow_mag_mean = np.mean(flow_mag_grid)
    flow_shape = flow_mag_grid.shape
    x_coords, y_coords, z_coords = flow_field.get_x_coords(), flow_field.get_y_coords(), flow_field.get_z_coords()
    x_turbine_coords, y_turbine_coords, z_turbine_coords = [t.get_coords()[0] for t in turbines], [t.get_coords()[1] for t in turbines], [t.get_coords()[2] for t in turbines]
    origin_coords = np.array([0, 0, 0])
    
    for alpha in alpha_arr:
        power = 0
        
        flow_alpha = np.zeros(flow_shape).tolist()

        for x in range(x_turbine_coords):
            for y in range(y_turbine_coords):
                for z in range(z_turbine_coords):
                    i, j, k = relative_index(origin_coords, [x, y, z], flow_field)
                    flow_alpha[i, j, k] = [np.cos(alpha), np.sin(alpha), 0] * flow_mag_mean

        flow_field_alpha = FlowField(x_coords, y_coords, z_coords, flow_alpha)

        plot_wakes(wake_field, wake_combination, turbine_field, flow_field, plane='xz')
    
