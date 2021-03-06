import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from helpers import *
from flow_field_model.flow import FlowField

def plot_flow_field(flow_field):
    # Make the grid. coordinates of the arrow locations
    x_coords, y_coords, z_coords = flow_field.get_coords()
    x_min, x_max, y_min, y_max, z_min, z_max = np.amin(x_coords), np.amax(x_coords), np.amin(y_coords), np.amax(y_coords), np.amin(z_coords), np.amax(z_coords)
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords)
    x_flatten, y_flatten, z_flatten = x_grid.flatten(), y_grid.flatten(), z_grid.flatten()
    flow = flow_field.get_flow()

    # Make the direction data for the arrows. x,y,z components of the arrow vectors
    flowFlatten = np.array(flow_field).flatten()
    flowLength = flowFlatten.size
    u, v, w = flowFlatten[0:flowLength:3], flowFlatten[1:flowLength:3], flowFlatten[2:flowLength:3]
   
    flowFig = plt.figure()
    flowAx = flowFig.gca(projection='3d')
    flowAx.quiver(x_flatten, y_flatten, z_flatten, u, v, w, label='Flow Vector', length=0.3)
    flowFig.suptitle(r'Flow Field', fontsize=20)
    flowAx.legend(loc='best')
    flowAx.set_xlabel('x (m)')
    flowAx.set_ylabel('y (m)')
    flowAx.set_zlabel('z (m)')
    flowAx.set_xlim3d(x_min, x_max)
    flowAx.set_ylim3d(y_min, y_max)
    flowAx.set_zlim3d(z_min, z_max)

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

def make_wake_figs(x_vec, r_vec, r_index, turbines, flow_field):

    wake_fig, wake_axes = plt.subplots(1, 2, constrained_layout = True)
    num_flows = 2
    origin_coords = np.array([0, 0, 0])
    num_turbines = len( turbines )
    
    for i in range( num_flows ):
        x_min = np.amin(x_vec); x_max = np.amax(x_vec)
        r_min = np.amin(r_vec); r_max = np.amax(r_vec)
        wake_axes[i].set_xlabel('Axial Distance, x (m)')
        wake_axes[i].set_ylabel('Radial Distance, r (m)')
        wake_axes[i].set_xlim(x_min, x_max)
        wake_axes[i].set_ylim(r_min, r_max)
        wake_axes[i].set_title('Undisturbed Flow Contour') if i == 0 else wake_axes[1].set_title('Disturbed Flow Contour')

        for t in range(num_turbines):
            turbine_radius = turbines[t].get_radius()
            turbine_coords = turbines[t].get_coords()
            turbine_direction = turbines[t].get_direction()
            turbine_angle = (180 / np.pi) * np.arctan( turbine_direction[r_index] / turbine_direction[0] )
            x, r = turbine_coords[0], turbine_coords[r_index]
            turbine_rect = Rectangle((x, r - turbine_radius), 1, 2 * turbine_radius, angle = turbine_angle, hatch = '//')
            wake_axes[i].add_patch(turbine_rect)

    return wake_fig, wake_axes

def plot_wakes( wake_axes, flow_contours, x_grid, r_grid, len_l ):
    
    num_flows = len(flow_contours)
    
    for i in range(num_flows):
        
        #step = 0.2
        for l in range( len_l ):

            f_grid = np.linalg.norm( flow_contours[i][:,:,l], 2, 2 )
            levels = np.arange( 0.0, max( f_grid.flatten() ) )
            contour = wake_axes[i].contourf( x_grid, r_grid, f_grid, levels, cmap='bwr', alpha = 0.2 )

    plt.show()
    
def make_wake_contours(wake_field, wake_combination, turbine_field, flow_field, plane='xz', wake_fig = None, wake_axes = None, plot = True):
    """
    param plane which plane to view wakes in. 'xz' produces a side-view, 'xy' produces a birdseye view
    """
    undisturbed_flow_grid = flow_field.get_flow()

    wakes = wake_field.get_wakes()
    turbines = turbine_field.get_turbines()
    turbines_coords = turbine_field.get_coords()
    num_turbines = turbine_field.get_num_turbines()
    
    disturbed_flow_grid = wake_combination.get_disturbed_flow_grid( flow_field, wake_field )
    
    undisturbed_flow_flatten = undisturbed_flow_grid.flatten()
    disturbed_flow_flatten = disturbed_flow_grid.flatten()
    flow_size = undisturbed_flow_flatten.size
    x_coords, y_coords, z_coords = flow_field.get_coords()
    len_x, len_y, len_z = undisturbed_flow_grid.shape[0:3]
    
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
    turbine_levels = np.unique( turbines_coords[ l_index ] )
    len_l = turbine_levels.size

    x_vec = x_coords

    # for an xz plane, select only the x and z flow components. for an xy plane, select only the x and y flow components.
    # included are all undisturbed flows in the plane of choice at each turbine coordinate not in the plane of choice
    # Reshape in a multi-dimensional array of shape x, r (be it z or y) and the number of layers (each corresponding to the y or z coordinates of the turbines)
    
    undisturbed_flow_contour, disturbed_flow_contour = np.zeros((len_x, len_r, len_l, 2)), np.zeros((len_x, len_r, len_l, 2))
    # u_undisturbed, u_disturbed, v_undisturbed, v_disturbed = np.array([]), np.array([]), np.array([]), np.array([])
    for s in range( 0, len_x * len_y * len_z * 3, 3 ):
        # (3 * len_z * len_y) is the number of flattened elements in each x row
        # (3 * len_z) is the number of flattened elements in each y column of each x row
        # (3) is the number of flattened elements in each z column of each y column of each x row
        x_contour_index = int( s / ( len_z * len_y * 3 ) )
        y_contour_index = int( ( s % ( len_z * len_y * 3 ) ) / ( len_z * 3 )  )
        z_contour_index = int( ( s % ( len_z * 3 ) ) / 3 )
        
        if plane == 'xz':
            r_contour_index = z_contour_index
            l_contour_index = y_contour_index
        else:
            r_contour_index = y_contour_index
            l_contour_index = z_contour_index

        # if the current l_contour_index corresponds to a level at which one or more turbines are located, set flow_contour points
        # (representing level number, where each level corresponds to a cross-section at
        # which one or more turbines are located)
        if any( l_vec[ l_contour_index ] == turbine_levels ):
            l_contour_index = np.argwhere( l_vec[ l_contour_index ] == turbine_levels )
            undisturbed_flow_contour[ x_contour_index, r_contour_index, l_contour_index ] = \
                                                      undisturbed_flow_flatten[s : s + end : increment]
            
            disturbed_flow_contour[ x_contour_index, r_contour_index, l_contour_index ] = \
                                                    disturbed_flow_flatten[s : s + end : increment]

    # set turbine coords of minimum x coordinate as origin
    origin_coords = np.array([0,0,0])
    
    ## PLOT UNDISTURBED AND DISTURBED FLOW CONTOURS
    # u and v are x and y or z components of quiver coordinates.
    x_grid, r_grid = np.meshgrid( x_vec, r_vec, indexing='ij' )
    flow_contours = [ undisturbed_flow_contour, disturbed_flow_contour ]
    print(disturbed_flow_contour[30, 25, 0])
    if plot == True:
        if wake_fig == None or wake_axes == None:
            wake_fig, wake_axes = make_wake_figs( x_vec, r_vec, r_index, turbines, flow_field )
        plot_wakes( wake_axes, flow_contours, x_grid, r_grid, len_l )

    return flow_contours

def plot_power_vs_flow_dir( wake_field, wake_combination, turbine_field, flow_field ):
    
    wakes = wake_field.get_wakes()
    turbines = turbine_field.get_turbines()
    num_frames = 15
    alpha_arr = np.linspace(0, 2 * np.pi, num_frames)
    power_arr = []
    flow = flow_field.get_flow()
    flow_mag_grid = np.linalg.norm(flow, 2, 3)
    flow_mag_mean = np.mean(flow_mag_grid)
    flow_shape = flow.shape
    x_coords, y_coords, z_coords = flow_field.get_coords()
    turbines_coords = turbine_field.get_coords().flatten()
    
    for alpha in alpha_arr:
        power = 0
        
        flow_alpha = np.ones( flow_shape )
        flow_alpha[ :, :, : ] = np.cos( alpha ) * flow_mag_mean, np.sin( alpha ) * flow_mag_mean, 0
        flow_field_alpha = FlowField( x_coords, y_coords, z_coords, flow_alpha )
        wake_combination.is_grid_outdated = True
        for w in wakes:
            w.is_grid_outdated = True
        
        for t in turbines:
            turbine_coords = t.get_coords()
            t.set_direction( [ np.cos( alpha ), np.sin( alpha ), 0 ] )
            disturbed_flow_mag_at_turbine = wake_combination.get_disturbed_flow_at_point( turbine_coords, flow_field_alpha, wake_field, True, False )
            power += t.calc_power_op( disturbed_flow_mag_at_turbine )
        power_arr.append(power)

    power_fig, power_axes = plt.subplots( constrained_layout = True )
    power_axes.set_title( 'Power Output vs Flow Direction' )
    power_axes.set_xlabel( 'XY Flow Direction, alpha (rad)' )
    power_axes.set_ylabel( 'Total Power Output, P (W)' )
    power_axes.plot( alpha_arr, power_arr )
    plt.show()

def plot_wakes_vs_flow_dir( wake_field, wake_combination, turbine_field, flow_field ):
    
    wakes = wake_field.get_wakes()
    turbines = turbine_field.get_turbines()
    num_frames = 15
    alpha_arr = np.linspace( 0, 2 * np.pi, num_frames )
    power_arr = []
    flow = flow_field.get_flow()
    flow_mag_grid = np.linalg.norm( flow, 2, 3 )
    flow_mag_mean = np.mean( flow_mag_grid )
    flow_shape = flow.shape
    x_coords, y_coords, z_coords = flow_field.get_coords()
    
    turbines_coords = turbine_field.get_coords()
    x_vec, r_vec, r_index, l_index = x_coords, y_coords, 1, 2
    turbine_levels = np.unique( turbines_coords[ l_index ] )
    len_l = turbine_levels.size
    
    step = 0.2
    levels = np.arange( flow_mag_mean )
    x_grid, r_grid = np.meshgrid( x_vec, r_vec, indexing='ij' )

    contour_alpha = []
    
    for alpha in alpha_arr:
        power = 0
        
        flow_alpha = np.ones(flow_shape)
        flow_alpha[:,:,:] = np.cos(alpha) * flow_mag_mean, np.sin(alpha) * flow_mag_mean, 0

        flow_field_alpha = FlowField( x_coords, y_coords, z_coords, flow_alpha )
        
        wake_combination.is_grid_outdated = True
        for w in wakes:
            w.is_grid_outdated = True

        wake_fig, wake_axes = make_wake_figs( x_vec, r_vec, r_index, turbines, flow_field_alpha )
        wake_axes = wake_axes[ 1 ]
        
        contour_alpha = make_wake_contours( wake_field, wake_combination, turbine_field, flow_field_alpha, 'xy', wake_fig, wake_axes, False )[ 1 ]

        for l in range( len_l ):
            f_grid = np.linalg.norm( contour_alpha[ :, :, l ], 2, 2 )
            contour = wake_axes.contourf( x_grid, r_grid, f_grid, levels, cmap='bwr', alpha = 0.2 )
            
        wake_axes.set_title( "alpha = %f" % alpha ) 

    plt.show()
