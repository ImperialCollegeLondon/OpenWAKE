import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

def plot_flow_field(flow):
    # Make the grid. coordinates of the arrow locations
    x, y, z = flow.get_x_coords(), flow.get_y_coords(), flow.get_z_coords()
    xMin, xMax, yMin, yMax, zMin, zMax = np.amin(x), np.amax(x), np.amin(y), np.amax(y), np.amin(z), np.amax(z)
    x, y, z = np.meshgrid(x, y, z)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    flow_field = flow.get_flow()

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

def plot_turbine_location(turbine):
    coords = turbine.get_coords()
    x = coords[0]
    y = coords[1]
    locPlt, locAx = plt.subplots()
    locAx.plot(x, y, 'o', label='Turbine Location')
    locPlt.suptitle(r'Turbine Array Grid', fontsize=20)
    locAx.legend(loc='best')
    locAx.set_xlabel('x (m)')
    locAx.set_ylabel('y (m)')
    locAx.set_xlim(0,5)
    locAx.set_ylim(0,5)
    plt.show()

def plot_wake_2d(wakeInst, turbine, flow):
 
    flowArr = flow.get_flow()
    flowArrShape = flowArr.shape
    turbine_radius = turbine.get_radius()

    # for every point in flow, recompute based on velocity reduction factor
    # provide with vector of vel_red_factors

    xCoords = flow.get_x_coords()
    zCoords = flow.get_z_coords()

    # for 2d plot, set y equal to incoming freestream velocity y to convert to 2d
    #y = flowInst.get_flow_at_point(turbineInst.get_coords())[1]
    y = turbine.get_coords()[1]
    # x and z components of quiver coordinates
    x, r = [], []
    undisturbedFlowContour = np.zeros((flowArrShape[0],flowArrShape[1],2))
    disturbedFlowContour = np.zeros((flowArrShape[0],flowArrShape[1],2))
    for i in np.arange(flowArrShape[0]):
        for k in np.arange(flowArrShape[2]):
            rel_pos = wakeInst.relative_position([xCoords[i], y, zCoords[k]])
            # select x, z plane parallel to y coordinate of turbine, then dispense with y-component of flow, assuming that turbine will turn to face it
            undisturbedFlowContour[i][k] = flow.get_flow_at_point([xCoords[i], y, zCoords[k]])[0:3:2]
            disturbedFlowContour[i][k] = flow.get_flow_at_point([xCoords[i], y, zCoords[k]], wakeInst.calc_flow_at_point)[0:3:2]
    
            if i == 0:
                r.append(rel_pos[2])
        x.append(rel_pos[0])

    # x and z components of quiver directions
    # in multi-dimensional flow array, first dim = x, second dim = y, third dim = z
    # but in meshgrid, these axis are swapped visually (different 'rows' correspond to r axis)
    # therefore, flow contours are transposed for contour plots
    # and meshgrids are transposed for quiver plots
    u, w = np.meshgrid(x, r)
    xMin = np.amin(x); xMax = np.amax(x)
    rMin = np.amin(r); rMax = np.amax(r)
    
    undisturbedWakePlot, undisturbedWakeAx = plt.subplots()
    turbineRect = Rectangle((0, -turbine_radius), 1, 2 * turbine_radius, hatch = '//')
    undisturbedWakeAx.add_patch(turbineRect)
    undisturbedWakePlot.suptitle(r'Wake at Turbine', fontsize=20)
    #wakeAx.legend(loc='best')
    undisturbedWakeAx.set_xlabel('Longitudinal Distance, r (m)')
    undisturbedWakeAx.set_ylabel('Transverse Distance, x (m)')
    undisturbedWakeAx.set_xlim(xMin, xMax)
    undisturbedWakeAx.set_ylim(rMin, rMax)
    undisturbedWakeAx.contourf(u, w, np.transpose(np.linalg.norm(undisturbedFlowContour,2,2)), cmap='bwr')
    plt.show()
    
    disturbedWakePlot, disturbedWakeAx = plt.subplots()
    turbineRect = Rectangle((0, -turbine_radius), 1, 2 * turbine_radius, hatch = '//')
    disturbedWakeAx.add_patch(turbineRect)
    disturbedWakePlot.suptitle(r'Wake at Turbine', fontsize=20)
    #wakeAx.legend(loc='best')
    disturbedWakeAx.set_xlabel('Longitudinal Distance, r (m)')
    disturbedWakeAx.set_ylabel('Transverse Distance, x (m)')
    disturbedWakeAx.set_xlim(xMin, xMax)
    disturbedWakeAx.set_ylim(rMin, rMax)
    disturbedWakeAx.contourf(u, w, np.transpose(np.linalg.norm(disturbedFlowContour,2,2)), cmap='bwr')
    plt.show()

    x, r = np.meshgrid(x, r)

    undisturbedFlowFlattened = np.array(undisturbedFlowContour).flatten()
    u = [i for i in undisturbedFlowFlattened][0:undisturbedFlowFlattened.size:2]
    w = [i for i in undisturbedFlowFlattened][1:undisturbedFlowFlattened.size:2]
    
    undisturbedFlowPlt, undisturbedFlowAx = plt.subplots()
    undisturbedFlowAx.quiver(r, x, u, w, scale=90, alpha = 0.9)
    turbineRect = Rectangle((0, -turbine_radius), 1, 2 * turbine_radius, hatch = '//')
    undisturbedFlowAx.add_patch(turbineRect)
    undisturbedFlowAx.set_xlabel('Longitudinal Distance, r (m)')
    undisturbedFlowAx.set_ylabel('Transverse Distance, x (m)')
    undisturbedFlowAx.set_xlim(xMin, xMax)
    undisturbedFlowAx.set_ylim(rMin, rMax)
    plt.show()

    disturbedFlowFlattened = np.array(disturbedFlowContour).flatten()
    u = [i for i in disturbedFlowFlattened][0:disturbedFlowFlattened.size:2]
    w = [i for i in disturbedFlowFlattened][1:disturbedFlowFlattened.size:2]
    
    disturbedFlowPlt, disturbedFlowAx = plt.subplots()
    disturbedFlowAx.quiver(r, x, u, w, scale=90, alpha = 0.9)
    turbineRect = Rectangle((0, -turbine_radius), 1, 2 * turbine_radius, hatch = '//')
    disturbedFlowAx.add_patch(turbineRect)
    disturbedFlowAx.set_xlabel('Longitudinal Distance, r (m)')
    disturbedFlowAx.set_ylabel('Transverse Distance, x (m)')
    disturbedFlowAx.set_xlim(xMin, xMax)
    disturbedFlowAx.set_ylim(rMin, rMax)
    plt.show()
