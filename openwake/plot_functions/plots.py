import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_wake_2d(wakeInst, turbineInst, flowInst):
    wakePlot, wakeAx = plt.subplots()
    wakePlot.suptitle(r'Wake at Turbine', fontsize=20)
    #wakeAx.legend(loc='best')
    wakeAx.set_xlabel('Longitudinal Distance, r (m)')
    wakeAx.set_ylabel('Transverse Distance, x (m)')
    flowArr = flowInst.get_flow()
    flowArrShape = flowArr.shape
    #xDelta = (xRange[1]-xRange[0])/flowArrShape[0]
    #yDelta = (yRange[1]-yRange[0])/flowArrShape[1]
    #x = np.arange(xRange[0],xRange[1],xDelta)
    #y = np.arange(yRange[0],yRange[1],yDelta)

    turbineRect = Rectangle((0, -turbineInst.get_radius()), 1, 2 * turbineInst.get_radius(), hatch = '//')
    wakeAx.add_patch(turbineRect)

    # for every point in flow, recompute based on velocity reduction factor
    # provide with vector of vel_red_factors

    xCoords = flowInst.get_x_coords()
    yCoords = flowInst.get_y_coords()
    zCoords = flowInst.get_z_coords()

    # for 2d plot, set y equal to incoming freestream velocity y
    #y = flowInst.get_flow_at_point(turbineInst.get_coords())[1]
    y = turbineInst.get_coords()[1]
    
    x, r = [], []
    flowContour = np.zeros((flowArrShape[0],flowArrShape[1]))
    for i in np.arange(flowArrShape[0]):
        for k in np.arange(flowArrShape[2]):
            rel_pos = wakeInst.relative_position([xCoords[i], y, zCoords[k]])
            flowContour[i][k] = flowInst.get_flow_at_point([xCoords[i], y, zCoords[k]], wakeInst.calc_vrf_at_point)
            if i == 0:
                r.append(rel_pos[2])
        x.append(rel_pos[0])

    u, v = np.meshgrid(x, r)
    xRange = max([abs(i) for i in x])
    rRange = max([abs(i) for i in r])
    
    wakeAx.set_xlim(-xRange, xRange)
    wakeAx.set_ylim(-rRange, rRange)
    
    #flowContour = np.linalg.norm(flowContour,2)
    wakeAx.contour(u, v, flowContour)
    plt.show()

    flowPlt, flowAx = plt.subplots()
    flowAx.quiver(x, r, u, v, flowContour, scale=90, alpha = 0.9)
    #flowAx.quiver(x, r, u, v, np.linalg.norm(flowArr,2), scale=90, alpha = 0.9)
    turbineRect = Rectangle((0, -turbineInst.get_radius()), 1, 2 * turbineInst.get_radius(), hatch = '//')
    flowAx.add_patch(turbineRect)
    flowAx.set_xlim(-xRange, xRange)
    flowAx.set_ylim(-rRange, rRange)
    plt.show()
