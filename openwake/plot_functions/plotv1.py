import numpy as np
import matplotlip.pyplot as plt
"""
from mpl_toolkits.mplot3d import Axes3D
from IPython import embed
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
"""

def plot_2d_wake():
    wakePlot, wakeAx = plt.subplots()
    wakePlot.suptitle(r'Wake at Turbine', fontsize=20)
    wakeAx.legend(loc='best')
    wakeAx.set_xlabel('Longitudinal Distance (m)')
    wakeAx.set_ylabel('Transverse Distance (m)')
    xRange = [-20,200]
    yRange = [-40,40]
    wakeAx.set_xlim(xRange)
    wakeAx.set_ylim(yRange)
    xDelta = (xRange[1]-xRange[0])/flowArrShape[0]
    yDelta = (yRange[1]-yRange[0])/flowArrShape[1]
    x = np.arange(xRange[0],xRange[1],xDelta)
    y = np.arange(yRange[0],yRange[1],yDelta)

    turbineRect = Rectangle((0, -baseTurbineInst.get_radius()), 5, 2 * baseTurbineInst.get_radius(), hatch = '//')
    wakeAx.add_patch(turbineRect)
    u, v = np.meshgrid(x, y)

    wakeAx.contour(x, y, z)
    plt.show()

def plot_contour(x, r, wake):
    fig = plt.figure(figsize=(10, 10))
    plt.contourf(x, r, wake, 100)
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(r.min(), r.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

class SuperImpose(object):
    """ given a meshgrid, ambient velocity and turbine location, calculates the 
    wake and superimposes it onto the ambient velocity - for visualisation 
    """
    def __init__(self, mesh_x, mesh_r, ambient_velocity, turbine_position, turbine_diameter):
        
        def find_index(array, value):
            # Find index of nearest value in an array
            return (np.abs(array-value)).argmin()
        
        dx = abs(mesh_x[0,0] - mesh_x[0,1])
        dr = abs(mesh_r[0,0] - mesh_r[1,0])

        parameters = Larsen.default_parameters()
        parameters.turbine_radius = turbine_diameter/2.
#        lwm = Larsen(parameters, ambient_velocity, dx, dr)
        #TODO TODO TODO this turbine position is incorrect!!!!!!!!!!!!!
        tix = 0
        tir = len(mesh_x)/2
        flow = ambient_velocity[tir,tix]
        reduction = np.ones(ambient_velocity.shape) * flow #TODO take resultant
        lwm = Larsen(parameters, reduction, dx, dr)      
        
        #y_min = find_index(mesh_x[0,:], turbine_position[0])
        #x_min = find_index(mesh_x[0], turbine_position[1]-3*turbine_diameter)
        #x_max = find_index(mesh_x[0], turbine_position[1]+3*turbine_diameter)
        
        
        from copy import deepcopy
        ambient = deepcopy(ambient_velocity)
#        turbine_position = (0, max(mesh_r[:,0])/2)
        for i in range(len(mesh_x)):
            for j in range(len(mesh_x[0])):
#                ambient_velocity[i][j] *= (lwm.individual_factor(turbine_position, 
#                                                                 np.array([mesh_x[i][j], 
#                                                                           mesh_r[i][j]])))
                reduction[i,j] = (lwm.individual_factor(turbine_position,
                                                   np.array([mesh_x[i][j], 
                                                             mesh_r[i][j]])))

        # TODO fix this tidying step to get rid of 'halo' 
        x_cut = find_index(mesh_x[0], 10)
        for i in range(0, x_cut):
            for j in range(len(mesh_x)):
                if j < find_index(mesh_r[:,0],
                        turbine_position[1]-turbine_diameter*0.75) or \
                                j > find_index(mesh_r[:,0],
                                        turbine_position[1]+turbine_diameter*0.75):
                    #ambient_velocity[j,i] = ambient[j,i]
                    reduction[j,i] = 0

        embed()
        #plot(mesh_x, mesh_r, ambient_velocity)
        self.wake = ambient_velocity

if __name__ == '__main__':
    
    turbine_diameter = 20.
    size_x = 500.
    size_r = turbine_diameter * 2    
    dx = 0.5
    dr = 0.5  
    range_x = np.linspace(0, size_x, size_x/dx)
    range_r = np.linspace(0, size_r, size_r/dr) 
    mesh_x, mesh_r = np.meshgrid(range_x, range_r)      
    wake = np.ones(mesh_x.shape)
    turbine_position = np.array([10, size_r/2.])
    
    test = SuperImpose(mesh_x, mesh_r, wake, turbine_position, turbine_diameter)    
    
    
#    parameters = Larsen.default_parameters()
#    parameters.turbine_radius = turbine_diameter/2.
#    lwm = Larsen(parameters, wake, dx, dr)
#    
#    for j in range(int(turbine_position[0]), len(mesh_x[0,:]-int(turbine_position[0]))):
#        for i in range(len(mesh_x[:,0])):
#            if turbine_position[1]-7*turbine_diameter < i < turbine_position[1]+7*turbine_diameter:       
#                wake[i][j] = lwm.individual_factor(turbine_position, np.array([mesh_x[i][j], mesh_r[i][j]]))
#            
#    plot(mesh_x, mesh_r, wake)
