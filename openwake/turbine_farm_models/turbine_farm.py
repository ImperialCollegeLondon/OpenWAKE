from windIO.Plant import WTLayout
import numpy as np
import plotly as pl
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode() # run at the start of every ipython notebook to use plotly.offline
                     # this injects the plotly.js source files into the notebook
print(pl.__version__) # requires version >= 1.9.0

farm_name = "Middelgrunden"
file_name = 'middelgrunden.yml'


import os
import sys

src_dir = '/'.join([os.getcwd().split('/tests')[0],'openwake'])
if src_dir not in sys.path:
    sys.path.append(src_dir)
    
farm_dir = '/'.join([src_dir,'turbine_farm_models'])

wtl = WTLayout('/'.join([farm_dir,file_name]))

wtl.plot_location((32,'U'),layout={'title': farm_name+' location'})

wtl.plot_layout(layout={'title': farm_name+' layout'})

turbine = wtl['turbine_types'][0]

ct = np.array(turbine['c_t_curve'])

iplot(Figure(
    data=[{'x':ct[:,0], 'y':ct[:,1]}],
    layout={
        'xaxis':{'title':"Wind Speed [m/s]"},
        'yaxis':{'title':'$C_T$'},
        'title': '$C_T$ curve of the %s'%(turbine['name']),            
        }
     ))

power = np.array(turbine['power_curve'])
iplot(Figure(
    data=[{ 'x':power[:,0], 'y':power[:,1]}],
    layout={
        'xaxis':{'title':"Wind Speed [m/s]"},
        'yaxis':{'title':'Power [kW]'},
        'title': 'Power Curve of the %s'%(turbine['name']),
        }
     ))
