import numpy as np

class TurbineField(object):
    def __init__(self, turbine_list = []):
        self.turbines = []
        self.coords = [[],[],[]]
        self.add_turbines(turbine_list)

    def add_turbines(self, turbine_list = []):
        self.turbines = self.turbines + [t for t in turbine_list]
        for i in range(3):
            self.coords[i] = self.coords[i] + [t.get_coords()[i] for t in turbine_list]

    def get_turbines(self):
        return np.array(self.turbines)

    def get_num_turbines(self):
        return self.get_turbines().size

    def get_coords(self):
        return self.coords
