import numpy as np

class WakeField(object):
    def __init__(self, wake_list = []):
        self.wakes = []
        self.add_wakes(wake_list)

    def add_wakes(self, wake_list = []):
        self.wakes = self.wakes + [w for w in wake_list]

    def get_wakes(self):
        return np.array(self.wakes)

    def get_num_wakes(self):
        return self.get_wakes().size
