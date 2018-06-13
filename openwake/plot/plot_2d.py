import numpy as np
import matplotlip.pyplot as plt

def plot2d(x, r, wake):
    fig = plt.figure(figsize=(10, 10))
    plt.contourf(x, r, wake, 100)
    plt.colorbar()
    plt.xlim(x.min(), x.max())
    plt.ylim(r.min(), r.max())
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
