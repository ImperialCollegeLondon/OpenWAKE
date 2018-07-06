__author__="Georgios Deskos <g.deskos14@imperial.ac.uk"
__date__="2018-06-23"

import os
import sys
import numpy as np
import matplotlib as mp
version(mp)
exit()
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pan

# Load windIO
WINDIO = True
try:
    from windIO.Plant import WTLayout
except Exception as e:
    WINDIO = False
    print("WARNING: WINDIO isn't installed correctly:", e)
