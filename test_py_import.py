
from __future__ import print_function
# line above is for python2 retro-compatibility

# the following modules should be standard in a python installation
# allows you to interact with your operating system
import os, sys
from time import time

# print out the used python version
v1,v2,v3 = sys.version_info[0:3]
print("code       | version")
print("--------------------")
print("python     | {}.{}.{}".format(v1,v2,v3))

# popular modules for scientific computation
import numpy as np
print("numpy      |",np.__version__)
import scipy
print("scipy      |",scipy.__version__)

# typical module for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
print("matplotlib |",mpl.__version__)

# wide spread module for cosmological data manipulation/computation
# we will use this mainly to import constants and units
import astropy
from astropy import units,constants
print("astropy    |",astropy.__version__)

# the following can be used to speed up some python computations (warning, module is in active development)
import numba
from numba import njit
print("numba      |",numba.__version__)
print("--------------------")


