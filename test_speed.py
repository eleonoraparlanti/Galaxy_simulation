
from numba import jit,njit
import numpy as np
from time import time

def check_pure_python(a_test):
  nn = len(a_test)
  for ii in range(nn):
    a_test[ii] = a_test[ii] +1
  return a_test

@njit
def check_numba(a_test):
  nn = len(a_test)
  for ii in range(nn):
    a_test[ii] = a_test[ii] +1
  return a_test

jitted_version = jit(check_pure_python, nopython=True)

nn     = 1000000
a_test = np.zeros(nn)

t1 = time()
a_test =  check_pure_python(a_test)
t2 = time() - t1
print("time pure python",t2,"s")

__  =  jitted_version(np.zeros(3))
t1 = time()
a_test =  jitted_version(a_test)
t2 = time() - t1
print("time numba",t2,"s")

t1 = time()
a_test[:] = a_test[:] +1
t2 = time() - t1
print("time numpy",t2,"s")



