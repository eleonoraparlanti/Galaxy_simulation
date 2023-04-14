
from numba import njit
import numpy as np

def compute_kinetic_energy(masses,velocities):

  n_dim, n_pt = np.shape(velocities)
  out         = np.zeros(n_pt)
  
  for i_pt in range(n_pt):
      out[i_pt] = 0.5*masses[i_pt] * np.sum(velocities[:,i_pt]**2)

  return out

@njit
def compute_potential(masses,positions, G_gravity= 1.0):

  n_dim, n_pt = np.shape(positions)
  out         = np.zeros(n_pt)
  pot_j       = 0.0
  pos_ij      = np.zeros(n_dim)
  modulus_ij  = 0.0

  for i_pt in range(n_pt):

     for j_pt in range(n_pt):

       if i_pt != j_pt:
         pos_ij[:]   = positions[:,j_pt] - positions[:,i_pt]

         modulus_ij  = 0.0
         for i_dim in range(n_dim):
           modulus_ij  = modulus_ij + pos_ij[i_dim]**2
         modulus_ij = np.sqrt(modulus_ij)

         pot_j     = - masses[j_pt]/modulus_ij

         out[i_pt] = out[i_pt] + pot_j

  out[:] = out[:] * G_gravity

  return out


def compute_acceleration(masses,positions, G_gravity= 1.0):

  n_dim, n_pt = np.shape(positions)

  out         = np.zeros((n_dim,n_pt))
  acc_j       = np.zeros(n_dim)
  pos_ij      = np.zeros(n_dim)
  modulus_ij  = 0.0

  for i_pt in range(n_pt):

     for j_pt in range(n_pt):

       if i_pt != j_pt:
         pos_ij[:]   = positions[:,j_pt] - positions[:,i_pt]

         modulus_ij  = 0.0
         for i_dim in range(n_dim):
           modulus_ij  = modulus_ij + pos_ij[i_dim]**2
         modulus_ij = np.sqrt(modulus_ij)

         acc_j[:]    = masses[j_pt] * pos_ij[:]/ modulus_ij**(3.0)

         out[:,i_pt] = out[:,i_pt] + acc_j[:]

  out[:] = out[:] * G_gravity

  return out

def compute_jerk(masses,positions,velocities, G_gravity= 1.0):

  n_dim, n_pt = np.shape(positions)

  out         = np.zeros((n_dim,n_pt))
  jerk_j      = np.zeros(n_dim)
  pos_ij      = np.zeros(n_dim)
  vel_ij      = np.zeros(n_dim)
  v_times_r   = 0.0
  modulus_ij  = 0.0

  for i_pt in range(n_pt):

     for j_pt in range(n_pt):

       if i_pt != j_pt:
         pos_ij[:]   = positions[:,j_pt]  - positions[:,i_pt]
         vel_ij[:]   = velocities[:,j_pt] - velocities[:,i_pt]

         v_times_r   = np.sum(vel_ij[:]*pos_ij[:])

         modulus_ij  = 0.0
         for i_dim in range(n_dim):
           modulus_ij  = modulus_ij + pos_ij[i_dim]**2
         modulus_ij = np.sqrt(modulus_ij)

         jerk_j[:] = masses[j_pt]*(vel_ij[:]/ modulus_ij**(3.0)  + \
                                   3.0*v_times_r*pos_ij[:]/ modulus_ij**(5.0)
                                  )

         out[:,i_pt] = out[:,i_pt] + jerk_j[:]

  out[:] = out[:] * G_gravity

  return out

if __name__ == "__main__":

  import matplotlib.pyplot as plt
  from time import time

  n_dim     = 3
  max_level = 12
  n_pt      = int(2**max_level)
  test_pos  = np.zeros((n_dim,n_pt))
  test_mass = np.ones(n_pt)

  arr_n_part  = np.zeros(max_level)
  arr_cputime = np.zeros(max_level)

  for i_dim in range(n_dim):
    test_pos[i_dim,:] = np.random.normal(loc=0.0,scale=1.0, size=n_pt)
  
  t_star = time()
  test_potential = compute_potential(masses=test_mass[0:2],positions=test_pos[:,0:2])
  t_end = time()

  print("jit compilation: ",t_end-t_star,"s")

  for i_lev in range(max_level):

    npart_now      = int(2**i_lev)
    #
    t_star         = time()
    test_potential = compute_potential(masses=test_mass[0:npart_now],positions=test_pos[:,0:npart_now])
    t_end          = time()
    #
    arr_n_part[i_lev]  = npart_now
    arr_cputime[i_lev] = t_end-t_star

  fig = plt.figure()
  ax  = fig.add_subplot(111)

  ax.plot(np.log10(arr_n_part),np.log10(arr_cputime),marker='o',ls=':')

  ax.set_xlabel(r"$\log N_p$")
  ax.set_ylabel(r"$\log \rm CPUtime/s$")

  plt.show()




