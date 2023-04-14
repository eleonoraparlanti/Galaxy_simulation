
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  import numpy as np
  from gravity_tree import tree, build_tree_from_particles,compute_mass_on_tree,compute_potential_tree
  from gravity_bruteforce import compute_potential
  from time import time


  n_dim           = 2

  #--------------------------------
  # parameters of the tree
  #--------------------------------
  n_part_per_cell = 10
  critical_angle  = 2.0*np.pi*(10.0/360.0)
  n_grid          = 5000
  iter_max        = 200
 
  #--------------------------------
  # set up particles
  #--------------------------------
  n_pt      = 1000
  test_pos  = np.zeros((n_dim,n_pt))
  test_mass = np.ones(n_pt)

  for i_dim in range(n_dim):
    test_pos[i_dim,:] = np.random.normal(loc=0.0,scale=1.0, size=n_pt)
 
  # renormalize point position (in a bad way, unit_l should be defined)
  for i_dim in range(n_dim):
      test_pos[i_dim,:] = (test_pos[i_dim,:]-np.min(test_pos[i_dim,:]))/(np.max(test_pos[i_dim,:])-np.min(test_pos[i_dim,:])) - 0.5

  #--------------------------------
  # compile tree and N^2 computation
  #-------------------------------

  comp_tree = tree(n_dim = n_dim,n_grid=n_grid,n_part_per_cell = n_part_per_cell, mass_pt=test_mass[0:2], pos_pt = test_pos[:,0:2])
  comp_tree = build_tree_from_particles(tree_in=comp_tree,i_iter_max=iter_max)
  comp_tree = compute_mass_on_tree(tree_in=comp_tree)
  __        = compute_potential_tree(tree_in=comp_tree,critical_angle=critical_angle, G_gravity= 1.0)
  __        = compute_potential(masses=test_mass,positions=test_pos, G_gravity= 1.0)

  #--------------------------------
  # tree computation
  #-------------------------------

  test_tree      = tree(n_dim = n_dim,n_grid=n_grid,n_part_per_cell = n_part_per_cell, mass_pt=test_mass, pos_pt = test_pos)
  t_start        = time()
  test_tree      = build_tree_from_particles(tree_in=test_tree,i_iter_max=iter_max)
  t_1            = time()
  test_tree      = compute_mass_on_tree(tree_in=test_tree)
  t_2            = time()
  potential_tree = compute_potential_tree(tree_in=test_tree,critical_angle=critical_angle, G_gravity= 1.0, verbose = True)
  t_end          = time()
  time_tree      = t_end - t_start

  print('time tree ',time_tree)
  print('  build tree     ', 100.0*(t_1 - t_start)/time_tree,'%')
  print('  mass assignment', 100.0*(t_2 - t_1)/time_tree,'%')
  print('  potential      ', 100.0*(t_end - t_2)/time_tree,'%')

  #--------------------------------
  # N^2 computation
  #-------------------------------

  t_start        = time()
  potential_bruteforce = compute_potential(masses=test_mass,positions=test_pos, G_gravity= 1.0)
  t_end          = time()
  time_N2        = t_end - t_start 
  print('time bruteforce',time_N2)

  err_array            = np.zeros(n_pt)
  err_array[:]         = np.abs(potential_tree[:] -  potential_bruteforce[:])/np.abs(potential_bruteforce[:])
  mean_prec, std_prec  = np.mean(err_array), np.std(err_array)

  print("mean fractional error",mean_prec,'pm',std_prec)
  print("speedup              ",time_N2/time_tree)

