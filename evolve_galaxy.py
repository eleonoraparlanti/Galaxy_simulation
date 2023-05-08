
import numpy as np
from astropy import units, constants
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

  import matplotlib.pyplot as plt
  from time import time
  
  # check IC
  from gravity_tree import tree, build_tree_from_particles, compute_mass_on_tree, compute_potential_tree, compute_acceleration_tree
  from tree_module.part_2_tree import get_level_per_particle
  from gravity_bruteforce import compute_acceleration,compute_jerk,compute_kinetic_energy, compute_potential
  from evolve_system import evolve_dt_galaxy, evolve_to_t_end_galaxy
  
  # units
  unit_m  = constants.M_sun
  unit_l  = constants.pc
  unit_t = (constants.G * unit_m/unit_l**3)**(-0.5)
  
  # For normalization
  pos_normalization  = 20000 
  vel_normalization  = (units.km/units.s).to(units.pc/units.Myr)
  
  # system setup
  M_tot_msun      = 1e10   # msun
  r_0_pc          = 100.0 # pc
  N_pt            = int(1e2)
  N_pt_star       = int(1e2)
  N_pt_dm         = int(1e1)
  # N_pt       = int(8e2)   for 3d check  
  
  n_grid = 10000
  n_part_per_cell = 8
  n_dim = 3
  
  # time steps 
  dt_myr     = 0.01   # in 'Myr'
  t_end_myr  = 2.0    # in 'Myr'
  f_snap_myr = 10 
  
  G_gravity  = 0.0044985022   ## In unit of (pc^3/Msun/Myr/Myr)
  G_gravity  = G_gravity  / pos_normalization / pos_normalization / pos_normalization  ## (  " / (2 * r_cut_star) "  ) is mormalization for the position (see pos_normed_star) 
    
  ############################################################
  #### Reading Initial condition from the file 
  ############################################################
  
  print("Reading IC for star particles from : ic_star_N",N_pt_star,".txt")
  ic_star        = np.genfromtxt("ic_star_N"+str(N_pt_star)+".txt")
  mass_star      = ic_star[:,0]
  x_pos_star     = ic_star[:,1]
  y_pos_star     = ic_star[:,2]
  z_pos_star     = ic_star[:,3]
  vx_star        = ic_star[:,4] * vel_normalization
  vy_star        = ic_star[:,5] * vel_normalization
  vz_star        = ic_star[:,6] * vel_normalization
  
  v_star         = np.zeros((3, N_pt_star))
  v_star[0, :]   = vx_star
  v_star[1, :]   = vy_star
  v_star[2, :]   = vz_star
  pos_star       = np.zeros((3, N_pt_star))
  pos_star[0, :] = x_pos_star
  pos_star[1, :] = y_pos_star
  pos_star[2, :] = z_pos_star
  
  print("Reading IC for DM particles from   : ic_dm_N",N_pt_dm,".txt")
  ic_dm          = np.genfromtxt("ic_dm_N"+str(N_pt_dm)+".txt")
  mass_dm        = ic_dm[:,0]
  x_pos_dm       = ic_dm[:,1]
  y_pos_dm       = ic_dm[:,2]
  z_pos_dm       = ic_dm[:,3]
  vx_dm          = ic_dm[:,4] * vel_normalization
  vy_dm          = ic_dm[:,5] * vel_normalization
  vz_dm          = ic_dm[:,6] * vel_normalization
  
  v_dm           = np.zeros((3, N_pt_dm))
  v_dm[0, :]     = vx_dm
  v_dm[1, :]     = vy_dm
  v_dm[2, :]     = vz_dm
  pos_dm         = np.zeros((3, N_pt_dm))
  pos_dm[0, :]   = x_pos_dm
  pos_dm[1, :]   = y_pos_dm
  pos_dm[2, :]   = z_pos_dm
  
  mass_particle  = M_tot_msun /N_pt
  print("----------------------------")
  print("Imported IC succesfully !!")
  print("----------------------------")
  print("  n particles   : ",N_pt)
  print("  total mass    : ",M_tot_msun,'Msun')
  print("  particle mass : ",mass_particle,'Msun')
  
  # ------------------------------------
  # Evolving for the first few time step
  # ------------------------------------
    
  print("\n")  
  print("Evolution started!!")

  # get particle level using a tree
  comp_tree = tree(n_dim=n_dim, n_grid=n_grid, n_part_per_cell=n_part_per_cell, mass_pt=mass_star, pos_pt=pos_star)

  comp_tree = build_tree_from_particles(tree_in=comp_tree, i_iter_max=10000)
  acceleration_stars = compute_acceleration_tree(tree_in=comp_tree,critical_angle=0.1, G_gravity= G_gravity, n_iter_max=10000, verbose= False)
  t_now, i_iter, pt_mass, pt_pos, pt_vel, pt_acc, pos_snap, vel_snap, e_snap , t_snap = evolve_to_t_end_galaxy(t_end=t_end_myr, pt_mass=mass_star, pt_pos=pos_star, pt_vel=v_star, pt_acc=acceleration_stars, comp_tree=comp_tree, G_gravity=G_gravity, max_iterations=1e6, n_snap=1000, f_snap=f_snap_myr, eta_time=dt_myr)
    
  
