
import numpy as np
from gravity_bruteforce import compute_acceleration,compute_jerk,compute_kinetic_energy, compute_potential


def drift(dt, pos_in, vel_in):

  n_dim, n_pt = np.shape(pos_in)
  pos_out     = np.zeros((n_dim,n_pt))

  pos_out[:,:] = pos_in[:,:] + dt * vel_in[:,:]

  return pos_out

def kick(dt, vel_in, acc_in):

  n_dim, n_pt = np.shape(vel_in)

  vel_out      = np.zeros((n_dim,n_pt))
  vel_out[:,:] = vel_in[:,:] + dt * acc_in[:,:]

  return vel_out

def compute_time_step(pt_mass, pt_pos, pt_vel, pt_acc,pt_jerk,eta_time = 0.01):

  n_dim, n_pt = np.shape(pt_vel)
  dt_vec      = np.zeros(n_pt)

  for i_pt in range(n_pt):

      acc_i        = np.sum(pt_acc[: ,i_pt]**2)
      jerk_i       = np.sum(pt_jerk[:,i_pt]**2)
      dt_vec[i_pt] = eta_time * np.sqrt(acc_i/jerk_i)

  return np.min(dt_vec)

def evolve_to_t_end(t_end, pt_mass, pt_pos, pt_vel, pt_acc,pt_jerk,G_gravity=1.0, max_iterations=1000, n_snap=1000, f_snap=1, eta_time=1.e-3):

  t_now  = 0.0
  i_iter = 0

  n_dim, n_pt = np.shape(pt_pos)
  pos_snap    = np.zeros((n_dim,n_pt,n_snap))
  vel_snap    = np.zeros((n_dim,n_pt,n_snap))
  t_snap      = np.zeros(n_snap)
  e_snap      = np.zeros(n_snap)

  i_snap = 0

  while (i_iter <= max_iterations) and (t_now <= t_end):

    dt = compute_time_step(pt_mass, pt_pos, pt_vel, pt_acc,pt_jerk , eta_time = eta_time)

    dt, pt_mass, pt_pos, pt_vel, pt_acc, pt_jerk = evolve_dt(dt, pt_mass, pt_pos, pt_vel, pt_acc,pt_jerk, G_gravity=G_gravity)

    t_now  = t_now  + dt
    i_iter = i_iter + 1

    if (i_iter % f_snap == 0) and (i_snap < n_snap):
      # produce some snapshot

      e_k                   = np.sum(compute_kinetic_energy(masses=pt_mass,velocities=pt_vel))
      e_p                   = np.sum(compute_potential(masses=pt_mass,positions=pt_pos, G_gravity= G_gravity))

      e_snap[i_snap]        = e_k + e_p

      pos_snap[:,:,i_snap]  = pt_pos[:,:]
      vel_snap [:,:,i_snap] = pt_vel[:,:]

      t_snap[i_snap]        = t_now
      
      i_snap                = i_snap + 1

  # trim the snapshots
  pos_snap= pos_snap[:,:, 0: i_snap-1]
  vel_snap= vel_snap[:,:, 0: i_snap-1]
  e_snap  = e_snap[0: i_snap-1]
  t_snap  = t_snap[0: i_snap-1]
  
  return t_now, i_iter, pt_mass, pt_pos, pt_vel, pt_acc, pt_jerk,  pos_snap, vel_snap, e_snap , t_snap

def evolve_dt(dt, pt_mass, pt_pos, pt_vel, pt_acc,pt_jerk, G_gravity=1.0):

  """
  leapfrog time step
  """

  n_dim, n_pt = np.shape(pt_pos)
  
  # update state of the system
  #new_pt_mass = np.zeros(n_pt)
  new_pt_pos  = np.zeros((n_dim,n_pt))
  new_pt_vel  = np.zeros((n_dim,n_pt))
  #
  new_pt_acc   = np.zeros((n_dim,n_pt))
  new_pt_jerk  = np.zeros((n_dim,n_pt))

  new_pt_vel[:,:] = kick(dt=dt/2.0, vel_in = pt_vel[:,:], acc_in = pt_acc[:,:])
  new_pt_pos[:,:] = drift(dt=dt   , pos_in = pt_pos[:,:], vel_in = new_pt_vel[:,:])
  new_pt_acc[:,:] = compute_acceleration(masses=pt_mass[:],positions=new_pt_pos[:,:], G_gravity= G_gravity)
  new_pt_jerk[:,:]= compute_jerk(masses=pt_mass[:],positions= new_pt_pos[:,:],velocities=new_pt_vel[:,:], G_gravity= G_gravity) 
  new_pt_vel[:,:] = kick(dt=dt/2.0, vel_in = new_pt_vel[:,:], acc_in = new_pt_acc[:,:])

  return dt, pt_mass, new_pt_pos, new_pt_vel, new_pt_acc, new_pt_jerk



