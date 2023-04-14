
import numpy as np
import matplotlib.pyplot as plt
from astropy import units, constants
from time import time

from evolve_system import evolve_to_t_end
from gravity_bruteforce import compute_acceleration,compute_jerk,compute_kinetic_energy, compute_potential

# system setting
n_dim   = 3
n_pt    = 2

# units
unit_m  = constants.M_sun
unit_l  = constants.au
unit_t = (constants.G * unit_m/unit_l**3)**(-0.5)

unit_v  = unit_l/unit_t
unit_e  = unit_m * unit_v**2

# state of the system
pt_mass = np.zeros(n_pt)
pt_pos  = np.zeros((n_dim,n_pt))
pt_vel  = np.zeros((n_dim,n_pt))
#
pt_acc   = np.zeros((n_dim,n_pt))
pt_jerk  = np.zeros((n_dim,n_pt))
e_k      = 0.0
e_p      = 0.0

# IC
if 0:
  pt_mass[0]  =  constants.M_sun.to(unit_m).value
  pt_mass[1]  =  constants.M_earth.to(unit_m).value

  pt_pos[0,0] = -(0.5*constants.au.to(unit_l)).value
  pt_pos[0,1] = +(0.5*constants.au.to(unit_l)).value

  pt_vel[1,0] = -(20.0*units.km/units.s).to(unit_v).value  # vy of part 1
  pt_vel[1,1] = +(20.0*units.km/units.s).to(unit_v).value  # vy of part 2
else:

  pt_mass[0]  = 1.0
  pt_mass[1]  = 1.0

  pt_pos[0,0] = -1.0
  pt_pos[0,1] = +1.0

  pt_vel[1,0] = -0.5
  pt_vel[1,1] = +0.5


pt_acc  = compute_acceleration(masses=pt_mass,positions=pt_pos, G_gravity= 1.0)
pt_jerk = compute_jerk(masses=pt_mass,positions= pt_pos,velocities=pt_vel, G_gravity= 1.0) 

e_k     = np.sum(compute_kinetic_energy(masses=pt_mass,velocities=pt_vel))
e_p     = np.sum(compute_potential(masses=pt_mass,positions=pt_pos, G_gravity= 1.0))

t_end   = units.yr.to(unit_t)

print("system stats")
print("masses")
for i_pt in range(n_pt):
  print("  ",i_pt, pt_mass[i_pt],unit_m.to('g').value)

print("kinetic   energy",e_k*unit_e.to(constants.M_sun*(units.km/units.s)**2).value,'Msun km/s**2')
print("potential energy",e_p*unit_e.to(constants.M_sun*(units.km/units.s)**2).value,'Msun km/s**2')
print("total           ",(e_k+e_p)*unit_e.to(constants.M_sun*(units.km/units.s)**2).value,'Msun km/s**2')
print("t end           ",(t_end*unit_t).to("yr").value,"yr")

t_start_cpu = time()
t_now, i_iter, pt_mass, pt_pos, pt_vel, pt_acc, pt_jerk,  pos_snap, vel_snap, e_snap , t_snap = evolve_to_t_end(t_end, pt_mass, pt_pos, pt_vel, pt_acc,pt_jerk,G_gravity=1.0, max_iterations=10000, n_snap = 10000, f_snap = 1)

t_end_cpu = time()
print("CPU time taken       ",t_end_cpu-t_start_cpu)
print("last time            ",(t_now*unit_t).to("yr").value,"yr")
print("iterations           ",i_iter)
print("average energy error ",100.0*np.average(e_snap[:] - (e_k+e_p))/(e_k+e_p),"%")

#---------------------------------------------------------------
# plot trajectory

fig = plt.figure()
ax  = fig.add_subplot(111)

# remove center of mass
for i_snap in range(np.shape(pos_snap[:,:,:])[2]):
  pos_cm               = np.sum(pt_mass[np.newaxis,:]*pos_snap[:,:,i_snap],axis=1)/np.sum(pt_mass[:])
  pos_snap[:,:,i_snap] = pos_snap[:,:,i_snap] - pos_cm[:,np.newaxis]

l_lim = np.max(np.abs(pos_snap[:,:,:]))
for i_pt in range(n_pt):
  ax.plot(pos_snap[0,i_pt,:], pos_snap[1,i_pt,:],label="particle "+str(i_pt+1),ls='-',marker='o')

ax.legend(frameon=False)
ax.set_xlim(-l_lim,l_lim)
ax.set_ylim(-l_lim,l_lim)


#---------------------------------------------------------------
# plot energy
e_tot = e_k + e_p

fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(t_snap, e_snap )
ax.axhline(e_k+e_p, ls='--')


plt.show()


