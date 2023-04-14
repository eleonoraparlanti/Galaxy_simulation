
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from time import time

"""

  dt               = d a/\dot{a}

  t = \int_0^{t} dt = \int_0^1 d a/\dot{a}

"""


def compute_dot_a(aexp, H_0, Omega_m, Omega_L, Omega_r):

  """
  H^2     = H_0^2 (Omega_m/a^3 Omega_r/a^4 + Omega_L)
  H       = \dot{a}/a
  \dot{a} = H_0 a \times \sqrt{Omega_m/a^3 Omega_r/a^4 + Omega_L}
  """
  out     = H_0 * np.sqrt(Omega_m*aexp + Omega_r  + Omega_L * aexp**4) / aexp

  return out

def compute_inverse_dot_a(aexp, H_0, Omega_m, Omega_L, Omega_r):

  """
  1/\dot{a} = 1/(H_0 a \times \sqrt{Omega_m/a^3 Omega_r/a^4 + Omega_L})
  """
  out     = aexp / np.sqrt(Omega_m*aexp + Omega_r  + Omega_L * aexp**4) / H_0

  return out

if __name__ == "__main__":

  # after having imported the cosmological parameters
  # check them

  Omega_L   = cosmo.Ode0
  Omega_r   = cosmo.Ogamma0 + cosmo.Onu0
  Omega_m   = cosmo.Om0
  Omega_tot = Omega_L + Omega_r + Omega_m

  unit_t    = 1/cosmo.H0

  print("Omega_L  ",Omega_L)
  print("Omega_r  ",Omega_r)
  print("Omega_m  ",Omega_m)
  print("Omega_tot",Omega_tot) # ~ 1
  #-----------------------------------------------

  check = False

  # visual check for the equation
  if check:
    n_bins     = 1000
    aexp       = np.linspace(1.0, 0.0 , n_bins)
    inv_dot_a  = compute_inverse_dot_a(aexp=aexp, H_0=1.0, Omega_m=Omega_m, Omega_L=Omega_L, Omega_r=Omega_r)

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.plot(aexp, inv_dot_a ,ls='-')

    ax.set_xlabel("a (expansion factor)" , size= 24)
    ax.set_ylabel("$\dot{a}^-1 / H_0$" , size=24)
    plt.show()

  #-----------------------------------------------

  """
  indeptennt variable t
  explicit time integration
  eventually use RK4
  start from a = 1
  integrate backward in time
  set the timestep somewhow

  """

  # set up the workspace
  n_bins     = 100000
  aexp       = np.zeros(n_bins)  # expansion factor
  proper_t   = np.zeros(n_bins)  # look back time

  # IC @ today
  aexp[0]     = 1.0
  proper_t[0] = 0.0
  aexp_now    = aexp[0]
  i_step = 0

  # dt \dot{a} = d a
  # choose dt
  dt = -1.e-5

  # define the numerical flux
  def flux(aexp_now,dt):
    out = dt * compute_dot_a(aexp=aexp_now, H_0=1.0, Omega_m=Omega_m, Omega_L=Omega_L, Omega_r=Omega_r)

    return out

  #-----------------------------------------------
  # integration

  t_cpu_start = time()

  while aexp_now > 0 and i_step < n_bins:

    # get current status
    aexp_now = aexp[i_step]
    t_now    = proper_t[i_step]
    # increment time (independent variable)
    t_new    = t_now + dt
    # solve numerically for expansion
    k1       = flux(aexp_now = aexp_now   , dt= dt)
    k2       = flux(aexp_now = aexp_now+k1, dt= dt)
    # RK2
    aexp_new = aexp_now + (k1 + k2)/2.0

    # storing the update
    aexp[i_step+1]     = aexp_new
    proper_t[i_step+1] = t_new

    # increment the counter
    i_step = i_step+1
 
  t_cpu_end = time() - t_cpu_start

  print("CPU time taken",t_cpu_end,"s")
  #-----------------------------------------------

  # trim out excess workspace
  aexp     = aexp[0:i_step-1]
  proper_t = proper_t[0:i_step-1]

  redshift  = 1.0/aexp - 1

  # shift time axis
  proper_t[:] = proper_t[:]  -proper_t[-1]
  # get back the units
  proper_t[:] = proper_t[:] * (unit_t.to('Gyr')).value
  dt_gyr      = np.abs(dt)* (unit_t.to('Gyr')).value

  # grab a reference
  reference = (cosmo.age(z=redshift).to('Gyr')).value

  delta  = 100.0*(proper_t - reference)/reference

  print("Age at z=0",proper_t[0],"Gyr")

  fig  = plt.figure()
  ax_up   = fig.add_subplot(211)
  ax_low  = fig.add_subplot(212)

  
  ax_up.plot(np.log10(redshift+1), np.log10(proper_t)  ,ls='-' , label='RK2')
  ax_up.plot(np.log10(redshift+1), np.log10(reference) ,ls='-' , label='astropy')

  ax_up.axhline(np.log10(dt_gyr),ls='--',label='fixed time step')
  ax_low.plot(np.log10(redshift+1), delta)

  ax_low.set_xlabel("$\log(z + 1)$" , size= 24)
  ax_up.set_ylabel("$\log t /Gyr$" , size=24)
  ax_low.set_ylabel("$\delta [\%] $" , size=24)

  ax_up.legend(frameon=False)

  plt.show()






