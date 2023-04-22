import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from astropy import units, constants
import scipy.special
import matplotlib as mpl
import h5py
from projection import projection
from evolve_system import evolve_dt_galaxy
from mpl_toolkits.mplot3d import Axes3D
#mpl.use('macosx')

from time import time
from evolve_system import evolve_dt_galaxy, evolve_to_t_end_galaxy

"""
def extract_radius_from_plummer(N_pt, M_tot_msun=1.e5,sigma_bound = 200, r_0_pc = 100.0,i_iter_max=100000):

  AA              = get_plummer_density(radius=0.0, M_tot=M_tot_msun, r_0=r_0_pc)/get_gaussian(xx=0.0, sigma_0=sigma_bound,mean=0.0)

  out             = np.zeros(N_pt)

  i_pt     = 0
  i_iter  = 0

  print("extracting particles with plummer density with A/R")

  while i_pt < N_pt:
      
    assert i_iter < i_iter_max

    # extraction from the boundary function
    x_0             = np.random.normal(loc=0.0,scale=sigma_bound)
    f_of_x_0        = get_gaussian(xx=x_0, sigma_0=sigma_bound,mean=0.0)
    f_of_x_0        = AA*f_of_x_0

    # second extraction
    mm              = f_of_x_0*np.random.random()

    # compare
    accept = mm <= get_plummer_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc)

    if accept:
        out[i_pt] = x_0
        i_pt      = i_pt +1

    i_iter = i_iter + 1

  print("  efficiency   ",100.0*float(N_pt)/float(i_iter),"%")

  return out

"""


def get_sersic_density(radius, M_tot=1.0, r_0=1.0, n_sersic=1):
    """
    Density from Terzic, Graham 2005
    :param radius: radius array
    :param M_tot: total mass of the sersic profile
    :param r_0: effective radius of the sersic profile
    :param n_sersic: serisc index
    :return: density at each radius
    """
    norm = 3.0 * M_tot / (4.0 * np.pi * r_0 ** 3)
    b = 2 * n_sersic - 1 / 3 + 0.009876 / n_sersic  # Prugniel & Simien 1997
    p = 1 - 0.6097 * (1 / n_sersic) + 0.05563 * (1 / n_sersic) ** 2
    density_out = norm * (radius / r_0) ** (-p) * np.exp(-b * (radius / r_0) ** (1 / n_sersic))

    return density_out


def extract_radius_from_sersic(N_pt, M_tot_msun=1.e5, sigma_bound=200, r_0_pc=100.0, n_sersic=1, i_iter_max=100000,
                               min_star_radius=10):
    AA = get_sersic_density(radius=min_star_radius, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic) / get_gaussian(
        xx=min_star_radius,
        sigma_0=sigma_bound,
        mean=0.0)

    out = np.zeros(N_pt)

    i_pt = 0
    i_iter = 0

    print("extracting particles with sersic density with A/R")

    while i_pt < N_pt:

        assert i_iter < i_iter_max

        # extraction from the boundary function
        x_0 = np.random.normal(loc=0.0, scale=sigma_bound)
        x_0 = np.abs(x_0)
        f_of_x_0 = get_gaussian(xx=x_0, sigma_0=sigma_bound, mean=0.0)
        f_of_x_0 = AA * f_of_x_0

        # second extraction
        mm = f_of_x_0 * np.random.random()

        # compare
        accept = mm <= get_sersic_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic)
        # print(mm, get_sersic_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic), accept)

        if accept:
            out[i_pt] = x_0
            i_pt = i_pt + 1

        i_iter = i_iter + 1

    print("  efficiency   ", 100.0 * float(N_pt) / float(i_iter), "%")

    return out


def get_nfw_density(radius, M_tot=1.0, r_0=1.0, concentration=10):
    norm = 3.0 * M_tot / (4.0 * np.pi * r_0 ** 3)
    A = np.log(1 + concentration) - concentration / (1 + concentration)
    xx = radius / r_0
    density_out = norm / (3 * A * xx * (concentration ** -1 + xx) ** 2)
    return density_out


def extract_radius_from_nfw(N_pt, M_tot_msun=1.e5, sigma_bound=200, r_0_pc=100.0, i_iter_max=100000, min_radius=1e-10,
                            concentration=10):
    AA = get_nfw_density(radius=min_radius, M_tot=M_tot_msun, r_0=r_0_pc, concentration=10) / get_gaussian(
        xx=min_radius,
        sigma_0=sigma_bound,
        mean=0.0)

    out = np.zeros(N_pt)

    i_pt = 0
    i_iter = 0

    print("extracting particles with nfw density with A/R")

    while i_pt < N_pt:

        # assert i_iter < i_iter_max

        # extraction from the boundary function
        x_0 = np.random.normal(loc=0.0, scale=sigma_bound)
        x_0 = np.abs(x_0)

        f_of_x_0 = get_gaussian(xx=x_0, sigma_0=sigma_bound, mean=0.0)
        f_of_x_0 = AA * f_of_x_0

        # second extraction
        mm = f_of_x_0 * np.random.random()

        # compare
        accept = mm <= get_nfw_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc)
        print(mm, get_sersic_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic), accept)

        if accept:
            out[i_pt] = x_0
            i_pt = i_pt + 1

        i_iter = i_iter + 1

    print("  efficiency   ", 100.0 * float(N_pt) / float(i_iter), "%")

    return out


def extract_z_from_gaussian(N_pt, scale_height, sigma_bound=200, i_iter_max=100000):
    AA = get_gaussian(xx=0.0, sigma_0=scale_height, mean=0) / get_gaussian(xx=0.0, sigma_0=scale_height,
                                                                           mean=0.0)

    out = np.zeros(N_pt)

    i_pt = 0
    i_iter = 0

    print("extracting particles with gaussian density")

    while i_pt < N_pt:
        # assert i_iter < i_iter_max

        # extraction from the boundary function
        x_0 = np.random.normal(loc=0.0, scale=scale_height)
        out[i_pt] = x_0
        i_pt = i_pt + 1

        """
        f_of_x_0 = get_gaussian(xx=x_0, sigma_0=scale_height, mean=0.0)
        f_of_x_0 = AA * f_of_x_0

        # second extraction
        mm = f_of_x_0 * np.random.random()

        # compare
        accept = mm <= get_gaussian(xx=x_0, sigma_0=scale_height, mean=0.0)

        if accept:
            out[i_pt] = x_0
            i_pt = i_pt + 1
        """

        # i_iter = i_iter + 1

    # print("  efficiency   ", 100.0 * float(N_pt) / float(i_iter), "%")

    return out


def velocity_sersic(N_pt, radius_array, reff=1, M_tot_msun=1e10, n_sersic=1, ):
    """
    :param N_pt: number of particles
    :param radius_array: array of radius for each particle [pc]
    :param reff: effective radius of the sersic profile [pc]
    :param M_tot_msun: total mass of the sersic profile
    :param n_sersic: sersic index
    :return: Array of the velocity for each particle due to the sersic in km/s
    """

    out = np.zeros(N_pt)
    for ii in range(0, N_pt):
        # Lima Neto et al 1999
        p_par = 1 - 0.6097 * (1 / n_sersic) + 0.05563 * (1 / n_sersic) ** 2

        # Prigniel & Simien 1997
        b_par = 2 * n_sersic - 1 / 3 + 0.009876 / n_sersic

        # Terzic & Graham 2005
        num = constants.G.value * M_tot_msun * constants.M_sun.value * scipy.special.gammainc(n_sersic * (3 - p_par),
                                                                                              b_par * (
                                                                                                          radius_array / reff) ** (
                                                                                                      1 / n_sersic)
                                                                                              ) * scipy.special.gamma(
            n_sersic * (3 - p_par))
        den = (radius_array * units.pc).to(units.m).value * scipy.special.gamma(n_sersic * (3 - p_par))

        out = 1.0e-3 * (num / den) ** 0.5

    return out


def velocity_nfw(N_pt, radius_array, r_vir=1, M_tot_msun=1e10, concentration=10):
    """
    :param N_pt: number of particles
    :param radius_array: array of radius for each particle [pc]
    :param r_vir: virial_radius [pc]
    :param M_tot_msun: virial mass of the dm halo
    :param concentration: concentration parameter
    :return: Array of the velocity for each particle due to the dm halo
    """
    xx = radius_array / r_vir
    r_vir_in_meter = (r_vir * units.pc).to(units.m).value

    out = np.zeros(N_pt)
    for ii in range(0, N_pt):
        num = constants.G.value * M_tot_msun * constants.M_sun.value * np.log(
            1 + concentration * xx) - concentration * xx / (1 + concentration * xx)
        den = r_vir_in_meter * (np.log(1 + concentration) - concentration / (1 + concentration))

        out = 1.0e-3 * (num / den) ** 0.5

    return out


def initial_velocity(N_pt, radius_array, r_vir=1, reff=1, M_tot_dm=1e10, M_tot_star=1e10, n_sersic=1, concentration=10,
                     velocity_dispersion=0):
    v_star = velocity_sersic(N_pt, radius_array, reff=reff, M_tot_msun=M_tot_star, n_sersic=n_sersic)
    v_dm = velocity_nfw(N_pt, radius_array, r_vir=r_vir, M_tot_msun=M_tot_dm, concentration=concentration)

    # circular velocity due to the potential
    print(v_star)
    print(v_dm)
    v_circ = np.sqrt(v_star ** 2 + v_dm ** 2)

    # adding a random velocity field

    v_r = np.random.normal(loc=0.0, scale=velocity_dispersion, size=N_pt)
    v_theta = v_circ + np.random.normal(loc=0.0, scale=velocity_dispersion, size=N_pt)
    v_phi = np.random.normal(loc=0.0, scale=velocity_dispersion, size=N_pt)

    return v_r, v_theta, v_phi


def extract_angles_from_sphere(N_pt):
    # d Omega = d phi d cos(theta)
    phi = 2.0 * np.pi * np.random.random(size=N_pt)
    theta = np.arccos(2.0 * np.random.random(size=N_pt) - 1)

    return phi, theta


def extract_angles_from_circle(N_pt):
    theta = 2.0 * np.pi * np.random.random(size=N_pt)

    return theta


def convert_spherical_to_cartesian(radius, phi, theta):
    xx = radius * np.cos(phi) * np.sin(theta)
    yy = radius * np.sin(phi) * np.sin(theta)
    zz = radius * np.cos(theta)

    return xx, yy, zz


def convert_polar_to_cartesian(radius, theta):
    xx = radius * np.cos(theta)
    yy = radius * np.sin(theta)

    return xx, yy


def get_gaussian(xx, sigma_0=1.0, mean=0.0):
    norm = 1.0 / np.sqrt(2.0 * np.pi * sigma_0 ** 2)
    out = ((xx - mean) / sigma_0) ** 2
    out = np.exp(-out)
    out = out * norm

    return out


"""
def get_plummer_density(radius, M_tot=1.0, r_0=1.0):

    norm        = 3.0*M_tot/(4.0*np.pi * r_0**3)

    density_out = norm / (1 +  (radius/r_0)**2.0 )**(5.0/2.0)

    return density_out
"""
if __name__ == "__main__":
    # check IC
    from gravity_tree import tree, build_tree_from_particles, compute_mass_on_tree, compute_potential_tree, compute_acceleration_tree
    from tree_module.part_2_tree import get_level_per_particle
    from gravity_bruteforce import compute_potential

    # units
    unit_m = constants.M_sun
    unit_l = constants.pc
    unit_t = (constants.G * unit_m / unit_l ** 3) ** (-0.5)

    # system setup
    min_star_radius = 100  # pc
    M_star_tot_msun = 1e10  # msun
    r_eff_star_pc = 2000.0  # pc
    N_star_pt = int(1e2)
    n_sersic = 1
    h_scale_star = 40  # pc

    M_dm_tot_msun = 1e12  # msun
    r_eff_dm_pc = 200e3  # pc
    N_dm_pt = int(1e2)
    min_radius_nfw = 100  # pc
    concentration = 10  # concentration parameter

    mass_particle_star = M_star_tot_msun / N_star_pt
    mass_particle_dm = M_dm_tot_msun / N_dm_pt
    print("-----------")
    print("generate IC ")
    print("-----------")
    print("  n particles  ", N_star_pt)
    print("  total    mass", M_star_tot_msun, 'Msun')
    print("  particle mass", mass_particle_star, 'Msun')

    # analytical profile
    n_bins = 10000
    rr_star_pc = np.linspace(min_star_radius, 4.0 * r_eff_star_pc, n_bins)
    density_serisc = get_sersic_density(radius=rr_star_pc, M_tot=M_star_tot_msun, r_0=r_eff_star_pc, n_sersic=n_sersic)

    rr_dm_pc = np.linspace(min_radius_nfw, r_eff_dm_pc, n_bins)
    density_nfw = get_nfw_density(radius=rr_dm_pc, M_tot=M_dm_tot_msun, r_0=r_eff_dm_pc)

    # set bound
    sigma_bound_star = 3000
    AA = get_sersic_density(radius=min_star_radius, M_tot=M_star_tot_msun, r_0=r_eff_star_pc,
                            n_sersic=n_sersic) / get_gaussian(
        xx=min_star_radius, sigma_0=sigma_bound_star, mean=0.0)
    test_bound = AA * get_gaussian(xx=rr_star_pc, sigma_0=sigma_bound_star, mean=0.0)

    sigma_bound_dm = 10000
    AA_1 = get_nfw_density(radius=min_radius_nfw, M_tot=M_dm_tot_msun, r_0=r_eff_dm_pc) / get_gaussian(
        xx=min_radius_nfw, sigma_0=sigma_bound_dm, mean=0.0)
    test_bound_1 = AA_1 * get_gaussian(xx=rr_dm_pc, sigma_0=sigma_bound_dm, mean=0.0)

    """Test"""

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(rr_star_pc, density_serisc, ls='-', color='k', label='sersic')
    ax.plot(rr_star_pc, test_bound, ls='-', label='bound for A/R sampling')
    plt.legend()

    # ax.set_xlim(0, 100)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(rr_dm_pc, density_nfw, ls='-', color='k', label='NFW')
    ax.plot(rr_dm_pc, test_bound_1, ls='-', label='bound for A/R sampling')
    plt.legend()
    # ax.set_xlim(0, 200)
    plt.show()

    # extract the disk
    r_extract_star = extract_radius_from_sersic(N_pt=N_star_pt, M_tot_msun=M_star_tot_msun, n_sersic=n_sersic,
                                                sigma_bound=sigma_bound_star, r_0_pc=r_eff_star_pc, i_iter_max=200000,
                                                min_star_radius=min_star_radius)
    r_extract_star = np.abs(r_extract_star)
    # extract the angle
    theta_star = extract_angles_from_circle(N_pt=N_star_pt)
    # extract the z position
    z_extract_star = extract_z_from_gaussian(N_pt=N_star_pt, scale_height=h_scale_star)
    x_extract_star, y_extract_star = convert_polar_to_cartesian(radius=r_extract_star, theta=theta_star)
    density_check_star, bin_edges_star = np.histogram(r_extract_star, bins=100, density=True)
    BB_star = get_sersic_density(radius=min_star_radius, M_tot=M_star_tot_msun, r_0=r_eff_star_pc, n_sersic=n_sersic) / \
              density_check_star[0]
    density_check_star = density_check_star * BB_star

    # extract the dm halo
    r_extract_dm = extract_radius_from_nfw(N_pt=N_dm_pt, M_tot_msun=M_dm_tot_msun, sigma_bound=sigma_bound_dm,
                                           r_0_pc=r_eff_dm_pc, i_iter_max=200000, min_radius=min_radius_nfw)
    r_extract_dm = np.abs(r_extract_dm)
    phi_dm, theta_dm = extract_angles_from_sphere(N_dm_pt)
    x_extract_dm, y_extract_dm, z_extract_dm = convert_spherical_to_cartesian(radius=r_extract_dm, theta=theta_dm,
                                                                             phi=phi_dm)
    density_check_dm, bin_edges_dm = np.histogram(r_extract_dm, bins=100, density=True)
    """
    plt.hist(r_extract_dm, alpha = 0.3)
    plt.hist(x_extract_dm, alpha = 0.3)
    plt.hist(y_extract_dm, alpha = 0.3)
    plt.hist(z_extract_dm, alpha = 0.3)

    plt.show()
    """
    BB_dm = get_nfw_density(radius=min_radius_nfw, M_tot=M_dm_tot_msun, r_0=r_eff_dm_pc) / density_check_dm[0]
    density_check_dm = density_check_dm * BB_dm

    # ----------
    # check IC for stars
    # ----------

    r_cut_star = 4.0 * r_eff_star_pc
    mask_star = r_extract_star < r_cut_star
    N_pt_cut_star = len(r_extract_star[mask_star])

    pos_normed_star = np.zeros((3, N_pt_cut_star))
    pos_normed_star[0, :] = x_extract_star[mask_star]
    pos_normed_star[1, :] = y_extract_star[mask_star]
    pos_normed_star[2, :] = z_extract_star[mask_star]
    pos_normed_star[:, :] = pos_normed_star[:, :] / (2 * r_cut_star)

    mass_norm_star = np.ones(N_pt_cut_star) * mass_particle_star

    n_grid = 10000
    n_part_per_cell = 8
    n_dim = 3

    # ----------
    # check IC for dm halo
    # ----------

    r_cut_dm = 4.0 * r_eff_dm_pc
    mask_dm = r_extract_dm < r_cut_dm
    N_pt_cut_dm = len(r_extract_dm[mask_dm])

    pos_normed_dm = np.zeros((3, N_pt_cut_dm))
    pos_normed_dm[0, :] = x_extract_dm[mask_dm]
    pos_normed_dm[1, :] = y_extract_dm[mask_dm]
    pos_normed_dm[2, :] = z_extract_dm[mask_dm]
    pos_normed_dm[:, :] = pos_normed_dm[:, :] / (2 * r_cut_dm)

    mass_norm_dm = np.ones(N_pt_cut_dm) * mass_particle_dm

    # get particle level using a tree
    comp_tree = tree(n_dim=n_dim, n_grid=n_grid, n_part_per_cell=n_part_per_cell, mass_pt=mass_norm_star,
                     pos_pt=pos_normed_star)
    comp_tree = build_tree_from_particles(tree_in=comp_tree, i_iter_max=10000)
    particle_level = get_level_per_particle(tree_in=comp_tree, i_iter_max=10000)

    # estimate density for each particle
    # should use part in that cell, not the max
    particle_density = n_part_per_cell * mass_particle_star / ((2 * r_cut_star) * 0.5 ** particle_level) ** 3

    # estimate free fall time
    t_ff = 1.0 / np.sqrt(constants.G * particle_density * unit_m / unit_l ** 3)

    print("----------------")
    print("estimate CPUtime")
    print("----------------")

    print("assuming a tree with  ", n_part_per_cell, "max n_part_per_cell")
    print("  min particle level  ", np.max(particle_level))
    print("  min particle level  ", np.min(particle_level))

    # estimate CPU
    dt_myr = 0.01 * np.min(t_ff.to('Myr').value)
    t_end_myr = 50.0
    n_time_steps = t_end_myr / dt_myr

    print("using free fall for dt estimate")
    print("  max t free fall     ", np.max(t_ff.to('Myr')))
    print("  min t free fall     ", np.min(t_ff.to('Myr')))
    print("simulation time       ", t_end_myr, "Myr")
    print("using min(t_tp) for the time step")
    print('expected n steps      ', n_time_steps)

    __ = compute_potential(masses=mass_norm_star[:4], positions=pos_normed_star[:, :4], G_gravity=1.0)

    t_start = time()
    __ = compute_potential(masses=mass_norm_star[:], positions=pos_normed_star[:, :], G_gravity=1.0)

    t_end = time()
    cpu_time_dt = t_end - t_start
    print('assuming N^2 potential computation for dt cost')
    print('estimated CPUtime     ', cpu_time_dt * n_time_steps / 3600, 'hr')
	
    t_start = time()
    acceleration = compute_acceleration_tree(tree_in=comp_tree,critical_angle=0.1, G_gravity= 1.0, n_iter_max=10000, verbose= False)
    t_end = time()
    cpu_time_dt = t_end - t_start
    print('assuming tree potential computation for dt cost')
    print('estimated CPUtime     ', cpu_time_dt * n_time_steps / 3600, 'hr')
	
    # ----------------------------------------
    # setting up initial velocity of the stars
    # ----------------------------------------
    
    v_r_star, v_theta_star, v_phi_star = initial_velocity(N_star_pt, r_extract_star, r_vir=r_eff_dm_pc,
                                                          reff=r_eff_star_pc, M_tot_dm=M_dm_tot_msun,
                                                          M_tot_star=M_star_tot_msun, n_sersic=n_sersic,
                                                          concentration=concentration,
                                                          velocity_dispersion=5)
    vx_star, vy_star, vz_star = convert_spherical_to_cartesian(radius=v_r_star,phi=v_phi_star,theta=v_theta_star)
	
    v_tot_star = np.zeros((3, N_pt_cut_star))
    v_tot_star[0, :] = vx_star[mask_star]
    v_tot_star[1, :] = vy_star[mask_star]
    v_tot_star[2, :] = vz_star[mask_star]
    
    print("min v")
    print(np.nanmin(v_theta_star))

    plt.plot(r_extract_star, v_theta_star, ls="", marker="o", color="black", alpha=0.2)

    # ----------------------------------------
    # setting up initial velocity of DM
    # ----------------------------------------

    v_r_dm, v_theta_dm, v_phi_dm = initial_velocity(N_dm_pt, r_extract_dm, r_vir=r_eff_dm_pc, reff=r_eff_star_pc,
                                                    M_tot_dm=M_dm_tot_msun, M_tot_star=M_star_tot_msun,
                                                    n_sersic=n_sersic, concentration=concentration,
                                                    velocity_dispersion=0)
    vx_dm, vy_dm, vz_dm = convert_spherical_to_cartesian(radius=v_r_dm,phi=v_phi_dm,theta=v_theta_dm)
    v_tot_dm = np.zeros((3, N_pt_cut_dm))
    v_tot_dm[0, :] = vx_dm[mask_dm]
    v_tot_dm[1, :] = vy_dm[mask_dm]
    v_tot_dm[2, :] = vz_dm[mask_dm]
    
    plt.plot(r_extract_dm, v_theta_dm, ls="", marker="o", color="red", alpha=0.2)

    # expected velocity field
    v_star = velocity_sersic(100, rr_dm_pc, reff=r_eff_star_pc, M_tot_msun=M_star_tot_msun, n_sersic=1, )
    v_dm = velocity_nfw(100, rr_dm_pc, r_vir=r_eff_dm_pc, M_tot_msun=M_dm_tot_msun, concentration=concentration)
    v_circ = np.sqrt(v_star ** 2 + v_dm ** 2)
    plt.plot(rr_dm_pc, v_star, ls="--", color="green")
    plt.plot(rr_dm_pc, v_dm, ls="--", color="blue")
    plt.plot(rr_dm_pc, v_circ, ls="--", color="magenta")
    plt.ylim(-1, 600)
    plt.xlabel("radius [pc]")
    plt.ylabel("circular velocity [km/s]")
    plt.title("Circular velocity")
    plt.show()

    # --------------
    # check density star
    # --------------

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(rr_star_pc, density_serisc, ls='-', color='k', label='Sersic')
    ax.plot(rr_star_pc, test_bound, ls='-', label='bound for A/R sampling')
    ax.plot(0.5 * (bin_edges_star[0:-1] + bin_edges_star[1:]), density_check_star, label='extraction')
    ax.axvline(r_eff_star_pc, ls='--', color='k')
    ax.legend(frameon=False)
    ax.set_ylabel(r'$\rho/M_\odot \, \rm pc^{-3}$')
    ax.set_xlabel(r'$r \, \rm pc$')
    plt.tight_layout()
    plt.show()

    # --------------
    # check density dm
    # --------------

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(rr_dm_pc, density_nfw, ls='-', color='k', label='NFW')
    ax.plot(rr_dm_pc, test_bound_1, ls='-', label='bound for A/R sampling')
    ax.plot(0.5 * (bin_edges_dm[0:-1] + bin_edges_dm[1:]), density_check_dm, label='extraction')
    ax.axvline(r_eff_dm_pc, ls='--', color='k')
    ax.legend(frameon=False)
    ax.set_ylabel(r'$\rho/M_\odot \, \rm pc^{-3}$')
    ax.set_xlabel(r'$r \, \rm pc$')
    plt.tight_layout()
    plt.show()
    
    # ---------------------------------------
	# Checking 3D distribution with HDF5
	# ---------------------------------------
		
    box_left_edge   = np.min(pos_normed_star)/8 
    box_right_edge  = np.max(pos_normed_star)/8
    N_grid  = 128
    
    print("Projection using cloud in cell")
    projected_field = projection(box_left_edge=box_left_edge,box_right_edge=box_right_edge,N_grid=N_grid,pos_array=pos_normed_star,field_array=v_theta_star,normalization=1.0)
    print("Projection completed! snapshot saved!")
    print("min projected density = ",np.min(projected_field),"max projected_field = ",np.max(projected_field))
    
    #h = h5py.File('star_velocity_theta.h5', 'w')
    #dset = h.create_dataset('velocity', data=projected_field)
    
    # --------------
    # check 3D distribution
    # --------------

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(x_extract_star, y_extract_star, z_extract_star, marker='x', ls='', alpha=0.1, color="k")

    ax.plot(x_extract_dm, y_extract_dm, z_extract_dm, marker='x', ls='', alpha=0.1, color="r")

    plt.tight_layout()

    plt.show()

    # -------
    # plot density
    # ---------
    from matplotlib import colors

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist2d(x_extract_star, y_extract_star, bins=[200, 200], norm=colors.LogNorm())
    # ax.hist(y_extract)
    plt.tight_layout()

    plt.show()
    
    # ------------------------------------
    # Evolving for the first few time step
    # ------------------------------------
    
    print("Evolution started!!")
    
    dt_myr     = 1
    t_end_myr  = 2.0 
    f_snap_myr = 1.0 
    
    # get particle level using a tree
    comp_tree = tree(n_dim=n_dim, n_grid=n_grid, n_part_per_cell=n_part_per_cell, mass_pt=mass_norm_star, pos_pt=pos_normed_star)
    
    comp_tree = build_tree_from_particles(tree_in=comp_tree, i_iter_max=10000)
    acceleration_stars = compute_acceleration_tree(tree_in=comp_tree,critical_angle=0.1, G_gravity= 1.0, n_iter_max=10000, verbose= False)
    t_now, i_iter, pt_mass, pt_pos, pt_vel, pt_acc, pos_snap, vel_snap, e_snap , t_snap = evolve_to_t_end_galaxy(t_end=t_end_myr, pt_mass=mass_norm_star, pt_pos=pos_normed_star, pt_vel=v_tot_star, pt_acc=acceleration_stars, comp_tree=comp_tree, G_gravity=1.0, max_iterations=1000, n_snap=1000, f_snap=f_snap_myr, eta_time=dt_myr)
    
    # --------------
    # check 3D distribution
    # --------------

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(x_extract_star, y_extract_star, z_extract_star, marker='x', ls='', alpha=0.1, color="k")

    ax.plot(x_extract_dm, y_extract_dm, z_extract_dm, marker='x', ls='', alpha=0.1, color="r")

    plt.tight_layout()

    plt.show()

    
