import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from astropy import units, constants

import matplotlib as mpl

mpl.use('macosx')

from time import time

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
    norm = 3.0 * M_tot / (4.0 * np.pi * r_0 ** 3)
    b = 2 * n_sersic - 1 / 3 + 0.009876 / n_sersic  # Prugniel & Simien 1997
    density_out = norm * np.exp(-b * (radius / r_0) ** (1 / n_sersic))

    return density_out


def extract_radius_from_sersic(N_pt, M_tot_msun=1.e5, sigma_bound=200, r_0_pc=100.0, n_sersic=1, i_iter_max=100000):
    AA = get_sersic_density(radius=0.0, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic) / get_gaussian(xx=0.0,
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
        f_of_x_0 = get_gaussian(xx=x_0, sigma_0=sigma_bound, mean=0.0)
        f_of_x_0 = AA * f_of_x_0

        # second extraction
        mm = f_of_x_0 * np.random.random()

        # compare
        accept = mm <= get_sersic_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic)
        print(mm, get_sersic_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc, n_sersic=n_sersic), accept)

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


def extract_radius_from_nfw(N_pt, M_tot_msun=1.e5, sigma_bound=200, r_0_pc=100.0, i_iter_max=100000, min_radius=1e-10):
    AA = get_nfw_density(radius=min_radius, M_tot=M_tot_msun, r_0=r_0_pc) / get_gaussian(xx=min_radius,
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
        f_of_x_0 = get_gaussian(xx=x_0, sigma_0=sigma_bound, mean=0.0)
        f_of_x_0 = AA * f_of_x_0

        # second extraction
        mm = f_of_x_0 * np.random.random()

        # compare
        accept = mm <= get_nfw_density(radius=x_0, M_tot=M_tot_msun, r_0=r_0_pc)

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

    print("extracting particles with gaussian density with A/R")

    while i_pt < N_pt:

        assert i_iter < i_iter_max

        # extraction from the boundary function
        x_0 = np.random.normal(loc=0.0, scale=sigma_bound)
        f_of_x_0 = get_gaussian(xx=x_0, sigma_0=sigma_bound, mean=0.0)
        f_of_x_0 = AA * f_of_x_0

        # second extraction
        mm = f_of_x_0 * np.random.random()

        # compare
        accept = mm <= get_gaussian(xx=x_0, sigma_0=scale_height, mean=0.0)

        if accept:
            out[i_pt] = x_0
            i_pt = i_pt + 1

        i_iter = i_iter + 1

    print("  efficiency   ", 100.0 * float(N_pt) / float(i_iter), "%")

    return out


def extract_angles_from_sphere(N_pt):
    # d Omega = d phi d cos(theta)
    phi = 2.0 * np.pi * np.random.random(size=N_pt)
    theta = np.arccos(2.0 * np.random.random(size=N_pt) - 1)

    return phi, theta


def extract_angles_from_circle(N_pt):
    theta = 2.0 * np.pi * np.random.random(size=N_pt)

    return theta


def covert_spherical_to_cartesian(radius, phi, theta):
    xx = radius * np.cos(phi) * np.sin(theta)
    yy = radius * np.sin(phi) * np.sin(theta)
    zz = radius * np.cos(theta)

    return xx, yy, zz


def covert_polar_to_cartesian(radius, theta):
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
    from gravity_tree import tree, build_tree_from_particles, compute_mass_on_tree, compute_potential_tree
    from tree_module.part_2_tree import get_level_per_particle
    from gravity_bruteforce import compute_potential

    # units
    unit_m = constants.M_sun
    unit_l = constants.pc
    unit_t = (constants.G * unit_m / unit_l ** 3) ** (-0.5)

    # system setup
    M_star_tot_msun = 1e10  # msun
    r_eff_star_pc = 2000.0  # pc
    N_star_pt = int(1e4)
    n_sersic = 1
    h_scale_star = 40  # pc

    M_dm_tot_msun = 1e12  # msun
    r_eff_dm_pc = 200e3  # pc
    N_dm_pt = int(1e4)
    min_radius_nfw = 10  # pc

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
    rr_star_pc = np.linspace(0.0, 4.0 * r_eff_star_pc, n_bins)
    density_serisc = get_sersic_density(radius=rr_star_pc, M_tot=M_star_tot_msun, r_0=r_eff_star_pc, n_sersic=n_sersic)

    rr_dm_pc = np.linspace(min_radius_nfw, r_eff_dm_pc, n_bins)
    density_nfw = get_nfw_density(radius=rr_dm_pc, M_tot=M_dm_tot_msun, r_0=r_eff_dm_pc)

    # set bound
    sigma_bound_star = 3000
    AA = get_sersic_density(radius=0.0, M_tot=M_star_tot_msun, r_0=r_eff_star_pc, n_sersic=n_sersic) / get_gaussian(
        xx=0.0, sigma_0=sigma_bound_star, mean=0.0)
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
                                                sigma_bound=sigma_bound_star, r_0_pc=r_eff_star_pc, i_iter_max=100000)
    r_extract_star = np.abs(r_extract_star)
    # extract the angle
    theta_star = extract_angles_from_circle(N_pt=N_star_pt)
    #extract the z position
    z_extract_star = extract_z_from_gaussian(N_pt=N_star_pt, scale_height=h_scale_star)
    x_extract_star, y_extract_star = covert_polar_to_cartesian(radius=r_extract_star, theta=theta_star)
    density_check_star, bin_edges_star = np.histogram(r_extract_star, bins=100, density=True)
    BB_star = get_sersic_density(radius=0.0, M_tot=M_star_tot_msun, r_0=r_eff_star_pc, n_sersic=n_sersic) / \
              density_check_star[0]
    density_check_star = density_check_star * BB_star

    # extract the dm halo
    r_extract_dm = extract_radius_from_nfw(N_pt=N_dm_pt, M_tot_msun=M_dm_tot_msun, sigma_bound=sigma_bound_dm,
                                           r_0_pc=r_eff_dm_pc, i_iter_max=100000, min_radius=min_radius_nfw)
    r_extract_dm = np.abs(r_extract_dm)
    phi_dm, theta_dm = extract_angles_from_sphere(N_dm_pt)
    x_extract_dm, y_extract_dm, z_extract_dm = covert_spherical_to_cartesian(radius=r_extract_dm, theta=theta_dm,
                                                                             phi=phi_dm)
    density_check_dm, bin_edges_dm = np.histogram(r_extract_dm, bins=1000, density=True)
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

    ax.plot(rr_dm_pc,density_nfw,ls='-',color='k',label='NFW')
    ax.plot(rr_dm_pc,test_bound_1, ls='-',label='bound for A/R sampling')
    ax.plot(0.5 * (bin_edges_dm[0:-1] + bin_edges_dm[1:]), density_check_dm, label='extraction')
    ax.axvline(r_eff_dm_pc, ls='--', color='k')
    ax.legend(frameon=False)
    ax.set_ylabel(r'$\rho/M_\odot \, \rm pc^{-3}$')
    ax.set_xlabel(r'$r \, \rm pc$')
    plt.tight_layout()
    plt.show()

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
