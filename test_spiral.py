from spiral import *

if __name__=="__main__":

    """
    Spirals are produced by generating a number of ellipses
    with 
    1. increasing semi-major axis
    2. regular angle offset with respect to the previous ellipse
    3. evolving eccentricity, from 0 to inner eccentricity, to
    outer eccentricity, back to 0
    """

    # init galaxy parameters
    CORE_RAD     = 5      # radius of core
    GAL_RAD      = 20     # radius of visible spiral arms
    DIST_RAD     = 70     # end radius of galaxy
    INNER_E      = 0.4    # inner eccentricity
    OUTER_E      = 0.9    # outer eccentricity
    ANG_OFF      = 4      # 'coiledness' of spiral arms
    N_ELLIPSES   = 1000   # number of ellipses to produce

    # make ellipses
    ellipses = make_ellipses(core_radius=CORE_RAD, galaxy_radius=GAL_RAD, \
                                distant_radius=DIST_RAD, inner_eccentricity=INNER_E, \
                                outer_eccentricity=OUTER_E, angular_offset=ANG_OFF, \
                                n_ellipses=N_ELLIPSES)
    
    # make galaxy from ellipses
    gxy = make_galaxy(ellipses)

    # save galaxy for reference
    save_galaxy(gxy)

    # load saved galaxy
    galaxy = load_galaxy()

    # replace this with sersic profile
    r_sample = np.random.exponential(GAL_RAD, int(1e4))

    # map sampled radii to galaxy
    x,y,vx,vy = map_r_to_galaxy(r_sample, galaxy)

    # plot galaxy
    plt.plot(x, y, 'b.', alpha=0.05)
    plt.xlim(-70, 70)
    plt.ylim(-70, 70)
    plt.gca().set_aspect('equal')
    plt.title("Star Particles", fontsize=20)
    plt.ylabel("Distance [kpc]", fontsize=15)
    plt.xlabel("Distance [kpc]", fontsize=15)
    plt.show()