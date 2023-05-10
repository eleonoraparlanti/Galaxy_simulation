import numpy as np
import matplotlib.pyplot as plt


def eccentricity(r, core_radius=5, galaxy_radius=20, distant_radius=70, inner_eccentricity=0.5, outer_eccentricity=0.99):

    # Core region of the galaxy. Innermost part is round
    # eccentricity increasing linear to the border of the core.
    if  r < core_radius:
        return (r / core_radius) * inner_eccentricity

    elif r > core_radius and r <= galaxy_radius:
        return (outer_eccentricity - inner_eccentricity) \
                * (r - core_radius) / (galaxy_radius - core_radius) \
                + inner_eccentricity

    # eccentricity is slowly reduced to 0.
    elif r > galaxy_radius and r < distant_radius:
        return -1*outer_eccentricity \
                * (r - galaxy_radius) / (distant_radius - galaxy_radius) \
                + outer_eccentricity

    else:
        return 0
        

def ellipse_radius(theta, eccentricity=0.5, a=1):
    if eccentricity == 0:
        eccentricity = 1e-5
    return a/np.sqrt(np.cos(theta)**2+(np.sin(theta)**2)/(1-(eccentricity**2)))


def ellipse_slope(x, a, e):
    if x == a:
        return -1*np.inf
    elif x == -1*a:
        return np.inf
    else:
        return -1*x*np.sqrt((1-e**2)/(1-(x/a)**2))
    

def v_r(r):
    """
    Velocity of a star particle at a given radius.
    This is just a placeholder, the quantity should be
    calculated using the DM, stellar, and gas distribution.
    """
    return np.sqrt(np.log(1+r) - (1+r**(-1))**(-1))


def make_ellipses(core_radius, galaxy_radius, distant_radius, \
                inner_eccentricity, outer_eccentricity, angular_offset,\
                n_ellipses):
    
    # initialize tracked parameters
    rotate = 0
    ellipses = []
    a_range = np.linspace(1e-2, distant_radius, n_ellipses)
    theta = np.linspace(0, 2*np.pi-1e-3, n_ellipses)

    angular_offset *= np.pi/(1.8 * n_ellipses)
    
    for a in a_range:
        ecc = eccentricity(a, core_radius=core_radius, galaxy_radius=galaxy_radius,\
                           distant_radius=distant_radius, inner_eccentricity=inner_eccentricity,\
                           outer_eccentricity=outer_eccentricity)
        r = ellipse_radius(theta, eccentricity=ecc, a=a)

        x0 = r*np.cos(theta)
        y0 = r*np.sin(theta)

        # calculation of velocity distribution
        vr = v_r(r)

        x = x0*np.cos(rotate) - y0*np.sin(rotate)
        y = x0*np.sin(rotate) + y0*np.cos(rotate)

        dx = x - np.roll(x, -1)
        dy = y - np.roll(y, -1)
        phi = np.arctan(dy / dx)
        iota = np.pi/2 - np.arctan(np.abs(y)/np.abs(x))\
            - np.arctan(np.abs(dy)/np.abs(dx))
        va = vr/np.cos(iota)
        vx = va*np.cos(phi)
        vy = va*np.sin(phi)
        vx = np.abs(vx)
        vy = np.abs(vy)
        vx[dx<0] *= -1
        vy[dy<0] *= -1

        ellipses += [(x, y, vx, vy)]

        rotate += angular_offset
    
    return ellipses

def make_galaxy(ellipses):
    ex   = np.asarray([e[0] for e in ellipses]).flatten()
    ey   = np.asarray([e[1] for e in ellipses]).flatten()
    evx   = np.asarray([e[2] for e in ellipses]).flatten()
    evy   = np.asarray([e[3] for e in ellipses]).flatten()
    er   = np.sqrt(ex**2+ey**2)
    et   = np.arctan(ey/ex)
    glxy = np.stack([er, et, ex, ey, evx, evy])
    return glxy

def save_galaxy(galaxy, fn='./galaxy.npy'):
    np.savetxt(fn, galaxy)

def load_galaxy(fn='./galaxy.npy'):
    return np.loadtxt(fn)

def map_r_to_galaxy(radii, galaxy):
    """
    Takes a sequence of radii and maps them to point in the 
    galaxy matrix of nearest radii. This roughly maps to the
    correct distribution in theta while preserving
    the radial distribution, assuming a sufficient number
    of ellipses are used in generating the galaxy.

    Exceptions are made for points in the galactic core and
    beyond the spiral arms. In these regions, angles are
    drawn from a uniform distribution.

    NOTE core (min_r) is hard-set to 50.
    """
    idx = []
    er = np.asarray(galaxy[0])
    ex = np.asarray(galaxy[2])
    ey = np.asarray(galaxy[3])
    evx = np.asarray(galaxy[4])
    evy = np.asarray(galaxy[5])
    max_r = np.max(er)
    min_r = 50
    r_in_galaxy = radii[radii<max_r]
    r_in_galaxy = r_in_galaxy[r_in_galaxy>min_r]
    r_out_galaxy = np.concatenate([radii[radii>=max_r], \
                                   radii[radii<min_r]])
    for r in r_in_galaxy:
        idx += [np.argmin(abs(er - r))]
    mapped_x_in_galaxy = ex[idx]
    mapped_y_in_galaxy = ey[idx]
    mapped_vx_in_galaxy = evx[idx]
    mapped_vy_in_galaxy = evy[idx]
    theta_out_galaxy = np.random.uniform(size=len(r_out_galaxy))\
        *2*np.pi
    x_out_galaxy = r_out_galaxy * np.cos(theta_out_galaxy)
    y_out_galaxy = r_out_galaxy * np.sin(theta_out_galaxy)
    vx_out_galaxy = v_r(r_out_galaxy) * np.sin(theta_out_galaxy)
    vy_out_galaxy = -1 * v_r(r_out_galaxy) * np.cos(theta_out_galaxy)
    mapped_x = np.concatenate([mapped_x_in_galaxy, \
                              x_out_galaxy])
    mapped_y = np.concatenate([mapped_y_in_galaxy,\
                              y_out_galaxy])
    mapped_vx = np.concatenate([mapped_vx_in_galaxy, \
                              vx_out_galaxy])
    mapped_vy = np.concatenate([mapped_vy_in_galaxy,\
                              vy_out_galaxy])
    return mapped_x, mapped_y, mapped_vx, mapped_vy


if __name__=="__main__":

    # initialize variables

    CORE_RAD     = 100
    GAL_RAD      = 1000
    DIST_RAD     = 5000
    INNER_E      = 0.4
    OUTER_E      = 0.9
    ANG_OFF      = 4
    N_ELLIPSES   = 50

    ellipses = make_ellipses(core_radius=CORE_RAD, galaxy_radius=GAL_RAD, \
                             distant_radius=DIST_RAD, inner_eccentricity=INNER_E, \
                             outer_eccentricity=OUTER_E, angular_offset=ANG_OFF, \
                             n_ellipses=N_ELLIPSES)

    for x, y, vx, vy in ellipses:
        # print(x.max(), y.max(), vx.max(), vy.max())
        plt.plot(x, y, 'b-', alpha=0.02*(1000/N_ELLIPSES))
        # plt.plot(x, y, 'b.', alpha=0.002*(1000/N_ELLIPSES))
    # plt.xlim(-70, 70)
    # plt.ylim(-70, 70)
    # plt.gca().set_aspect('equal')
    # plt.title("Spiral Template", fontsize=20)
    # plt.ylabel("Distance [kpc]", fontsize=15)
    # plt.xlabel("Distance [kpc]", fontsize=15)
    # plt.show()

    ex = np.asarray([e[0] for e in ellipses]).flatten()
    ey = np.asarray([e[1] for e in ellipses]).flatten()

    er = np.sqrt(ex**2+ey**2)
    et = np.arctan(ey/ex)

    r_sample = np.random.exponential(GAL_RAD, int(1e4))

    r_in_galaxy = r_sample[r_sample<70]
    r_out_galaxy = r_sample[r_sample>=70]

    idx = []
    for r in r_in_galaxy:
        idx += [np.argmin(abs(er - r))]

    galaxy = make_galaxy(ellipses)

    x, y, vx, vy = map_r_to_galaxy(r_sample, galaxy)
    T_TOT = np.sum(vx[r_sample<70]**2 + vy[r_sample<70]**2)
    T_REF_TOT = np.sum(v_r(r_in_galaxy)**2)
    print("Kinetic Energy Surplus", 100*T_TOT/T_REF_TOT - 100, "%")

    # plt.plot(ex[idx], ey[idx], 'b.', alpha=0.05)
    # plt.plot(x, y, 'b.', alpha=0.05)
    plt.quiver(x, y, vx*100, vy*100, scale=np.ones(len(x))/5, angles='xy', units='xy', color='red', alpha=0.2)#, 'b.', alpha=0.05)
    plt.xlim(-500, 500)
    plt.ylim(-500, 500)
    plt.gca().set_aspect('equal')
    plt.title("Star Particles", fontsize=20)
    plt.ylabel("Distance [kpc]", fontsize=15)
    plt.xlabel("Distance [kpc]", fontsize=15)
    plt.show()

    bins = np.histogram(np.hstack((er[idx], r_in_galaxy)), bins=100)[1] #get the bin edges
    P = np.histogram(er[idx], bins, density=True)[0]
    Q = np.histogram(r_in_galaxy, bins, density=True)[0]

    print("KLD (radius)", np.sum(P[P>0]*np.log10(P[P>0]/Q[P>0])))

    rt = np.random.uniform(size=100000)*np.pi - np.pi/2
    bins = np.histogram(np.hstack((et[idx], rt)), bins=100)[1] #get the bin edges
    P = np.histogram(et[idx], bins, density=True)[0]
    Q = np.histogram(rt, bins, density=True)[0]

    print("KLD (theta)", np.sum(P[P>0]*np.log10(P[P>0]/Q[P>0])))