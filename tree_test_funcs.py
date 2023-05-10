import matplotlib.pyplot as plt
from gravity_tree import *

def normalize_field(n_dim, test_pos):
    """
    Returns normalized field and normalization factor
    """
    norm_vector = np.ones(n_dim)
    for i_dim in range(n_dim):
        norm_vector[i_dim] = np.abs(np.max(test_pos[i_dim,:]) - np.min(test_pos[i_dim,:]))
        test_pos[i_dim,:] = (test_pos[i_dim,:]-np.min(test_pos[i_dim,:]))\
            /(np.max(test_pos[i_dim,:])-np.min(test_pos[i_dim,:])) - 0.5
    return test_pos, norm_vector

def init_particle_field(n_part_per_cell=6, n_dim=2, n_grid=200, iter_max=200, n_pt=320):

    #--------------------------------
    # set up particles
    #--------------------------------
    test_pos  = np.zeros((n_dim,n_pt))
    test_mass = np.ones(n_pt)

    for i_dim in range(n_dim):
        test_pos[i_dim,:] = np.random.normal(loc=0.0,scale=1.0, size=n_pt)

    test_pos, norm_vector = normalize_field(n_dim, test_pos)
    return test_pos, test_mass, n_part_per_cell, n_dim, n_grid, iter_max, n_pt, norm_vector

def init_trees(n_dim, n_grid, n_part_per_cell, test_mass, test_pos, iter_max, critical_angle):
    # produce test tree with masses
    test_tree = tree(n_dim=n_dim, n_grid=n_grid, n_part_per_cell=n_part_per_cell, \
                    mass_pt=test_mass, pos_pt=test_pos)
    test_tree   = build_tree_from_particles(tree_in=test_tree, i_iter_max=iter_max)
    test_tree   = compute_mass_on_tree(tree_in=test_tree)

    potential_tree = compute_potential_tree(tree_in=test_tree,critical_angle=0.1)
    
    return test_tree, potential_tree

def plot_tree(test_tree, test_pos, norm_vector):
    #--------------------------------
    # visualize the tree
    #--------------------------------
    fig = plt.figure()
    ax  = fig.add_subplot(111)

    for i_cell in range(test_tree.n_now):

        col    = 'k'
        zorder = 0

        dx    = 0.5**(test_tree.levels[i_cell]+1) * norm_vector[0]
        dy    = 0.5**(test_tree.levels[i_cell]+1) * norm_vector[1]

        tree_center_x = test_tree.center[0] - test_tree.center[0].mean()
        tree_center_y = test_tree.center[1] - test_tree.center[1].mean()
        # determination of the edges of the square can be cleaned
        xedge = [tree_center_x[i_cell]*norm_vector[0] - dx, tree_center_x[i_cell]*norm_vector[0] + dx,\
                tree_center_x[i_cell]*norm_vector[0] + dx,\
                tree_center_x[i_cell]*norm_vector[0] - dx,\
                tree_center_x[i_cell]*norm_vector[0] - dx]
        yedge = [tree_center_y[i_cell]*norm_vector[1] + dy, tree_center_y[i_cell]*norm_vector[1] + dy,\
                tree_center_y[i_cell]*norm_vector[1] - dy,\
                tree_center_y[i_cell]*norm_vector[1] - dy,
                tree_center_y[i_cell]*norm_vector[1] + dy]
        
        ax.plot(tree_center_x[i_cell]*norm_vector[0], tree_center_y[i_cell]*norm_vector[1], marker='o', color=col, zorder=zorder)

        ax.plot(xedge,yedge,ls='-',marker='',color=col,zorder=zorder)

    ax.plot((test_pos[0,:]-test_pos[0,:].mean())*norm_vector[0], \
            (test_pos[1,:]-test_pos[1,:].mean())*norm_vector[1], ls='', marker='x', color='b', alpha=0.6)
    
def normal_to_physical(n_dim, acceleration_tree, test_pos, norm_vector):
    for i_dim in range(n_dim):
            acceleration_tree[i_dim] /= norm_vector[i_dim]**2
            test_pos[i_dim] *= norm_vector[i_dim]
            test_pos[i_dim] -= test_pos[i_dim].mean()
    return acceleration_tree, test_pos