
"""
# starting point for pure python implementation
class node:
    def __init__(self):
        self.level    = 0
        self.parent   = None
        self.children = [None,None,None,None]

"""

from numba import int64   as NB_INT
from numba import float64 as NB_FLOAT
from numpy import int64   as NP_INT
from numpy import float64 as NP_FLOAT

import numpy as np
from numba.experimental import jitclass
from numba import njit

from tree_module.tree_structure import is_leaf, split_target_oct
from tree_module.tree_info import get_structure
from tree_module.part_2_tree import get_leaf_cell_id, is_position_in_cell_i, node_need_to_be_opened

spec_tree = {
               "n_dim"           : NB_INT
              ,"n_children"      : NB_INT
              ,"n_max"           : NB_INT
              ,"n_now"           : NB_INT
              ,"n_part_per_cell" : NB_INT
              ,"geometry"        : NB_FLOAT[:,:]
              # tree structure
              ,"levels"          : NB_INT[:]
              ,"parents"         : NB_INT[:]
              ,"childrens"       : NB_INT[:,:]
              # tree data
              ,"center"          : NB_FLOAT[:,:]
              ,"mass_moment_0"   : NB_FLOAT[:]
              ,"mass_moment_1"     : NB_FLOAT[:,:]
              # part 2 tree arrays
              ,"id_parts"        : NB_INT[:,:]
              ,"n_parts"         : NB_INT[:]
              # particle data
              ,"n_pt"            : NB_INT
              ,"mass_pt"         : NB_FLOAT[:]
              ,"pos_pt"          : NB_FLOAT[:,:]
            }

@jitclass(spec_tree)
class tree:

    def __init__(self,n_dim=2,n_grid = 10,n_part_per_cell=1, mass_pt=np.zeros(2),pos_pt=np.zeros((2,2))):

        self.n_dim           = n_dim
        self.n_children      = int(2**n_dim)
        self.n_max           = 1+self.n_children*n_grid                               # max number of octs
        self.n_now           = 0                                                      # n octs currently stored
        self.n_part_per_cell = n_part_per_cell                                        # max n particle per node
        self.geometry        = np.zeros((self.n_dim,self.n_children),dtype= NP_FLOAT)

        self.levels    = np.zeros(self.n_max,dtype=NP_INT)
        self.parents   = np.zeros(self.n_max,dtype=NP_INT)                            # "pointer" to the parent
        self.childrens = np.zeros((self.n_children,self.n_max),dtype=NP_INT)          # "pointers" to the children
   
        self.center       = np.zeros((self.n_dim,self.n_max),dtype= NP_FLOAT)         # center of the cell
        #                                                                             # following sum are extend to the particle within the cell
        self.mass_moment_0= np.zeros(self.n_max,dtype= NP_FLOAT)                      # \sum_i m_i           , total mass within the cell
        self.mass_moment_1= np.zeros((self.n_dim,self.n_max),dtype= NP_FLOAT)         # \sum_i m_i \vec{x}_i , can be used to compute center of mass

        self.id_parts  = np.zeros((self.n_part_per_cell,self.n_max),dtype=NP_INT)     # "pointers" to the particles
        self.n_parts   = np.zeros(self.n_max,dtype=NP_INT)                            # n particles currently in a leaf

        self.n_pt      = len(mass_pt)                                                 # particels within the tree
        self.mass_pt   = np.zeros(self.n_pt,dtype= NP_FLOAT)                          # particles mass
        self.pos_pt    = np.zeros((self.n_dim,self.n_pt),dtype= NP_FLOAT)             # particels positions

        # set up geometry

        assert n_dim <=3, "geometry not implemented"
        
        if n_dim  == 1:
            self.geometry[0,0] = -0.5                    # first  child on the left
            self.geometry[0,1] = +0.5                    # second child on the right
        elif n_dim == 2:
            self.geometry[:,0] = np.array([-0.5, +0.5])  # first  child on the top    left
            self.geometry[:,1] = np.array([+0.5, +0.5])  # second child on the top    right
            self.geometry[:,2] = np.array([-0.5, -0.5])  # first  child on the bottom left
            self.geometry[:,3] = np.array([+0.5, -0.5])  # second child on the bottom right

        elif n_dim == 3:
            self.geometry[:, 0] = np.array([-0.5, +0.5, -0.5])
            self.geometry[:, 1] = np.array([+0.5, +0.5, -0.5])
            self.geometry[:, 2] = np.array([-0.5, -0.5, -0.5])
            self.geometry[:, 3] = np.array([+0.5, -0.5, -0.5])
            self.geometry[:, 4] = np.array([-0.5, +0.5, +0.5])
            self.geometry[:, 5] = np.array([+0.5, +0.5, +0.5])
            self.geometry[:, 6] = np.array([-0.5, -0.5, +0.5])
            self.geometry[:, 7] = np.array([+0.5, -0.5, +0.5])

        # store particle data
        
        assert self.n_dim == np.shape(pos_pt)[0]

        self.mass_pt[:]  = mass_pt[:]
        self.pos_pt[:,:] = pos_pt[:,:]

        # reset the tree
        
        self.levels[:]        = -1
        self.parents[:]       = -1
        self.childrens[:,:]   = -1
        #
        self.mass_moment_0[:] = 0.0
        self.mass_moment_1[:,:] = 0.0
        #        
        self.id_parts[:]      = -1

        # root initialization
        
        self.levels[0]   =  0
        self.parents[0]  = -1              # root has not parent
        self.center[:,0] = np.zeros(n_dim) # root should stay in the center of the box
        self.n_now       =  1



@njit
def compute_mass_on_tree(tree_in, i_iter_max=100):

  """
  once tree has bee build, particle mass is assigned to the tree
  """

  # assign mass to the leaves
  for i_pt in range(tree_in.n_pt):
    i_cell, __                    = get_leaf_cell_id(i_pt, tree_in, i_iter_max)
    tree_in.mass_moment_0[i_cell] = tree_in.mass_moment_0[i_cell] + tree_in.mass_pt[i_pt]
    tree_in.mass_moment_1[:,i_cell] = tree_in.mass_moment_1[:,i_cell] + tree_in.mass_pt[i_pt]*tree_in.pos_pt[:,i_pt]

  # get all leaf cells in the tree
  for i_cell in range(tree_in.n_now):
    if(is_leaf(i_cell,tree_in)):
      i_parent  = tree_in.parents[i_cell]
      i_current = i_cell
      # propagate from the leaf to the split cells
      while i_parent > 0:
        tree_in.mass_moment_0[i_parent] = tree_in.mass_moment_0[i_parent] +tree_in.mass_moment_0[i_current]
        tree_in.mass_moment_1[:,i_parent] = tree_in.mass_moment_1[:,i_parent] +tree_in.mass_moment_1[:,i_current]
        # go to the next parent
        i_current = i_parent
        i_parent  = tree_in.parents[i_parent]

  return tree_in
 

@njit
def build_tree_from_particles(tree_in,i_iter_max=100):

  """
  build tree on a particle by particle basis
  not super efficient at the moment e.g.:
   -- rassignment does not consider current particle position in a cell
   -- mass is not assigned while the tree is build (easier to implement with a level by level assigment)
  """

  # init ids for particles to be assigne
  ids_parts_to_assign     = np.zeros(tree_in.n_pt,dtype=NP_INT)
  ids_parts_to_assign_new = np.zeros(tree_in.n_pt,dtype=NP_INT)
  n_pt_to_assign          = tree_in.n_pt
  n_pt_to_assign_new      = tree_in.n_pt
  for i_pt in range(n_pt_to_assign):
    ids_parts_to_assign[i_pt] = i_pt

  i_iter = 0

  # loop on particles till there are some left to be assigned
  while n_pt_to_assign > 0:

    n_pt_to_assign_new = 0
    assert i_iter < i_iter_max

    # loop on the current particles
    for i_pt in range(n_pt_to_assign):
    
      id_to_assign = ids_parts_to_assign[i_pt]
      i_cell, __   = get_leaf_cell_id(id_to_assign, tree_in, i_iter_max=i_iter_max)
      n_current    = tree_in.n_parts[i_cell]

      # check if current leaf cell holds too many particles
      if n_current >= tree_in.n_part_per_cell:
        tree_in                = split_target_oct(i_target=i_cell, tree_in=tree_in)
        # current particle should be reassinged during the next sweep
        ids_parts_to_assign_new[n_pt_to_assign_new] = id_to_assign
        n_pt_to_assign_new                          = n_pt_to_assign_new +1
        # particles in the old leaf should be reassigned during the next sweep
        for j_pt in range(n_current):
          ids_parts_to_assign_new[n_pt_to_assign_new] = tree_in.id_parts[j_pt,i_cell]
          n_pt_to_assign_new                          = n_pt_to_assign_new +1
        # reset particle in the leaf
        tree_in.id_parts[:,i_cell]  = -1
        tree_in.n_parts[i_cell]     = 0

      else:
        # record the new particle id
        tree_in.id_parts[n_current,i_cell]   = id_to_assign
        tree_in.n_parts[i_cell]              = tree_in.n_parts[i_cell] + 1
   
    # update the particles that are left to be assigned
    ids_parts_to_assign[0:n_pt_to_assign_new] = ids_parts_to_assign_new[0:n_pt_to_assign_new]
    n_pt_to_assign                            = n_pt_to_assign_new

    i_iter = i_iter + 1

  return tree_in

@njit
def compute_potential_tree(tree_in,critical_angle=0.0, G_gravity= 1.0, n_iter_max=5000, verbose= False):

  out            = np.zeros(tree_in.n_pt)

  # temporary n_dim arrays
  center_of_mass = np.zeros(tree_in.n_dim)
  pos_i_part     = np.zeros(tree_in.n_dim)
  pos_j_part     = np.zeros(tree_in.n_dim)

  # lists
  id_cells_to_check     = np.zeros(tree_in.n_max,dtype=NP_INT)
  id_cells_to_check_new = np.zeros(tree_in.n_max,dtype=NP_INT)
  n_cells_to_check      = 0
  n_cells_to_check_new  = 0

  # counters
  n_iter_tot        = 0.0
  n_exact_tot       = 0.0
  n_approx_tot      = 0.0

  if verbose:
    print("calling compute_potential_tree()")
    print("  particle number            ",tree_in.n_pt)
    print("  number of dimensions       ",tree_in.n_dim)
    print("  number of cells            ",tree_in.n_now)
    print("  max level                  ",np.max(tree_in.levels))
    print("  opening angle              ",critical_angle)

  # loop on the particle
  for i_part in range(tree_in.n_pt):

    # pick a particle
    pos_i_part[:]        = tree_in.pos_pt[:,i_part]

    # init the pointers for the cells
    n_cells_to_check_new = 0
    n_cells_to_check     = 1
    id_cells_to_check[0] = 0

    # counters to check the efficiency
    i_exact              = 0 
    i_iter               = 0
    i_approx             = 0

    # while there are cells that can be opened/pruned
    while(n_cells_to_check > 0):
    
      assert i_iter< n_iter_max
    
      # loop on the current cells
      for ii in range(n_cells_to_check):

        i_iter           = i_iter + 1 
      
        # pick a cell
        i_current        = id_cells_to_check[ii]

        # check if we need to open the cell
        if node_need_to_be_opened(i_cell=i_current, pos_in= pos_i_part,tree_in=tree_in,critical_angle=critical_angle):
        
          # if it is a leaf, do a bruteforce computation
          if is_leaf(i_current,tree_in):
            # do a N^2 sum within the cell
            for jj in range(tree_in.n_parts[i_current]):
              j_part = tree_in.id_parts[jj,i_current]
              # avoid self potential
              if j_part != i_part:
                pos_j_part[:] = tree_in.pos_pt[:,j_part]
                dist          = np.sqrt(np.sum((pos_j_part[:]- pos_i_part[:])**2))
                out[i_part]   = out[i_part] - tree_in.mass_pt[j_part]/dist
                # update counter for exact computation
                i_exact       = i_exact + 1
          # if it is not a leaf, add all the childrens
          else:
            for i_child in range(tree_in.n_children):
              id_cells_to_check_new[n_cells_to_check_new + i_child] = tree_in.childrens[i_child,i_current]
            # update the new sets of pointers
            n_cells_to_check_new = n_cells_to_check_new + tree_in.n_children
        else:
          # prune out the branch (higher levels of the current cell)
          if(tree_in.mass_moment_0[i_current]>0):
            # compute center of mass and distance
            center_of_mass[:] = tree_in.mass_moment_1[:,i_current]/tree_in.mass_moment_0[i_current]
            dist              = np.sqrt(np.sum((center_of_mass[:] - pos_i_part[:])**2))
            # compute the potential with monopole approximation
            out[i_part]       = out[i_part] - tree_in.mass_moment_0[i_current]/dist
            
            i_approx          = i_approx + 1
            
      # loop on current cells is done
      # update the cell id list for next iteration
      id_cells_to_check[0:n_cells_to_check_new] = id_cells_to_check_new[0:n_cells_to_check_new]
      n_cells_to_check                          = n_cells_to_check_new
      # reset list for next cells
      n_cells_to_check_new                      = 0

    if verbose:
      n_iter_tot  = n_iter_tot   + float(i_iter)
      n_exact_tot = n_exact_tot  + float(i_exact)
      n_approx_tot= n_approx_tot + float(i_approx)
  
  if verbose:
    print("  average iterations         ",n_iter_tot/tree_in.n_pt)
    print("  average exact  calculations",n_exact_tot/tree_in.n_pt)
    print("  average approx calculations",n_approx_tot/tree_in.n_pt)
    print("  theoretical speedup        ",tree_in.n_pt**2/n_iter_tot)

  out[:] = out[:] * G_gravity

  return out

@njit
def compute_acceleration_tree(tree_in,critical_angle=0.0, G_gravity= 1.0, n_iter_max=5000, verbose= False):

  out            = np.zeros((tree_in.n_dim,tree_in.n_pt))

  # temporary n_dim arrays
  center_of_mass = np.zeros(tree_in.n_dim)
  pos_i_part     = np.zeros(tree_in.n_dim)
  pos_j_part     = np.zeros(tree_in.n_dim)
  pos_ij_part    = np.zeros(tree_in.n_dim)
  pos_cmi_part   = np.zeros(tree_in.n_dim)   ##cm = center of mass
  acc_j          = np.zeros(tree_in.n_dim)
  
  # lists
  id_cells_to_check     = np.zeros(tree_in.n_max,dtype=NP_INT)
  id_cells_to_check_new = np.zeros(tree_in.n_max,dtype=NP_INT)
  n_cells_to_check      = 0
  n_cells_to_check_new  = 0

  # counters
  n_iter_tot        = 0.0
  n_exact_tot       = 0.0
  n_approx_tot      = 0.0

  if verbose:
    print("calling compute_force_tree()")
    print("  particle number            ",tree_in.n_pt)
    print("  number of dimensions       ",tree_in.n_dim)
    print("  number of cells            ",tree_in.n_now)
    print("  max level                  ",np.max(tree_in.levels))
    print("  opening angle              ",critical_angle)

  # loop on the particle
  for i_part in range(tree_in.n_pt):

    # pick a particle
    pos_i_part[:]        = tree_in.pos_pt[:,i_part]

    # init the pointers for the cells
    n_cells_to_check_new = 0
    n_cells_to_check     = 1
    id_cells_to_check[0] = 0

    # counters to check the efficiency
    i_exact              = 0 
    i_iter               = 0
    i_approx             = 0

    # while there are cells that can be opened/pruned
    while(n_cells_to_check > 0):
    
      assert i_iter< n_iter_max
    
      # loop on the current cells
      for ii in range(n_cells_to_check):

        i_iter           = i_iter + 1 
      
        # pick a cell
        i_current        = id_cells_to_check[ii]

        # check if we need to open the cell
        if node_need_to_be_opened(i_cell=i_current, pos_in= pos_i_part,tree_in=tree_in,critical_angle=critical_angle):
        
          # if it is a leaf, do a bruteforce computation
          if is_leaf(i_current,tree_in):
            # do a N^2 sum within the cell
            for jj in range(tree_in.n_parts[i_current]):
              j_part = tree_in.id_parts[jj,i_current]
              # avoid self force
              if j_part != i_part:
                pos_j_part[:] = tree_in.pos_pt[:,j_part]
                pos_ij_part[:]= pos_j_part[:] - pos_i_part[:]
                dist          = np.sqrt(np.sum((pos_j_part[:]- pos_i_part[:])**2))
                acc_j[:]      = tree_in.mass_pt[j_part] * pos_ij_part[:]/ dist**(3.0)

                out[:,i_part] = out[:,i_part] + acc_j[:]
                
                # update counter for exact computation
                i_exact       = i_exact + 1
          # if it is not a leaf, add all the childrens
          else:
            for i_child in range(tree_in.n_children):
              id_cells_to_check_new[n_cells_to_check_new + i_child] = tree_in.childrens[i_child,i_current]
            # update the new sets of pointers
            n_cells_to_check_new = n_cells_to_check_new + tree_in.n_children
        else:
          # prune out the branch (higher levels of the current cell)
          if(tree_in.mass_moment_0[i_current]>0):
            # compute center of mass and distance
            center_of_mass[:] = tree_in.mass_moment_1[:,i_current]/tree_in.mass_moment_0[i_current]
            pos_cmi_part[:]   = center_of_mass[:] - pos_i_part[:]
            dist              = np.sqrt(np.sum((center_of_mass[:] - pos_i_part[:])**2))
            # compute the potential with monopole approximation
            acc_j[:]          = tree_in.mass_moment_0[i_current] * pos_cmi_part[:]/dist**(3.0)
            out[:,i_part]     = out[:,i_part] + acc_j[:]
            i_approx          = i_approx + 1
            
      # loop on current cells is done
      # update the cell id list for next iteration
      id_cells_to_check[0:n_cells_to_check_new] = id_cells_to_check_new[0:n_cells_to_check_new]
      n_cells_to_check                          = n_cells_to_check_new
      # reset list for next cells
      n_cells_to_check_new                      = 0

    if verbose:
      n_iter_tot  = n_iter_tot   + float(i_iter)
      n_exact_tot = n_exact_tot  + float(i_exact)
      n_approx_tot= n_approx_tot + float(i_approx)
  
  if verbose:
    print("  average iterations         ",n_iter_tot/tree_in.n_pt)
    print("  average exact  calculations",n_exact_tot/tree_in.n_pt)
    print("  average approx calculations",n_approx_tot/tree_in.n_pt)
    print("  theoretical speedup        ",tree_in.n_pt**2/n_iter_tot)

  out[:] = out[:] * G_gravity

  return out

if __name__ == "__main__":

  """
  todo list:
    check if mass assignment to the tree is ok
    open the tree with critical angle condition, to compute the potential
    test speed and precision wrt the bruteforce gravity
  """

  import matplotlib.pyplot as plt

  n_part_per_cell = 6
  n_dim           = 2
  n_grid          = 200
  iter_max        = 200
  
  check_finder    = True

  #--------------------------------
  # set up particles
  #--------------------------------
  n_pt      = 32
  test_pos  = np.zeros((n_dim,n_pt))
  test_mass = np.ones(n_pt)

  for i_dim in range(n_dim):
    test_pos[i_dim,:] = np.random.normal(loc=0.0,scale=1.0, size=n_pt)
 
  # renormalize point position (in a bad way, unit_l should be defined)
  for i_dim in range(n_dim):
      test_pos[i_dim,:] = (test_pos[i_dim,:]-np.min(test_pos[i_dim,:]))/(np.max(test_pos[i_dim,:])-np.min(test_pos[i_dim,:])) - 0.5
  #--------------------------------


  #--------------------------------
  # init tree
  #--------------------------------
  test_tree = tree(n_dim = n_dim,n_grid=n_grid,n_part_per_cell = n_part_per_cell, mass_pt=test_mass, pos_pt = test_pos)
  if False:
    # manually split the tree
    test_tree =  split_target_oct(i_target=0, ree_in=test_tree)
    test_tree =  split_target_oct(i_target=1, tree_in=test_tree)
  else:
    test_tree   = build_tree_from_particles(tree_in=test_tree,i_iter_max=iter_max)

  test_tree      = compute_mass_on_tree(tree_in=test_tree)

  potential_tree = compute_potential_tree(tree_in=test_tree,critical_angle=0.1)
  print(potential_tree)

  if check_finder:
    id_target    = 0
    id_cell_check ,i_count= get_leaf_cell_id(i_target=id_target, tree_in=test_tree)

  #--------------------------------
  # visualize the tree
  #--------------------------------
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  
  for i_cell in range(test_tree.n_now):

    col    = 'k'
    zorder = 0

    if check_finder:
      if i_cell == id_cell_check:
        col    = 'r'
        zorder = 5

    ax.plot(test_tree.center[0,i_cell],test_tree.center[1,i_cell],marker='o',color=col,zorder=zorder)

    dx    = 0.5**(test_tree.levels[i_cell]+1)
    # determination of the edges of the square can be cleaned
    xedge = [test_tree.center[0,i_cell] - dx, test_tree.center[0,i_cell] + dx,test_tree.center[0,i_cell] + dx,test_tree.center[0,i_cell] - dx,test_tree.center[0,i_cell] - dx]
    yedge = [test_tree.center[1,i_cell] + dx, test_tree.center[1,i_cell] + dx,test_tree.center[1,i_cell] - dx,test_tree.center[1,i_cell] - dx,test_tree.center[1,i_cell] + dx]
    ax.plot(xedge,yedge,ls='-',marker='',color=col,zorder=zorder)

  ax.plot(test_pos[0,:],test_pos[1,:],ls='',marker='x',color='b',alpha=0.6)

  if check_finder:
    ax.plot(test_pos[0,id_target],test_pos[1,id_target],ls='',marker='x',color='r',alpha=0.6)

  #ax.set_xlim(-0.5,0.5)
  #ax.set_ylim(-0.5,0.5)
  #ax.set_xlim(-0.6,0.6)
  #ax.set_ylim(-0.6,0.6)

  #--------------------------------
  # visualize structure tree
  #--------------------------------

  level_max_plus_one, cell_per_level, leaf_per_level = get_structure(tree_in=test_tree)

  fig = plt.figure()
  ax  = fig.add_subplot(111)

  levels_in_the_tree = np.arange(0,level_max_plus_one,1)

  ax.plot(levels_in_the_tree, cell_per_level,marker='o',label='all  cells',ls='')
  ax.plot(levels_in_the_tree, leaf_per_level,marker='s',label='leaf cells',ls='')

  ax.legend(frameon=False)
  ax.set_xlabel('levels')
  ax.set_ylabel('number of cells')

  plt.show()


