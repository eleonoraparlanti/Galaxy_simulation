
from numba import njit
import numpy as np
from tree_module.tree_structure import is_leaf
from numpy import int64   as NP_INT

@njit
def is_position_in_cell_i(pos_in, tree_in, i_cell):

  """
  check if target pos_in[:] is in cell i_cell
  """

  is_in   = True

  half_dx = 0.5**(tree_in.levels[i_cell] +1)

  for i_dim in range(tree_in.n_dim):

    is_in = is_in and np.abs(pos_in[i_dim] - tree_in.center[i_dim,i_cell]) <= half_dx

  return is_in

@njit
def get_leaf_cell_id(i_target, tree_in, i_iter_max = 100):

    """
    get the id of the cell the target particle i_target belong to
    """

    # get the position of the target
    pos_target = tree_in.pos_pt[:,i_target]

    # check if the particle is in the root
    assert is_position_in_cell_i(pos_in=pos_target, tree_in=tree_in, i_cell=0)

    i_iter    = 0
    i_current = 0
    go        = not is_leaf(i_cell=i_current,tree_in=tree_in)

    while go:

      assert i_iter < i_iter_max 

      i_new = -1

      # check the children
      for i_child in range(tree_in.n_children):

        id_check = tree_in.childrens[i_child,i_current]

        if is_position_in_cell_i(pos_in=pos_target, tree_in=tree_in, i_cell=id_check):
          i_new = id_check

      # update the loop for next level
      i_current = i_new
      i_iter    = i_iter + 1
      go        = not is_leaf(i_cell=i_current,tree_in=tree_in)

    return i_current, i_iter

@njit
def get_level_per_particle(tree_in, i_iter_max = 10000):

  n_pt = tree_in.n_pt

  out  = np.zeros(n_pt, dtype=NP_INT)

  for i_pt in range(n_pt):
    id_cell , __   = get_leaf_cell_id(i_target=i_pt, tree_in=tree_in, i_iter_max = i_iter_max)
    out[i_pt]      = tree_in.levels[id_cell]

  return out

@njit
def node_need_to_be_opened(i_cell , pos_in ,tree_in , critical_angle=0.0):

   # if particle is in, open it
   out = is_position_in_cell_i(pos_in=pos_in, tree_in=tree_in, i_cell=i_cell)

   # check critical angle condition
   dist = np.sqrt(np.sum((tree_in.center[:,i_cell] - pos_in[:])**2))
   size = 0.5**(tree_in.levels[i_cell])
   out  = out or size/dist >= critical_angle

   return out    
   
   
