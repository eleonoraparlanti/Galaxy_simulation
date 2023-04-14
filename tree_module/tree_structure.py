
import numpy as np
from numba import njit

@njit
def is_leaf(i_cell,tree_in):

  """
  check if target cell i_cell is a leaf
  """

  out = True

  # can be done in a faster way (check only the first)
  for i_child in range(tree_in.n_children):
      out = out and tree_in.childrens[i_child,i_cell] < 0

  return out

@njit
def split_target_oct(tree_in, i_target):

  """
  generate octs by splitting (refining) the target cell
  """

  assert tree_in.n_now + tree_in.n_children <= tree_in.n_max, "Please increase n grid"

  tmp_pos   = np.zeros(tree_in.n_dim)     # temporary array for positions

  new_level = tree_in.levels[i_target]+1  # new level
  new_dx    = 0.5**new_level              # new size of the octs

  for i_child in range(tree_in.n_children):

    # get the new avaiable ids for the child
    id_child                            = tree_in.n_now + i_child
    # set up pointers
    tree_in.childrens[i_child,i_target] = id_child
    tree_in.parents[id_child]           = i_target
    # set up the level
    tree_in.levels[id_child]            = tree_in.levels[i_target]+1 
    # set up geometry of the children
    id_child                            = tree_in.childrens[i_child,i_target]
    tmp_pos[:]                          = tree_in.center[:,i_target] + tree_in.geometry[:, i_child]*new_dx
    tree_in.center[:,id_child]          = tmp_pos[:]

  # increase current occupation
  tree_in.n_now = tree_in.n_now+ tree_in.n_children

  return tree_in

