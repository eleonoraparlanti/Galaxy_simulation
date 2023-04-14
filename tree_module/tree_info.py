
from tree_module.tree_structure import is_leaf
import numpy as np

def get_structure(tree_in):

    level_max_plus_one      = np.max(tree_in.levels) + 1

    cell_per_level = np.zeros(level_max_plus_one)
    leaf_per_level = np.zeros(level_max_plus_one)

    for i_cell in range(tree_in.n_now):
        i_level                 = tree_in.levels[i_cell]
        cell_per_level[i_level] = cell_per_level[i_level] + 1
        if(is_leaf(i_cell=i_cell,tree_in=tree_in)):
          leaf_per_level[i_level] = leaf_per_level[i_level] + 1

    return level_max_plus_one, cell_per_level, leaf_per_level

