import numpy as np
import numba


@numba.njit
def _get_num_muts(
    ts_num_nodes,
    tree_nodes_preorder,
    tree_parent_array,
    tree_nodes_num_mutations,
):
    num_muts = np.zeros(ts_num_nodes, dtype=np.int32)
    for node in tree_nodes_preorder:
        pa = tree_parent_array[node]
        if pa > -1:
            num_muts[node] = num_muts[pa]
        num_muts[node] += tree_nodes_num_mutations[node]
    return num_muts


def get_num_muts(ts):
    num_muts_all_trees = np.zeros(ts.num_nodes, dtype=np.int32)
    for tree in ts.trees():
        tree_nodes_preorder = tree.preorder()
        assert np.min(tree_nodes_preorder) >= 0
        tree_parent_array = tree.parent_array
        mut_pos = ts.sites_position[ts.mutations_site]
        is_mut_in_tree = (tree.interval.left <= mut_pos) & (
            mut_pos < tree.interval.right
        )
        tree_nodes_num_muts = np.bincount(
            ts.mutations_node[is_mut_in_tree],
            minlength=ts.num_nodes,
        )
        num_muts_all_trees += _get_num_muts(
            ts_num_nodes=ts.num_nodes,
            tree_nodes_preorder=tree_nodes_preorder,
            tree_parent_array=tree_parent_array,
            tree_nodes_num_mutations=tree_nodes_num_muts,
        )
    return num_muts_all_trees


@numba.njit
def _get_root_path(parent, node):
    u = node
    path = []
    while u != -1:
        path.append(u)
        u = parent[u]
    return path


def get_root_path(tree, node):
    return _get_root_path(tree.parent_array, node)


@numba.njit
def _get_path_mrca(path1, path2, node_time):
    j1 = 0
    j2 = 0
    while True:
        if path1[j1] == path2[j2]:
            return path1[j1]
        elif node_time[path1[j1]] < node_time[path2[j2]]:
            j1 += 1
        elif node_time[path2[j2]] < node_time[path1[j1]]:
            j2 += 1
        else:
            # Time is equal, but the nodes differ
            j1 += 1
            j2 += 1


def get_path_mrca(path1, path2, node_time):
    assert path1[-1] == path2[-1]
    return _get_path_mrca(
        np.array(path1, dtype=np.int32), np.array(path2, dtype=np.int32), node_time
    )
