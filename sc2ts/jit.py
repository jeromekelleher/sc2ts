import dataclasses
import logging


import numpy as np
import numba

logger = logging.getLogger(__name__)


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


# NOTE: we're not actually using this function, so it can be deleted
# at some point if we don't find a purpose for it. It has some reasonable
# test coverage, or I'd just delete it now.
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
    if node >= len(tree.parent_array):
        raise ValueError("node {node} out of bounds")
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


spec = [
    ("num_edges", numba.int64),
    ("sequence_length", numba.float64),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("edge_insertion_order", numba.int32[:]),
    ("edge_removal_order", numba.int32[:]),
    ("edge_insertion_index", numba.int64),
    ("edge_removal_index", numba.int64),
    ("interval", numba.float64[:]),
    ("in_range", numba.int64[:]),
    ("out_range", numba.int64[:]),
]


@numba.experimental.jitclass(spec)
class TreePosition:
    def __init__(
        self,
        num_edges,
        sequence_length,
        edges_left,
        edges_right,
        edge_insertion_order,
        edge_removal_order,
    ):
        self.num_edges = num_edges
        self.sequence_length = sequence_length
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.edge_insertion_index = 0
        self.edge_removal_index = 0
        self.interval = np.zeros(2)
        self.in_range = np.zeros(2, dtype=np.int64)
        self.out_range = np.zeros(2, dtype=np.int64)

    def next(self):  # noqa
        left = self.interval[1]
        j = self.in_range[1]
        k = self.out_range[1]
        self.in_range[0] = j
        self.out_range[0] = k
        M = self.num_edges
        edges_left = self.edges_left
        edges_right = self.edges_right
        out_order = self.edge_removal_order
        in_order = self.edge_insertion_order

        while k < M and edges_right[out_order[k]] == left:
            k += 1
        while j < M and edges_left[in_order[j]] == left:
            j += 1
        self.out_range[1] = k
        self.in_range[1] = j

        right = self.sequence_length
        if j < M:
            right = min(right, edges_left[in_order[j]])
        if k < M:
            right = min(right, edges_right[out_order[k]])
        self.interval[:] = [left, right]
        return j < M or left < self.sequence_length


# Helper function to make it easier to communicate with the numba class
def alloc_tree_position(ts):
    return TreePosition(
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
    )


@dataclasses.dataclass
class ArgCounts:
    nodes_max_descendant_samples: np.ndarray
    mutations_num_inheritors: np.ndarray
    mutations_num_descendants: np.ndarray
    mutations_num_parents: np.ndarray


@numba.njit()
def _compute_mutations_num_parents(mutations_parent):
    N = mutations_parent.shape[0]
    num_parents = np.zeros(N, dtype=np.int32)

    for j in range(N):
        u = j
        while mutations_parent[u] != -1:
            num_parents[j] += 1
            u = mutations_parent[u]
    return num_parents


@numba.njit()
def _compute_inheritance_counts(
    tree_pos,
    num_nodes,
    num_mutations,
    edges_parent,
    edges_child,
    samples,
    mutations_position,
    mutations_node,
    mutations_parent,
):
    parent = np.zeros(num_nodes, dtype=np.int32) - 1
    num_samples = np.zeros(num_nodes, dtype=np.int32)
    num_samples[samples] = 1
    nodes_max_descendant_samples = np.zeros(num_nodes, dtype=np.int32)
    nodes_max_descendant_samples[samples] = 1
    mutations_num_descendants = np.zeros(num_mutations, dtype=np.int32)
    mutations_num_inheritors = np.zeros(num_mutations, dtype=np.int32)

    mut_id = 0

    while tree_pos.next():
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            p = edges_parent[e]
            parent[c] = -1
            u = p
            while u != -1:
                num_samples[u] -= num_samples[c]
                u = parent[u]

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            c = edges_child[e]
            parent[c] = p
            u = p
            while u != -1:
                num_samples[u] += num_samples[c]
                nodes_max_descendant_samples[u] = max(
                    nodes_max_descendant_samples[u], num_samples[u]
                )
                u = parent[u]

        left, right = tree_pos.interval
        while mut_id < num_mutations and mutations_position[mut_id] < right:
            assert mutations_position[mut_id] >= left
            mutation_node = mutations_node[mut_id]
            descendants = num_samples[mutation_node]
            mutations_num_descendants[mut_id] = descendants
            mutations_num_inheritors[mut_id] = descendants
            # Subtract this number of descendants from the parent mutation. We are
            # guaranteed to list parents mutations before their children
            mut_parent = mutations_parent[mut_id]
            if mut_parent != -1:
                mutations_num_inheritors[mut_parent] -= descendants
            mut_id += 1

    return (
        nodes_max_descendant_samples,
        mutations_num_inheritors,
        mutations_num_descendants,
        _compute_mutations_num_parents(mutations_parent),
    )


def count(ts):
    logger.info("Computing inheritance counts")
    tree_pos = alloc_tree_position(ts)
    mutations_position = ts.sites_position[ts.mutations_site].astype(int)
    return ArgCounts(
        *_compute_inheritance_counts(
            tree_pos,
            ts.num_nodes,
            ts.num_mutations,
            ts.edges_parent,
            ts.edges_child,
            ts.samples(),
            mutations_position,
            ts.mutations_node,
            ts.mutations_parent,
        )
    )
