import dataclasses
import logging

import numpy as np
import numba
import tskit.jit.numba as tskit_numba
import tskit

NODE_IS_SAMPLE = tskit.NODE_IS_SAMPLE

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
    numba_ts,
):
    num_nodes = numba_ts.num_nodes
    num_mutations = numba_ts.num_mutations
    edges_parent = numba_ts.edges_parent
    edges_child = numba_ts.edges_child
    mutations_node = numba_ts.mutations_node
    mutations_parent = numba_ts.mutations_parent
    mutations_position = numba_ts.sites_position[numba_ts.mutations_site].astype(
        np.int32
    )
    nodes_flags = numba_ts.nodes_flags

    parent = np.zeros(num_nodes, dtype=np.int32) - 1
    num_samples = np.zeros(num_nodes, dtype=np.int32)
    nodes_max_descendant_samples = np.zeros(num_nodes, dtype=np.int32)

    for node in range(num_nodes):
        if (nodes_flags[node] & NODE_IS_SAMPLE) != 0:
            num_samples[node] = 1
            nodes_max_descendant_samples[node] = 1
    mutations_num_descendants = np.zeros(num_mutations, dtype=np.int32)
    mutations_num_inheritors = np.zeros(num_mutations, dtype=np.int32)

    mut_id = 0
    tree_index = numba_ts.tree_index()

    while tree_index.next():
        out_range = tree_index.out_range
        for j in range(out_range.start, out_range.stop):
            e = out_range.order[j]
            c = edges_child[e]
            p = edges_parent[e]
            parent[c] = -1
            u = p
            while u != -1:
                num_samples[u] -= num_samples[c]
                u = parent[u]

        in_range = tree_index.in_range
        for j in range(in_range.start, in_range.stop):
            e = in_range.order[j]
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

        left, right = tree_index.interval
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
    numba_ts = tskit_numba.jitwrap(ts)
    return ArgCounts(
        *_compute_inheritance_counts(
            numba_ts,
        )
    )


# FIXME make cache optional.
@numba.njit(cache=True)
def encode_alleles(h):
    # Just so numba knows this is a constant string.
    alleles = "ACGT-RYSWKMBDHV."
    n = h.shape[0]
    a = np.full(n, -1, dtype=np.int8)
    for j in range(n):
        if h[j] == "N":
            a[j] = -1
        else:
            for k, c in enumerate(alleles):
                if c == h[j]:
                    break
            else:
                raise ValueError(f"Allele {h[j]} not recognised")
            a[j] = k
    return a
