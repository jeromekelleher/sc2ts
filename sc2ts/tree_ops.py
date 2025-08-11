"""
Miscellanous tree operations we need for sc2ts inference.
"""

import collections
import logging
import dataclasses
from typing import List

import tskit
import tqdm
import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy
import biotite.sequence.phylo as bsp

from . import core
from . import stats

logger = logging.getLogger(__name__)


def reroot(pi, new_root):
    # Note: we don't really need to store the path here, but I'm
    # in a hurry and it's easier.
    path = []
    u = new_root
    while u != -1:
        path.append(u)
        u = pi[u]
    for j in range(len(path) - 1):
        child = path[j]
        parent = path[j + 1]
        pi[parent] = child
    pi[new_root] = -1


def reroot_ts(ts, new_root, scale_time=False):
    """
    Reroot the tree around the specified node, keeping node IDs
    the same.
    """
    assert ts.num_trees == 1
    tree = ts.first()
    pi = tree.parent_array.copy()
    reroot(pi, new_root)

    tables = ts.dump_tables()
    tables.edges.clear()
    # NOTE: could be done with numpy so this will work for large trees.
    for u in range(ts.num_nodes):
        if pi[u] != -1:
            tables.edges.add_row(0, ts.sequence_length, pi[u], u)
    set_tree_time(tables, unit_scale=scale_time)
    tables.sort()
    return tables.tree_sequence()


def biotite_to_tskit_tables(tree, tables):
    """
    Updates the specified set of tables with the biotite tree.
    """
    L = tables.sequence_length
    pi, n = biotite_to_oriented_forest(tree)
    assert n == len(tables.nodes)
    for _ in range(n, len(pi)):
        tables.nodes.add_row()
    for u, parent in enumerate(pi):
        if parent != -1:
            tables.edges.add_row(0, L, parent, u)
    set_tree_time(tables, unit_scale=True)
    tables.sort()


def biotite_to_tskit(tree):
    """
    Returns a tskit tree with the topology of the specified biotite tree.

    This is just a wrapper to facilitate testing.
    """
    tables = tskit.TableCollection(1)
    for node in tree.leaves:
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE)
    biotite_to_tskit_tables(tree, tables)
    return tables.tree_sequence().first()


def max_leaf_distance(pi, n):
    tau = np.zeros(len(pi))
    for j in range(n):
        u = j
        t = 0
        while u != -1:
            # TODO we can also stop this loop early, I just haven't
            # thought about it.
            tau[u] = max(tau[u], t)
            t += 1
            u = pi[u]
    return tau


def set_tree_time(tables, unit_scale=False):
    # Add times using max number of hops from leaves
    pi = np.full(len(tables.nodes), -1, dtype=int)
    tau = np.full(len(tables.nodes), -1, dtype=float)
    pi[tables.edges.child] = tables.edges.parent
    for sample in np.where(tables.nodes.flags == tskit.NODE_IS_SAMPLE)[0]:
        t = 0
        u = sample
        while u != -1:
            tau[u] = max(tau[u], t)
            t += 1
            u = pi[u]
    if unit_scale:
        tau /= max(1, np.max(tau))
    tables.nodes.time = tau


def biotite_to_oriented_forest(tree):
    node_map = {}
    for u, node in enumerate(tree.leaves):
        node_map[node] = u
        assert u == node.get_indices()[0]
    n = len(node_map)
    pi = [-1] * n

    stack = [tree.root]
    while len(stack) > 0:
        node = stack.pop()
        if not node.is_leaf():
            assert node not in node_map
            node_map[node] = len(pi)
            pi.append(-1)
            for child in node.children:
                stack.append(child)
        if node.parent is not None:
            pi[node_map[node]] = node_map[node.parent]
    return pi, n


def add_tree_to_tables(tables, pi, tau):
    # add internal nodes
    for j in range(len(tables.nodes), len(tau)):
        tables.nodes.add_row(time=tau[j])
    L = tables.sequence_length
    for u, parent in enumerate(pi):
        if parent != -1:
            tables.edges.add_row(0, L, parent, u)


def infer_binary_topology(ts, tables):
    assert ts.num_trees == 1
    assert ts.num_mutations > 0

    if ts.num_samples < 2:
        return tables.tree_sequence()

    samples = ts.samples()
    tree = ts.first()
    # Include the root as a sample in the tree building
    samples = np.concatenate((samples, [tree.root]))
    G = ts.genotype_matrix(samples=samples, isolated_as_missing=False)

    # Hamming distance should be suitable here because it's giving the overall
    # number of differences between the observations. Euclidean is definitely
    # not because of the allele encoding (difference between 0 and 4 is not
    # greater than 0 and 1).
    Y = scipy.spatial.distance.pdist(G.T, "hamming")

    if ts.num_samples < 3:
        # NJ fails with < 4
        biotite_tree = bsp.upgma(scipy.spatial.distance.squareform(Y))
    else:
        biotite_tree = bsp.neighbor_joining(scipy.spatial.distance.squareform(Y))
    pi, n = biotite_to_oriented_forest(biotite_tree)
    # Node n - 1 is the pre-specified root, so force rerooting around that.
    reroot(pi, n - 1)

    assert n == len(tables.nodes) + 1
    tau = max_leaf_distance(pi, n)
    tau /= max(1, np.max(tau))
    add_tree_to_tables(tables, pi, tau)
    tables.sort()

    return tables.tree_sequence()


# TODO rename this to infer_sample_group_tree
def infer_binary(ts):
    """
    Infer a strictly binary tree from the variation data in the
    specified tree sequence.
    """
    assert ts.num_trees == 1
    assert list(ts.samples()) == list(range(ts.num_samples))
    tables = ts.dump_tables()
    # Don't clear populations for simplicity
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()
    # Preserve the samples
    tables.nodes.truncate(ts.num_samples)

    # Update the tables with the topology
    infer_binary_topology(ts, tables)
    binary_ts = tables.tree_sequence()

    # Now add on mutations under parsimony
    tree = binary_ts.first()
    for v in ts.variants():
        anc, muts = tree.map_mutations(
            v.genotypes, v.alleles, ancestral_state=v.site.ancestral_state
        )
        site = tables.sites.add_row(v.site.position, anc)
        for mut in muts:
            tables.mutations.add_row(
                site=site,
                node=mut.node,
                derived_state=mut.derived_state,
            )
    tables.compute_mutation_parents()
    new_ts = tables.tree_sequence()
    # print(new_ts.draw_text())
    return new_ts


def trim_branches(ts):
    """
    Remove branches from the tree that have no mutations.
    """
    assert ts.num_trees == 1
    tree = ts.first()
    nodes_to_keep = set(ts.samples()) | {tree.root}
    for mut in tree.mutations():
        nodes_to_keep.add(mut.node)

    parent = {}
    for u in tree.postorder()[:-1]:
        if u in nodes_to_keep:
            p = tree.parent(u)
            while p not in nodes_to_keep:
                p = tree.parent(p)
            parent[u] = p

    tables = ts.dump_tables()
    tables.edges.clear()
    for c, p in parent.items():
        tables.edges.add_row(0, ts.sequence_length, parent=p, child=c)

    tables.sort()
    # FIXME not sure this compute_mutation_parents is needed, check
    tables.build_index()
    tables.compute_mutation_parents()
    # Get rid of unreferenced nodes
    tables.simplify()
    return tables.tree_sequence()


def update_tables(tables, edges_to_delete, mutations_to_delete):
    # Updating the mutations is a real faff, and the only way I
    # could get it to work is by using the time values. This should
    # be easier...
    mutations_to_keep = np.ones(len(tables.mutations), dtype=bool)
    mutations_to_keep[mutations_to_delete] = False
    tables.mutations.replace_with(tables.mutations[mutations_to_keep])
    # Set the parent values to -1 and recompute them later.
    tables.mutations.parent = np.full_like(tables.mutations.parent, -1)

    edges_to_keep = np.ones(len(tables.edges), dtype=bool)
    edges_to_keep[edges_to_delete] = False
    tables.edges.replace_with(tables.edges[edges_to_keep])

    logger.debug("Update tables: sorting and indexing final tables.")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


def full_span_sibs(ts, nodes):
    """
    Return the set of nodes that are sibs of the specified nodes over
    the full span of the tree sequence (including the nodes themselves).
    """
    full_span = np.logical_and(ts.edges_left == 0, ts.edges_right == ts.sequence_length)
    edges_parent = ts.edges_parent[full_span]
    edges_child = ts.edges_child[full_span]
    select = np.isin(edges_child, nodes)
    parents = np.unique(edges_parent[select])
    select = np.isin(edges_parent, parents)
    return np.unique(edges_child[select])


def coalesce_mutations(ts, samples=None, date="1999-01-01", show_progress=False):
    """
    Examine all time-0 samples (or specified nodes) and their (full-sequence)
    sibs and create new nodes to represent overlapping sets of mutations. The
    algorithm is greedy and makes no guarantees about uniqueness or optimality.
    Also note that we don't recurse and only reason about mutation sharing at a
    single level in the tree.
    """
    tables = ts.dump_tables()
    # Set the mutations time to the time of the node. We rely on this
    # for sorting and to make sure we don't violate constraints during
    # parsimony updates
    tables.mutations.time = ts.nodes_time[ts.mutations_node]

    if samples is None:
        samples = ts.samples(time=0)

    sibs = full_span_sibs(ts, samples)
    logger.debug(f"Computing mutation descriptors for {len(sibs)} sibs")
    node_mutations = nodes_mutation_descriptors(ts, sibs, show_progress=show_progress)
    # remove any nodes that are not in the sibs from our sib-groups
    samples = set(samples) & set(sibs)

    tree = ts.first()

    # For each sample, what is the ("a" more accurately - this is greedy)
    # maximum mutation overlap with one of its sibs?
    max_sample_overlap = {}
    for sample in tqdm.tqdm(samples, disable=not show_progress, desc="MMO"):
        u = tree.parent(sample)
        max_overlap = set()
        for v in tree.children(u):
            if v != sample and v in node_mutations:
                overlap = set(node_mutations[sample]) & set(node_mutations[v])
                if len(overlap) > len(max_overlap):
                    max_overlap = overlap
        max_sample_overlap[sample] = max_overlap

    # Group the maximum mutation overlaps by the parent and mutation pattern
    sib_groups = collections.defaultdict(set)
    # Make sure we don't use the same node in more than one sib-set
    used_nodes = set()
    for sample in tqdm.tqdm(samples, disable=not show_progress, desc="SGO"):
        u = tree.parent(sample)
        sample_overlap = frozenset(max_sample_overlap[sample])
        key = (u, sample_overlap)
        if len(sample_overlap) > 0:
            for v in tree.children(u):
                if v in node_mutations and v not in used_nodes:
                    if sample_overlap.issubset(set(node_mutations[v])):
                        sib_groups[key].add(v)
                        used_nodes.add(v)
        # Avoid creating a new node when there's only one node in the sib group
        if len(sib_groups[key]) < 2:
            del sib_groups[key]

    logger.debug(f"Found {len(sib_groups)} sib groups")

    mutations_to_delete = []
    edges_to_delete = []
    for (_, overlap), sibs in sib_groups.items():
        for mut_desc in overlap:
            for sib in sibs:
                mutations_to_delete.append(node_mutations[sib][mut_desc])
                edges_to_delete.append(tree.edge(sib))

    for (parent, overlap), sibs in tqdm.tqdm(
        sib_groups.items(), disable=not show_progress, desc="Add"
    ):
        group_parent = len(tables.nodes)
        tables.edges.add_row(0, ts.sequence_length, parent, group_parent)
        max_sib_time = 0
        for sib in sibs:
            max_sib_time = max(max_sib_time, ts.nodes_time[sib])
            tables.edges.add_row(0, ts.sequence_length, group_parent, sib)
        parent_time = ts.nodes_time[parent]
        assert max_sib_time < parent_time
        diff = parent_time - max_sib_time
        group_parent_time = max_sib_time + diff / 2
        assert group_parent_time < parent_time

        md_overlap = [
            f"{x.inherited_state}{int(ts.sites_position[x.site])}{x.derived_state}"
            for x in overlap
        ]
        md_sibs = [int(sib) for sib in sibs]
        tables.nodes.add_row(
            flags=core.NODE_IS_MUTATION_OVERLAP,
            time=group_parent_time,
            metadata={
                "sc2ts": {
                    "mutations": md_overlap,
                    "sibs": md_sibs,
                    "date_added": date,
                }
            },
        )
        for mut_desc in overlap:
            tables.mutations.add_row(
                site=mut_desc.site,
                derived_state=mut_desc.derived_state,
                node=group_parent,
                time=group_parent_time,
                metadata={"sc2ts": {"type": "overlap"}},
            )

    num_del_mutations = len(mutations_to_delete)
    num_new_nodes = len(tables.nodes) - ts.num_nodes
    logger.info(
        f"Coalescing mutations: delete {num_del_mutations} mutations; "
        f"add {num_new_nodes} new nodes"
    )
    return update_tables(tables, edges_to_delete, mutations_to_delete)


# NOTE: "samples" is a bad name here, this is actually the set of attach_nodes
# that we get from making a local tree from a group.
def push_up_reversions(ts, samples, date="1999-01-01", show_progress=False):
    # We depend on mutations having a time below.
    assert np.all(np.logical_not(np.isnan(ts.mutations_time)))

    tree = ts.first()
    # Get the samples that span the whole sequence and also have
    # parents that span the full sequence. No reason we couldn't
    # update the algorithm to work with partial edges, it's just easier
    # this way and it covers the vast majority of simple reversions
    # that we see
    full_span_samples = []
    for u in samples:
        parent = tree.parent(u)
        assert parent != -1
        full_edge = True
        for v in [u, parent]:
            assert v != -1
            e = tree.edge(v)
            if e == -1:
                # The parent is the root
                full_edge = False
                break
            edge = ts.edge(e)
            if edge.left != 0 or edge.right != ts.sequence_length:
                full_edge = False
                break
        if full_edge:
            full_span_samples.append(u)

    logger.info(f"Pushing reversions for {len(full_span_samples)} full-span samples")

    all_nodes = set()
    for child in full_span_samples:
        parent = tree.parent(child)
        all_nodes.add(child)
        all_nodes.add(parent)

    descriptors = nodes_mutation_descriptors(
        ts, list(all_nodes), show_progress=show_progress
    )

    # For each node check if it has an immediate reversion
    sib_groups = collections.defaultdict(list)
    for child in full_span_samples:
        parent = tree.parent(child)
        child_muts = {desc.site: desc for desc in descriptors[child]}
        parent_muts = {desc.site: desc for desc in descriptors[parent]}
        reversions = []
        for site in child_muts:
            if site in parent_muts:
                if child_muts[site].is_reversion_of(parent_muts[site]):
                    reversions.append(
                        ReversionDescriptor(
                            site, child, child_muts[site], parent_muts[site]
                        )
                    )
        # Pick the maximum set of reversions per sib group so that we're not
        # trying to resolve incompatible reversion sets.
        if len(reversions) > len(sib_groups[parent]):
            sib_groups[parent] = reversions

    logger.debug(f"Found {len(sib_groups)} sib groups")

    tables = ts.dump_tables()
    edges_to_delete = set()
    mutations_to_delete = []
    for parent, reversions in tqdm.tqdm(sib_groups.items(), disable=not show_progress):
        if len(reversions) == 0:
            continue

        sample = reversions[0].child_node
        assert all(x.child_node == sample for x in reversions)
        # Remove the edges above the sample and its parent
        e = {tree.edge(sample), tree.edge(parent)}
        if len(e & edges_to_delete) > 0:
            # One of these edges has already been altered - skip!
            continue
        edges_to_delete.update(e)

        sites = [x.site for x in reversions]
        # Create new node that is fractionally older than the current
        # parent that will be the parent of both nodes.
        grandparent = tree.parent(parent)
        # Arbitrarily make it 1/8 of the branch_length. Probably should
        # make it proportional to the number of mutations or something.
        eps = tree.branch_length(parent) * 0.125
        w_time = tree.time(parent) + eps
        parent_summary = []
        for r in reversions:
            position = int(ts.sites_position[r.site])
            parent_summary.append(
                f"{r.parent_mutation.inherited_state}{position}{r.parent_mutation.derived_state}"
            )
        w = tables.nodes.add_row(
            flags=core.NODE_IS_REVERSION_PUSH,
            time=w_time,
            metadata={
                "sc2ts": {
                    "date_added": date,
                    # Store the parent mutation, that is the mutations we were trying
                    # to revert
                    "mutations": parent_summary,
                }
            },
        )
        # Add new edges to join the sample and parent to w, and then
        # w to the grandparent.
        tables.edges.add_row(0, ts.sequence_length, parent=w, child=parent)
        tables.edges.add_row(0, ts.sequence_length, parent=w, child=sample)
        tables.edges.add_row(0, ts.sequence_length, parent=grandparent, child=w)

        # Move any non-reversions mutations above the parent to the new node.
        for mut in np.where(ts.mutations_node == parent)[0]:
            row = tables.mutations[mut]
            if row.site not in sites:
                tables.mutations[mut] = row.replace(node=w, time=w_time)
        for site in sites:
            # Delete the reversion mutations above the sample
            muts = np.where(
                np.logical_and(ts.mutations_node == sample, ts.mutations_site == site)
            )[0]
            assert len(muts) == 1
            mutations_to_delete.extend(muts)

    num_del_mutations = len(mutations_to_delete)
    num_new_nodes = len(tables.nodes) - ts.num_nodes
    logger.info(
        f"Push reversions: delete {num_del_mutations} mutations; "
        f"add {num_new_nodes} new nodes"
    )
    return update_tables(tables, list(edges_to_delete), mutations_to_delete)


@dataclasses.dataclass(frozen=True)
class MutationDescriptor:
    site: float
    derived_state: str
    inherited_state: str
    parent: int

    def is_reversion_of(self, other) -> bool:
        """
        Returns True if this mutation is a reversion of the other.
        """
        assert self.site == other.site
        return self.derived_state == other.inherited_state


@dataclasses.dataclass
class ReversionDescriptor:
    site: int
    child_node: int
    child_mutation: MutationDescriptor
    parent_mutation: MutationDescriptor


def nodes_mutation_descriptors(ts, nodes, show_progress=False):
    dfm = stats.mutation_data(ts, inheritance_stats=False).set_index("node")
    ret = {node: {} for node in nodes}
    nodes = np.sort(np.array(nodes))
    present_nodes = np.intersect1d(nodes, dfm.index.unique())
    subset = dfm.loc[present_nodes]
    for node, row in tqdm.tqdm(
        subset.iterrows(),
        total=subset.shape[0],
        disable=not show_progress,
        desc="mutdesc",
    ):
        desc = MutationDescriptor(
            row["site_id"],
            row["derived_state"],
            row["inherited_state"],
            row["parent"],
        )
        ret[node][desc] = row["mutation_id"]
    return ret
