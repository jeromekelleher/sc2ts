"""
Miscellanous tree operations we need for sc2ts inference.
"""

import collections
import logging
import dataclasses

import tskit
import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy
import biotite.sequence.phylo as bsp

from . import core

logger = logging.getLogger(__name__)


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


def biotite_to_tskit_tables(tree, tables):
    """
    Updates the specified set of tables with the biotite tree.
    """
    L = tables.sequence_length
    node_map = {}
    for u, node in enumerate(tree.leaves):
        node_map[node] = u
        assert u == node.get_indices()[0]

    stack = [tree.root]
    while len(stack) > 0:
        node = stack.pop()
        if not node.is_leaf():
            node_map[node] = tables.nodes.add_row()
            for child in node.children:
                stack.append(child)
        if node.parent is not None:
            tables.edges.add_row(
                0, L, parent=node_map[node.parent], child=node_map[node]
            )
    set_tree_time(tables, unit_scale=True)
    tables.sort()


def infer_binary_topology(ts, tables):
    assert ts.num_trees == 1
    assert ts.num_mutations > 0

    if ts.num_samples < 2:
        return tables.tree_sequence()

    # epsilon = 1e-6  # used in rerooting: separate internal nodes from samples by this
    # assert ts.num_trees == 1

    # # can only use simplify later to match the samples if the originals are at the start
    # # assert set(ts.samples()) == set(np.arange(ts.num_samples))
    # assert not ts.node(root).is_sample()

    # Include the root as a sample node for tree-rerooting purposes
    # sample_indexes = np.concatenate((ts.samples(), [root]))
    # G = ts.genotype_matrix(samples=sample_indexes, isolated_as_missing=False)
    G = ts.genotype_matrix()

    # Hamming distance should be suitable here because it's giving the overall
    # number of differences between the observations. Euclidean is definitely
    # not because of the allele encoding (difference between 0 and 4 is not
    # greater than 0 and 1).
    Y = scipy.spatial.distance.pdist(G.T, "hamming")

    # nj_tree = bsp.neighbor_joining(scipy.spatial.distance.squareform(Y))
    nj_tree = bsp.upgma(scipy.spatial.distance.squareform(Y))
    # print(nj_tree)
    biotite_to_tskit_tables(nj_tree, tables)
    # print(tsk_tree.draw_text())

    # # Extract the NJ tree, but rooted at the last leaf
    # root = nj_tree.leaves[-1]  # root is the last entry of the distance matrix
    # time = 1
    # parent = tables.nodes.add_row(time=time)
    # L = tables.sequence_length
    # stack = [(root, None, parent, time)]
    # node_map = {}
    # while len(stack) > 0:
    #     node, prev_node, parent, time = stack.pop()
    #     for new_node in _biotite_attached_nodes(node):
    #         print(new_node)
    #         assert new_node is not None
    #         if new_node is not prev_node:
    #             if new_node.is_leaf():
    #                 ts_node = ts.node(sample_indexes[new_node.index])
    #                 new_time = ts_node.time
    #                 u = tables.nodes.append(ts_node)
    #             else:
    #                 new_time = time - epsilon
    #                 u = tables.nodes.add_row(time=new_time)
    #             assert new_time < tables.nodes[parent].time
    #             tables.edges.add_row(parent=parent, child=u, left=0, right=L)
    #             if new_node.is_leaf():
    #                 node_map[sample_indexes[new_node.index]] = u
    #                 # print("added internal", u, f"at time {time} (parent is {parent})")
    #             else:
    #                 stack.append((new_node, node, u, new_time))
    #                 # print("made leaf", u, f"(was {sample_indexes[new_node.index]}) at
    #                 # time {time} (parent is {parent})")
    # tables.sort()
    # # Line below makes nodes in the new TS map to those in the old
    # tables.simplify([node_map[u] for u in np.arange(ts.num_samples)])
    # new_ts = tables.tree_sequence()

    # assert list(new_ts.samples()) == list(ts.samples())
    # assert new_ts.num_samples == ts.num_samples
    # assert new_ts.num_trees == 1
    return tables.tree_sequence()


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
    # Get rid of unreferenced nodes
    tables.simplify()
    return tables.tree_sequence()


def update_tables(tables, edges_to_delete, mutations_to_delete):
    # Updating the mutations is a real faff, and the only way I
    # could get it to work is by setting the time values. This should
    # be easier...
    # NOTE: this should be easier to do now that we have the "keep_rows" methods
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


def coalesce_mutations(ts, samples=None):
    """
    Examine all time-0 samples and their (full-sequence) sibs and create
    new nodes to represent overlapping sets of mutations. The algorithm
    is greedy and makes no guarantees about uniqueness or optimality.
    Also note that we don't recurse and only reason about mutation sharing
    at a single level in the tree.
    """
    # We depend on mutations having a time below.
    assert np.all(np.logical_not(np.isnan(ts.mutations_time)))
    if samples is None:
        samples = ts.samples(time=0)

    tree = ts.first()

    # Get the samples that span the whole sequence
    keep_samples = []
    for u in samples:
        e = tree.edge(u)
        assert e != -1
        edge = ts.edge(e)
        if edge.left == 0 and edge.right == ts.sequence_length:
            keep_samples.append(u)
    samples = keep_samples
    logger.info(f"Coalescing mutations for {len(samples)} full-span samples")

    # For each node in one of the sib groups, the set of mutations.
    node_mutations = {}
    for sample in samples:
        u = tree.parent(sample)
        for v in tree.children(u):
            # Filter out non-tree like things. If the edge spans the whole genome
            # then it must be present in the first tree.
            edge = ts.edge(tree.edge(v))
            assert edge.child == v and edge.parent == u
            if edge.left == 0 and edge.right == ts.sequence_length:
                if v not in node_mutations:
                    node_mutations[v] = node_mutation_descriptors(ts, v)

    # For each sample, what is the ("a" more accurately - this is greedy)
    # maximum mutation overlap with one of its sibs?
    max_sample_overlap = {}
    for sample in samples:
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
    for sample in samples:
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

    mutations_to_delete = []
    edges_to_delete = []
    for (_, overlap), sibs in sib_groups.items():
        for mut_desc in overlap:
            for sib in sibs:
                mutations_to_delete.append(node_mutations[sib][mut_desc])
                edges_to_delete.append(tree.edge(sib))

    tables = ts.dump_tables()
    for (parent, overlap), sibs in sib_groups.items():
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

        md_overlap = [(x.site, x.inherited_state, x.derived_state) for x in overlap]
        md_sibs = [int(sib) for sib in sibs]
        tables.nodes.add_row(
            flags=core.NODE_IS_MUTATION_OVERLAP,
            time=group_parent_time,
            metadata={
                "sc2ts": {
                    "overlap": md_overlap,
                    "sibs": md_sibs,
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
def push_up_reversions(ts, samples, date="1999-01-01"):
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

    # For each node check if it has an immediate reversion
    sib_groups = collections.defaultdict(list)
    for child in full_span_samples:
        parent = tree.parent(child)
        child_muts = {desc.site: desc for desc in node_mutation_descriptors(ts, child)}
        parent_muts = {
            desc.site: desc for desc in node_mutation_descriptors(ts, parent)
        }
        reversions = []
        for site in child_muts:
            if site in parent_muts:
                if child_muts[site].is_reversion_of(parent_muts[site]):
                    reversions.append((site, child))
        # Pick the maximum set of reversions per sib group so that we're not
        # trying to resolve incompatible reversion sets.
        if len(reversions) > len(sib_groups[parent]):
            sib_groups[parent] = reversions

    tables = ts.dump_tables()
    edges_to_delete = []
    mutations_to_delete = []
    for parent, reversions in sib_groups.items():
        if len(reversions) == 0:
            continue

        sample = reversions[0][1]
        assert all(x[1] == sample for x in reversions)
        sites = [x[0] for x in reversions]
        # Remove the edges above the sample and its parent
        edges_to_delete.extend([tree.edge(sample), tree.edge(parent)])
        # Create new node that is fractionally older than the current
        # parent that will be the parent of both nodes.
        grandparent = tree.parent(parent)
        # Arbitrarily make it 1/8 of the branch_length. Probably should
        # make it proportional to the number of mutations or something.
        eps = tree.branch_length(parent) * 0.125
        w_time = tree.time(parent) + eps
        w = tables.nodes.add_row(
            flags=core.NODE_IS_REVERSION_PUSH,
            time=w_time,
            metadata={
                "sc2ts": {
                    # FIXME it's not clear how helpful the metadata is here
                    # If we had separate pass for each group, it would probably
                    # be easier to reason about.
                    "sites": [int(x) for x in sites],
                    "date_added": date,
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
    return update_tables(tables, edges_to_delete, mutations_to_delete)


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


def node_mutation_descriptors(ts, u):
    """
    Return a mapping of unique mutations
    (site, inherited_state, derived_state, parent_id) to the corresponding
    mutation IDs that are on the specified node.
    """
    descriptors = {}
    for mut_id in np.where(ts.mutations_node == u)[0]:
        mut = ts.mutation(mut_id)
        inherited_state = ts.site(mut.site).ancestral_state
        if mut.parent != -1:
            parent_mut = ts.mutation(mut.parent)
            if parent_mut.node == u:
                raise ValueError("Multiple mutations on same branch not supported")
            inherited_state = parent_mut.derived_state
        assert inherited_state != mut.derived_state
        desc = MutationDescriptor(
            mut.site, mut.derived_state, inherited_state, mut.parent
        )
        assert desc not in descriptors
        descriptors[desc] = mut_id
    return descriptors
