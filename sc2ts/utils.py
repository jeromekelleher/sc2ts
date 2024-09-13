"""
Utilities for examining sc2ts output.
"""
import collections
import dataclasses
import itertools
import operator
import warnings
import datetime
import logging

import tskit
import tszip
import numpy as np
import pandas as pd

# TODO where do we use this? This is a *great* example of why not to use
# this style, because we have loads of variables called "tree" in this file.
from sklearn import tree
import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import Markdown, HTML
import networkx as nx
import numba

import sc2ts
from . import core
from . import lineages


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


def get_recombinant_edges(ts):
    """
    Return the partial edges from the tree sequence grouped by child (which must
    be flagged as a recombinant node).
    """
    partial_edges = np.where(
        np.logical_or(ts.edges_left != 0, ts.edges_right != ts.sequence_length)
    )[0]
    edges = collections.defaultdict(list)
    for edge_id in partial_edges:
        edge = ts.edge(edge_id)
        assert ts.nodes_flags[edge.child] == sc2ts.NODE_IS_RECOMBINANT
        edges[edge.child].append(edge)

    # Check that they are in order and completely cover the region
    for child_edges in edges.values():
        child_edges.sort(key=lambda e: e.left)
        assert len(child_edges) >= 2
        assert child_edges[0].left == 0
        assert child_edges[-1].right == ts.sequence_length
        last_edge = child_edges[0]
        for edge in child_edges[1:]:
            assert edge.left == last_edge.right
            last_edge = edge
    return edges


def get_recombinant_mrca_table(ts):
    """
    Return a pandas data frame of the recombinant breakpoints from the
    specified tree sequence. For each partial edge (which must have a
    node marked as NODE_IS_RECOMBINANT as child), return a row in
    the dataframe giving the breakpoint, the left parent, right parent
    and the most recent common ancestor of these parent nodes.
    """

    recombinant_edges = get_recombinant_edges(ts)
    # Split these up into adjacent pairs
    breakpoint_pairs = []
    for child, child_edges in recombinant_edges.items():
        for j in range(len(child_edges) - 1):
            assert child_edges[j].child == child
            breakpoint_pairs.append((child_edges[j], child_edges[j + 1]))
    assert len(breakpoint_pairs) >= len(recombinant_edges)

    data = []
    tree = ts.first()
    for left_edge, right_edge in sorted(breakpoint_pairs, key=lambda x: x[1].left):
        assert left_edge.right == right_edge.left
        assert left_edge.child == right_edge.child
        recombinant_node = left_edge.child
        bp = left_edge.right
        tree.seek(bp)
        assert tree.interval.left == bp
        right_path = get_root_path(tree, right_edge.parent)
        tree.prev()
        assert tree.interval.right == bp
        left_path = get_root_path(tree, left_edge.parent)
        mrca = get_path_mrca(left_path, right_path, ts.nodes_time)
        row = {
            "recombinant_node": recombinant_node,
            "breakpoint": bp,
            "left_parent": left_edge.parent,
            "right_parent": right_edge.parent,
            "mrca": mrca,
        }
        data.append(row)
    return pd.DataFrame(data, dtype=np.int32)


@dataclasses.dataclass
class HmmRun:
    breakpoints: list
    parents: list
    parent_imputed_lineages: list
    mutations: list


@dataclasses.dataclass
class ArgRecombinant:
    breakpoints: list
    breakpoint_intervals: list
    parents: list
    parent_imputed_lineages: list
    mrcas: list


@dataclasses.dataclass
class Recombinant:
    causal_strain: str
    causal_date: str
    causal_lineage: str
    hmm_runs: dict
    arg_info: ArgRecombinant
    node: int
    max_descendant_samples: int

    def data_summary(self):
        d = self.asdict()
        del d["hmm_runs"]
        del d["arg_info"]
        d["num_parents"] = self.num_parents
        d["total_cost"] = self.total_cost
        d["is_hmm_mutation_consistent"] = self.is_hmm_mutation_consistent()
        d["is_arg_hmm_path_identical"] = self.is_arg_hmm_path_identical()
        d[
            "is_arg_hmm_path_length_consistent"
        ] = self.is_arg_hmm_path_length_consistent()
        d["is_path_length_consistent"] = self.is_path_length_consistent()
        d["is_parent_lineage_consistent"] = self.is_parent_lineage_consistent()
        return d

    @property
    def total_cost(self, num_mismatches):
        """
        How different is the causal sequence from the rest, roughly?
        """
        fwd = self.hmm_runs["forward"]
        bck = self.hmm_runs["backward"]
        cost_fwd = num_mismatches * (len(fwd.parents) - 1) + len(fwd.mutations)
        cost_bck = num_mismatches * (len(bck.parents) - 1) + len(bck.mutations)
        assert cost_fwd == cost_bck
        return cost_fwd

    @property
    def num_parents(self):
        """
        The ARG version is definitive.
        """
        return len(self.arg_info.parents)

    def is_hmm_mutation_consistent(self):
        """
        Do we get the same set of mutations in the HMM in the back and
        forward runs?
        """
        fwd = self.hmm_runs["forward"]
        bck = self.hmm_runs["backward"]
        return fwd.mutations == bck.mutations

    def is_arg_hmm_path_identical(self):
        """
        Does this recombinant have the same path in the forwards HMM run, and in the ARG?
        """
        fwd = self.hmm_runs["forward"]
        arg = self.arg_info
        return fwd.parents == arg.parents and fwd.breakpoints == arg.breakpoints

    def is_arg_hmm_path_length_consistent(self):
        """
        Does this recombinant have the same path length in the forwards HMM run,
        and in the ARG?
        """
        fwd = self.hmm_runs["forward"]
        arg = self.arg_info
        return len(fwd.parents) == len(arg.parents)

    def is_path_length_consistent(self):
        """
        Returns True if all the HMM runs agree on the number of parents.
        """
        fwd = self.hmm_runs["forward"]
        bck = self.hmm_runs["backward"]
        return (
            len(fwd.parents) == len(bck.parents)
            and self.is_arg_hmm_path_length_consistent()
        )

    def is_parent_lineage_consistent(self):
        """
        Returns True if all the HMM runs agree on the imputed pango lineage status of
        parents. Implies is_path_length_consistent.
        """
        fwd = self.hmm_runs["forward"]
        bck = self.hmm_runs["backward"]
        arg = self.arg_info
        return (
            fwd.parent_imputed_lineages
            == bck.parent_imputed_lineages
            == arg.parent_imputed_lineages
        )

    def asdict(self):
        return dataclasses.asdict(self)


# https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]
        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
        return run_values, run_starts, run_lengths




def pad_sites(ts):
    """
    Fill in missing sites with the reference state.
    """
    ref = core.get_reference_sequence()
    missing_sites = set(np.arange(1, len(ref)))
    missing_sites -= set(ts.sites_position.astype(int))
    tables = ts.dump_tables()
    for pos in missing_sites:
        tables.sites.add_row(pos, ref[pos])
    tables.sort()
    return tables.tree_sequence()


def examine_recombinant(strain, ts, alignment_store, num_mismatches=3):
    # We need to do this because tsinfer won't work on the mirrored
    # coordinates unless we have all positions in the site-table.
    # This is just an annoying detail of tsinfer's implementation.
    ts = pad_sites(ts)
    num_mismatches = num_mismatches
    data = []
    for mirror in [True, False]:
        sample = sc2ts.Sample({"strain": strain})
        samples = sc2ts.match(
            samples=[sample],
            alignment_store=alignment_store,
            base_ts=ts,
            num_mismatches=num_mismatches,
            precision=14,
            num_threads=0,
            mirror_coordinates=mirror,
        )
        assert len(samples) == 1
        sample = samples[0]
        data.append(
            {
                "strain": strain,
                "num_mismatches": num_mismatches,
                "direction": ["backward", "forward"][int(not mirror)],
                "breakpoints": sample.breakpoints,
                "parents": sample.parents,
                "mutations": [str(mut) for mut in sample.mutations],
            }
        )
    return data


def get_recombinant_samples(ts):
    """
    Returns a map of recombinant nodes and their causal samples IDs.
    Only one causal strain per recombinant node is returned, chosen arbitrarily.
    """
    recomb_nodes = get_recombinants(ts)
    tree = ts.first()
    out = {}
    for u in recomb_nodes:
        node = ts.node(u)
        recomb_date = node.metadata["date_added"]
        causal_sample = -1
        # Search the subtree for a causal sample.
        for v in tree.nodes(u, order="levelorder"):
            child = ts.node(v)
            if child.is_sample() and child.metadata["date"] <= recomb_date:
                edge = ts.edge(tree.edge(v))
                assert edge.left == 0 and edge.right == ts.sequence_length
                causal_sample = child
                break
        assert causal_sample != -1
        out[u] = causal_sample.id
    assert len(set(out.values())) == len(recomb_nodes)
    assert len(out) == len(recomb_nodes)
    return out


def detach_singleton_recombinants(ts, filter_nodes=False):
    """
    Return a new tree sequence where recombinant samples that are the sole
    descendant of a recombination node have been marked as non-samples,
    causing those nodes and the recombination nodes above them
    to be detached from the topology.

    In order to ensure that that node IDs remain stable, by default,
    detached nodes are *not* filtered from the resulting tree sequence. To
    remove the nodes completely, set filter_nodes=True (but note that
    this may render invalid any node IDs used within the tree sequence's
    metadata).
    """

    is_sample_leaf = np.zeros(ts.num_nodes, dtype=bool)
    is_sample_leaf[ts.samples()] = True
    is_sample_leaf[ts.edges_parent] = False
    # parent IDs of sample leaves
    sample_leaf_parents = ts.edges_parent[
        np.isin(ts.edges_child, np.flatnonzero(is_sample_leaf))
    ]
    # get repeated parent IDs, one for each edge leading to the parent
    sample_leaf_parents = ts.edges_parent[np.isin(ts.edges_parent, sample_leaf_parents)]
    sample_leaf_parents, counts = np.unique(sample_leaf_parents, return_counts=True)
    single_sample_leaf_parents = sample_leaf_parents[counts == 1]
    # Find the ones that are also recombination nodes
    re_nodes = np.flatnonzero(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)
    single_sample_leaf_re = np.intersect1d(re_nodes, single_sample_leaf_parents)
    bad_samples = ts.edges_child[np.isin(ts.edges_parent, single_sample_leaf_re)]
    # All of these should be samples, because they were defined via single edges above a sample
    assert len(np.setdiff1d(bad_samples, ts.samples())) == 0
    keep = np.setdiff1d(ts.samples(), bad_samples)
    return ts.simplify(
        keep,
        keep_unary=True,
        filter_sites=False,
        filter_nodes=filter_nodes,
    )


def node_path_to_samples(
    nodes, ts, rootwards=True, ignore_initial=True, stop_at_recombination=False
):
    """
    Given a list of nodes, traverse rootwards (if rootwards is True) or
    tipwards (if rootwards is False) to the nearest sample nodes,
    returning all nodes on the path, including the nodes passed in.
    Note that this does not account for genomic intervals, so parent
    or child edges can be followed even if no genetic material links
    them to the passed-in nodes.

    :param rootwards bool: If True, ascend rootwards, otherwise descend tipwards.
    :param ignore_initial bool: If True, the initial nodes passed in are not considered
        as samples for the purposes of stopping the traversal.
    :param stop_at_recombination bool: If True, stop the traversal at recombination nodes.
    """
    nodes = np.array(list(nodes))
    ret = {n: True for n in nodes}  # Use a dict not a set, to maintain order
    if not ignore_initial:
        nodes = nodes[(ts.nodes_flags[nodes] & tskit.NODE_IS_SAMPLE) == 0]
        if stop_at_recombination:
            nodes = nodes[(ts.nodes_flags[nodes] & sc2ts.NODE_IS_RECOMBINANT) == 0]
    while len(nodes) > 0:
        if rootwards:
            nodes = ts.edges_parent[np.isin(ts.edges_child, nodes)]
        else:
            nodes = ts.edges_child[np.isin(ts.edges_parent, nodes)]
        ret.update({n: True for n in nodes})
        nodes = nodes[(ts.nodes_flags[nodes] & tskit.NODE_IS_SAMPLE) == 0]
        if stop_at_recombination:
            nodes = nodes[(ts.nodes_flags[nodes] & sc2ts.NODE_IS_RECOMBINANT) == 0]
    return np.array(list(ret.keys()), dtype=ts.edges_child.dtype)


def edges_for_nodes(ts, nodes, include_external=False):
    """
    Returns the edges that connect the specified numpy array of nodes in the ts.
    """
    edges = np.logical_and(
        np.isin(ts.edges_child, nodes),
        np.isin(ts.edges_parent, nodes),
    )
    return np.flatnonzero(edges)


def to_nx_subgraph(ts, nodes, return_external_edges=False):
    """
    Return a networkx graph relating the specified nodes.
    If return_external_edges is true, also return a tuple
    of (parent_edge_list, child_edge_list) giving the edges
    from the nodes that are *not* in the graph (because they
    connect to nodes not in ``nodes``)
    """
    G = nx.DiGraph()
    for u in nodes:
        G.add_node(u)
    edges = edges_for_nodes(ts, nodes)
    for parent, child in zip(ts.edges_parent[edges], ts.edges_child[edges]):
        G.add_edge(parent, child)
    if return_external_edges:
        parent_e = np.setdiff1d(np.flatnonzero(np.isin(ts.edges_child, nodes)), edges)
        parent_e = parent_e[np.argsort(ts.edges_child[parent_e])]  # Sort by child id
        child_e = np.setdiff1d(np.flatnonzero(np.isin(ts.edges_parent, nodes)), edges)
        child_e = child_e[np.argsort(ts.edges_parent[child_e])]  # Sort by parent id

        return G, (parent_e, child_e)
    return G


def plot_subgraph(
    nodes,
    ts,
    ti=None,
    mutations_json_filepath=None,
    filepath=None,
    *,
    ax=None,
    node_size=None,
    exterior_edge_len=None,
    node_colours=None,
    colour_metadata_key=None,
    ts_id_labels=None,
    node_metadata_labels=None,
    sample_metadata_labels=None,
    show_descendant_samples=None,
    edge_labels=None,
    edge_font_size=None,
    node_font_size=None,
    label_replace=None,
    node_positions=None,
):
    """
    Draws out a subgraph of the ARG defined by the provided node ids and the
    edges connecting them.

    :param list nodes: A list of node ids used in the subgraph. Only edges connecting
        these nodes will be drawn.
    :param tskit.TreeSequence ts: The tree sequence to use.
    :param TreeInfo ti: The TreeInfo instance associated with the tree sequence. If
        ``None`` calculate the TreeInfo within this function. However, as
        calculating the TreeInfo class takes some time, if you have it calculated
        already, it is far more efficient to pass it in here.
    :param str mutations_json_filepath: The path to a list of mutations (only relevant
        if ``edge_labels`` is ``None``). If provided, only mutations in this file will
        be listed on edges of the plot, with others shown as "+N mutations". If ``None``
        (default), list all mutations. If "", only plot the number of mutations.
    :param str filepath: If given, save the plot to this file path.
    :param plt.Axes ax: a matplotlib axis object on which to plot the graph.
        This allows the graph to be placed as a subplot or the size and aspect ratio
        to be adjusted. If ``None`` (default) plot to the current axis with some
        sensible figsize defaults, calling ``plt.show()`` once done.
    :param int node_size: The size of the node circles. Default:
        ``None``, treated as 2800.
    :param bool exterior_edge_len: The relative length of the short dotted lines,
        representing missing edges to nodes that we have not drawn. If ``0``,
        do not plot such lines. Default: ``None``, treated as ``0.4``.
    :param bool ts_id_labels: Should we label nodes with their tskit node ID? If
        ``None``, show the node ID only for sample nodes. If ``True``, show
        it for all nodes. If ``False``, do not show. Default: ``None``.
    :param str node_metadata_labels: Should we label all nodes with a value from their
        metadata: Default: ``None``, treated as ``"Imputed_GISAID_lineage"``. If ``""``,
        do not plot any all-node metadata.
    :param str sample_metadata_labels: Should we additionally label sample nodes with a
        value from their metadata: Default: ``None``, treated as ``"gisaid_epi_isl"``.
    :param str show_descendant_samples: Should we label nodes with the maximum number
        of samples descending from them in any tree (in the format "+XXX samples").
        If ``"samples"``, only label sample nodes. If "tips", label all tip nodes.
        If ``"sample_tips"` label all tips that are also samples. If ``"all"``, label
        all nodes. If ``""`` or False, do not show labels. Default: ``None``, treated
        as ``"sample_tips"``. If a node has no descendant samples, a label is not placed.
    :param dict edge_labels: a mapping of {(parent_id, child_id): "label")} with which
        to label the edges. If ``None``, label with mutations or (if above a
        recombination node) with the edge interval. If ``{}``, do not plot
        edge labels.
    :param float edge_font_size: The font size for edge labels.
    :param float node_font_size: The font size for node labels.
    :param dict label_replace: A dict of ``{key: value}`` such that node or edge
        labels containing the string ``key`` have that string replaced with
        ``value``. For example, the word "Unknown" can be removed from
        the plot, by specifying ``{"Unknown": "", "Unknown ": ""}``.
    :param dict node_colours: A dict mapping nodes to colour values. The keys of the
        dictionary can be integer node IDs, strings, or None. If the key is a string,
        it is compared to the value of ``node.metadata[colour_metadata_key]`` (see
        below). If no relevant key exists, the fill colour is set to the value of
        ``node_colours[None]``, or is set to empty if there is no key of ``None``.
        However, if ``node_colours`` is itself ``None``, use the default colourscheme
        which distinguishes between sample nodes, recombination nodes, and all others.
    :param dict colour_metadata_key: A key in the metadata, to use when specifying
        bespoke node colours. Default: ``None``, treated as "strain".
    :param dict node_positions: A dictionary of ``node_id: [x, y]`` positions, for
        example obtained in a previous call to this function. If ``None`` (default)
        calculate the positions using ``nx_agraph.graphviz_layout(..., prog="dot")``.


    :return: The networkx Digraph and the positions of nodes in the digraph as a dict of
        ``{node_id : (x, y), ...}``
    :rtype:  tuple(nx.DiGraph, dict)

    """

    def sort_mutation_label(s):
        """
        Mutation labels are like "A123T", "+1 mutation", or "3",
        """
        try:
            return float(s)
        except ValueError:
            if s[0] == "$":
                # matplotlib mathtext - remove the $ and the formatting
                s = (
                    s.replace("$", "")
                    .replace(r"\bf", "")
                    .replace("\it", "")
                    .replace("{", "")
                    .replace("}", "")
                )
            try:
                return float(s[1:-1])
            except ValueError:
                return np.inf  # put at the end

    if ti is None:
        ti = TreeInfo(ts)
    if node_size is None:
        node_size = 2800
    if edge_font_size is None:
        edge_font_size = 5
    if node_font_size is None:
        node_font_size = 6
    if node_metadata_labels is None:
        node_metadata_labels = "Imputed_GISAID_lineage"
    if sample_metadata_labels is None:
        sample_metadata_labels = "gisaid_epi_isl"
    if show_descendant_samples is None:
        show_descendant_samples = "sample_tips"
    if colour_metadata_key is None:
        colour_metadata_key = "strain"
    if exterior_edge_len is None:
        exterior_edge_len = 0.4

    if show_descendant_samples not in {
        "samples",
        "tips",
        "sample_tips",
        "all",
        "",
        False,
    }:
        raise ValueError(
            "show_descendant_samples must be one of 'samples', 'tips', 'sample_tips', 'all', or '' / False"
        )

    # Read in characteristic mutations info
    linmuts_dict = None
    if mutations_json_filepath is not None:
        if mutations_json_filepath == "":
            TmpClass = collections.namedtuple("TmpClass", ["all_positions"])
            linmuts_dict = TmpClass({})  # an empty dict
        else:
            linmuts_dict = lineages.read_in_mutations(mutations_json_filepath)

    exterior_edges = None
    if exterior_edge_len != 0:
        G, exterior_edges = to_nx_subgraph(ts, nodes, return_external_edges=True)
    else:
        G = to_nx_subgraph(ts, nodes)

    nodelabels = collections.defaultdict(list)
    shown_tips = []
    for u, out_deg in G.out_degree():
        node = ts.node(u)
        if node_metadata_labels:
            nodelabels[u].append(node.metadata[node_metadata_labels])
        if ts_id_labels or (ts_id_labels is None and node.is_sample()):
            nodelabels[u].append(f"tsk{node.id}")
        if node.is_sample():
            if sample_metadata_labels:
                nodelabels[u].append(node.metadata[sample_metadata_labels])
        if show_descendant_samples:
            show = True if show_descendant_samples == "all" else False
            is_tip = out_deg == 0
            if show_descendant_samples == "tips" and is_tip:
                show = True
            elif node.is_sample():
                if show_descendant_samples == "samples":
                    show = True
                elif show_descendant_samples == "sample_tips" and is_tip:
                    show = True
            if show:
                s = ti.nodes_max_descendant_samples[u]
                if node.is_sample():
                    s -= 1  # don't count self
                if s > 0:
                    nodelabels[u].append(f"+{s} {'samples' if s > 1 else 'sample'}")

    nodelabels = {k: "\n".join(v) for k, v in nodelabels.items()}

    interval_labels = {k: collections.defaultdict(str) for k in ("lft", "mid", "rgt")}
    mutation_labels = collections.defaultdict(set)

    ## Details for mutations (labels etc)
    mut_nodes = set()
    mutation_suffix = collections.defaultdict(set)
    used_edges = set(edges_for_nodes(ts, nodes))
    for m in ts.mutations():
        if m.edge in used_edges:
            mut_nodes.add(m.node)
            if edge_labels is None:
                edge = ts.edge(m.edge)
                pos = int(ts.site(m.site).position)
                includemut = False
                if m.parent == tskit.NULL:
                    inherited_state = ts.site(m.site).ancestral_state
                else:
                    inherited_state = ts.mutation(m.parent).derived_state

                if ti.mutations_is_reversion[m.id]:
                    mutstr = f"$\\bf{{{inherited_state.lower()}{pos}{m.derived_state.lower()}}}$"
                elif ts.mutations_parent[m.id] != tskit.NULL:
                    mutstr = f"$\\bf{{{inherited_state.upper()}{pos}{m.derived_state.upper()}}}$"
                else:
                    mutstr = f"{inherited_state.upper()}{pos}{m.derived_state.upper()}"
                if linmuts_dict is None or pos in linmuts_dict.all_positions:
                    includemut = True
                if includemut:
                    mutation_labels[(edge.parent, edge.child)].add(mutstr)
                else:
                    mutation_suffix[(edge.parent, edge.child)].add(mutstr)
    for key, value in mutation_suffix.items():
        mutation_labels[key].add(
            ("" if len(mutation_labels[key]) == 0 else "+")
            + f"{len(value)} mutation{'s' if len(value) > 1 else ''}"
        )

    multiline_mutation_labels = False
    for key, value in mutation_labels.items():
        mutation_labels[key] = "\n".join(sorted(value, key=sort_mutation_label))
        if len(value) > 1:
            multiline_mutation_labels = True

    if edge_labels is None:
        for pc in G.edges():
            if ts.node(pc[1]).flags & sc2ts.NODE_IS_RECOMBINANT:
                for e in edges_for_nodes(ts, pc):
                    edge = ts.edge(e)
                    lpos = "mid"
                    if edge.left == 0 and edge.right < ts.sequence_length:
                        lpos = "lft"
                    elif edge.left > 0 and edge.right == ts.sequence_length:
                        lpos = "rgt"
                    # Add spaces between or in front of labels if
                    # multiple lft or rgt labels (i.e. intervals) exist for an edge
                    if interval_labels[lpos][pc]:  # between same side labels
                        interval_labels[lpos][pc] += "  "
                    if (
                        lpos == "rgt" and interval_labels["lft"][pc]
                    ):  # in front of rgt label
                        interval_labels[lpos][pc] = "  " + interval_labels[lpos][pc]
                    interval_labels[lpos][pc] += f"{int(edge.left)}…{int(edge.right)}"
                    if (
                        lpos == "lft" and interval_labels["rgt"][pc]
                    ):  # at end of lft label
                        interval_labels[lpos][pc] += "  "

    if label_replace is not None:
        for search, replace in label_replace.items():
            for k, v in nodelabels.items():
                nodelabels[k] = v.replace(search, replace)
            for k, v in mutation_labels.items():
                mutation_labels[k] = v.replace(search, replace)
            for key in interval_labels.keys():
                for k, v in interval_labels[key].items():
                    interval_labels[key][k] = v.replace(search, replace)

    # Shouldn't need this once https://github.com/jeromekelleher/sc2ts/issues/132 fixed
    unary_nodes_to_remove = set()
    for (k, in_deg), (k2, out_deg) in zip(G.in_degree(), G.out_degree()):
        assert k == k2
        flags = ts.node(k).flags
        if (
            in_deg == 1
            and out_deg == 1
            and k not in mut_nodes
            and not (flags & sc2ts.NODE_IS_RECOMBINANT)
        ):
            G.add_edge(*G.predecessors(k), *G.successors(k))
            for d in [mutation_labels, *list(interval_labels.values()), edge_labels]:
                if d is not None and (k, *G.successors(k)) in d:
                    d[(*G.predecessors(k), *G.successors(k))] = d.pop(
                        (k, *G.successors(k))
                    )
            unary_nodes_to_remove.add(k)
    [G.remove_node(k) for k in unary_nodes_to_remove]
    nodelabels = {k: v for k, v in nodelabels.items() if k not in unary_nodes_to_remove}

    if node_positions is None:
        node_positions = nx.nx_agraph.graphviz_layout(G, prog="dot")
    if ax is None:
        dim_x = len(set(x for x, y in node_positions.values()))
        dim_y = len(set(y for x, y in node_positions.values()))
        fig, ax = plt.subplots(1, 1, figsize=(dim_x * 1.5, dim_y * 1.1))

    if exterior_edges is not None:
        # Draw a short dotted line above nodes with extra parent edges to show that more
        # topology exists above them. For simplicity we assume when calculating how to
        # space the lines that no other parent edges from this node have been plotted.
        # Parent edges are sorted by child id, so we can use this to groupby

        # parent-child dist
        av_y = np.mean(
            [node_positions[u][1] - node_positions[v][1] for u, v in G.edges()]
        )
        # aspect_ratio = np.divide(*np.ptp([[x, y] for x, y in node_positions.values()], axis=0))
        aspect_ratio = 1.0
        for child, edges in itertools.groupby(
            exterior_edges[0], key=lambda e: ts.edge(e).child
        ):
            edges = list(edges)[:6]  # limit to 6 lines, otherwise it gets messy
            for x in [0] if len(edges) < 2 else np.linspace(-1, 1, len(edges)):
                dx = x * aspect_ratio * av_y * exterior_edge_len
                dy = av_y * exterior_edge_len
                # make lines the same length
                hypotenuse = np.sqrt(dx**2 + dy**2)
                dx *= dy / hypotenuse
                dy *= dy / hypotenuse
                ax.plot(
                    [node_positions[child][0], node_positions[child][0] + dx],
                    [node_positions[child][1], node_positions[child][1] + dy],
                    marker="",
                    linestyle=":",
                    color="gray",
                    zorder=-1,
                )

        # Draw a short dotted line below nodes with extra child edges to show that more
        # topology exists below them. For simplicity we assume when calculating how to
        # space the lines that no other child edges from this node have been plotted.
        # Child edges are sorted by child id, so we can use this to groupby
        for parent, edges in itertools.groupby(
            exterior_edges[1], key=lambda e: ts.edge(e).parent
        ):
            edges = list(edges)[:6]  # limit to 6 lines, otherwise it gets messy
            for x in [0] if len(edges) < 2 else np.linspace(-1, 1, len(edges)):
                dx = x * aspect_ratio * av_y * exterior_edge_len
                dy = av_y * exterior_edge_len
                # make lines the same length
                hypotenuse = np.sqrt(dx**2 + dy**2)
                dx *= dy / hypotenuse
                dy *= dy / hypotenuse
                ax.plot(
                    [node_positions[parent][0], node_positions[parent][0] + dx],
                    [node_positions[parent][1], node_positions[parent][1] - dy],
                    marker="",
                    linestyle=":",
                    color="gray",
                    zorder=-1,
                )

    fill_cols = []
    if node_colours is None:
        for u in G.nodes:
            fill_cols.append(
                "k" if ts.node(u).flags & sc2ts.NODE_IS_RECOMBINANT else "white"
            )
    else:
        default_colour = node_colours.get(None, "None")
        for u in G.nodes:
            try:
                fill_cols.append(node_colours[u])
            except KeyError:
                md_val = ts.node(u).metadata.get(colour_metadata_key, None)
                fill_cols.append(node_colours.get(md_val, default_colour))

    # Put a line around the point if white or transparent
    stroke_cols = [
        "black"
        if col == "None" or np.mean(colors.ColorConverter.to_rgb(col)) > 0.99
        else col
        for col in fill_cols
    ]
    fill_cols = np.array(fill_cols)
    stroke_cols = np.array(stroke_cols)

    is_sample = np.array([ts.node(u).is_sample() for u in G.nodes])
    # Use a loop so allow possiblity of different shapes for samples and non-samples
    for use_sample, shape, size in zip(
        [True, False], ["o", "o"], [node_size, node_size / 3]
    ):
        node_list = np.array(list(G.nodes))
        use = is_sample == use_sample
        nx.draw_networkx_nodes(
            G,
            node_positions,
            nodelist=node_list[use],
            ax=ax,
            node_color=fill_cols[use],
            edgecolors=stroke_cols[use],
            node_size=size,
            node_shape=shape,
        )
    nx.draw_networkx_edges(
        G,
        node_positions,
        ax=ax,
        node_size=np.where(is_sample, node_size, node_size / 3),
        arrowstyle="-",
    )

    black_labels = {}
    white_labels = {}
    for node, col in zip(list(G), fill_cols):
        if node in nodelabels:
            if col == "None" or np.mean(colors.ColorConverter.to_rgb(col)) > 0.2:
                black_labels[node] = nodelabels[node]
            else:
                white_labels[node] = nodelabels[node]
    if black_labels:
        nx.draw_networkx_labels(
            G,
            node_positions,
            ax=ax,
            labels=black_labels,
            font_size=node_font_size,
            font_color="k",
        )
    if white_labels:
        nx.draw_networkx_labels(
            G,
            node_positions,
            ax=ax,
            labels=white_labels,
            font_size=node_font_size,
            font_color="w",
        )
    av_dy = np.median(
        [
            # We could use the minimum y diff here, but then could be susceptible to
            # pathological cases where the y diff is very small.
            np.abs(node_positions[u][1] - node_positions[v][1])
            for u, v in G.edges
        ]
    )
    ax_height = np.diff(ax.get_ylim())
    height_pts = (
        ax.get_position().transformed(ax.get_figure().transFigure).height
        * 72
        / ax.get_figure().dpi
    )
    node_height = np.sqrt(node_size) / height_pts * ax_height
    if multiline_mutation_labels:
        # Bottom align mutations: useful when there are multiple lines of mutations
        mut_pos = node_height / 2 / av_dy
        mut_v_align = "bottom"
    else:
        # Center align mutations, still placed near the child if possible
        font_height = edge_font_size / height_pts * ax_height
        mut_pos = (node_height / 2 + font_height / 2) / av_dy
        if mut_pos > 0.5:
            # Never go further up the line than the middle
            mut_pos = 0.5
        mut_v_align = "center"

    for name, (labels, position, valign, halign) in {
        "mutations": [mutation_labels, mut_pos, mut_v_align, "center"],
        "user": [edge_labels, 0.5, "center", "center"],
        "intervals_l": [interval_labels["lft"], 0.6, "top", "right"],
        "intervals_m": [interval_labels["mid"], 0.6, "top", "center"],
        "intervals_r": [interval_labels["rgt"], 0.6, "top", "left"],
    }.items():
        if labels:
            font_color = "darkred" if name == "mutations" else "k"
            nx.draw_networkx_edge_labels(
                G,
                node_positions,
                ax=ax,
                edge_labels=labels,
                label_pos=position,
                verticalalignment=valign,
                horizontalalignment=halign,
                font_color=font_color,
                rotate=False,
                font_size=edge_font_size,
                bbox={"facecolor": "white", "pad": 0.5, "edgecolor": "none"},
            )
    if filepath:
        plt.savefig(filepath)
    elif ax is None:
        plt.show()
    return G, node_positions


def sample_subgraph(sample_node, ts, ti=None, **kwargs):
    """
    Returns a subgraph of the tree sequence containing the specified nodes.
    """
    # Ascend up from input node
    up_nodes = node_path_to_samples([sample_node], ts)
    # Descend from these
    nodes = sc2ts.node_path_to_samples(
        up_nodes, ts, rootwards=False, ignore_initial=False
    )
    # Ascend again, to get parents of downward nonsamples
    up_nodes = sc2ts.node_path_to_samples(nodes, ts, ignore_initial=False)
    nodes = np.append(nodes, up_nodes)
    # Remove duplicates
    _, idx = np.unique(nodes, return_index=True)
    nodes = nodes[np.sort(idx)]

    return plot_subgraph(nodes, ts, ti, **kwargs)


def imputation_setup(filepath, verbose=False):
    """
    Reads in JSON of lineage-defining mutations and constructs decision tree classifier
    JSON can be downloaded from covidcg.org -> 'Compare AA mutations' -> Download -> 'Consensus mutations'
    (setting mutation type to 'NT' and consensus threshold to 0.9)
    """
    linmuts_dict = lineages.read_in_mutations(filepath)
    df, df_ohe, ohe = lineages.read_in_mutations_json(filepath)

    # Get decision tree
    y = df_ohe.index  # lineage labels
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(df_ohe, y)

    if verbose:
        # Check tree works and that lineages-defining mutations are unique for each lineage
        y_pred = clf.predict(df_ohe)
        correct = incorrect = lineage_definition_issue = 0
        for yy, yy_pred in zip(y, y_pred):
            if yy == yy_pred:
                correct += 1
            else:
                incorrect += 1
                if linmuts_dict.get_mutations(yy) == linmuts_dict.get_mutations(
                    yy_pred
                ):
                    lineage_definition_issue += 1
                    print(yy_pred, "same mutations as", yy)
        print(
            "Correct:",
            correct,
            "incorrect:",
            incorrect,
            "of which due to lineage definition ambiguity:",
            lineage_definition_issue,
        )

    return linmuts_dict, df, df_ohe, ohe, clf


def lineage_imputation(filepath, ts, ti, internal_only=False, verbose=False):
    """
    Runs lineage imputation on input ts
    """
    linmuts_dict, df, df_ohe, ohe, clf = imputation_setup(filepath, verbose)
    print("Recording relevant mutations for each node...")
    node_to_mut_dict = lineages.get_node_to_mut_dict(ts, ti, linmuts_dict)
    edited_ts = lineages.impute_lineages(
        ts, ti, node_to_mut_dict, df, ohe, clf, "Nextclade_pango", internal_only
    )
    edited_ts = lineages.impute_lineages(
        edited_ts, ti, node_to_mut_dict, df, ohe, clf, "GISAID_lineage", internal_only
    )
    return edited_ts


def add_gisaid_lineages_to_ts(ts, node_gisaid_lineages, linmuts_dict):
    """
    Adds lineages from GISAID to ts metadata (as 'GISAID_lineage').
    """
    tables = ts.tables
    new_metadata = []
    ndiffs = 0
    for node in ts.nodes():
        md = node.metadata
        if node_gisaid_lineages[node.id] is not None:
            if node_gisaid_lineages[node.id] in linmuts_dict.names:
                md["GISAID_lineage"] = str(node_gisaid_lineages[node.id])
            else:
                md["GISAID_lineage"] = md["Nextclade_pango"]
                ndiffs += 1
        new_metadata.append(md)
    validated_metadata = [
        tables.nodes.metadata_schema.validate_and_encode_row(row)
        for row in new_metadata
    ]
    tables.nodes.packset_metadata(validated_metadata)
    edited_ts = tables.tree_sequence()
    print("Filling in missing GISAID lineages with Nextclade lineages:", ndiffs)
    return edited_ts


# NOTE: this is broken since moving to Viridian metadata, we no longer have
# GISAID EPI ISL in the metadata
def check_lineages(
    ts,
    ti,
    gisaid_data,
    linmuts_dict,
    diff_filehandle="lineage_disagreement",
):
    n_diffs = 0
    total = 0
    diff_file = diff_filehandle + ".csv"
    node_gisaid_lineages = [None] * ts.num_nodes
    with tqdm.tqdm(total=len(gisaid_data)) as pbar:
        with open(diff_file, "w") as file:
            file.write("sample_node,gisaid_epi_isl,gisaid_lineage,ts_lineage\n")
            for gisaid_id, gisaid_lineage in gisaid_data:
                if gisaid_id in ti.epi_isl_map:
                    sample_node = ts.node(ti.epi_isl_map[gisaid_id])
                    if gisaid_lineage != sample_node.metadata["Nextclade_pango"]:
                        n_diffs += 1
                        file.write(
                            str(sample_node.id)
                            + ","
                            + gisaid_id
                            + ","
                            + gisaid_lineage
                            + ","
                            + sample_node.metadata["Nextclade_pango"]
                            + "\n"
                        )
                    node_gisaid_lineages[sample_node.id] = gisaid_lineage
                    total += 1
                pbar.update(1)
    print("ts number of samples:", ts.num_samples)
    print("number matched to gisaid data:", total)
    print("number of differences:", n_diffs)
    print("proportion:", n_diffs / total)

    edited_ts = add_gisaid_lineages_to_ts(ts, node_gisaid_lineages, linmuts_dict)

    return edited_ts


def compute_left_bound(ts, parents, right):
    right_index = np.searchsorted(ts.sites_position, right)
    assert ts.sites_position[right_index] == right
    variant = tskit.Variant(ts, samples=parents, isolated_as_missing=False)
    variant.decode(right_index)
    assert variant.genotypes[0] != variant.genotypes[1]
    left_index = right_index - 1
    variant.decode(left_index)
    while left_index >= 0 and variant.genotypes[0] == variant.genotypes[1]:
        left_index -= 1
        variant.decode(left_index)
    assert variant.genotypes[0] != variant.genotypes[1]
    left_index += 1
    return int(ts.sites_position[left_index])


def add_breakpoints_to_recombinant_metadata(ts):
    """
    Compute the recombinant breakpoint intervals, and write out to the
    metadata.
    """
    tables = ts.dump_tables()
    iterator = get_recombinant_edges(ts).items()
    for child, edges in tqdm.tqdm(iterator):
        intervals = []
        for j in range(len(edges) - 1):
            right = int(edges[j].right)
            parents = [edges[j].parent, edges[j + 1].parent]
            left = compute_left_bound(ts, parents, right)
            # The interval is right-exclusive, and the current right coord
            # the rightmost value that it can be.
            assert left <= right
            intervals.append((left, right + 1))
        row = tables.nodes[child]
        md = row.metadata
        md["breakpoint_intervals"] = intervals
        # Note this isn't very efficient - would possibly be better to flush
        # the whole column out at the end
        tables.nodes[child] = row.replace(metadata=md)
    return tables.tree_sequence()
