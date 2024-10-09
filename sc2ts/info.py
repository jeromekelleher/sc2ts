import collections
import logging
import json
import warnings
import dataclasses
import datetime
import re
from typing import List

import numba
import tskit
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import Markdown, HTML

from . import core
from . import utils


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LineageDetails:
    """
    Details about major known lineages that we can use for QC purposes.

    https://github.com/jeromekelleher/sc2ts/issues/290

    The month and year of the first detected sample are taken from
    https://www.ecdc.europa.eu/en/covid-19/variants-concern
    """

    pango_lineage: str
    nextstrain_clade: str
    who_label: str
    date: str
    mutations: list


# TODO reduce precision on these dates to month
major_lineages = [
    LineageDetails(
        "B.1.1.7",
        "20I",
        "Alpha",
        "2020-09",
        ["C5388A", "C3267T"],
    ),
    LineageDetails(
        "B.1.351",
        "20H",
        "Beta",
        "2020-09",
        None,  # TODO: From which source?
    ),
    LineageDetails(
        "B.1.617.2",
        "21A",
        "Delta",
        "2020-12",
        [
            "C23012G",
            "T26767C",
            "A28461G",
            "C22995A",
            "C27752T",
        ],
    ),
    LineageDetails(
        "P.1",
        "20J",
        "Gamma",
        "2020-12",
        None,  # TODO: From which source?
    ),
    LineageDetails(
        "BA.1",
        "21K",
        "Omicron",
        "2021-11",
        [
            "C21762T",
            "C2790T",
            "A11537G",
            "A26530G",
            "T22673C",
            "G23048A",
            "C24130A",
            "C23202A",
            "C24503T",
            "T13195C",
            "C25584T",
            "C15240T",
            "G8393A",
            "C25000T",
        ],
    ),
    LineageDetails(
        "BA.2",
        "21L",
        "Omicron",
        "2021-11",
        [
            "C10198T",
            "T22200G",
            "C17410T",
            "A22786C",
            "C21618T",
            "C19955T",
            "A20055G",
            "C25584T",
            "A22898G",
            "C25000T",
        ],
    ),
    LineageDetails(
        "BA.4",
        "22A",
        "Omicron",
        "2022-01",
        ["C28724T"],
    ),
    LineageDetails(
        "BA.5",
        "22B",
        "Omicron",
        "2022-02",
        ["T27383A", "C27382G"],
    ),
]


def tally_lineages(ts, metadata_db, show_progress=False):
    cov_lineages = core.get_cov_lineages_data()

    md = ts.metadata["sc2ts"]
    date = md["date"]
    # Take the exact matches into account also.
    counter = collections.Counter(md["num_exact_matches"])
    key = "Viridian_pangolin"
    iterator = tqdm.tqdm(
        ts.samples()[1:],
        desc="ARG metadata",
        disable=not show_progress,
    )
    for u in iterator:
        node = ts.node(u)
        counter[node.metadata[key]] += 1

    # print(counter)
    result = metadata_db.query(
        f"SELECT {key}, COUNT(*) FROM samples "
        f"WHERE date <= '{date}'"
        f" GROUP BY {key}"
    )
    data = []
    today = datetime.datetime.fromisoformat(date)
    for row in result:
        pango = row[key]
        if pango in cov_lineages:
            lin_data = cov_lineages[pango]
        else:
            logger.warning(f"Lineage {pango} not in cov-lineages dataset")
            lin_data = core.CovLineage(".", date, date, "")
        # Some lineages don't have an earliest date
        if lin_data.earliest_date == "":
            logger.warning(f"Lineage {pango} has no earliest date")
            lin_data.earliest_date = "2019-12-01"
        if lin_data.latest_date == "":
            logger.warning(f"Lineage {pango} has no latest date")
            lin_data.earliest_date = "2029-12-01"
        earliest_date = datetime.datetime.fromisoformat(lin_data.earliest_date)
        data.append(
            {
                "arg_count": counter[pango],
                "db_count": row["COUNT(*)"],
                "earliest_date": lin_data.earliest_date,
                "latest_date": lin_data.latest_date,
                "earliest_date_offset": (today - earliest_date).days,
                "pango": pango,
            }
        )
    return pd.DataFrame(data).sort_values("arg_count", ascending=False)


def get_recombinant_samples(ts):
    """
    Returns a map of recombinant nodes and their causal samples IDs.
    Only one causal strain per recombinant node is returned, chosen arbitrarily.
    """
    recomb_nodes = np.where((ts.nodes_flags & core.NODE_IS_RECOMBINANT) > 0)[0]
    tree = ts.first()
    out = {}
    for u in recomb_nodes:
        node = ts.node(u)
        recomb_date = node.metadata["sc2ts"]["date_added"]
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
        assert ts.nodes_flags[edge.child] == core.NODE_IS_RECOMBINANT
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


def max_descendant_samples(ts, show_progress=True):
    """
    Returns the maximum number of descendant samples for each node in the
    tree as an array.
    """
    tree = ts.first()
    num_samples = np.zeros(ts.num_nodes, dtype=np.int32)
    iterator = tqdm.tqdm(
        tree.preorder(),
        desc="Counting descendants ",
        total=ts.num_nodes,
        disable=not show_progress,
    )
    for u in iterator:
        num_samples[u] = tree.num_samples(u)
    iterator = ts.edge_diffs()
    # Skip the first tree to make things a bit quicker.
    next(iterator)
    tree.next()
    for (left, right), _, edges_in in iterator:
        assert tree.interval == (left, right)
        for edge in edges_in:
            u = edge.parent
            while u != -1:
                num_samples[u] = max(num_samples[u], tree.num_samples(u))
                u = tree.parent(u)
        tree.next()
    return num_samples


class TreeInfo:
    def __init__(
        self, ts, *, quick=False, show_progress=True, pango_source="Viridian_pangolin"
    ):
        self.ts = ts
        self.pango_source = pango_source
        self.strain_map = {}
        self.recombinants = np.where(ts.nodes_flags == core.NODE_IS_RECOMBINANT)[0]

        self.nodes_max_descendant_samples = None
        self.nodes_date = None
        self.nodes_num_missing_sites = None
        self.nodes_metadata = None

        top_level_md = ts.metadata["sc2ts"]
        self.date = top_level_md["date"]
        samples = ts.samples()
        self.strain_map = dict(zip(top_level_md["samples_strain"], ts.samples()))

        self.sites_num_mutations = np.bincount(
            ts.mutations_site, minlength=ts.num_sites
        )
        self.nodes_num_mutations = np.bincount(
            ts.mutations_node, minlength=ts.num_nodes
        )
        self.nodes_num_parents = np.bincount(ts.edges_child, minlength=ts.num_edges)

        # The number of samples per day in time-ago (i.e., the nodes_time units).
        self.num_samples_per_day = np.bincount(ts.nodes_time[samples].astype(int))

        if not quick:
            self._preprocess_nodes(show_progress)
            self._preprocess_sites(show_progress)
            self._preprocess_mutations(show_progress)

    def node_counts(self):
        mc_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_MUTATION_OVERLAP)
        pr_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_REVERSION_PUSH)
        re_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_RECOMBINANT)
        exact_matches = np.sum((self.ts.nodes_flags & core.NODE_IS_EXACT_MATCH) > 0)
        sg_nodes = np.sum((self.ts.nodes_flags & core.NODE_IN_SAMPLE_GROUP) > 0)
        rsg_nodes = np.sum(
            (self.ts.nodes_flags & core.NODE_IN_RETROSPECTIVE_SAMPLE_GROUP) > 0
        )
        immediate_reversion_marker = np.sum(
            (self.ts.nodes_flags & core.NODE_IS_IMMEDIATE_REVERSION_MARKER) > 0
        )

        nodes_with_zero_muts = np.sum(self.nodes_num_mutations == 0)
        return {
            "sample": self.ts.num_samples,
            "ex": exact_matches,
            "mc": mc_nodes,
            "pr": pr_nodes,
            "re": re_nodes,
            "sg": sg_nodes,
            "rsg": rsg_nodes,
            "imr": immediate_reversion_marker,
            "zero_muts": nodes_with_zero_muts,
        }

    def _preprocess_nodes(self, show_progress):
        ts = self.ts
        self.nodes_max_descendant_samples = max_descendant_samples(
            ts, show_progress=show_progress
        )
        self.nodes_date = np.zeros(ts.num_nodes, dtype="datetime64[D]")
        self.nodes_num_missing_sites = np.zeros(ts.num_nodes, dtype=np.int32)
        self.nodes_num_deletion_sites = np.zeros(ts.num_nodes, dtype=np.int32)
        self.nodes_metadata = {}
        self.nodes_sample_group = collections.defaultdict(list)
        samples = ts.samples()
        last_sample = ts.node(samples[-1])

        self.nodes_date[last_sample.id] = last_sample.metadata["date"]
        self.time_zero_as_date = self.nodes_date[last_sample.id]
        self.earliest_pango_lineage = {}
        self.pango_lineage_samples = collections.defaultdict(list)

        iterator = tqdm.tqdm(
            ts.nodes(),
            desc="Indexing metadata    ",
            total=ts.num_nodes,
            disable=not show_progress,
        )
        for node in iterator:
            md = node.metadata
            self.nodes_metadata[node.id] = md
            group_id = None
            sc2ts_md = md["sc2ts"]
            group_id = sc2ts_md.get("group_id", None)
            if group_id is not None:
                self.nodes_sample_group[group_id].append(node.id)
            if node.is_sample():
                self.nodes_date[node.id] = md["date"]
                pango = md.get(self.pango_source, "unknown")
                self.pango_lineage_samples[pango].append(node.id)
                self.nodes_num_missing_sites[node.id] = sc2ts_md.get(
                    "num_missing_sites", 0
                )
                try:
                    deletions = sc2ts_md["alignment_composition"].get("-", 0)
                except KeyError:
                    deletions = -1
                self.nodes_num_deletion_sites[node.id] = deletions
            else:
                # Rounding down here, might be misleading
                self.nodes_date[node.id] = self.time_zero_as_date - int(
                    self.ts.nodes_time[node.id]
                )

    def _preprocess_sites(self, show_progress):
        self.sites_num_missing_samples = np.full(self.ts.num_sites, -1, dtype=int)
        self.sites_num_deletion_samples = np.full(self.ts.num_sites, -1, dtype=int)
        for site in self.ts.sites():
            md = site.metadata
            try:
                self.sites_num_missing_samples[site.id] = md["sc2ts"]["missing_samples"]
                self.sites_num_deletion_samples[site.id] = md["sc2ts"][
                    "deletion_samples"
                ]
            except KeyError:
                # Both of these keys were added at the same time, so no point
                # in doing two try/catches here.
                pass

    def _preprocess_mutations(self, show_progress):
        ts = self.ts

        # Mutation states
        # https://github.com/tskit-dev/tskit/issues/2631
        tables = self.ts.tables
        assert np.all(
            tables.mutations.derived_state_offset == np.arange(ts.num_mutations + 1)
        )
        derived_state = tables.mutations.derived_state.view("S1").astype(str)
        assert np.all(
            tables.sites.ancestral_state_offset == np.arange(ts.num_sites + 1)
        )
        ancestral_state = tables.sites.ancestral_state.view("S1").astype(str)
        del tables
        inherited_state = ancestral_state[ts.mutations_site]
        mutations_with_parent = ts.mutations_parent != -1

        parent = ts.mutations_parent[mutations_with_parent]
        assert np.all(parent >= 0)
        inherited_state[mutations_with_parent] = derived_state[parent]
        self.mutations_derived_state = derived_state
        self.mutations_inherited_state = inherited_state

        self.sites_ancestral_state = ancestral_state
        assert np.all(self.mutations_inherited_state != self.mutations_derived_state)
        self.mutations_position = self.ts.sites_position[self.ts.mutations_site].astype(
            int
        )

        N = ts.num_mutations
        # The number of samples that descend from this mutation
        mutations_num_descendants = np.zeros(N, dtype=int)
        # The number of samples that actually inherit this mutation
        mutations_num_inheritors = np.zeros(N, dtype=int)
        # The depth of the mutation tree - i.e., how long the chain of
        # mutations is back to the ancestral state.
        mutations_num_parents = np.zeros(N, dtype=int)
        # A mutation is a reversion if its derived state is equal to the
        # inherited state of its parent
        is_reversion = np.zeros(ts.num_mutations, dtype=bool)
        # An immediate reversion is one which occurs on the immediate
        # parent in the tree.
        is_immediate_reversion = np.zeros(ts.num_mutations, dtype=bool)
        # Classify transitions and tranversions
        mutations_is_transition = np.zeros(ts.num_mutations, dtype=bool)
        mutations_is_transversion = np.zeros(ts.num_mutations, dtype=bool)
        # TODO maybe we could derive these later rather than storing?
        sites_num_transitions = np.zeros(ts.num_sites, dtype=int)
        sites_num_transversions = np.zeros(ts.num_sites, dtype=int)

        transitions = {("A", "G"), ("G", "A"), ("T", "C"), ("C", "T")}
        transversions = set()
        for b1 in "ACGT":
            for b2 in "ACGT":
                if b1 != b2 and (b1, b2) not in transitions:
                    transversions.add((b1, b2))

        tree = ts.first()
        iterator = tqdm.tqdm(
            np.arange(N), desc="Classifying mutations", disable=not show_progress
        )
        for mut_id in iterator:
            tree.seek(self.mutations_position[mut_id])
            mutation_node = ts.mutations_node[mut_id]
            descendants = tree.num_samples(mutation_node)
            mutations_num_descendants[mut_id] = descendants
            mutations_num_inheritors[mut_id] = descendants
            # Subtract this number of descendants from the parent mutation. We are
            # guaranteed to list parents mutations before their children
            parent = ts.mutations_parent[mut_id]
            if parent != -1:
                mutations_num_inheritors[parent] -= descendants
                is_reversion[mut_id] = inherited_state[parent] == derived_state[mut_id]
                if ts.mutations_node[parent] == tree.parent(mutation_node):
                    is_immediate_reversion[mut_id] = True

            num_parents = 0
            while parent != -1:
                num_parents += 1
                parent = ts.mutations_parent[parent]
            mutations_num_parents[mut_id] = num_parents
            # Ts/Tvs
            key = (inherited_state[mut_id], derived_state[mut_id])
            mutations_is_transition[mut_id] = key in transitions
            mutations_is_transversion[mut_id] = key in transversions
            site = ts.mutations_site[mut_id]
            sites_num_transitions[site] += mutations_is_transition[mut_id]
            sites_num_transversions[site] += mutations_is_transversion[mut_id]

        # Note: no real good reason for not just using self.mutations_num_descendants
        # etc above
        self.mutations_num_descendants = mutations_num_descendants
        self.mutations_num_inheritors = mutations_num_inheritors
        self.mutations_num_parents = mutations_num_parents
        self.mutations_is_reversion = is_reversion
        self.mutations_is_immediate_reversion = is_immediate_reversion
        self.mutations_is_transition = mutations_is_transition
        self.mutations_is_transversion = mutations_is_transversion
        self.sites_num_transitions = sites_num_transitions
        self.sites_num_transversions = sites_num_transversions

    def summary(self):
        # TODO use the node_counts function above
        mc_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_MUTATION_OVERLAP)
        pr_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_REVERSION_PUSH)
        re_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_RECOMBINANT)
        exact_matches = np.sum((self.ts.nodes_flags & core.NODE_IS_EXACT_MATCH) > 0)
        imr_nodes = np.sum(
            (self.ts.nodes_flags == core.NODE_IS_IMMEDIATE_REVERSION_MARKER)
        )

        samples = self.ts.samples()[1:]  # skip reference
        nodes_with_zero_muts = np.sum(self.nodes_num_mutations == 0)
        sites_with_zero_muts = np.sum(self.sites_num_mutations == 0)
        latest_sample = self.nodes_date[samples[-1]]
        missing_sites_per_sample = self.nodes_num_missing_sites[samples]
        deletion_sites_per_sample = self.nodes_num_deletion_sites[samples]
        non_samples = (self.ts.nodes_flags & tskit.NODE_IS_SAMPLE) == 0
        max_non_sample_mutations = np.max(self.nodes_num_mutations[non_samples])
        insertions = np.sum(self.mutations_inherited_state == "-")
        deletions = np.sum(self.mutations_derived_state == "-")

        data = [
            ("latest_sample", latest_sample),
            ("samples", self.ts.num_samples),
            ("nodes", self.ts.num_nodes),
            ("exact_matches", exact_matches),
            ("mc_nodes", mc_nodes),
            ("pr_nodes", pr_nodes),
            ("re_nodes", re_nodes),
            ("imr_nodes", imr_nodes),
            ("mutations", self.ts.num_mutations),
            ("recurrent", np.sum(self.ts.mutations_parent != -1)),
            ("reversions", np.sum(self.mutations_is_reversion)),
            ("immediate_reversions", np.sum(self.mutations_is_immediate_reversion)),
            ("private_mutations", np.sum(self.mutations_num_descendants == 1)),
            ("transitions", np.sum(self.mutations_is_transition)),
            ("transversions", np.sum(self.mutations_is_transversion)),
            ("insertions", insertions),
            ("deletions", deletions),
            ("max_mutations_parents", np.max(self.mutations_num_parents)),
            ("nodes_with_zero_muts", nodes_with_zero_muts),
            ("sites_with_zero_muts", sites_with_zero_muts),
            ("max_mutations_per_site", np.max(self.sites_num_mutations)),
            ("mean_mutations_per_site", np.mean(self.sites_num_mutations)),
            ("median_mutations_per_site", np.median(self.sites_num_mutations)),
            ("max_mutations_per_node", np.max(self.nodes_num_mutations)),
            ("max_mutations_per_non_sample_node", max_non_sample_mutations),
            ("max_missing_sites_per_sample", np.max(missing_sites_per_sample)),
            ("mean_missing_sites_per_sample", np.mean(missing_sites_per_sample)),
            ("max_missing_samples_per_site", np.max(self.sites_num_missing_samples)),
            ("mean_missing_samples_per_site", np.mean(self.sites_num_missing_samples)),
            ("max_deletion_sites_per_sample", np.max(deletion_sites_per_sample)),
            ("mean_deletion_sites_per_sample", np.mean(deletion_sites_per_sample)),
            ("max_deletion_samples_per_site", np.max(self.sites_num_deletion_samples)),
            (
                "mean_deletion_samples_per_site",
                np.mean(self.sites_num_deletion_samples),
            ),
            ("max_samples_per_day", np.max(self.num_samples_per_day)),
            ("mean_samples_per_day", np.mean(self.num_samples_per_day)),
        ]
        df = pd.DataFrame(
            {"property": [d[0] for d in data], "value": [d[1] for d in data]}
        )
        return df.set_index("property")

    def _node_mutation_summary(self, u, child_mutations=True):
        mutations_above = self.ts.mutations_node == u
        assert np.sum(mutations_above) == self.nodes_num_mutations[u]

        data = {
            "mutations": self.nodes_num_mutations[u],
            "reversions": np.sum(self.mutations_is_reversion[mutations_above]),
            "immediate_reversions": np.sum(
                self.mutations_is_immediate_reversion[mutations_above]
            ),
        }
        if child_mutations:
            children = self.ts.edges_child[self.ts.edges_parent == u]
            num_child_reversions = 0
            num_child_mutations = 0
            for child in np.unique(children):
                child_mutations = self.ts.mutations_node == child
                num_child_mutations += np.sum(child_mutations)
                num_child_reversions += np.sum(
                    self.mutations_is_reversion[child_mutations]
                )
            data["child_mutations"] = num_child_mutations
            data["child_reversions"] = num_child_reversions
        return data

    def _node_summary(self, u, child_mutations=True):
        md = self.nodes_metadata[u]
        qc_map = {"good": "0", "mediocre": "1"}
        qc_fields = [
            "qc.missingData.status",
            "qc.frameShifts.status",
            "qc.mixedSites.status",
            "qc.stopCodons.status",
        ]
        qc = ""
        for qc_type in qc_fields:
            status = "-"  # missing
            if qc_type in md:
                status = qc_map[md[qc_type]]
            qc += status
        flags = self.ts.nodes_flags[u]
        strain = ""
        if (flags & tskit.NODE_IS_SAMPLE) != 0:
            strain = md["strain"]
        elif flags == 1 << 21:
            if "overlap" in md:
                strain = f"Overlap {len(md['overlap'])} mut {len(md['sibs'])} sibs"
            else:
                strain = "Overlap debug missing"
        elif flags == 1 << 22:
            if "sites" in md:
                strain = f"Push {len(md['sites'])} reversions"
            else:
                strain = "Push debug missing"
        elif "date_added" in md:
            strain = f"Added {md['date_added']}"

        pango = md.get(self.pango_source, None)
        imputed_pango = md.get("Imputed_" + self.pango_source, None)
        if pango is not None:
            if imputed_pango is not None and imputed_pango != pango:
                pango = f"MISMATCH: {pango} != {imputed_pango}"
        elif imputed_pango is not None:
            pango = imputed_pango
        else:
            pango = ""

        return {
            "node": u,
            "strain": strain,
            "pango": pango,
            "parents": np.sum(self.ts.edges_child == u),
            "children": np.sum(self.ts.edges_parent == u),
            "descendants": self.nodes_max_descendant_samples[u],
            "date": self.nodes_date[u],
            "qc": qc,
            **self._node_mutation_summary(u, child_mutations=child_mutations),
        }

    def _children_summary(self, u):
        u_children = self.ts.edges_child[self.ts.edges_parent == u]
        counter = collections.Counter(
            dict(zip(u_children, self.nodes_max_descendant_samples[u_children]))
        )

        # Count the mutations on the parent and those on each child
        u_mutations = self.node_mutations(u)
        data = []
        for v, _ in counter.most_common(10):
            v_mutations = self.node_mutations(v)
            same_site = set(u_mutations.keys()) & set(v_mutations.keys())
            # NOTE: these are immediate reversions we're counting
            # FIXME probably not much point in going through these here
            # as we've counted all immediate reversions above already.
            reversions = 0
            for site in same_site:
                u_mut = u_mutations[site]
                v_mut = v_mutations[site]
                assert len(u_mut) == len(v_mut) == 3
                if u_mut[-1] == v_mut[0] and v_mut[-1] == u_mut[0]:
                    reversions += 1
            summary = self._node_summary(v)
            summary["new_muts"] = len(v_mutations)
            summary["same_site_muts"] = len(same_site)
            summary["reversions"] = reversions
            summary["branch_length"] = self.ts.nodes_time[u] - self.ts.nodes_time[v]
            data.append(summary)
        df = pd.DataFrame(data)
        return [
            Markdown(
                "### Children \n"
                f"Node {u} has {len(counter)} children. "
                "Showing top-10 by descendant count"
            ),
            df,
        ]

    def _collect_node_data(self, nodes):
        data = []
        for u in nodes:
            data.append(self._node_summary(u))
        return pd.DataFrame(data)

    def site_mutation_data(self, position):
        site = self.ts.site(position=int(position))
        data = []
        for mut in site.mutations:
            data.append(self._mutation_summary(mut.id))
        return pd.DataFrame(data)

    def site_summary(self, position):
        site = self.ts.site(position=position)
        reversions = 0
        immediate_reversions = 0
        state_changes = collections.Counter()
        df_muts = self.site_mutation_data(position)
        for _, row in df_muts.iterrows():
            key = (row.inherited_state, row.derived_state)
            state_changes[key] += 1
        data = [
            ("id", site.id),
            ("position", int(site.position)),
            ("ancestral_state", site.ancestral_state),
            ("num_mutations", len(df_muts)),
            ("private", np.sum(df_muts.descendants == 1)),
            ("max_inheritors", np.max(df_muts.inheritors)),
            ("reversions", np.sum(df_muts.is_reversion)),
            ("immediate_reversions", np.sum(df_muts.is_immediate_reversion)),
            ("transitions", np.sum(df_muts.is_transition)),
            ("transversions", np.sum(df_muts.is_transversion)),
            ("insertions", np.sum(df_muts.is_insertion)),
            ("deletions", np.sum(df_muts.is_deletion)),
        ]
        for (a, b), value in state_changes.most_common():
            data.append((f"{a}>{b}", value))
        return pd.DataFrame(
            {"property": [d[0] for d in data], "value": [d[1] for d in data]}
        )

    def recombinants_summary(self):
        data = []
        for u in self.recombinants:
            md = self.nodes_metadata[u]["sc2ts"]
            group_id = md["group_id"]
            # NOTE this is overlapping quite a bit with the SampleGroupInfo
            # class functionality here, but we just want something quick for
            # now here.
            causal_lineages = collections.Counter()
            for v in self.nodes_sample_group[group_id]:
                if self.ts.nodes_flags[v] & tskit.NODE_IS_SAMPLE > 0:
                    pango = self.nodes_metadata[v].get(self.pango_source, "Unknown")
                    causal_lineages[pango] += 1
            data.append(
                {
                    "recombinant": u,
                    "parents": self.nodes_num_parents[u],
                    "descendants": self.nodes_max_descendant_samples[u],
                    "causal_pango": dict(causal_lineages),
                    **md,
                }
            )
        return pd.DataFrame(data)

    def deletions_summary(self):
        deletion_ids = np.where(self.mutations_derived_state == "-")[0]
        df = pd.DataFrame(
            {
                "mutation": deletion_ids,
                "position": self.mutations_position[deletion_ids],
                "node": self.ts.mutations_node[deletion_ids],
            }
        )
        df = df.sort_values(["position", "node"])
        events = {}
        for row in df.itertuples():
            if row.node not in events:
                events[row.node] = [
                    DeletionEvent(row.position, row.node, 1, [row.mutation])
                ]
            else:
                for e in events[row.node]:
                    if row.position == e.start + e.length:
                        e.length += 1
                        e.mutations.append(row.mutation)
                        break
                else:
                    # Didn't find an event to extend, add another one
                    events[row.node].append(
                        DeletionEvent(row.position, row.node, 1, [row.mutation])
                    )
        # Now unwrap the events and compute summaries
        data = []
        for event_list in events.values():
            for e in event_list:
                num_inheritors = self.mutations_num_inheritors[e.mutations]
                data.append(
                    {
                        "start": e.start,
                        "node": e.node,
                        "length": e.length,
                        "max_inheritors": np.max(num_inheritors),
                        "min_inheritors": np.min(num_inheritors),
                    }
                )

        return pd.DataFrame(data)

    def combine_recombinant_info(self):
        def get_imputed_pango(u, pango_source):
            # Can set pango_source to "Nextclade_pango" or "GISAID_lineage"
            key = "Imputed_" + pango_source
            if key not in self.nodes_metadata[u]:
                raise ValueError(
                    f"{key} not available. You may need to run the imputation pipeline"
                )
            lineage = self.nodes_metadata[u]["Imputed_" + pango_source]
            return lineage

        df_arg = get_recombinant_mrca_table(self.ts)
        arg_info = collections.defaultdict(list)
        for _, row in df_arg.iterrows():
            arg_info[row.recombinant_node].append(row)

        output = []
        for u, rows in arg_info.items():
            md = self.nodes_metadata[u]
            match_info = md["match_info"]
            strain = match_info[0]["strain"]
            assert len(match_info) == 2
            assert strain == match_info[1]["strain"]

            hmm_runs = {}
            for record in match_info:
                parents = record["parents"]
                hmm_runs[record["direction"]] = HmmRun(
                    breakpoints=record["breakpoints"],
                    parents=parents,
                    mutations=record["mutations"],
                    parent_imputed_lineages=[
                        get_imputed_pango(x, self.pango_source) for x in parents
                    ],
                )
            # TODO it's confusing that the breakpoints array is bracketed by
            # 0 and L. We should just remove these from all the places that
            # we're using them.
            breakpoints = [0]
            parents = []
            mrcas = []
            for row in rows:
                mrcas.append(row.mrca)
                breakpoints.append(row.breakpoint)
                parents.append(row.left_parent)
            parents.append(row.right_parent)
            breakpoints.append(int(self.ts.sequence_length))
            arg_rec = utils.ArgRecombinant(
                breakpoints=breakpoints,
                breakpoint_intervals=md["breakpoint_intervals"],
                parents=parents,
                mrcas=mrcas,
                parent_imputed_lineages=[
                    get_imputed_pango(x, self.pango_source) for x in parents
                ],
            )
            causal_node = self.strain_map[strain]
            causal_lineage = self.nodes_metadata[causal_node].get(
                self.pango_source, "unknown"
            )
            rec = Recombinant(
                causal_strain=strain,
                causal_date=md["date_added"],
                causal_lineage=causal_lineage,
                node=u,
                hmm_runs=hmm_runs,
                max_descendant_samples=self.nodes_max_descendant_samples[u],
                arg_info=arg_rec,
            )
            output.append(rec)
        return output

    def export_recombinant_breakpoints(self):
        """
        Make a dataframe with one row per recombination node breakpoint,
        giving information about that the left and right parent of that
        break, and their MRCA.

        Recombinants with multiple breaks are represented by multiple rows.
        You can only list the breakpoints in recombination nodes that have 2 parents by
        doing e.g. df.drop_duplicates('node', keep=False)
        """
        recombs = self.combine_recombinant_info()
        data = []
        for rec in recombs:
            arg = rec.arg_info
            for j in range(len(arg.parents) - 1):
                row = rec.data_summary()
                mrca = arg.mrcas[j]
                row["breakpoint"] = arg.breakpoints[j + 1]
                row["breakpoint_interval_left"] = arg.breakpoint_intervals[j][0]
                row["breakpoint_interval_right"] = arg.breakpoint_intervals[j][1]
                row["left_parent"] = arg.parents[j]
                row["right_parent"] = arg.parents[j + 1]
                row["left_parent_imputed_lineage"] = arg.parent_imputed_lineages[j]
                row["right_parent_imputed_lineage"] = arg.parent_imputed_lineages[j + 1]
                row["mrca"] = mrca
                row["mrca_date"] = self.nodes_date[mrca]
                data.append(row)
        return pd.DataFrame(sorted(data, key=operator.itemgetter("node")))

    def mutators_summary(self, threshold=10):
        mutator_nodes = np.where(self.nodes_num_mutations > threshold)[0]
        df = self._collect_node_data(mutator_nodes)
        df.sort_values("mutations", inplace=True)
        return df

    def reversion_push_summary(self):
        nodes = np.where(self.ts.nodes_flags == 1 << 22)[0]
        df = self._collect_node_data(nodes)
        df.sort_values("descendants", inplace=True, ascending=False)
        return df

    def mutation_coalescing_summary(self):
        nodes = np.where(self.ts.nodes_flags == 1 << 21)[0]
        df = self._collect_node_data(nodes)
        df.sort_values("descendants", inplace=True, ascending=False)
        return df

    def immediate_reversions_summary(self):
        nodes = self.ts.mutations_node[self.mutations_is_immediate_reversion]
        df = self._collect_node_data(np.unique(nodes))
        df.sort_values("immediate_reversions", inplace=True)
        return df

    def node_mutations(self, node):
        muts = {}
        for mut_id in np.where(self.ts.mutations_node == node)[0]:
            pos = int(self.ts.sites_position[self.ts.mutations_site[mut_id]])
            assert pos not in muts
            state0 = self.mutations_inherited_state[mut_id]
            state1 = self.mutations_derived_state[mut_id]
            muts[pos] = f"{state0}>{state1}"
        return muts

    def _copying_table(self, node, edges):
        def css_cell(allele):
            # function for the cell style - nucleotide colours faded from SNiPit
            cols = {"A": "#869eb5", "T": "#adbda8", "C": "#d19394", "G": "#c3dde6"}
            return (
                ' style="border: 1px solid black; background-color: '
                + cols.get(allele, "white")
                + '; border-collapse: collapse;"'
            )

        vrl = ' style="writing-mode: vertical-rl; transform: rotate(180deg)"'

        parent_cols = {}
        samples = [node]
        for edge in edges:
            if edge.parent not in parent_cols:
                parent_cols[edge.parent] = len(parent_cols) + 1
                samples.append(edge.parent)

        # Can't have missing data here, so we're OK.
        variants = self.ts.variants(samples=samples, isolated_as_missing=False)
        mutations = self.node_mutations(node)

        positions = []
        ref = []
        parents = [[] for _ in range(len(parent_cols))]
        child = []
        extra_mut = []
        for var in variants:
            if len(np.unique(var.genotypes)) > 1:
                pos = int(var.site.position)
                positions.append(f"<td><span{vrl}>{pos}</span></td>")
                ref.append(f"<td>{var.site.ancestral_state}</td>")
                allele = var.alleles[var.genotypes[0]]
                child.append(f"<td{css_cell(allele)}>{allele}</td>")

                edge_index = np.searchsorted(edges.left, pos, side="right") - 1
                parent_col = parent_cols[edges[edge_index].parent]
                for j in range(1, len(var.genotypes)):
                    allele = var.alleles[var.genotypes[j]]
                    css = css_cell(allele) if j == parent_col else ""
                    parents[j - 1].append(f"<td{css}>{allele}</td>")

                extra_mut.append(f"<td><span{vrl}>{mutations.get(pos, '')}</span></td>")

        html = '<tr style="font-size: 70%"><th>pos</th>' + "".join(positions) + "</tr>"
        html += "<tr><th>ref</th>" + "".join(ref) + "</tr>"
        html += "<tr><th>P0</th>" + "".join(parents.pop(0)) + "</tr>"
        html += "<tr><th>C</th>" + "".join(child) + "</tr>"
        for i, parent in enumerate(parents):
            html += f"<tr><th>P{i + 1}</th>" + "".join(parent) + "</tr>"
        html += '<tr style="font-size: 75%"><th>mut</th>' + "".join(extra_mut) + "</tr>"

        return f"<table>{html}</table>"

    def _show_parent_copying(self, child):
        edge_list = [
            self.ts.edge(eid) for eid in np.where(self.ts.edges_child == child)[0]
        ]
        edges = tskit.EdgeTable()
        for e in sorted(edge_list, key=lambda e: e.left):
            edges.append(e)

        data = []
        summary = self._node_summary(child)
        summary["role"] = "Child"
        summary["branch_length"] = 0
        data.append(summary)
        for u in list(edges.parent):
            summary = self._node_summary(u)
            summary["role"] = "Parent"
            summary["branch_length"] = self.ts.nodes_time[u] - self.ts.nodes_time[child]
            data.append(summary)
        return [
            Markdown("### Node data "),
            pd.DataFrame(data),
            Markdown("### Edges"),
            edges,
            Markdown("### Copying pattern"),
            HTML(self._copying_table(child, edges)),
        ]

    def _get_closest_recombinant(self, tree, node):
        u = node
        closest_recombinant = -1
        path_length = 0
        while u != 0:
            e = self.ts.edge(tree.edge(u))
            if e.left != 0 or e.right != self.ts.sequence_length:
                if closest_recombinant == -1:
                    closest_recombinant = u
                    break
            path_length += 1
            u = tree.parent(u)
        return closest_recombinant, path_length

    def _show_path_to_root(self, tree, node):
        u = node
        closest_recombinant = -1
        data = []
        while u != 0:
            e = self.ts.edge(tree.edge(u))
            if e.left != 0 or e.right != self.ts.sequence_length:
                if closest_recombinant == -1:
                    closest_recombinant = u
            row = self._node_summary(u, child_mutations=False)
            row["branch_length"] = tree.branch_length(u)
            data.append(row)
            u = tree.parent(u)
        return closest_recombinant, pd.DataFrame(data)

    def _show_paths_to_root(self, node):
        closest_recombinant, df = self._show_path_to_root(self.ts.first(), node)
        items = []
        if closest_recombinant != -1:
            items.append(Markdown("## Left path to root"))
            items.append(Markdown(f"### Closest recombinant: {closest_recombinant}"))
            items.append(df)
            items.append(Markdown("## Right path to root"))
            closest_recombinant, df = self._show_path_to_root(self.ts.last(), node)
            items.append(Markdown(f"### Closest recombinant: {closest_recombinant}"))
            items.append(df)
        else:
            items = [Markdown("## Path to root")]
            items.append(df)
        return items

    def _tree_mutation_path(self, tree, node):
        u = node
        site_in_tree = np.logical_and(
            self.mutations_position >= tree.interval.left,
            self.mutations_position < tree.interval.right,
        )
        ret = []
        while u != -1:
            # Get all mutations for this node on this tree
            condition = np.logical_and(self.ts.mutations_node == u, site_in_tree)
            ret.extend(np.where(condition)[0])
            u = tree.parent(u)
        return ret

    def mutation_path(self, node):
        data = []
        for tree in self.ts.trees():
            for mut_id in self._tree_mutation_path(tree, node):
                data.append(self._mutation_summary(mut_id))
        return pd.DataFrame(data).sort_values("time")

    def _show_mutation_path(self, node):
        df = self.mutation_path(node)
        return [Markdown("## Mutation path"), df]

    def _mutation_summary(self, mut_id):
        return {
            "site": self.mutations_position[mut_id],
            "node": self.ts.mutations_node[mut_id],
            "descendants": self.mutations_num_descendants[mut_id],
            "inheritors": self.mutations_num_inheritors[mut_id],
            "inherited_state": self.mutations_inherited_state[mut_id],
            "derived_state": self.mutations_derived_state[mut_id],
            "is_reversion": self.mutations_is_reversion[mut_id],
            "is_immediate_reversion": self.mutations_is_immediate_reversion[mut_id],
            "is_transition": self.mutations_is_transition[mut_id],
            "is_transversion": self.mutations_is_transversion[mut_id],
            "is_insertion": self.mutations_inherited_state[mut_id] == "-",
            "is_deletion": self.mutations_derived_state[mut_id] == "-",
            "parent": self.ts.mutations_parent[mut_id],
            "num_parents": self.mutations_num_parents[mut_id],
            "time": self.ts.mutations_time[mut_id],
            "id": mut_id,
            "metadata": self.ts.mutation(mut_id).metadata,
        }

    def node_report(self, node_id=None, strain=None):
        if strain is not None:
            node_id = self.strain_map[strain]
        # node_summary = pd.DataFrame([self._node_summary(node_id)])
        # TODO improve this for internal nodes
        node_summary = [self.ts.node(node_id).metadata]
        items = [Markdown(f"# Report for {node_id}"), node_summary]
        items += self._show_parent_copying(node_id)
        items += self._show_paths_to_root(node_id)
        items += self._children_summary(node_id)
        items += self._show_mutation_path(node_id)
        return items

    def pango_lineages_report(self):
        data = []
        for lineage in self.pango_lineage_samples.keys():
            node = self.pango_lineage_samples[lineage][0]
            row = {
                "total_samples": len(self.pango_lineage_samples[lineage]),
                **self._node_summary(node),
            }
            data.append(row)
        return pd.DataFrame(data)

    def pango_recombinant_lineages_report(self):
        nodes = []
        for lineage in self.pango_lineage_samples.keys():
            if lineage.startswith("X"):
                node = self.pango_lineage_samples[lineage][0]
                nodes.append(node)
        return self.recombinant_samples_report(nodes)

    def recombinant_samples_report(self, nodes):
        tree = self.ts.first()
        data = []
        for node in nodes:
            node_summary = self._node_summary(node)
            closest_recombinant, path_length = self._get_closest_recombinant(tree, node)
            sample_is_recombinant = False
            if closest_recombinant != -1:
                recomb_date = self.ts.node(closest_recombinant).metadata["date_added"]
                sample_is_recombinant = recomb_date == str(node_summary["date"])
            summary = {
                "recombinant": closest_recombinant,
                "direct": sample_is_recombinant,
                "path_length": path_length,
                **node_summary,
            }
            data.append(summary)
        return pd.DataFrame(data)

    def _repr_html_(self):
        return self.summary()._repr_html_()

    def _histogram(self, data, title, bins=None, xlabel=None, ylabel=None):
        fig, ax = plt.subplots(1, 1)
        fig.suptitle(title)
        ax.hist(data, rwidth=0.9, bins=bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, [ax]

    def plot_mutations_per_node_distribution(self):
        nodes_with_many_muts = np.sum(self.nodes_num_mutations >= 10)
        return self._histogram(
            self.nodes_num_mutations,
            title=f"Nodes with >= 10 muts: {nodes_with_many_muts}",
            bins=range(10),
            xlabel="Number of mutations",
            ylabel="Number of nodes",
        )

    def plot_missing_sites_per_sample(self):
        return self._histogram(
            self.nodes_num_missing_sites[self.ts.samples()],
            title="Missing sites per sample",
        )

    def plot_deletion_sites_per_sample(self):
        return self._histogram(
            self.nodes_num_deletion_sites[self.ts.samples()],
            title="Deletion sites per sample",
        )

    def plot_branch_length_distributions(
        self, log_scale=True, min_value=1, exact_match=False, max_value=400
    ):
        fig, ax = plt.subplots(1, 1)
        ts = self.ts
        branch_length = ts.nodes_time[ts.edges_parent] - ts.nodes_time[ts.edges_child]
        select = branch_length >= min_value
        if exact_match:
            select &= ts.nodes_flags[ts.edges_child] & core.NODE_IS_EXACT_MATCH > 0
        ax.hist(branch_length[select], range(min_value, max_value))
        ax.set_xlabel("Length of branches")
        if log_scale:
            ax.set_yscale("log")
        return fig, [ax]

    def plot_mutations_per_site_distribution(self):
        fig, ax = plt.subplots(1, 1)
        sites_with_many_muts = np.sum(self.sites_num_mutations >= 10)
        ax.set_title(f"Sites with >= 10 muts: {sites_with_many_muts}")
        ax.hist(self.sites_num_mutations, range(10), rwidth=0.9)
        ax.set_xlabel("Number of mutations")
        ax.set_ylabel("Number of site")
        return fig, [ax]

    def plot_mutation_spectrum(self, min_inheritors=1):
        counter = self.get_mutation_spectrum(min_inheritors)
        fig, ax = plt.subplots(1, 1)
        types = ["C>T", "G>A", "G>T", "G>C", "C>A", "T>A"]
        rev_types = [t[::-1] for t in types]
        x = range(len(types))
        ax.bar(x, [counter[t] for t in types])
        ax.bar(x, [-counter[t] for t in rev_types], bottom=0)

        ax2 = ax.secondary_xaxis("top")
        ax2.tick_params(axis="x")
        ax2.set_xticks(x)

        ax2.set_xticklabels(types)
        ax.set_xticks(x)
        ax.set_xticklabels(rev_types)

        y = max(counter.values())
        step = y / 10
        for key in ["C>T", "G>T"]:
            rev_key = key[::-1]
            ratio = counter[key] / max(1, counter[rev_key])  # avoid division by zero
            text = f"{key} / {rev_key}={ratio:.2f}"
            y -= step
            ax.text(4, y, text)
        return fig, [ax]

    def get_mutation_spectrum(self, min_inheritors=1):
        keep = self.mutations_num_inheritors >= min_inheritors
        inherited = self.mutations_inherited_state[keep]
        derived = self.mutations_derived_state[keep]
        sep = inherited.copy()
        sep[:] = ">"
        x = np.char.add(inherited, sep)
        x = np.char.add(x, derived)
        return collections.Counter(x)

    def _add_genes_to_axis(self, ax):
        genes = core.get_gene_coordinates()
        mids = []
        for j, (gene, (left, right)) in enumerate(genes.items()):
            mids.append(left + (right - left) / 2)
            # FIXME totally arbitrary choice of colours, use something better!
            colour = "black"
            if j % 2 == 1:
                colour = "green"
            ax.axvspan(left, right, color=colour, alpha=0.1, zorder=0)

        ax2 = ax.secondary_xaxis("top")
        ax2.tick_params(axis="x")
        ax2.set_xticks(mids, minor=False)
        ax2.set_xticklabels(list(genes.keys()), rotation="vertical")

    def _wide_plot(self, *args, **kwargs):
        return plt.subplots(*args, figsize=(16, 4), **kwargs)

    def plot_ts_tv_per_site(self, annotate_threshold=0.9):
        nonzero = self.sites_num_transversions != 0
        ratio = (
            self.sites_num_transitions[nonzero] / self.sites_num_transversions[nonzero]
        )
        pos = self.ts.sites_position[nonzero]

        fig, ax = self._wide_plot(1, 1)
        ax.plot(pos, ratio)
        self._add_genes_to_axis(ax)

        threshold = np.max(ratio) * annotate_threshold
        top_sites = np.where(ratio > threshold)[0]
        for site in top_sites:
            plt.annotate(
                f"{int(pos[site])}", xy=(pos[site], ratio[site]), xycoords="data"
            )
        ax.set_ylabel("Ts/Tv")
        ax.set_xlabel("Position on genome")
        return fig, [ax]

    def _plot_per_site_count(self, count, annotate_threshold):
        fig, ax = self._wide_plot(1, 1)
        pos = self.ts.sites_position
        ax.plot(pos, count)
        self._add_genes_to_axis(ax)
        threshold = np.max(count) * annotate_threshold

        # Show runs of sites exceeding threshold
        for v, start, length in zip(*find_runs(count > threshold)):
            if v:
                end = start + length
                x, y = int(pos[start]), int(pos[min(self.ts.num_sites - 1, end)])
                if x == y - 1:
                    label = f"{x}"
                else:
                    label = f"{x}-{y}"
                plt.annotate(label, xy=(x, count[start]), xycoords="data")
        ax.set_xlabel("Position on genome")
        return fig, ax

    def plot_mutations_per_site(self, annotate_threshold=0.9, select=None):
        if select is None:
            count = self.sites_num_mutations
        else:
            count = np.bincount(
                self.ts.mutations_site[select], minlength=self.ts.num_sites
            )
        fig, ax = self._plot_per_site_count(count, annotate_threshold)
        zero_fraction = np.sum(count == 0) / self.ts.num_sites
        ax.annotate(
            f"{zero_fraction * 100:.2f}% sites have 0 mutations",
            xy=(self.ts.sites_position[0], np.max(count)),
            xycoords="data",
        )
        ax.set_ylabel("Number of mutations")
        return fig, [ax]

    def plot_missing_samples_per_site(self, annotate_threshold=0.5):
        fig, ax = self._plot_per_site_count(
            self.sites_num_missing_samples, annotate_threshold
        )
        ax.set_ylabel("Number missing samples")
        return fig, [ax]

    def plot_deletion_samples_per_site(self, annotate_threshold=0.5):
        fig, ax = self._plot_per_site_count(
            self.sites_num_deletion_samples, annotate_threshold
        )
        ax.set_ylabel("Number deletion samples")
        return fig, [ax]

    def plot_samples_per_day(self):
        fig, ax = self._wide_plot(1, 1)
        t = np.arange(self.num_samples_per_day.shape[0])
        ax.plot(self.time_zero_as_date - t, self.num_samples_per_day)
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of samples")
        return fig, [ax]

    def plot_resources(self, start_date="2020-04-01"):
        ts = self.ts
        fig, ax = self._wide_plot(2, sharex=True)
        timestamp = np.zeros(ts.num_provenances, dtype="datetime64[s]")
        date = np.zeros(ts.num_provenances, dtype="datetime64[D]")
        num_samples = np.zeros(ts.num_provenances, dtype=int)
        for j in range(ts.num_provenances):
            p = ts.provenance(j)
            timestamp[j] = p.timestamp
            record = json.loads(p.record)
            text_date = record["parameters"]["args"][2]
            date[j] = text_date

            days_ago = self.time_zero_as_date - date[j]
            # Avoid division by zero
            num_samples[j] = max(1, self.num_samples_per_day[days_ago.astype(int)])

        keep = date >= np.array([start_date], dtype="datetime64[D]")

        wall_time = np.append([0], np.diff(timestamp).astype(float))
        ax[0].plot(date[keep], wall_time[keep] / 60)
        ax[1].set_xlabel("Date")
        ax[0].set_ylabel("Elapsed time (mins)")
        ax[1].plot(date[keep], wall_time[keep] / num_samples[keep])
        ax[1].set_ylabel("Elapsed time per sample (s)")
        return fig, ax

    def fixme_plot_recombinants_per_day(self):
        counter = collections.Counter()
        for u in self.recombinants:
            date = np.datetime64(self.nodes_metadata[u]["date_added"])
            counter[date] += 1

        samples_per_day = np.zeros(len(counter))
        sample_date = self.nodes_date[self.ts.samples()]
        for j, date in enumerate(counter.keys()):
            samples_per_day[j] = np.sum(sample_date == date)
        x = np.array(list(counter.keys()))
        y = np.array(list(counter.values()))

        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        ax1.plot(x, y)
        ax2.plot(x, y / samples_per_day)
        ax2.set_xlabel("Date")
        ax1.set_ylabel("Number of recombinant samples")
        ax2.set_ylabel("Fraction of samples recombinant")
        ax2.set_ylim(0, 0.01)

    def draw_pango_lineage_subtree(
        self,
        lineage,
        position=None,
        collapse_tracked=None,
        remove_clones=None,
        *,
        pack_untracked_polytomies=True,
        time_scale="rank",
        title=None,
        mutation_labels=None,
        size=None,
        style="",
        **kwargs,
    ):
        if position is None:
            position = 21563  # pick the start of the spike
        if size is None:
            size = (1000, 1000)
        if remove_clones:
            # TODO
            raise NotImplementedError("remove_clones not implemented")
        # remove mutation times, so they get spaced evenly along a branch
        tables = self.ts.dump_tables()
        time = tables.mutations.time
        time[:] = tskit.UNKNOWN_TIME
        tables.mutations.time = time
        ts = tables.tree_sequence()
        tracked_nodes = self.pango_lineage_samples[lineage]
        tree = ts.at(position, tracked_samples=tracked_nodes)
        order = np.array(
            list(
                tskit.drawing._postorder_tracked_minlex_traversal(
                    tree, collapse_tracked=collapse_tracked
                )
            )
        )
        if title is None:
            simplified_ts = ts.simplify(
                order[np.where(ts.nodes_flags[order] & tskit.NODE_IS_SAMPLE)[0]]
            )
            num_trees = simplified_ts.num_trees
            tree_pos = simplified_ts.at(position).index
            title = (
                f"Sc2ts genealogy of {len(tracked_nodes)} {lineage} samples "
                f"at position {position} (tree {tree_pos}/{num_trees})"
                # f" --- file: "  # TODO - show filename
            )

        # Find the actually shown nodes (i.e. if polytomies are packed, we may not
        # see some tips. This is copied from tskit.drawing.SvgTree.assign_x_coordinates
        shown_nodes = order
        if pack_untracked_polytomies:
            shown_nodes = []
            untracked_children = collections.defaultdict(list)
            prev = tree.virtual_root
            for u in order:
                parent = tree.parent(u)
                assert parent != prev
                if tree.parent(prev) != u:  # is a tip
                    if tree.num_tracked_samples(u) == 0:
                        untracked_children[parent].append(u)
                    else:
                        shown_nodes.append(u)
                else:
                    if len(untracked_children[u]) == 1:
                        # If only a single non-focal lineage, we might as well show it
                        for child in untracked_children[u]:
                            shown_nodes.append(child)
                    shown_nodes.append(u)
                prev = u

        if mutation_labels is None:
            mutation_labels = collections.defaultdict(list)
            multiple_mutations = []
            reverted_mutations = []
            use_mutations = np.where(np.isin(ts.mutations_node, shown_nodes))[0]
            sites = ts.mutations_site[use_mutations]
            for mut_id in use_mutations:
                # TODO Viz the recurrent mutations
                mut = ts.mutation(mut_id)
                site = ts.site(mut.site)
                if len(sites == site.id) > 1:
                    multiple_mutations.append(mut.id)
                inherited_state = site.ancestral_state
                if mut.parent >= 0:
                    parent = ts.mutation(mut.parent)
                    inherited_state = parent.derived_state
                    parent_inherited_state = site.ancestral_state
                    if parent.parent >= 0:
                        parent_inherited_state = ts.mutation(
                            parent.parent
                        ).derived_state
                    if parent_inherited_state == mut.derived_state:
                        reverted_mutations.append(mut.id)
                # Reverse map label name to mutation id, so we can count duplicates
                label = f"{inherited_state}{int(site.position)}{mut.derived_state}"
                mutation_labels[label].append(mut.id)
            # If more than one mutation has the same label, add a prefix with the counts
            mutation_labels = {
                m_id: label + (f" ({i+1}/{len(ids)})" if len(ids) > 1 else "")
                for label, ids in mutation_labels.items()
                for i, m_id in enumerate(ids)
            }
        # some default styles
        styles = [
            "".join(f".n{u} > .sym {{fill: cyan}}" for u in tracked_nodes),
            ".lab.summary {font-size: 12px}",
            ".polytomy {font-size: 10px}",
            ".mut .lab {font-size: 10px}",
            ".y-axis .lab {font-size: 12px}",
            ".mut .lab {fill: darkred} .mut .sym {stroke: darkred} .background path {fill: white}",
        ]
        if len(multiple_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in multiple_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in multiple_mutations)
            styles.append(lab_css + "{fill: red}" + sym_css + "{stroke: red}")
        if len(reverted_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in reverted_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in reverted_mutations)
            styles.append(lab_css + "{fill: magenta}" + sym_css + "{stroke: magenta}")

        return tree.draw_svg(
            time_scale=time_scale,
            y_axis=True,
            x_axis=False,
            title=title,
            size=size,
            order=order,
            mutation_labels=mutation_labels,
            all_edge_mutations=True,
            symbol_size=4,
            pack_untracked_polytomies=pack_untracked_polytomies,
            style="".join(styles) + style,
            **kwargs,
        )

    def get_sample_group_info(self, group_id):
        samples = []

        for u in self.nodes_sample_group[group_id]:
            if self.ts.nodes_flags[u] & tskit.NODE_IS_SAMPLE > 0:
                samples.append(u)

        tree = self.ts.first()
        while self.nodes_metadata[u]["sc2ts"].get("group_id", None) == group_id:
            u = tree.parent(u)
        attach_date = self.nodes_date[u]
        ts = self.ts.simplify(samples + [u])
        tables = ts.dump_tables()
        # Wipe out the sample strain index as its out of date.
        tables.metadata = {}

        # Remove the mutations above the root (these are the mutations going all
        # the way back up the tree) UNLESS there's a recurrent mutation within
        # the subtree
        tree = ts.first()
        u = ts.samples()[-1]
        keep_mutations = np.ones(ts.num_mutations, dtype=bool)
        keep_mutations[ts.mutations_node == u] = False
        for recurrent_mut in np.where(ts.mutations_parent != -1)[0]:
            if ts.mutations_node[recurrent_mut] != u:
                keep_mutations[ts.mutations_parent[recurrent_mut]] = True
        tables.mutations.keep_rows(keep_mutations)
        tables.nodes[u] = tables.nodes[u].replace(flags=0)
        # Space the mutations evenly along branches for viz
        time = tables.mutations.time
        time[:] = tskit.UNKNOWN_TIME
        tables.mutations.time = time

        return SampleGroupInfo(
            group_id,
            self.nodes_sample_group[group_id],
            ts=tables.tree_sequence(),
            attach_date=attach_date,
        )


@dataclasses.dataclass
class DeletionEvent:
    start: int
    node: int
    length: int
    mutations: List


def country_abbr(country):
    return {
        "United Kingdom": "UK",
        "United States": "USA",
        "South Africa": "SA",
        "Australia": "AUS",
        "Brazil": "BRA",
        "Denmark": "DEN",
        "France": "FRA",
        "Germany": "GER",
        "India": "IND",
        "Italy": "ITA",
    }.get(country, country)


@dataclasses.dataclass
class SampleGroupInfo:
    group_id: str
    nodes: List[int]
    ts: tskit.TreeSequence
    attach_date: None

    def draw_svg(
        self,
        size=(800, 600),
        time_scale=None,
        y_axis=True,
        mutation_labels=None,
        style=None,
        highlight_universal_mutations=None,
        x_regions=None,
        node_labels=None,
        **kwargs,
    ):
        """
        Draw an SVG representation of the tree of samples that trace to a single origin.

        The default style is to colour mutations such that sites with a single
        mutation in the tree are dark red, whereas sites with multiple mutations
        show those mutations in red or magenta (magenta when a mutation immediately
        reverts its parent mutation). Any identical mutations (from the same inherited
        to derived state at the same site, i.e. recurrent mutations) have the count of
        recurrent mutations appended to the label, e.g. "C842T (1/2)".

        If highlight_universal_mutations is set, then mutations in the ancestry of all
        the samples (i.e. between the root and the MRCA of all the samples) are highlighted
        in bold and with thicker symbol lines

        If genetic_regions is set, it should be a dictionary mapping (start, end) tuples
        to region names. These will be drawn as coloured rectangles on the x-axis. If None,
        a default selection of SARS-CoV-2 genes will be used.
        """
        if x_regions is None:
            x_regions = {
                (266, 13468): "ORF1a",
                (13468, 21555): "ORF1b",
                (21563, 25384): "Spike",
            }
        pango_md = "Viridian_pangolin"
        ts = self.ts
        if style is None:
            style = ""

        if node_labels == "strain":
            node_labels = {
                n.id: n.metadata["strain"] for n in ts.nodes() if "strain" in n.metadata
            }
        elif node_labels in ("pango", pango_md):
            node_labels = {
                n.id: n.metadata[pango_md] for n in ts.nodes() if pango_md in n.metadata
            }
        elif node_labels == "Country":
            node_labels = {
                n.id: n.metadata["Country"]
                for n in ts.nodes()
                if "Country" in n.metadata
            }
        elif node_labels == "Country_abbr":
            node_labels = {
                n.id: country_abbr(n.metadata["Country"])
                for n in ts.nodes()
                if "Country" in n.metadata
            }
        elif node_labels == "pango+country":
            node_labels = {
                n.id: f"{n.metadata.get(pango_md, '')}:{country_abbr(n.metadata.get('Country', ''))}"
                for n in ts.nodes()
                if pango_md in n.metadata or "Country" in n.metadata
            }

        assert ts.num_trees == 1
        y_ticks = {
            ts.nodes_time[u]: ts.node(u).metadata["date"] for u in list(ts.samples())
        }
        y_ticks[ts.nodes_time[ts.first().root]] = self.attach_date
        if time_scale == "rank":
            times = list(np.unique(ts.nodes_time))
            y_ticks = {times.index(k): v for k, v in y_ticks.items()}
        shared_nodes = []
        if highlight_universal_mutations is not None:
            # find edges above
            tree = ts.first()
            shared_nodes = [tree.root]
            while tree.num_children(shared_nodes[-1]) == 1:
                shared_nodes.append(tree.children(shared_nodes[-1])[0])

        multiple_mutations = []
        universal_mutations = []
        reverted_mutations = []
        if mutation_labels is None:
            mutation_labels = collections.defaultdict(list)
            for site in ts.sites():
                # TODO Viz the recurrent mutations
                for mut in site.mutations:
                    if mut.node in shared_nodes:
                        universal_mutations.append(mut.id)
                    if len(site.mutations) > 1:
                        multiple_mutations.append(mut.id)
                    inherited_state = site.ancestral_state
                    if mut.parent >= 0:
                        parent = ts.mutation(mut.parent)
                        inherited_state = parent.derived_state
                        parent_inherited_state = site.ancestral_state
                        if parent.parent >= 0:
                            parent_inherited_state = ts.mutation(
                                parent.parent
                            ).derived_state
                        if parent_inherited_state == mut.derived_state:
                            reverted_mutations.append(mut.id)
                    # Reverse map label name to mutation id, so we can count duplicates
                    label = f"{inherited_state}{int(site.position)}{mut.derived_state}"
                    mutation_labels[label].append(mut.id)
            # If more than one mutation has the same label, add a prefix with the counts
            mutation_labels = {
                m_id: label + (f" ({i+1}/{len(ids)})" if len(ids) > 1 else "")
                for label, ids in mutation_labels.items()
                for i, m_id in enumerate(ids)
            }
        # some default styles
        styles = [
            ".mut .lab {fill: darkred} .mut .sym {stroke: darkred} .background path {fill: white}"
        ]
        if len(multiple_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in multiple_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in multiple_mutations)
            styles.append(lab_css + "{fill: red}" + sym_css + "{stroke: red}")
        if len(reverted_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in reverted_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in reverted_mutations)
            styles.append(lab_css + "{fill: magenta}" + sym_css + "{stroke: magenta}")
        if len(universal_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in universal_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in universal_mutations)
            sym_ax_css = ", ".join(
                f".x-axis .mut.m{m} .sym" for m in universal_mutations
            )
            styles.append(
                lab_css + "{font-weight: bold}" + sym_css + "{stroke-width: 3}"
            )
            styles.append(sym_ax_css + "{stroke-width: 8}")
        svg = self.ts.draw_svg(
            size=size,
            time_scale=time_scale,
            y_axis=y_axis,
            mutation_labels=mutation_labels,
            y_ticks=y_ticks,
            node_labels=node_labels,
            style="".join(styles) + style,
            **kwargs,
        )

        # Hack to add genes to the X axis: we can replace this with the proper
        # calls once https://github.com/tskit-dev/tskit/pull/3002 is in tskit
        if len(x_regions) > 0:
            assert svg.startswith("<svg")
            header = svg[: svg.find(">") + 1]
            footer = "</svg>"

            # Find SVG positions of the X axis
            m = re.search(
                r'class="x-axis".*?class="ax-line" x1="([\d\.]+)" x2="([\d\.]+)" y1="([\d\.]+)"',
                svg,
            )
            assert m is not None
            x1, x2, y1 = float(m.group(1)), float(m.group(2)), float(m.group(3))
            xdiff = x2 - x1
            x_box_svg = '<rect fill="yellow" stroke="black" x="{x}" width="{w}" y="{y}" height="{h}" />'
            x_name_svg = '<text text-anchor="middle" alignment-baseline="hanging" x="{x}" y="{y}">{name}</text>'
            x_scale = xdiff / ts.sequence_length
            x_boxes = [
                x_box_svg.format(
                    x=x1 + p1 * x_scale, w=(p2 - p1) * x_scale, y=y1, h=20
                )  # height of the box: hardcoded for now to match font height
                for p1, p2 in x_regions.keys()
            ]
            x_names = [
                x_name_svg.format(
                    x=x1 + (p[0] + p[1]) / 2 * x_scale, y=y1 + 2, name=name
                )
                for p, name in x_regions.items()
            ]
            # add the new SVG to the old
            svg = (header + "".join(x_boxes) + "".join(x_names) + footer) + svg
            # Now wrap both in another SVG
            svg = header + svg + footer

        return tskit.drawing.SVGString(svg)

    def get_sample_metadata(self, key):
        ret = []
        for u in self.ts.samples():
            node = self.ts.node(u)
            ret.append(node.metadata[key])
        return ret

    @property
    def lineages(self):
        return self.get_sample_metadata("Viridian_pangolin")

    @property
    def strains(self):
        return self.get_sample_metadata("strain")

    @property
    def sample_dates(self):
        return np.array(self.get_sample_metadata("date"), dtype="datetime64[D]")

    @property
    def num_mutations(self):
        return self.ts.num_mutations

    @property
    def num_recurrent_mutations(self):
        return np.sum(self.ts.mutations_parent != -1)
