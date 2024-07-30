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


def get_recombinants(ts):
    partial_edges = np.logical_or(
        ts.edges_left != 0, ts.edges_right != ts.sequence_length
    )
    recomb_nodes = np.unique(ts.edges_child[partial_edges])
    return recomb_nodes


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
    def __init__(self, ts, show_progress=True, pango_source="Nextclade_pango"):
        # Can current set pango_source to "Nextclade_pango" or "GISAID_lineage"
        self.ts = ts
        self.pango_source = pango_source
        self.epi_isl_map = {}
        self.strain_map = {}
        self.recombinants = get_recombinants(ts)
        self.nodes_max_descendant_samples = max_descendant_samples(ts)
        self.nodes_date = np.zeros(ts.num_nodes, dtype="datetime64[D]")
        self.nodes_submission_date = np.zeros(ts.num_nodes, dtype="datetime64[D]")
        self.nodes_num_masked_sites = np.zeros(ts.num_nodes, dtype=np.int32)
        self.nodes_metadata = {}
        iterator = tqdm.tqdm(
            ts.nodes(),
            desc="Indexing metadata    ",
            total=ts.num_nodes,
            disable=not show_progress,
        )
        samples = ts.samples()
        last_sample = ts.node(samples[-1])

        self.nodes_date[last_sample.id] = last_sample.metadata["date"]
        self.time_zero_as_date = self.nodes_date[last_sample.id]
        self.earliest_pango_lineage = {}
        self.pango_lineage_samples = collections.defaultdict(list)
        for node in iterator:
            md = node.metadata
            self.nodes_metadata[node.id] = md
            if node.is_sample():
                self.epi_isl_map[md["gisaid_epi_isl"]] = node.id
                if md["gisaid_epi_isl"] is not None:
                    if "." in md["gisaid_epi_isl"]:
                        self.epi_isl_map[md["gisaid_epi_isl"].split(".")[0]] = node.id
                self.strain_map[md["strain"]] = node.id
                self.nodes_date[node.id] = md["date"]
                self.nodes_submission_date[node.id] = md["date_submitted"]
                pango = md.get(pango_source, "unknown")
                self.pango_lineage_samples[pango].append(node.id)
                if "sc2ts" in md:
                    qc = md["sc2ts"]["qc"]
                    self.nodes_num_masked_sites[node.id] = qc["num_masked_sites"]
                else:
                    warnings.warn("Node QC metadata not available")
            else:
                # Rounding down here, might be misleading
                self.nodes_date[node.id] = self.time_zero_as_date - int(
                    self.ts.nodes_time[node.id]
                )

        self.nodes_submission_delay = self.nodes_submission_date - self.nodes_date

        self.sites_num_masked_samples = np.zeros(self.ts.num_sites, dtype=int)
        if ts.table_metadata_schemas.site.schema is not None:
            for site in ts.sites():
                self.sites_num_masked_samples[site.id] = site.metadata["masked_samples"]
        else:
            warnings.warn("Site QC metadata unavailable")

        self.sites_num_mutations = np.bincount(
            self.ts.mutations_site, minlength=self.ts.num_sites
        )
        self.nodes_num_mutations = np.bincount(
            self.ts.mutations_node, minlength=self.ts.num_nodes
        )

        self._compute_mutation_stats()

        # The number of samples per day in time-ago (i.e., the nodes_time units).
        self.num_samples_per_day = np.bincount(ts.nodes_time[samples].astype(int))

        # sample_sets = list(self.pango_lineage_samples.values())
        # print(sample_sets)
        # for lineage, samples in self.pango_lineage_samples.items():
        #     print(lineage, samples)
        # FIXME this is wrong
        # X = ts.segregating_sites(sample_sets, mode="node", span_normalise=False)
        # self.node_pango_lineage_descendants = X.astype(int)
        # # Corresponding sample-set names for this array
        # self.pango_lineage_keys = np.array(list(self.pango_lineage_samples.keys()))

    def _compute_mutation_stats(self):
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
        iterator = tqdm.tqdm(np.arange(N), desc="Classifying mutations")
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
        mc_nodes = np.sum(self.ts.nodes_flags == sc2ts.NODE_IS_MUTATION_OVERLAP)
        pr_nodes = np.sum(self.ts.nodes_flags == sc2ts.NODE_IS_REVERSION_PUSH)
        re_nodes = np.sum(self.ts.nodes_flags == sc2ts.NODE_IS_RECOMBINANT)

        samples = self.ts.samples()
        nodes_with_zero_muts = np.sum(self.nodes_num_mutations == 0)
        sites_with_zero_muts = np.sum(self.sites_num_mutations == 0)
        latest_sample = self.nodes_date[samples[-1]]
        masked_sites_per_sample = self.nodes_num_masked_sites[samples]
        non_samples = self.ts.nodes_flags != tskit.NODE_IS_SAMPLE
        max_non_sample_mutations = np.max(self.nodes_num_mutations[non_samples])
        insertions = np.sum(self.mutations_inherited_state == "-")
        deletions = np.sum(self.mutations_derived_state == "-")

        data = [
            ("latest_sample", latest_sample),
            ("max_submission_delay", np.max(self.nodes_submission_delay[samples])),
            ("samples", self.ts.num_samples),
            ("nodes", self.ts.num_nodes),
            ("mc_nodes", mc_nodes),
            ("pr_nodes", pr_nodes),
            ("re_nodes", re_nodes),
            ("recombinants", len(self.recombinants)),
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
            ("max_masked_sites_per_sample", np.max(masked_sites_per_sample)),
            ("mean_masked_sites_per_sample", np.mean(masked_sites_per_sample)),
            ("max_masked_samples_per_site", np.max(self.sites_num_masked_samples)),
            ("mean_masked_samples_per_site", np.mean(self.sites_num_masked_samples)),
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
        if flags == tskit.NODE_IS_SAMPLE:
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
            "delay": self.nodes_submission_delay[u],
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
        df = self._collect_node_data(self.recombinants)
        sample_map = get_recombinant_samples(self.ts)
        causal_strain = []
        causal_pango = []
        causal_date = []
        for u in df.node:
            md = self.nodes_metadata[sample_map[u]]
            causal_strain.append(md["strain"])
            causal_pango.append(md["Nextclade_pango"])
            causal_date.append(md["date"])
        df["causal_strain"] = causal_strain
        df["causal_pango"] = causal_pango
        df["causal_date"] = causal_date
        return df

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

        df_arg = sc2ts.utils.get_recombinant_mrca_table(self.ts)
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
            arg_rec = ArgRecombinant(
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

    def node_report(self, node_id=None, strain=None, epi_isl=None):
        if strain is not None:
            node_id = self.strain_map[strain]
        if epi_isl is not None:
            node_id = self.epi_isl_map[epi_isl]
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

    # TODO fix these horrible tick labels by doing the histogram manually.
    def plot_mutations_per_node_distribution(self):
        nodes_with_many_muts = np.sum(self.nodes_num_mutations >= 10)
        plt.title(f"Nodes with >= 10 muts: {nodes_with_many_muts}")
        plt.hist(self.nodes_num_mutations, range(10), rwidth=0.9)
        plt.xlabel("Number of mutations")
        plt.ylabel("Number of nodes")

    def plot_masked_sites_per_sample(self):
        # plt.title(f"Nodes with >= 10 muts: {nodes_with_many_muts}")
        plt.hist(self.nodes_num_masked_sites[self.ts.samples()], rwidth=0.9)
        # plt.xlabel("Number of mutations")
        # plt.ylabel("Number of nodes")

    def plot_mutations_per_site_distribution(self):
        sites_with_many_muts = np.sum(self.sites_num_mutations >= 10)
        plt.title(f"Sites with >= 10 muts: {sites_with_many_muts}")
        plt.hist(self.sites_num_mutations, range(10), rwidth=0.9)
        plt.xlabel("Number of mutations")
        plt.ylabel("Number of site")

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
            ratio = counter[key] / counter[rev_key]
            text = f"{key} / {rev_key}={ratio:.2f}"
            y -= step
            ax.text(4, y, text)

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

    def plot_diversity(self, xlim=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        site_div = self.ts.diversity(windows="sites", mode="site")
        branch_div = self.ts.diversity(windows="sites", mode="branch")
        ax1.plot(self.ts.sites_position, site_div)
        ax2.plot(self.ts.sites_position, branch_div)
        ax2.set_xlabel("Genome position")
        ax1.set_ylabel("Site diversity")
        ax2.set_ylabel("Branch diversity")
        for ax in [ax1, ax2]:
            self._add_genes_to_axis(ax)
            if xlim is not None:
                ax.set_xlim(xlim)

    def plot_ts_tv_per_site(self, annotate_threshold=0.9, xlim=None):
        nonzero = self.sites_num_transversions != 0
        ratio = (
            self.sites_num_transitions[nonzero] / self.sites_num_transversions[nonzero]
        )
        pos = self.ts.sites_position[nonzero]

        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.plot(pos, ratio)
        self._add_genes_to_axis(ax)

        threshold = np.max(ratio) * annotate_threshold
        top_sites = np.where(ratio > threshold)[0]
        for site in top_sites:
            plt.annotate(
                f"{int(pos[site])}", xy=(pos[site], ratio[site]), xycoords="data"
            )
        plt.ylabel("Ts/Tv")
        plt.xlabel("Position on genome")
        if xlim is not None:
            plt.xlim(xlim)

    def plot_mutations_per_site(self, annotate_threshold=0.9):
        count = self.sites_num_mutations
        pos = self.ts.sites_position
        zero_fraction = np.sum(count == 0) / self.ts.num_sites

        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.plot(pos, count)
        self._add_genes_to_axis(ax)
        plt.annotate(
            f"{zero_fraction * 100:.2f}% sites have 0 mutations",
            xy=(pos[0], np.max(count)),
            xycoords="data",
        )
        threshold = np.max(count) * annotate_threshold
        top_sites = np.where(count > threshold)[0]
        for site in top_sites:
            plt.annotate(
                f"{int(pos[site])}", xy=(pos[site], count[site]), xycoords="data"
            )
        plt.ylabel("Number of mutations")
        plt.xlabel("Position on genome")

    def plot_masked_samples_per_site(self, annotate_threshold=0.5):
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        self._add_genes_to_axis(ax)
        count = self.sites_num_masked_samples
        pos = self.ts.sites_position
        ax.plot(pos, count)
        threshold = np.max(count) * annotate_threshold
        # Show runs of sites exceeding threshold
        for v, start, length in zip(*find_runs(count > threshold)):
            if v:
                end = start + length
                x, y = int(pos[start]), int(pos[min(self.ts.num_sites - 1, end)])
                plt.annotate(f"{x}-{y}", xy=(x, count[start]), xycoords="data")

        # problematic_sites = get_problematic_sites()
        # plt.plot(problematic_sites)
        plt.ylabel("Number masked samples")
        plt.xlabel("Position on genome")

    def plot_samples_per_day(self):
        plt.figure(figsize=(16, 4))
        t = np.arange(self.num_samples_per_day.shape[0])
        plt.plot(self.time_zero_as_date - t, self.num_samples_per_day)
        plt.xlabel("Date")
        plt.ylabel("Number of samples")

    def plot_recombinants_per_day(self):
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
            if child.is_sample() and child.metadata["date"] == recomb_date:
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
                s = s.replace("$", "").replace(r"\bf", "").replace("\it", "").replace("{", "").replace("}", "")
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

    if show_descendant_samples not in {"samples", "tips", "sample_tips", "all", "", False}:
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
                    s -= 1 # don't count self
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
                    if lpos == "rgt" and interval_labels["lft"][pc]: # in front of rgt label
                        interval_labels[lpos][pc] = "  " + interval_labels[lpos][pc]
                    interval_labels[lpos][pc] += f"{int(edge.left)}…{int(edge.right)}"
                    if lpos == "lft" and interval_labels["rgt"][pc]: # at end of lft label
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
