"""
Utilities for examining sc2ts output.
"""
import collections
import dataclasses
import operator
import warnings
import datetime

import tskit
import tszip
import numpy as np
import pandas as pd
from sklearn import tree
from collections import defaultdict
import tqdm
import matplotlib.pyplot as plt
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
    def total_cost(self):
        """
        How different is the causal sequence from the rest, roughly?
        """
        fwd = self.hmm_runs["forward"]
        bck = self.hmm_runs["backward"]
        cost_fwd = 3 * (len(fwd.parents) - 1) + len(fwd.mutations)
        cost_bck = 3 * (len(bck.parents) - 1) + len(bck.mutations)
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
                if "sc2ts_qc" in md:
                    qc = md["sc2ts_qc"]
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
        return df

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

    def export_single_break_recombinants(self):
        """
        Return a table of information about one-break recombinants that
        which consistently have two parents under all runs of the HMM,
        and are mutation consistent on the foward and backward
        additional HMM runs. We are strict here because we're primarily
        interested in the breakpoint intervals --- mutation consistency
        is required, for example, because we can have conflicting
        intervals otherwise (where left > right).
        """
        recombs = self.combine_recombinant_info()
        data = []
        for rec in recombs:
            condition = (
                rec.is_path_length_consistent()
                and rec.num_parents == 2
                and rec.is_hmm_mutation_consistent()
            )
            if condition:
                row = rec.data_summary()
                fwd = rec.hmm_runs["forward"]
                bck = rec.hmm_runs["backward"]
                arg = rec.arg_info
                row["num_mutations"] = len(fwd.mutations)
                row["interval_left"] = bck.breakpoints[1]
                row["interval_right"] = fwd.breakpoints[1]
                mrca = arg.mrcas[0]
                row["mrca"] = mrca
                row["mrca_date"] = self.nodes_date[mrca]
                data.append(row)
        return pd.DataFrame(data)

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


def sample_subgraph(
    sample_node,
    ts,
    ti,
    mutations_json_filepath,
    expand_down=True,
    filepath=None,
    *,
    ax=None,
    node_size=None,
    node_colours=None,
    colour_metadata_key=None,
    ts_id_labels=None,
    node_metadata_labels=None,
    sample_metadata_labels=None,
    edge_labels=None,
    node_label_replace=None,
):
    """
    Draws out a subgraph of the ARG above the given sample, including all nodes and
    edges on the path to the nearest sample nodes (showing any recombinations on
    the way).

    # TODO - document the rest of the parameters

    :param plt.Axes ax: a matplotlib axis object on which to plot the graph.
        This allows the graph to be placed as a subplot or the size and aspect ratio
        to be adjusted. If ``None`` (default) plot to the current axis with some
        sensible figsize defaults.
    :param int node_size: The size of the node circles. Default:
        ``None``, treated as 2800.
    :param bool ts_id_labels: Should we label nodes with their tskit node ID? If
        ``None``, show the node ID only for sample nodes. If ``True``, show
        it for all nodes. If ``False``, do not show. Default: ``None``.
    :param str node_metadata_labels: Should we label all nodes with a value from their
        metadata: Default: ``None``, treated as ``"Imputed_GISAID_lineage"``. If ``""``,
        do not plot any all-node metadata.
    :param str sample_metadata_labels: Should we additionally label sample nodes with a
        value from their metadata: Default: ``None``, treated as ``"gisaid_epi_isl"``.
        Notes representing multiple samples will have a label saying "XXX samples".
        If ``""``, do not plot any sample node metadata.
    :param dict edge_labels: a mapping of {(parent_id, child_id): "label")} with which
        to label the edges. If ``None``, label with mutations. If ``{}``, do not plot
        edge labels.
    :param dict node_label_replace: A dict of ``{key: value}`` such that labels
        containing the string ``key`` have that string replaced with ``value``. This
        is primarily to be able to remove the word "Unknown" from the plot, by
        specifying ``{"Unknown": "", "Unknown ": ""}``.
    :param dict node_colours: A dict mapping nodes to colour values. The keys of the
        dictionary can be integer node IDs, strings, or None. If the key is a string,
        it is compared to the value of ``node.metadata[colour_metadata_key]`` (see
        below). If no relevant key exists, the fill colour is set to the value of
        ``node_colours[None]``, or is set to empty if there is no key of ``None``.
        However, if ``node_colours`` is itself ``None``, use the default colourscheme
        which distinguishes between sample nodes, recombination nodes, and all others.
    :param dict colour_metadata_key: A key in the metadata, to use when specifying
        bespoke node colours. Default: ``None``, treated as "strain".


    :return: The networkx Digraph and the positions of nodes in the digraph as a dict of
        ``{node_id : (x, y), ...}``
    :rtype:  tuple(nx.DiGraph, dict)

    """
    def sort_edgelabel(s):
        """
        Edge labels are mutations (integer strings, but can have the last char as "R"),
        or edge intervals such as "≥0\n<1000"
        """
        try:
            return int(s)
        except ValueError:
            try:
                return int(s[:-1])  # Snip off a trailing R
            except ValueError:
                return -1  # return first in sort
            
    if node_size is None:
        node_size = 2800
    if node_metadata_labels is None:
        node_metadata_labels = "Imputed_GISAID_lineage"
    if sample_metadata_labels is None:
        sample_metadata_labels = "gisaid_epi_isl"
    if colour_metadata_key is None:
        colour_metadata_key = "strain"
    col_green = "#228833"
    col_red = "#EE6677"
    col_purp = "#AA3377"
    col_blue = "#66CCEE"
    col_yellow = "#CCBB44"
    col_indigo = "#4477AA"
    col_grey = "#BBBBBB"

    # Read in characteristic mutations info
    linmuts_dict = lineages.read_in_mutations(mutations_json_filepath)

    G = nx.DiGraph()
    related_nodes = defaultdict(set)
    nodes_to_search_up = {sample_node}
    nodes_to_search_down = set()
    nodelabels = collections.defaultdict(list)
    default_nodecolours = {}
    edgelabels = defaultdict(set)

    G.add_node(sample_node)
    default_nodecolours[sample_node] = col_green

    while nodes_to_search_up:
        node = ts.node(nodes_to_search_up.pop())
        if ts_id_labels or (ts_id_labels is None and node.is_sample()):
            nodelabels[node.id].append(str(node.id))
        if node_metadata_labels:
            nodelabels[node.id].append(node.metadata[node_metadata_labels])

        if (not node.is_sample()) or node.id == sample_node:
            parent_node = None
            for t in ts.trees():
                if t.parent(node.id) != parent_node:
                    parent_node = t.parent(node.id)
                    nodes_to_search_up.add(parent_node)
                    if parent_node not in G.nodes:
                        G.add_node(parent_node)
                        default_nodecolours[parent_node] = col_grey
                    G.add_edge(parent_node, node.id)
                    related_nodes[node.id].add((parent_node, node.id))
                    edge = ts.edge(t.edge(node.id))
                    if edge.right - edge.left != ts.sequence_length:
                        default_nodecolours[node.id] = col_red
                        edgelabels[(parent_node, node.id)].add(
                            f"≥{int(edge.left)}\n<{int(edge.right)}"
                        )
            nodes_to_search_down.add(node.id)
        else:
            default_nodecolours[node.id] = col_blue
            if sample_metadata_labels:
                nodelabels[node.id].append(node.metadata[sample_metadata_labels])

    if expand_down:
        while nodes_to_search_down:
            node = ts.node(nodes_to_search_down.pop())
            if (not node.is_sample()) or node.id == sample_node:
                for t in ts.trees():
                    for ch in t.children(node.id):
                        ch_node = ts.node(ch)
                        if ch not in G.nodes:
                            G.add_node(ch)
                            if (
                                ts_id_labels or
                                (ts_id_labels is None and ch_node.is_sample())
                            ):
                                nodelabels[ch].append(str(ch_node.id))
                            if node_metadata_labels:
                                nodelabels[ch].append(
                                    ch_node.metadata[node_metadata_labels]
                                )
                            if ch_node.is_sample():
                                default_nodecolours[ch] = col_blue
                                if sample_metadata_labels:
                                    nodelabels[ch].append(
                                        ch_node.metadata[sample_metadata_labels]
                                    )
                                if t.num_samples(ch) > 1:
                                    n = t.num_samples(ch) - 1
                                    nodelabels[ch].append(
                                        f"+{n} sample{'' if n == 1 else 's'}"
                                    )
                            else:
                                default_nodecolours[ch] = col_grey
                                nodes_to_search_down.add(ch)
                        G.add_edge(node.id, ch)
                        related_nodes[ch].add((node.id, ch))
                        edge = ts.edge(t.edge(ch))
                        if edge.right - edge.left != ts.sequence_length:
                            default_nodecolours[ch] = col_red
                            edgelabels[(node.id, ch)].add(
                                f"≥{int(edge.left)}\n<{int(edge.right)}"
                            )
            else:
                default_nodecolours[node.id] = col_blue
                if sample_metadata_labels:
                    nodelabels[node.id].append(node.metadata[sample_metadata_labels])

    nodelabels = {k: "\n".join(v) for k, v in nodelabels.items()}
    if node_label_replace is not None:
        for search, replace in node_label_replace.items():
            for k, v in nodelabels.items():
                nodelabels[k] = v.replace(search, replace)

    mut_nodes = set()
    if edge_labels is not None:
        # Place user-provided edge labels in the middle of the edge
        edge_labels = {k:(v, "mid") for k, v in edge_labels.items()}
    else:
        edge_labels = {}
        for m in ts.mutations():
            if m.node in related_nodes:
                mut_nodes.add(m.node)
                includemut = False
                pos = int(ts.site(m.site).position)
                mutstr = str(pos)
                if ti.mutations_is_reversion[m.id]:
                    mutstr += "R"
                if pos in linmuts_dict.all_positions:
                    includemut = True
                if includemut:
                    for edge in related_nodes[m.node]:
                        edgelabels[edge].add(mutstr)
        for key, value in edgelabels.items():
            sorted_labels = sorted(value, key=sort_edgelabel)
            if sorted_labels[0].startswith("≥"):  # child is a recombination node
                edge_labels[key] = ("\n".join(sorted_labels), "parent")
            else:
                # Placing mutations near the child end of the edge avoids label clashes
                edge_labels[key] = ("\n".join(sorted_labels), "child")

    unary_nodes_to_remove = set()
    for k, d in G.degree():
        if d == 2 and k not in mut_nodes:
            G.add_edge(*G.predecessors(k), *G.successors(k))
            if (k, *G.successors(k)) in edge_labels:
                edge_labels[(*G.predecessors(k), *G.successors(k))] = edge_labels.pop(
                    (k, *G.successors(k))
                )
            unary_nodes_to_remove.add(k)
    [G.remove_node(k) for k in unary_nodes_to_remove]
    nodelabels = {k: v for k, v in nodelabels.items() if k not in unary_nodes_to_remove}

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    if ax is None:
        dim_x = len(set(x for x, y in pos.values()))
        dim_y = len(set(y for x, y in pos.values()))
        plt.figure(1, figsize=(dim_x * 1.5, dim_y * 1.1))

    fill_cols = []
    if node_colours is None:
        for u in G.nodes:
            fill_cols.append(default_nodecolours[u])
    else:
        default_colour = node_colours.get(None, "None")
        for u in G.nodes:
            try:
                fill_cols.append(node_colours[u])
            except KeyError:
                md_val = ts.node(u).metadata.get(colour_metadata_key, None)
                fill_cols.append(node_colours.get(md_val, default_colour))

    stroke_cols = [("black" if c == "None" else c) for c in fill_cols]

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=True,
        labels=nodelabels,
        node_color=fill_cols,
        edgecolors=stroke_cols,
        node_size=node_size,
        font_size=6,
    )

    edge_label_pos = collections.defaultdict(dict)
    for k, (label, placement) in edge_labels.items():
        edge_label_pos[placement][k] = label
    
    positions = {"child": 0.25, "mid": 0.5, "parent": 0.75}
    vert_align = {"child": "bottom", "mid": "center", "parent": "top"}
    for placement, labels in edge_label_pos.items():
        if len(labels) > 0:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=labels,
                label_pos=positions[placement],
                verticalalignment=vert_align[placement],
                rotate=False,
                font_size=5
            )
    if filepath:
        plt.savefig(filepath)
    elif ax is None:
        plt.show()
    return G, pos


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
