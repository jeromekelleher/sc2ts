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

import sc2ts
from . import core
from . import lineages


@dataclasses.dataclass
class Match:
    breakpoints: list
    parents: list
    mutations: list


@dataclasses.dataclass
class Recombinant:
    strain: str
    matches: dict
    node: int = -1

    def is_hmm_consistent(self):
        return len(self.matches["forward"].parents) == len(self.matches["backward"].parents) and (
            self.matches["forward"].mutations == self.matches["backward"].mutations
        )


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
    def __init__(self, ts, show_progress=True):
        self.ts = ts
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
                self.strain_map[md["strain"]] = node.id
                self.nodes_date[node.id] = md["date"]
                self.nodes_submission_date[node.id] = md["date_submitted"]
                pango = md["Nextclade_pango"]
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

        pango = md.get("Nextclade_pango", None)
        imputed_pango = md.get("Imputed_lineage", None)
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

    # TODO this is a strange name, but trying to differentiate from the
    # summary function above.
    def export_recombinants(self):
        recombinants = []

        def get_imputed_pango(u):
            lineage = self.nodes_metadata[u]["Imputed_lineage"]
            # Recombinant might be misleading, Unknown seems more accurate?
            if lineage == "Recombinant":
                return "Unknown"
            return lineage

        for u in self.recombinants:
            md = self.nodes_metadata[u]
            row = md["match_info"]
            assert len(row) == 2
            assert row[0]["strain"] == row[1]["strain"]
            matches = {}
            for record in row:
                matches[record["direction"]] = Match(
                    breakpoints=record["breakpoints"],
                    parents=record["parents"],
                    mutations=record["mutations"],
                )
            rec = Recombinant(row[0]["strain"], matches, node=u)
            if len(rec.matches["forward"].parents) == 2 and rec.is_hmm_consistent():
                strain_node = self.strain_map[rec.strain]
                strain_date = self.nodes_metadata[strain_node]["date"]
                left_parent = rec.matches["forward"].parents[0]
                right_parent = rec.matches["forward"].parents[1]
                record = {
                    "node": u,
                    "strain": rec.strain,
                    "strain_date": strain_date,
                    "max_descendant_samples": self.nodes_max_descendant_samples[u],
                    "lineage_left": get_imputed_pango(left_parent),
                    "lineage_right": get_imputed_pango(right_parent),
                    "interval_left": rec.matches["backward"].breakpoints[1],
                    "interval_right": rec.matches["backward"].breakpoints[1],
                    "num_mutations": len(rec.matches["forward"].mutations),
                    "mutations": rec.matches["forward"].mutations,
                }
                recombinants.append(record)

        return pd.DataFrame(recombinants)

    def export_recombination_node_breakpoints(self):
        """
        Make a dataframe with one row per recombination node breakpoint
        Return info about the path to the mrca of left and right parents when travelling
        up the tree to the left of the the breakpoint and to the right of the breakpoint.
        
        You can only list the breakpoints in recombination nodes that have 2 parents by
        doing e.g. df.drop_duplicates('node', keep=False)
        
        """
        # Note that it is more efficient to find the MRCA nodes in a single batch
        # for all breakpoints, reusing the trees, rather than getting a new tree
        # for each recombination node

        # Sort recombinants by breakpoint location
        forward_breakpoints = collections.defaultdict(list)
        meta = self.nodes_metadata
        for u in self.recombinants:
            md = meta[u]
            row = md["match_info"]
            assert len(row) == 2
            assert row[0]["strain"] == row[1]["strain"]
            matches = {}
            for record in row:
                matches[record["direction"]] = Match(
                    breakpoints=record["breakpoints"],
                    parents=record["parents"],
                    mutations=record["mutations"],
                )
            rec = Recombinant(row[0]["strain"], matches, node=u)
            breaks = rec.matches["forward"].breakpoints[1:-1]
            for i in range(len(breaks)):
                forward_breakpoints[breaks[i]].append(
                    (u, rec.is_hmm_consistent(), rec.matches["forward"].parents[i:i+2])
                )
        sorted_breakpoints = sorted(forward_breakpoints.keys())
        tree_a_indexes = np.searchsorted(
            self.ts.breakpoints(as_array=True),
            sorted_breakpoints,
            side='left',
        ) - 1
        tree_a_indexes = np.searchsorted(
            self.ts.breakpoints(as_array=True),
            sorted_breakpoints,
            side='right',
        ) - 1

        causal_sample_map = get_recombinant_samples(self.ts)
        tree_a = self.ts.first()    
        tree_b = self.ts.first()
        node_times = self.ts.nodes_time
        data = []
        for i_a, i_b, brk in zip(tree_a_indexes, tree_a_indexes, sorted_breakpoints):
            tree_a.seek(i_a)
            tree_b.seek(i_b)
            for nd, hmm_cons, (left_parent, right_parent) in forward_breakpoints[brk]:
                num_nodes_in_path = 0
                node_a = left_parent
                node_b = right_parent
                while True:
                    if node_a == tskit.NULL or node_b == tskit.NULL:
                        break
                    if node_times[node_a] == node_times[node_b]:
                        if node_a == node_b:
                            break
                        else:
                            if tree_a.parent(node_a) < tree_b.parent(node_b):
                                node_a = tree_a.parent(node_a)
                            else:
                                node_b = tree_b.parent(node_b)
                            num_nodes_in_path += 1
                                
                    elif node_times[node_a] < node_times[node_b]:
                        node_a = tree_a.parent(node_a)
                        num_nodes_in_path += 1
                    elif node_times[node_b] < node_times[node_a]:
                        node_b = tree_b.parent(node_b)
                        num_nodes_in_path += 1
                mrca = node_a
                row = {
                    "node": nd,
                    "breakpoint": brk,
                    "HMM_consistent": hmm_cons,
                    "left_parent": left_parent,
                    "left_parent_pango": meta[left_parent].get("Imputed_lineage", ""),
                    "right_parent": right_parent,
                    "right_parent_pango": meta[right_parent].get("Imputed_lineage", ""),
                    "parents_mrca": mrca,
                    "tmrca": np.nan,
                    "tmrca_delta": np.nan,
                    "nodes_between_parents": np.nan,
                    # Pick one of the causal strains and report its Nextclade label
                    "origin_nextclade_pango": meta[causal_sample_map[nd]]["Nextclade_pango"]
                }
                if mrca != tskit.NULL:
                     row["tmrca"] = node_times[mrca]
                     row["tmrca_delta"] = node_times[mrca] - node_times[nd]
                     row["nodes_between_parents"] = num_nodes_in_path
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
            html += f"<tr><th>P{i+1}</th>" + "".join(parent) + "</tr>"
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


def sample_subgraph(sample_node, ts, filepath=None):
    """
    Draws out a subgraph of the ARG above the given sample, including
    all nodes and edges on the path to the nearest sample nodes (showing any recombinations on
    the way)
    """
    col_green = "#228833"
    col_red = "#EE6677"
    col_purp = "#AA3377"
    col_blue = "#66CCEE"
    col_yellow = "#CCBB44"
    col_indigo = "#4477AA"
    col_grey = "#BBBBBB"

    G = nx.DiGraph()
    related_nodes = set((sample_node,))
    nodes_to_search = set((sample_node,))
    nodelabels = {}
    nodecolours = {}
    edgelabels = defaultdict(set)

    G.add_node(sample_node)
    nodecolours[sample_node] = col_green

    while nodes_to_search:
        node = ts.node(nodes_to_search.pop())
        nodelabels[node.id] = str(node.id) + "\n" + node.metadata["Imputed_lineage"]
        if (not node.is_sample()) or node.id == sample_node:
            parent_node = None
            for t in ts.trees():
                if t.parent(node.id) != parent_node:
                    parent_node = t.parent(node.id)
                    nodes_to_search.add(parent_node)
                    related_nodes.add(parent_node)
                    if parent_node not in G.nodes:
                        G.add_node(parent_node)
                        nodecolours[parent_node] = col_grey
                    G.add_edge(parent_node, node.id)
                    edge = ts.edge(t.edge(node.id))
                    if edge.right - edge.left != ts.sequence_length:
                        nodecolours[node.id] = col_red
                        edgelabels[(parent_node, node.id)].add(
                            (int(edge.left), int(edge.right))
                        )
        else:
            nodecolours[node.id] = col_blue

    for key, value in edgelabels.items():
        edgelabels[key] = "\n".join([str(v) for v in value])

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    dim_x = len(set(x for x, y in pos.values()))
    dim_y = len(set(y for x, y in pos.values()))
    fig = plt.figure(1, figsize=(dim_x * 1.5, dim_y * 1.1))

    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        labels=nodelabels,
        node_color=[nodecolours[node] for node in G.nodes],
        node_size=1600,
        font_size=6,
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edgelabels, label_pos=0.5, rotate=False, font_size=6
    )
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


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
    il, edited_ts = lineages.impute_lineages(
        ts, ti, linmuts_dict, df, ohe, clf, internal_only
    )
    return il, edited_ts
