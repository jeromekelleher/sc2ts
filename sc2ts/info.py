import functools
import collections
import logging
import json
import warnings
import dataclasses
import datetime
import re
from typing import List

import tskit
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import humanize
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import Markdown, HTML

from . import jit
from . import core
from . import utils
from . import inference


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
    counter = collections.Counter(md["exact_matches"]["pango"])
    key = "Viridian_pangolin"
    iterator = tqdm(
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
    iterator = tqdm(
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
        self,
        ts,
        *,
        pango_source="Viridian_pangolin",
        sample_group_id_prefix_len=10,
        show_progress=False,
    ):
        self.ts = ts
        top_level_md = ts.metadata["sc2ts"]
        self.date = top_level_md["date"]
        self.pango_source = pango_source
        logger.info("Computing ARG counts")
        c = jit.count(ts)
        self.mutations = self._mutations(c)
        self.nodes = self._nodes(c)

    def old_stuff(self):
        self.scorpio_source = "Viridian_scorpio"
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

        self.sample_group_id_prefix_len = sample_group_id_prefix_len
        self.sample_group_nodes = collections.defaultdict(list)
        self.sample_group_mutations = collections.defaultdict(list)
        self.retro_sample_groups = {}
        for retro_group in top_level_md["retro_groups"]:
            gid = retro_group["group_id"][: self.sample_group_id_prefix_len]
            self.retro_sample_groups[gid] = retro_group

        if not quick:
            self._preprocess_nodes(show_progress)
            self._preprocess_sites(show_progress)
            self._preprocess_mutations(show_progress)

    def node_counts(self):
        mc_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_MUTATION_OVERLAP)
        pr_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_REVERSION_PUSH)
        re_nodes = np.sum(self.ts.nodes_flags == core.NODE_IS_RECOMBINANT)
        exact_matches = np.sum((self.ts.nodes_flags & core.NODE_IS_EXACT_MATCH) > 0)
        u_nodes = np.sum(
            (self.ts.nodes_flags & core.NODE_IS_UNCONDITIONALLY_INCLUDED) > 0
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
            "u": u_nodes,
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
        self.nodes_num_exact_matches = np.zeros(ts.num_nodes, dtype=np.int32)
        self.nodes_metadata = {}
        samples = ts.samples()

        self.time_zero_as_date = np.array([self.date], dtype="datetime64[D]")[0]
        self.pango_lineage_samples = collections.defaultdict(list)
        self.first_scorpio_sample = {}

        # NOTE: keyed by *string* because of JSON
        exact_matches = ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"][
            "node"
        ]

        iterator = tqdm(
            ts.nodes(),
            desc="Indexing metadata    ",
            total=ts.num_nodes,
            disable=not show_progress,
        )
        for node in iterator:
            md = node.metadata
            self.nodes_metadata[node.id] = md
            self.nodes_num_exact_matches[node.id] = exact_matches.get(str(node.id), 0)
            group_id = None
            sc2ts_md = md.get("sc2ts", {})
            group_id = sc2ts_md.get("group_id", None)
            if group_id is not None:
                # Shorten key for readability.
                gid = group_id[: self.sample_group_id_prefix_len]
                self.sample_group_nodes[gid].append(node.id)
            if node.is_sample():
                self.nodes_date[node.id] = md["date"]
                pango = md.get(self.pango_source, "unknown")
                scorpio = md.get(self.scorpio_source, ".")
                if scorpio != "." and scorpio not in self.first_scorpio_sample:
                    self.first_scorpio_sample[scorpio] = node.id
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
        iterator = tqdm(
            np.arange(N), desc="Classifying mutations", disable=not show_progress
        )
        mutation_table = ts.tables.mutations
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
                if is_reversion[mut_id] and ts.mutations_node[parent] == tree.parent(
                    mutation_node
                ):
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

            # classify by origin
            md = mutation_table[mut_id].metadata["sc2ts"]
            inference_type = md.get("type", None)
            if inference_type == "parsimony":
                gid = md["group_id"][: self.sample_group_id_prefix_len]
                self.sample_group_mutations[gid].append(mut_id)

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

    @functools.cached_property
    def sites(self):

        num_missing_samples = np.full(self.ts.num_sites, -1, dtype=int)
        num_deletion_samples = np.full(self.ts.num_sites, -1, dtype=int)
        for site in self.ts.sites():
            md = site.metadata
            try:
                num_missing_samples[site.id] = md["sc2ts"]["missing_samples"]
                num_deletion_samples[site.id] = md["sc2ts"]["deletion_samples"]
            except KeyError:
                # Both of these keys were added at the same time, so no point
                # in doing two try/catches here.
                pass

        tables = self.ts.tables
        assert np.all(
            tables.sites.ancestral_state_offset == np.arange(self.ts.num_sites + 1)
        )
        ancestral_state = tables.sites.ancestral_state.view("S1").astype(str)
        del tables

        num_mutations = np.bincount(self.ts.mutations_site, minlength=self.ts.num_sites)
        return pd.DataFrame(
            {
                "id": np.arange(self.ts.num_sites, dtype=int),
                "position": self.ts.sites_position.astype("int"),
                "ancestral_state": ancestral_state,
                "num_missing_samples": num_missing_samples,
                "num_deletion_samples": num_deletion_samples,
                "num_mutations": num_mutations,
            },
        ).astype({"ancestral_state": pd.StringDtype()})

    def _nodes(self, arg_counter):
        ts = self.ts

        time_zero_as_date = np.array([self.date], dtype="datetime64[D]")[0]
        # NOTE not sure the internal times are getting rounded in to right days etc,
        # but day precision is probably right anyway.
        date = time_zero_as_date - ts.nodes_time.astype("timedelta64[D]")

        return pd.DataFrame(
            {
                "id": np.arange(ts.num_nodes, dtype=int),
                "flags": ts.nodes_flags,
                "time": ts.nodes_time,
                "date": date,
                "max_descendant_samples": arg_counter.nodes_max_descendant_samples,
                "num_mutations": np.bincount(ts.mutations_node, minlength=ts.num_nodes),
            },
        )

    def _mutations(self, arg_counter):
        ts = self.ts
        mutations_position = ts.sites_position[ts.mutations_site].astype(int)
        tables = ts.tables
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

        assert np.all(inherited_state != derived_state)

        is_reversion = np.zeros(ts.num_mutations, dtype=bool)
        is_reversion[mutations_with_parent] = (
            derived_state[mutations_with_parent] == inherited_state[parent]
        )
        return pd.DataFrame(
            {
                "id": np.arange(self.ts.num_mutations, dtype=int),
                "site": self.ts.mutations_site,
                "position": self.ts.mutations_site,
                "inherited_state": inherited_state,
                "derived_state": derived_state,
                "parent": self.ts.mutations_parent,
                "node": self.ts.mutations_node,
                # TODO ADD EDGE
                "num_parents": arg_counter.mutations_num_parents,
                "num_descendants": arg_counter.mutations_num_descendants,
                "num_inheritors": arg_counter.mutations_num_inheritors,
                "is_reversion": is_reversion,
            }
        ).astype(
            {"derived_state": pd.StringDtype(), "inherited_state": pd.StringDtype()}
        )

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
            ("sample_groups", len(self.sample_group_nodes)),
            ("retro_sample_groups", len(self.retro_sample_groups)),
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
        flags = self.ts.nodes_flags[u]

        strain = ""
        if flags & (tskit.NODE_IS_SAMPLE | core.NODE_IS_REFERENCE) > 0:
            strain = md["strain"]
        else:
            md = md.get("sc2ts", {})
            if (
                flags & (core.NODE_IS_MUTATION_OVERLAP | core.NODE_IS_REVERSION_PUSH)
                > 0
            ):
                try:
                    strain = f"{md['date_added']}:{', '.join(md['mutations'])}"
                except KeyError:
                    strain = "debug missing"
            elif "group_id" in md:
                # FIXME clipping this artificially for now
                # see https://github.com/jeromekelleher/sc2ts/issues/434
                strain = md["group_id"][:8]

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
            "flags": core.flags_summary(flags),
            "strain": strain,
            "pango": pango,
            "parents": np.sum(self.ts.edges_child == u),
            "children": np.sum(self.ts.edges_parent == u),
            "exact_matches": self.nodes_num_exact_matches[u],
            "descendants": self.nodes_max_descendant_samples[u],
            "date": self.nodes_date[u],
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

    def samples_summary(self):
        data = []
        md = self.ts.metadata["sc2ts"]["daily_stats"]
        for date, daily_stats in md.items():
            for row in daily_stats["samples_processed"]:
                data.append({"date": date, **row})
        df = pd.DataFrame(data)
        df["inserted"] = df["total"] - df["rejected"] - df["exact_matches"]
        return df

    def sample_groups_summary(self):
        data = []
        for group_id, nodes in self.sample_group_nodes.items():
            samples = []
            full_hashes = []
            for u in nodes:
                node = self.ts.node(u)
                if node.is_sample():
                    samples.append(u)
                    full_hashes.append(node.metadata["sc2ts"]["group_id"])
            assert len(set(full_hashes)) == 1
            assert full_hashes[0].startswith(group_id)
            data.append(
                {
                    "group_id": group_id,
                    "nodes": len(nodes),
                    "samples": len(samples),
                    "mutations": len(self.sample_group_mutations[group_id]),
                    "is_retro": group_id in self.retro_sample_groups,
                }
            )
        return pd.DataFrame(data).set_index("group_id")

    def retro_sample_groups_summary(self):
        data = []
        for group_id, retro_group in self.retro_sample_groups.items():
            d = dict(retro_group)
            d["group_id"] = group_id
            d["dates"] = len(set(d["dates"]))
            d["samples"] = len(d.pop("strains"))
            d["pango_lineages"] = len(set(d["pango_lineages"]))
            data.append(d)
        return pd.DataFrame(data).set_index("group_id")

    def recombinants_summary(
        self, parent_pango_source=None, characterise_copying=False, show_progress=True
    ):
        if parent_pango_source is None:
            parent_pango_source = self.pango_source
        data = []
        for u in self.recombinants:
            md = dict(self.nodes_metadata[u]["sc2ts"])
            group_id = md["group_id"][: self.sample_group_id_prefix_len]
            md["group_id"] = group_id
            group_nodes = self.sample_group_nodes[group_id]
            md["group_size"] = len(group_nodes)

            samples = []
            for v in self.sample_group_nodes[group_id]:
                if self.ts.nodes_flags[v] & tskit.NODE_IS_SAMPLE > 0:
                    samples.append(v)

            causal_lineages = {}
            hmm_matches = []
            breakpoint_intervals = []
            for v in samples:
                causal_lineages[v] = self.nodes_metadata[v].get(
                    self.pango_source, "Unknown"
                )

            # Arbitrarily pick the first sample node as the representative
            v = samples[0]
            node_md = self.nodes_metadata[v]["sc2ts"]
            hmm_matches.append(node_md["hmm_match"])
            breakpoint_intervals.append(node_md["breakpoint_intervals"])

            # Only deal with 2 parents recombs for now.
            assert self.nodes_num_parents[u] == 2
            # assert len(set(hmm_matches)) == 1
            # assert len(set(breakpoint_intervals)) == 1
            hmm_match = hmm_matches[0]
            assert len(hmm_match["path"]) == 2
            interval = breakpoint_intervals[0]
            parent_left = hmm_match["path"][0]["parent"]
            parent_right = hmm_match["path"][1]["parent"]
            data.append(
                {
                    "recombinant": u,
                    "descendants": self.nodes_max_descendant_samples[u],
                    "sample": v,
                    "sample_pango": causal_lineages[v],
                    "num_samples": len(samples),
                    "distinct_sample_pango": len(set(causal_lineages.values())),
                    "interval_left": interval[0][0],
                    "interval_right": interval[0][1],
                    "parent_left": parent_left,
                    "parent_right": parent_right,
                    "parent_left_pango": self.nodes_metadata[parent_left].get(
                        parent_pango_source,
                        "Unknown",
                    ),
                    "parent_right_pango": self.nodes_metadata[parent_right].get(
                        parent_pango_source,
                        "Unknown",
                    ),
                    "num_mutations": len(hmm_match["mutations"]),
                    **md,
                }
            )
        # Compute the MRCAs by iterating along trees in order of
        # breakpoint. We use the right interval
        df = pd.DataFrame(data).sort_values("interval_right")
        tree = self.ts.first()
        mrca_data = []
        for _, row in df.iterrows():
            bp = row.interval_right
            tree.seek(bp)
            assert tree.interval.left == bp
            right_path = jit.get_root_path(tree, row.parent_right)
            assert tree.parent(row.recombinant) == row.parent_right
            tree.prev()
            assert tree.interval.right == bp
            left_path = jit.get_root_path(tree, row.parent_left)
            assert tree.parent(row.recombinant) == row.parent_left
            mrca = jit.get_path_mrca(left_path, right_path, self.ts.nodes_time)
            mrca_data.append(mrca)
        mrca_data = np.array(mrca_data)
        df["mrca"] = mrca_data
        df["t_mrca"] = self.ts.nodes_time[mrca_data]

        if characterise_copying:
            # Slow - don't do this unless we really want to.
            df = self._characterise_recombinant_copying(df, show_progress)

        return df

    def _characterise_recombinant_copying(self, df, show_progress):
        """
        Return a copy of the specified recombinants_summary data frame
        in which we add fields to summarise the variant sites among
        the parents and sample.
        """
        max_run_length_data = []
        diff_data = []
        # Extract the haplotypes for each recombinant to catch runs of
        # adjacent differences
        for _, row in tqdm(df.iterrows(), total=df.shape[0], disable=not show_progress):
            H = np.array(
                inference.extract_haplotypes(
                    self.ts, [row.recombinant, row.parent_left, row.parent_right]
                )
            )
            # This is ugly but effective
            diffs = []
            for j in np.where(np.sum(H, axis=0) != 0)[0]:
                if len(np.unique(H[:, j])) != 1:
                    diffs.append(j)
            diff_data.append(len(diffs))
            x = self.ts.sites_position[diffs].astype(int)
            values, _, lengths = find_runs(np.diff(x))
            max_run_length = 0
            if np.sum(values == 1) > 0:
                max_run_length = np.max(lengths[values == 1])
            max_run_length_data.append(max_run_length)

        return df.assign(diffs=diff_data, max_run_length=max_run_length_data)

    def deletions_summary(self):
        deletion_ids = np.where(self.mutations.derived_state == "-")[0]
        df = pd.DataFrame(
            {
                "mutation": deletion_ids,
                "position": self.mutations.position[deletion_ids],
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
                num_inheritors = self.mutations.num_inheritors[e.mutations]
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

    def mutators_summary(self, threshold=10):
        mutator_nodes = np.where(self.nodes.num_mutations > threshold)[0]
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

    def copying_table(
        self,
        node,
        show_bases=None,
        hide_extra_rows=None,
        hide_labels=False,
        hide_runlengths=False,
        child_label=None,
        colours=None,
        exclude_stylesheet=None,
    ):
        """
        Return an styled HTML table indicating bases that differ between the parents of
        a recombination node. This is suitable for display e.g. in a Jupyter notebook
        using the ``IPython.display.HTML`` function.
        
        :param node int:
            The node ID of the child node, usually a recombination node.
            This will be placed on the second row of the copying pattern, so that
            in the most used case of a recombination node with two parents, the
            child is the middle row.
        :param show_base bool:
            If True, show the allelic state (i.e. ``A``, ``C``,
            ``G``, ``T``, or `-`) for each position at which the parents differ.
            If True, do not plot a character, but simply show coloured table cells.
            If None (default), show an em-dash for deletions, but nothing else.
        :param hide_extra_rows bool:
            If True, hide the rows showing site positions, reference alleles, and
            de-novo mutation state changes. If False or None (default), show these rows.
        :param hide_runlengths bool:
            If True, omit the lower bar that indicates adjacent bases as runs of red
            (or orange for near-adjacent) and tickmarks. If False or None, show this bar.
        :param hide_labels bool:
            If False or None (default), label the rows with P0, P1, etc. If True, hide
            these row labels.
        :param child_label str:
            The label to use for the child node. If None (default), use "C".
        :param colours list:
            A list of at least 2 hex colours to use. ``colours[0]`` is used for a base in
            the child that is not present in any parent, ``colours[1]`` for a base that
            matches the first parent, ``colours[2]`` for the second parent, etc.
            Default: None, treated as ``["#FC0", "#8D8", "#6AD", "#B9D", "#A88"]``.
        :param exclude_stylesheet bool:
            If True, exclude the default stylesheet from the HTML output. This is useful
            simply to save space if you want to include the copying table in a larger HTML
            document (e.g. a Jupyter notebook) that already has one copying table shown with
            the standard stylesheet. If False or None (default), include the default stylesheet.
        :return str:
            An HTML string representing the copying table.
        """
        edges = tskit.EdgeTable()
        for e in sorted([self.ts.edge(i) for i in np.where(self.ts.edges_child==node)[0]], key=lambda e: e.left):
            edges.append(e)
        return self._copying_table(
            node,
            edges,
            show_bases=show_bases,
            hide_runlengths=hide_runlengths,
            hide_extra_rows=hide_extra_rows,
            hide_labels=hide_labels,
            child_label=child_label,
            colours=colours,
            exclude_stylesheet=exclude_stylesheet,
        )

    def _copying_table(
        self,
        node,
        edges,
        show_bases=True,
        hide_runlengths=None,
        hide_extra_rows=None,
        hide_labels=None,
        child_label="C",
        colours=None,
        exclude_stylesheet=None,
    ):
        # private interface, used internally
        def cell_attributes(col, outline_sides=None):
            css = [f"background-color:{col}"]
            if outline_sides is None:
                outline_sides = []
            elif isinstance(outline_sides, str):
                outline_sides = [outline_sides]
            for side in outline_sides:
                css.append(f"border-{side}-width:3px")
            return f' style="{";".join(css)}"'

        def line_cell(pos, prev_pos, next_pos):
            dist_to_left = pos - prev_pos
            dist_to_right = next_pos - pos
            if dist_to_left > 2:
                dist_to_left = 0
            if dist_to_right > 2:
                dist_to_right = 0
            return f'<td title="{pos}" class="run-{int(dist_to_left)}-{int(dist_to_right)}"></td>'

        def row_lab(txt):
            return "" if hide_labels else f"<th>{txt}</th>"

        def label(allele, default=""):
            if show_bases is None:
                return ("<b>&mdash;</b>" if allele=="-" else default)
            if show_bases:
                return allele
            return ''
        
        if colours is None:
            colours = [  # Chosen to be light enough that black text on top is readable
                "#FC0",  # Gold for de-novo mutations
                "#8D8",  # Copy from first parent: light green
                "#6AD",  # Copy from second parent: light blue
                "#B9D",  # Copy from third parent (if any): light purple
                "#A88",  # Copy from fourth parent (if any): light brown
            ]
        parent_colours = colours.copy()
        parent_colours[0] = "#FFF"  # white for non-matching

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
        runs = []
        parents = [[] for _ in range(len(parent_cols))]
        child = []
        extra_mut = []
        prev_parent_col = None
        for var in variants:
            if len(np.unique(var.genotypes)) > 1:
                pos = int(var.site.position)
                positions.append(pos)
                ref.append(f"<td>{var.site.ancestral_state}</td>")
                child_allele = var.alleles[var.genotypes[0]]

                edge_index = np.searchsorted(edges.left, pos, side="right") - 1
                parent_col = parent_cols[edges[edge_index].parent]
                is_switch = False if prev_parent_col is None else parent_col != prev_parent_col

                child_colour_index = 0
                for j in range(1, len(var.genotypes)):
                    parent_allele = var.alleles[var.genotypes[j]]
                    if parent_allele == child_allele:
                        child_colour_index=j

                for j in range(1, len(var.genotypes)):
                    parent_allele = var.alleles[var.genotypes[j]]
                    col=parent_colours[0]
                    if parent_allele == child_allele:
                        try:
                            col = parent_colours[j]
                        except IndexError as e:
                            raise ValueError(
                                "Displaying the copying path only deals with a max of "
                                f"{len(parent_colours)-1} parents"
                            ) from e
                    elif parent_allele == var.site.ancestral_state:
                        col = "#DDD"
                    outline_sides = []
                    if j == parent_col:
                        outline_sides.append("top" if j == 1 else "bottom")
                    if is_switch and max(parent_col, 2) >= j:
                        outline_sides.append("left")
                    attr = cell_attributes(col, outline_sides)
                    parents[j - 1].append(f"<td{attr}>{label(parent_allele)}</td>")
                    
                attr = cell_attributes(
                    colours[child_colour_index],
                    outline_sides="left" if is_switch else None,
                    #default_border_width=1  # uncomment to outline child bases with a border
                )
                child.append(f"<td{attr}>{label(child_allele)}</td>")
                extra_mut.append(f"<td><span{vrl}>{mutations.get(pos, '')}</span></td>")
                prev_parent_col = parent_col
        html = ""
        if not exclude_stylesheet:
            # a class like "run-1-2" means a cell which has a closest left hand neighbour
            # 1 bp away (i.e. adjacent) but a right hand neighbour 2 bp away.
            # "0" is the "null" value, so "run-0-0" means neither neighbour is nearby
            runlength_cols = ("white", "red", "orange")
            bg_im_src = (
                "background-image:linear-gradient(to right, {0} 50%, {1} 50%);"
                "background-image:-webkit-linear-gradient(left, {0} 50%, {1} 50%);"  # for imgkit/wkhtmltopdf
            )
            html += "<style>"
            html += ".copying-table .pattern td {border:0px solid black; text-align: center; width:1em}"
            html += ".copying-table {border-spacing: 0px; border-collapse: collapse}"
            html += ".copying-table .runlengths {font-size:3px; height:3px;}"
            html += ".copying-table .runlengths td {border-style: solid; background: white; border-width:0px 1px; border-color: black}"
            for left in range(len(runlength_cols)):
                for right in range(len(runlength_cols)):
                    html += (
                        f".copying-table .runlengths .run-{left}-{right}" +
                        "{" + bg_im_src.format(runlength_cols[left], runlength_cols[right]) + "}"
                    )
            html += "</style>"
        html += '<table class="copying-table">'
        if not hide_extra_rows:
            pos = [f"<td><span{vrl}>{p}</span></td>" for p in positions]
            html += '<tr style="font-size: 70%">' + row_lab("pos") + "".join(pos) + '</tr>'
            html += '<tr>' + row_lab("ref") + "".join(ref) + '</tr>'
        rowstyle = "font-size: 10px; border: 0px; height: 14px"
        html += f'<tr class="pattern" style="{rowstyle}">' + row_lab("P0") + "".join(parents.pop(0)) + '</tr>'
        html += f'<tr class="pattern" style="{rowstyle}">' + row_lab(child_label) + "".join(child) + '</tr>'
        for i, parent in enumerate(parents):
            html += f'<tr class="pattern" style="{rowstyle}">' + row_lab(f"P{i+1}") + "".join(parent) + '</tr>'
        if not hide_runlengths:
            p = np.concatenate(([-np.inf], positions, [np.inf]))
            runs = [line_cell(p[i+1], p[i], p[i+2]) for i in range(len(positions))]
            html += "<tr style='font-size: 3px; height: 3px'></tr>"
            html += "<tr class='runlengths'>" + row_lab("") + "".join(runs) + "</tr>"
        if not hide_extra_rows:
            html += '<tr style="font-size: 75%">' + row_lab("mut") + "".join(extra_mut) + "</tr>"
        return html + "</table>"


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
        nodes_with_many_muts = np.sum(self.nodes.num_mutations >= 10)
        return self._histogram(
            self.nodes.num_mutations,
            title=f"Nodes with >= 10 muts: {nodes_with_many_muts}",
            bins=range(10),
            xlabel="Number of mutations",
            ylabel="Number of nodes",
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
        sites_with_many_muts = np.sum(self.sites.num_mutations >= 10)
        ax.set_title(f"Sites with >= 10 muts: {sites_with_many_muts}")
        ax.hist(self.sites.num_mutations, range(10), rwidth=0.9)
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
        keep = self.mutations.num_inheritors >= min_inheritors
        inherited = self.mutations.inherited_state.values[keep].astype("S1")
        derived = self.mutations.derived_state.values[keep].astype("S1")
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

    def _wide_plot(self, *args, height=4, **kwargs):
        return plt.subplots(*args, figsize=(16, height), **kwargs)

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
            count = self.sites.num_mutations
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
            self.sites.num_missing_samples, annotate_threshold
        )
        ax.set_ylabel("Number missing samples")
        return fig, [ax]

    def plot_deletion_samples_per_site(self, annotate_threshold=0.5):
        fig, ax = self._plot_per_site_count(
            self.sites.num_deletion_samples, annotate_threshold
        )
        ax.set_ylabel("Number deletion samples")
        return fig, [ax]

    def compute_deletion_overlaps(self, df_del):
        ts = self.ts
        overlaps = np.zeros(int(ts.sequence_length))
        df_del = self.deletions_summary()
        for row in df_del.itertuples():
            overlaps[row.start : row.start + row.length] += 1
        return overlaps[ts.sites_position.astype(int)]

    def plot_deletion_overlaps(self, annotate_threshold=0.9):
        df_del = self.deletions_summary()
        fig, ax = self._plot_per_site_count(
            self.compute_deletion_overlaps(df_del), annotate_threshold
        )
        ax.set_ylabel("Overlapping deletions")
        return fig, [ax]

    def plot_samples_per_day(
        self, start_date="2020-01-01", end_date="3000-01-01", scorpio_fraction=0.05
    ):
        df = self.samples_summary()
        df = df[(df.date >= start_date) & (df.date < end_date)]

        dfa = df.groupby("date").sum().reset_index().astype({"date": "datetime64[s]"})
        dfa["mean_hmm_cost"] = dfa["total_hmm_cost"] / dfa["total"]

        fig, (ax1, ax2, ax3, ax4) = self._wide_plot(4, height=12, sharex=True)
        exact_col = "tab:red"
        in_col = "tab:purple"
        ax1.plot(dfa.date, dfa.inserted, label="In ARG", color=in_col)
        ax1.plot(dfa.date, dfa.total, label="Processed")
        ax1.plot(dfa.date, dfa.exact_matches, label="Exact matches", color=exact_col)

        ax2.plot(
            dfa.date,
            dfa.inserted / dfa.total,
            label="Fraction processed in ARG",
            color=in_col,
        )
        ax2.plot(
            dfa.date,
            dfa.exact_matches / dfa.total,
            label="Fraction processed exact matches",
            color=exact_col,
        )

        ax3.plot(dfa.date, dfa.rejected / dfa.total, label="Fraction excluded")
        ax3_2 = ax3.twinx()
        ax3_2.plot(
            dfa.date, dfa.mean_hmm_cost, label="mean HMM cost", color="tab:orange"
        )
        ax2.set_ylabel("Fraction of samples")
        ax3.set_ylabel("Fraction of samples")
        ax4.set_xlabel("Date")
        ax3_2.set_ylabel("Mean HMM cost")
        ax1.set_ylabel("Number of samples")
        ax1.legend()
        ax2.legend()
        ax3.legend(loc="upper right")
        ax3_2.legend(loc="upper left")

        for ax in [ax1, ax2, ax3]:
            ax.grid()

        df_scorpio = df.pivot_table(
            columns="scorpio", index="date", values="total", aggfunc="sum", fill_value=0
        ).reset_index()
        # Need force conversion back to datetime here for some reason
        df_scorpio = df_scorpio.astype({"date": "datetime64[s]"}).set_index("date")
        # convert to fractions
        df_scorpio = df_scorpio.divide(df_scorpio.sum(axis="columns"), axis="index")
        # Remove columns that don't have more than the threshold
        keep_cols = []
        first_scorpio_date = []
        for col in df_scorpio:
            if np.any(df_scorpio[col] >= scorpio_fraction):
                keep_cols.append(col)
                try:
                    first_date = self.nodes.date[self.first_scorpio_sample[col]]
                    first_scorpio_date.append((first_date, col))
                except KeyError:
                    warnings.warn(f"No samples for Scorpio {col} present")

        df_scorpio = df_scorpio[keep_cols]
        ax4.set_title("Scorpio composition of processed samples")
        ax4.stackplot(
            df_scorpio.index,
            *[df_scorpio[s] for s in df_scorpio],
            labels=[" ".join(s.split("_")) for s in df_scorpio],
        )
        ax4.legend(loc="upper left", ncol=2)

        j = 0
        n = 5
        for date, scorpio in sorted(first_scorpio_date):
            y = (j + 1) / n
            ax4.annotate(f"{scorpio}", xy=(date, y), xycoords="data")
            ax4.axvline(date, color="grey", alpha=0.5)
            j = (j + 1) % (n - 1)

        return fig, [ax1, ax2, ax3, ax4]

    def plot_resources(self, start_date="2020-01-01", end_date="3000-01-01"):
        ts = self.ts
        fig, ax = self._wide_plot(3, height=8, sharex=True)

        dfs = self.samples_summary()
        dfa = dfs.groupby("date").sum()
        dfa["mean_hmm_cost"] = dfa["total_hmm_cost"] / dfa["total"]
        df = dfa.join(self.resources_summary(), how="inner")
        df = df.rename(
            columns={"inserted": "smaples_in_arg", "total": "samples_processed"}
        )
        df = df[(df.index >= start_date) & (df.index < end_date)]

        df["cpu_time"] = df.user_time + df.sys_time
        x = np.array(df.index, dtype="datetime64[D]")

        total_elapsed = datetime.timedelta(seconds=np.sum(df.elapsed_time))
        total_cpu = datetime.timedelta(seconds=np.sum(df.cpu_time))
        title = (
            f"{humanize.naturaldelta(total_elapsed)} elapsed "
            f"using {humanize.naturaldelta(total_cpu)} of CPU time "
            f"(utilisation = {np.sum(df.cpu_time) / np.sum(df.elapsed_time):.2f})"
        )

        ax[0].set_title(title)
        ax[0].plot(x, df.elapsed_time / 60, label="elapsed time")
        ax[-1].set_xlabel("Date")
        ax_twin = ax[0].twinx()
        ax_twin.plot(
            x, df.samples_processed, color="tab:red", alpha=0.5, label="samples"
        )
        ax_twin.legend(loc="upper left")
        ax_twin.set_ylabel("Samples processed")
        ax[0].set_ylabel("Elapsed time (mins)")
        ax[0].legend()
        ax_twin.legend()
        ax[1].plot(
            x, df.elapsed_time / df.samples_processed, label="Mean time per sample"
        )
        ax[1].set_ylabel("Elapsed time per sample (s)")
        ax[1].legend(loc="upper right")

        ax_twin = ax[1].twinx()
        ax_twin.plot(
            x, df.mean_hmm_cost, color="tab:orange", alpha=0.5, label="HMM cost"
        )
        ax_twin.set_ylabel("HMM cost")
        ax_twin.legend(loc="upper left")
        ax[2].plot(x, df.max_memory / 1024**3)
        ax[2].set_ylabel("Max memory (GiB)")

        for a in ax:
            a.grid()
        return fig, ax

    def resources_summary(self):
        ts = self.ts
        data = []
        for p in ts.provenances():
            record = json.loads(p.record)
            text_date = record["parameters"]["date"]
            resources = record["resources"]
            data.append({"date": text_date, **resources})
        return pd.DataFrame(data).set_index("date")

    def node_type_summary(self):
        ts = self.ts
        nodes_num_children = np.bincount(ts.edges_parent, minlength=ts.num_nodes)
        data = []
        for fv in core.flag_values:
            select = (ts.nodes_flags & fv.value) > 0
            count = np.sum(nodes_num_children[select] > 0)
            total = max(1, np.sum(select))
            datum = {
                "flag_short": fv.short,
                "flag_long": fv.long,
                "total": total,
                "with_children": np.sum(nodes_num_children[select] > 0) / total,
                "with_exact_matches": np.sum(self.nodes_num_exact_matches[select] > 0)
                / total,
            }
            data.append(datum)
        return pd.DataFrame(data).set_index("flag_short")

    def draw_pango_lineage_subtree(
        self,
        pango_lineage,
        position=None,
        *args,
        **kwargs,
    ):
        """
        Draw a subtree of the tree sequence containing only the samples from a given
        Pango lineage, e.g. "B.1.1.7". This is a convenience function that calls
        draw_subtree with the
        appropriate set of samples. See that function for more details.
        """
        return self.draw_subtree(
            tracked_pango=[pango_lineage], position=position, *args, **kwargs
        )

    def draw_subtree(
        self,
        *,
        tracked_pango=None,
        tracked_strains=None,
        tracked_samples=None,
        position=None,
        collapse_tracked=None,
        remove_clones=None,
        extra_tracked_samples=None,
        pack_untracked_polytomies=True,
        time_scale="rank",
        y_ticks=None,
        y_label=None,
        date_format=None,
        title=None,
        mutation_labels=None,
        append_mutation_recurrence=None,
        size=None,
        style="",
        symbol_size=4,
        **kwargs,
    ):
        """
        Draw a subtree of the tree sequence at a given ``position``, focussed on the
        samples specified by `tracked_pango`, `tracked_strains` and `tracked_samples`
        (at least one must be specified, and all are combined into a single list).
        Clades containing only untracked nodes are visually collapsed, and (by default)
        untracked node lineages within polytomies are condensed into a dotted line.
        Clades containing more than a certain proportion of tracked nodes can also be
        collapsed (see the ``collapse_tracked`` parameter).

        Most parameters are passed directly to ``tskit.Tree.draw_svg()`` method, apart
        from the following:
        :param position int: The genomic position at which to draw the tree. If None,
            the start of the spike protein is used.
        :param tracked_pango list: A list of Pango lineages (e.g. ["B.1.1.7"])
            to track in the tree.
        :param tracked_strains list: A list of strains (e.g.
            ``["ERR4413600", "ERR4460507"]``) to track in the tree.
        :param tracked_samples list: A list of sample nodes to track in the tree.
        :param collapse_tracked Union(float): Determine when to collapse clades
            containing tracked nodes. If ``None`` (default), do not collapse
            any such clades , otherwise only collapse them when they contain more
            than a fraction `collapse_tracked` of tracked nodes.
        :param date_format str: How to format the displayed node date: one of "ts" (use
            the tree sequence time, usually where the most recent sample is at zero),
            "from_zero" (set the earliest sample to zero and count time as negative
            days from the start of the genealogy), or "cal" (use the calendar date from
            the sample metadata, hence do not display times for nonsample nodes).
            Default: ``None`` treated as "from_zero".
        :param remove_clones bool: Whether to remove samples that are clones of other
            samples (i.e. that have no mutations above them). Currently unimplemented.
        :param extra_tracked_samples list: Additional nodes to track in the tree, to
            allow the context of the tracked nodes to be seen. By default, extra tracked
            sample nodes are coloured differently to tracked nodes.
        :param pack_untracked_polytomies bool: When a polytomy exists involving lineages
            containing both tracked and untracked nodes, should the untracked lineages
            be placed on the right, and condensed into a dotted line? Default: ``True``
        :param mutation_labels dict: A dictionary mapping mutation IDs to labels. If not
            provided, mutation labels are generated automatically, in the form
            ``{inherited_state}{position}{derived_state}``
        :param append_mutation_dupes bool: If True (default), append a count to the
            mutation label indicating the number of other such mutations above the
            shown nodes that are at the same position and to the same derived state.
        :param time_scale str: As for the ``time_scale`` parameter of `draw_svg()`, but
            defaults to "rank".
        :param y_label str: As for the ``y_label`` parameter of `draw_svg()`.
        :param y_ticks array: As for the ``y_ticks`` parameter of `draw_svg()`. Cannot be
            combined with ``date_format="cal"``.

        .. note::
            By default, styles are set such that tracked pango / strain / sample nodes
            are coloured in cyan, and extra tracked nodes in orange. Mutations seen only
            once in the visualised tree are coloured in dark red, multiple mutations at
            the same position are coloured in red, and mutations that are immediate
            reversions of a mutation above them in the tree are coloured in magenta.
        """

        if position is None:
            position = 21563  # pick the start of the spike
        if size is None:
            size = (1000, 1000)
        if date_format is None:
            date_format = "from_zero"
        if y_label is None:
            if date_format == "cal":
                y_label = "Calendar date"
            elif date_format == "ts":
                y_label = f"Time ({self.ts.time_units} ago)"
            else:
                y_label = f"Time difference from earliest sample ({self.ts.time_units})"
        if append_mutation_recurrence is None:
            append_mutation_recurrence = True
        if remove_clones:
            # TODO
            raise NotImplementedError("remove_clones not implemented")
        # remove mutation times, so they get spaced evenly along a branch
        tables = self.ts.dump_tables()
        time = tables.mutations.time
        time[:] = tskit.UNKNOWN_TIME
        tables.mutations.time = time
        # rescale to negative times
        if date_format == "from_zero":
            for node in reversed(self.ts.nodes(order="timeasc")):
                if node.is_sample():
                    break
            tables.nodes.time = tables.nodes.time - node.time
        ts = tables.tree_sequence()

        tracked_nodes = []
        if tracked_pango is not None:
            for lineage in tracked_pango:
                tracked_nodes.extend(self.pango_lineage_samples[lineage])
                if title is None:
                    f"Sc2ts genealogy of {len(tracked_nodes)} {lineage} samples. "
        if tracked_strains is not None:
            for strain in tracked_strains:
                tracked_nodes.append(self.strain_map[strain])
        if tracked_samples is not None:
            tracked_nodes.extend(tracked_samples)
        if len(tracked_nodes) == 0:
            raise ValueError(
                "No tracked nodes specified: you must provide one or more of "
                "`tracked_pango`, `tracked_strains` or `tracked_samples`."
            )
        if title is None:
            f"Sc2ts genealogy of {len(tracked_nodes)} samples. "
        tracked_nodes = np.unique(tracked_nodes)

        if extra_tracked_samples is not None:
            tn_set = set(tracked_nodes)
            extra_tracked_samples = [
                e for e in extra_tracked_samples if e not in tn_set
            ]
            tracked_nodes = np.concatenate((tracked_nodes, extra_tracked_samples))
        tree = ts.at(position, tracked_samples=tracked_nodes)
        order = np.array(
            list(
                tskit.drawing._postorder_tracked_minlex_traversal(
                    tree, collapse_tracked=collapse_tracked
                )
            )
        )

        if title is None:
            title = f"Sc2ts genealogy of {len(tracked_nodes)} samples. "
        simplified_ts = ts.simplify(
            order[np.where(ts.nodes_flags[order] & tskit.NODE_IS_SAMPLE)[0]]
        )
        num_trees = simplified_ts.num_trees
        tree_pos = simplified_ts.at(position).index + 1
        # TODO - show filename
        title += f"Position {position} (tree {tree_pos}/{num_trees})"

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

        multiple_mutations = []
        reverted_mutations = []
        recurrent_mutations = collections.defaultdict(list)
        mut_labels = {} if mutation_labels is None else mutation_labels.copy()
        use_mutations = np.where(np.isin(ts.mutations_node, shown_nodes))[0]
        sites = ts.mutations_site[use_mutations]
        for mut_id in use_mutations:
            # TODO Viz the recurrent mutations
            mut = ts.mutation(mut_id)
            site = ts.site(mut.site)
            if np.sum(sites == site.id) > 1:
                multiple_mutations.append(mut.id)
            inherited_state = site.ancestral_state
            if mut.parent >= 0:
                parent = ts.mutation(mut.parent)
                inherited_state = parent.derived_state
                parent_inherited_state = site.ancestral_state
                if parent.parent >= 0:
                    parent_inherited_state = ts.mutation(parent.parent).derived_state
                if parent_inherited_state == mut.derived_state:
                    reverted_mutations.append(mut.id)
            pos = int(site.position)
            recurrent_mutations[(pos, mut.derived_state)].append(mut.id)
            if mutation_labels is None:
                mut_labels[mut.id] = f"{inherited_state}{pos}{mut.derived_state}"
            # If more than one mutation has the same label, add a prefix with the counts
        if append_mutation_recurrence:
            num_recurrent = {
                m_id: (i + 1, len(ids))
                for ids in recurrent_mutations.values()
                for i, m_id in enumerate(ids)
                if len(ids) > 1
            }
            for m_id, (i, n) in num_recurrent.items():
                if m_id in mut_labels:
                    mut_labels[m_id] += f" ({i}/{n})"
        # some default styles
        styles = [
            "".join(f".n{u} > .sym {{fill: cyan}}" for u in tracked_nodes),
            ".mut .lab, .mut.extra .lab{fill: darkred}",
            ".mut .sym, .mut.extra .sym{stroke: darkred}",
            ".background path {fill: white}",
            ".lab.summary {font-size: 12px}",
            ".polytomy {font-size: 10px}",
            ".mut .lab {font-size: 10px}",
            ".y-axis .lab {font-size: 12px}",
        ]
        if extra_tracked_samples is not None:
            styles.append(
                "".join(f".n{u} > .sym {{fill: orange}}" for u in extra_tracked_samples)
            )
        if len(multiple_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in multiple_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in multiple_mutations)
            styles.append(lab_css + "{fill: red}" + sym_css + "{stroke: red}")
        if len(reverted_mutations) > 0:
            lab_css = ", ".join(f".mut.m{m} .lab" for m in reverted_mutations)
            sym_css = ", ".join(f".mut.m{m} .sym" for m in reverted_mutations)
            styles.append(lab_css + "{fill: magenta}" + sym_css + "{stroke: magenta}")
        # Recombination nodes as larger open circles
        re_nodes = np.where(ts.nodes_flags & core.NODE_IS_RECOMBINANT)[0]
        styles.append(
            ",".join([f".node.n{u} > .sym" for u in re_nodes])
            + f"{{r:{symbol_size/2*1.5:.2f}px; stroke:black; fill:white}}"
        )
        if date_format == "cal":
            if y_ticks is not None:
                raise ValueError("Cannot set y_ticks when date_format='cal'")
            shown_times = ts.nodes_time[shown_nodes]
            if time_scale == "rank":
                _, index = np.unique(shown_times, return_index=True)
                y_ticks = {
                    i: ts.node(shown_nodes[t]).metadata.get("date", "")
                    for i, t in enumerate(index)
                }
            else:
                # only place ticks at the sample nodes
                y_ticks = {
                    t: ts.node(u).metadata.get("date", "")
                    for u, t in zip(shown_nodes, shown_times)
                }
        return tree.draw_svg(
            time_scale=time_scale,
            y_axis=True,
            x_axis=False,
            y_label=y_label,
            title=title,
            size=size,
            order=order,
            mutation_labels=mut_labels,
            all_edge_mutations=True,
            y_ticks=y_ticks,
            symbol_size=symbol_size,
            pack_untracked_polytomies=pack_untracked_polytomies,
            style="".join(styles) + style,
            **kwargs,
        )

    def get_sample_group_info(self, group_id):
        samples = []

        group_nodes = self.sample_group_nodes[group_id]
        for u in group_nodes:
            if self.ts.nodes_flags[u] & tskit.NODE_IS_SAMPLE > 0:
                samples.append(u)

        tree = self.ts.first()
        while u in group_nodes:
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
            group_nodes,
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
        symbol_size=3,
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
        # mutations shared by all the samples (above their mrca)
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
        # recombination nodes in larger open white circles
        re_nodes = np.where(ts.nodes_flags & core.NODE_IS_RECOMBINANT)[0]
        styles.append(
            ",".join([f".node.n{u} > .sym" for u in re_nodes])
            + f"{{r: {symbol_size/2*1.5:.2f}px; stroke: black; fill: white}}"
        )
        svg = self.ts.draw_svg(
            size=size,
            time_scale=time_scale,
            y_axis=y_axis,
            mutation_labels=mutation_labels,
            y_ticks=y_ticks,
            node_labels=node_labels,
            style="".join(styles) + style,
            symbol_size=symbol_size,
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
