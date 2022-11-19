import logging
import datetime
import dataclasses
import collections

import tqdm
import tskit
import tsinfer
import numpy as np

# TODO move constants into core
from . import constants
from . import core

logger = logging.getLogger(__name__)


def infer(
    sd,
    *,
    ancestors_ts=None,
    num_mismatches=None,
    show_progress=False,
    max_submission_delay=None,
    num_threads=None,
    precision=None,
):
    if num_mismatches is None:
        # Default to no recombination
        num_mismatches = 1000
    if max_submission_delay is None:
        max_submission_delay = 10**8  # Arbitrary large number of days.

    max_submission_delay = np.timedelta64(max_submission_delay, "D")

    date = []
    date_submitted = []
    for ind in sd.individuals():
        date.append(ind.metadata["date"])
        date_submitted.append(ind.metadata["date_submitted"])
    date = np.array(date, dtype=np.datetime64)
    current_date = date[0]
    if not np.all(date == current_date):
        raise ValueError("Only one day of data can be added at a time")

    date_submitted = np.array(date_submitted, dtype=np.datetime64)
    submission_delay = date_submitted - date
    submission_delay = submission_delay.astype("timedelta64[D]")

    ts = ancestors_ts
    previous_date = None
    increment = 1
    if ancestors_ts is not None:
        previous_date = np.datetime64(ts.node(ts.samples()[-1]).metadata["date"])
        diff = current_date - previous_date
        increment = diff.astype("timedelta64[D]").astype("int")
        assert increment > 0

    samples = np.where(submission_delay <= max_submission_delay)[0]
    num_samples = len(samples)
    num_rejected = sd.num_samples - num_samples
    fraction_rejected = num_rejected / sd.num_samples

    logger.info(
        f"Filtered {num_rejected} ({100 * fraction_rejected:.2f}%) samples "
        f"with submission_delay > {max_submission_delay}"
    )
    logger.info(f"Extending for {current_date} with {len(samples)} samples")

    return extend(
        sd,
        ancestors_ts=ancestors_ts,
        samples=samples,
        num_mismatches=num_mismatches,
        time_increment=increment,
        show_progress=show_progress,
        num_threads=num_threads,
        precision=precision,
    )


def solve_num_mismatches(ts, k):
    """
    Return the low-level LS parameters corresponding to accepting
    k mismatches in favour of a single recombination.
    """
    m = ts.num_sites
    n = ts.num_nodes  # We can match against any node in tsinfer
    if k == 0:
        # Pathological things happen when k=0
        r = 1e-3
        mu = 1e-20
    else:
        mu = 1e-6
        denom = (1 - mu) ** k + (n - 1) * mu**k
        r = n * mu**k / denom
        assert mu < 0.5
        assert r < 0.5

    ls_recomb = np.full(m - 1, r)
    ls_mismatch = np.full(m, mu)
    return ls_recomb, ls_mismatch


def make_initial_tables(sample_data):
    reference = core.get_reference_sequence()
    tables = tskit.TableCollection(sample_data.sequence_length)
    tables.time_units = constants.TIME_UNITS
    base_schema = tskit.MetadataSchema.permissive_json().schema

    tables.metadata_schema = tskit.MetadataSchema(base_schema)
    # TODO gene annotations to top level
    tables.nodes.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.sites.metadata_schema = tskit.MetadataSchema(base_schema)
    for site in sample_data.sites():
        assert site.ancestral_state == reference[int(site.position)]
        tables.sites.add_row(
            site.position, site.ancestral_state, metadata={"masked_samples": 0}
        )
    # TODO should probably make the ultimate ancestor time something less
    # plausible or at least configurable.
    # NOTE: adding the ultimate ancestor is an artefact of using
    # tsinfer's matching engine. This shouldn't be necessary when
    # we move over the tskit.
    for t in [1, 0]:
        tables.nodes.add_row(time=t)
    # TODO node 1 should be given metadata as the reference.
    tables.edges.add_row(0, sample_data.sequence_length, 0, 1)
    return tables


def get_ancestors_ts(sample_data, ancestors_ts, time_increment):
    if ancestors_ts is None:
        tables = make_initial_tables(sample_data)
    else:
        # Should do more checks here for suitability.
        if ancestors_ts.time_units != constants.TIME_UNITS:
            raise ValueError(
                f"Mismatched time_units: {ancestors_ts.time_units}",
            )
        tables = ancestors_ts.dump_tables()

    tables.nodes.time += time_increment
    tables.mutations.time += time_increment

    return tables.tree_sequence()


def run_tsinfer_matching(
    *,
    sample_data,
    ancestors_ts,
    samples,
    num_mismatches,
    show_progress,
    precision,
    num_threads,
):
    ls_recomb, ls_mismatch = solve_num_mismatches(ancestors_ts, num_mismatches)
    pm = tsinfer.inference._get_progress_monitor(
        show_progress,
        generate_ancestors=False,
        match_ancestors=False,
        match_samples=False,
    )
    manager = Matcher(
        sample_data,
        ancestors_ts,
        allow_multiallele=True,
        recombination=ls_recomb,
        mismatch=ls_mismatch,
        progress_monitor=pm,
        num_threads=num_threads,
        precision=precision,
    )
    return manager.run_match(np.array(samples))


def extend(
    sample_data,
    *,
    ancestors_ts,
    samples,
    num_mismatches,
    time_increment,
    show_progress=False,
    precision=None,
    num_threads=None,
):
    assert sample_data.num_individuals == sample_data.num_samples

    ancestors_ts = get_ancestors_ts(sample_data, ancestors_ts, time_increment)
    results = run_tsinfer_matching(
        samples=samples,
        sample_data=sample_data,
        ancestors_ts=ancestors_ts,
        num_mismatches=num_mismatches,
        precision=precision,
        show_progress=show_progress,
        num_threads=num_threads,
    )
    return add_matching_results(sample_data, samples, ancestors_ts, results)


def add_matching_results(sample_data, samples, ts, results):
    """
    Adds the specified matching results to the specified tree sequence
    and returns the updated tree sequence.
    """
    assert sample_data.num_sites == ts.num_sites
    tables = ts.dump_tables()
    node_metadata = sample_data.individuals_metadata

    # TODO factor out the tsinfer ResultBuffer here into something more
    # useful locally, and do this coordinate translation there. Then
    # our dependence on tsinfer is quite well isolated.
    coord_map = np.append(tables.sites.position, [tables.sequence_length])
    coord_map[0] = 0
    for sd_id in samples:
        node_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE, time=0, metadata=node_metadata[sd_id]
        )
        # TODO path compression - what paths are identical? But be careful
        # about the mutations.
        for left, right, parent in zip(*results.get_path(node_id)):
            tables.edges.add_row(
                coord_map[left], coord_map[right], parent=parent, child=node_id
            )

        for site, derived_state in zip(*results.get_mutations(node_id)):
            tables.mutations.add_row(
                site=site,
                node=node_id,
                time=0,
                derived_state=core.ALLELES[derived_state],
            )
    # Update the sites with metadata for these newly added samples.
    tables.sites.clear()
    for ts_site, sd_site in zip(ts.sites(), sample_data.sites()):
        md = ts_site.metadata
        md["masked_samples"] += sd_site.metadata["masked_samples"]
        tables.sites.append(ts_site.replace(metadata=md))

    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()

    ts = coalesce_mutations(ts)
    ts = push_up_reversions(ts)
    return ts


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
    descriptors = set()
    for mut_id in np.where(ts.mutations_node == u)[0]:
        mut = ts.mutation(mut_id)
        inherited_state = ts.site(mut.site).ancestral_state
        if mut.parent != -1:
            parent_mut = ts.mutation(mut.parent)
            if parent_mut.node == u:
                raise ValueError("Multiple mutations on same branch not supported")
            inherited_state = parent_mut.derived_state
        descriptors.add(
            MutationDescriptor(mut.site, mut.derived_state, inherited_state, mut.parent)
        )
    return descriptors


def update_tables(tables, edges_to_delete, mutations_to_delete):

    # Updating the mutations is a real faff, and the only way I
    # could get it to work is by setting the time values. This should
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


def coalesce_mutations(ts):
    """
    Examine all time-0 samples and their (full-sequence) sibs and create
    new nodes to represent overlapping sets of mutations. The algorithm
    is greedy and makes no guarantees about uniqueness or optimality.
    Also note that we don't recurse and only reason about mutation sharing
    at a single level in the tree.

    Note: this function will most likely move to sc2ts once it has moved
    over to tskit's hapotype matching engine.
    """
    # We depend on mutations having a time below.
    assert np.all(np.logical_not(np.isnan(ts.mutations_time)))

    tree = ts.first()

    # Get the samples that span the whole sequence
    samples = []
    for u in ts.samples(time=0):
        e = tree.edge(u)
        assert e != -1
        edge = ts.edge(e)
        if edge.left == 0 and edge.right == ts.sequence_length:
            samples.append(u)
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
                overlap = node_mutations[sample] & node_mutations[v]
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
                    if sample_overlap.issubset(node_mutations[v]):
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
                condition = np.logical_and(
                    ts.mutations_node == sib,
                    ts.mutations_site == mut_desc.site,
                    ts.mutations_parent == mut_desc.parent,
                )
                mutations_to_delete.extend(np.where(condition)[0])
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

        tables.nodes.add_row(
            flags=constants.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR,
            time=group_parent_time,
        )
        # TODO would be good to store some provenance here about what
        # motivated the creation of this node.
        # metadata={"mutations": overlap})
        for mut_desc in overlap:
            tables.mutations.add_row(
                site=mut_desc.site,
                derived_state=mut_desc.derived_state,
                node=group_parent,
                time=group_parent_time,
            )

    num_del_mutations = len(mutations_to_delete)
    num_new_nodes = len(tables.nodes) - ts.num_nodes
    logger.info(
        f"Coalescing mutation: delete {num_del_mutations} mutations; "
        f"add {num_new_nodes} new nodes"
    )
    return update_tables(tables, edges_to_delete, mutations_to_delete)


def push_up_reversions(ts):
    # We depend on mutations having a time below.
    assert np.all(np.logical_not(np.isnan(ts.mutations_time)))

    tree = ts.first()
    mutations_per_node = np.bincount(ts.mutations_node, minlength=ts.num_nodes)

    # First get all the time-0 samples that have mutations, or the unique parents
    # of those that do not.
    samples = set()
    for u in ts.samples(time=0):
        if mutations_per_node[u] == 0:
            u = tree.parent(u)
            if ts.nodes_flags[u] == constants.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR:
                # Not strictly a sample, but represents some time-0 samples
                samples.add(u)
        else:
            samples.add(u)

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
        w = tables.nodes.add_row(flags=1 << 22, time=w_time)
        # Add new edges to join the sample and parent to w, and then
        # w to the grandparent.
        tables.edges.add_row(0, ts.sequence_length, parent=w, child=parent)
        tables.edges.add_row(0, ts.sequence_length, parent=w, child=sample)
        tables.edges.add_row(0, ts.sequence_length, parent=grandparent, child=w)

        for site in sites:
            # Delete the reversion mutations above the sample
            muts = np.where(
                np.logical_and(ts.mutations_node == sample, ts.mutations_site == site)
            )[0]
            assert len(muts) == 1
            mutations_to_delete.extend(muts)
        # Move any non-reversions mutations above the parent to the new node.
        for mut in np.where(ts.mutations_node == parent)[0]:
            row = tables.mutations[mut]
            if row.site not in sites:
                tables.mutations[mut] = row.replace(node=w, time=w_time)

    num_del_mutations = len(mutations_to_delete)
    num_new_nodes = len(tables.nodes) - ts.num_nodes
    logger.info(
        f"Push reversions: delete {num_del_mutations} mutations; "
        f"add {num_new_nodes} new nodes"
    )
    return update_tables(tables, edges_to_delete, mutations_to_delete)


def _parse_date(date):
    return datetime.datetime.fromisoformat(date)


def _validate_dates(ts):
    """
    Check that the time in the ts is days-ago in sync with the date
    metadata field.
    """
    samples = ts.samples()
    today = _parse_date(ts.node(samples[-1]).metadata["date"])
    for u in samples:
        node = ts.node(u)
        date = _parse_date(node.metadata["date"])
        diff = today - date
        assert diff.seconds == 0
        assert diff.microseconds == 0


def validate(sd, ts, max_submission_delay=None, show_progress=False):
    """
    Check that the ts contains all the data in the sample data.
    """
    assert ts.time_units == "days_ago"
    assert ts.num_sites == sd.num_sites
    if max_submission_delay is None:
        max_submission_delay = 10**9 - 1
    max_submission_delay = datetime.timedelta(days=max_submission_delay)
    name_map = {ts.node(u).metadata["strain"]: u for u in ts.samples()}
    ts_samples = []
    sd_samples = []
    for j, ind in enumerate(sd.individuals()):
        strain = ind.metadata["strain"]
        submission_delay = _parse_date(ind.metadata["date_submitted"]) - _parse_date(
            ind.metadata["date"]
        )
        if submission_delay <= max_submission_delay:
            if strain not in name_map:
                raise ValueError(f"Strain {strain} not in ts nodes")
            sd_samples.append(j)
            ts_samples.append(name_map[strain])
        else:
            if strain in name_map:
                raise ValueError(f"Strain {strain} should have been filtered")
    sd_samples = np.array(sd_samples)
    ts_samples = np.array(ts_samples)

    _validate_dates(ts)

    reference = core.get_reference_sequence()

    ts_vars = ts.variants(samples=ts_samples)
    vars_iter = zip(ts_vars, sd.variants())
    with tqdm.tqdm(vars_iter, total=ts.num_sites, disable=not show_progress) as bar:
        for ts_var, sd_var in bar:
            pos = int(ts_var.site.position)
            assert ts_var.site.ancestral_state == reference[pos]
            assert sd_var.site.position == pos
            ts_a = np.array(ts_var.alleles)
            sd_a = np.array(sd_var.alleles)
            sd_genotypes = sd_var.genotypes[sd_samples]
            non_missing = sd_genotypes != -1
            # Convert to actual allelic observations here because
            # allele encoding isn't stable
            ts_chars = ts_a[ts_var.genotypes[non_missing]]
            sd_chars = sd_a[sd_genotypes[non_missing]]
            if not np.all(ts_chars == sd_chars):
                raise ValueError("Data mismatch")


class Matcher(tsinfer.SampleMatcher):
    """
    NOTE: this is using undocumented internal APIs as a way of accessing
    tsinfer's Li and Stephens matching engine. There are some awkward
    workaround involved in dealing with tsinfer's internal representation
    of the data, which are tightly coupled to implementation details within
    tsinfer.

    This implementation will be swapped out for tskit's LS engine in the
    near future, using fully documented and supported APIs.
    """

    def _match_samples(self, sample_indexes):
        # Some hacks here to work around the fact that tsinfer does a bunch
        # of stuff we don't want here. All we want are the matched paths and
        # mutations.
        num_samples = len(sample_indexes)
        self.match_progress = self.progress_monitor.get("ms_match", num_samples)
        if self.num_threads <= 0:
            self._SampleMatcher__match_samples_single_threaded(sample_indexes)
        else:
            self._SampleMatcher__match_samples_multi_threaded(sample_indexes)
        self.match_progress.close()

    def run_match(self, samples):
        builder = self.tree_sequence_builder
        for sd_id in samples:
            self.sample_id_map[sd_id] = builder.add_node(0)
        self._match_samples(samples)
        return self.results
