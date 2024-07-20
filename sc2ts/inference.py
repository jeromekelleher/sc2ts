from __future__ import annotations
import logging
import datetime
import dataclasses
import collections
import io
import pickle
import os

import tqdm
import tskit
import tsinfer
import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy
import zarr
import numba

from . import core
from . import alignments

logger = logging.getLogger(__name__)


def mirror(x, L):
    return L - x


def mirror_ts_coordinates(ts):
    """
    Returns a copy of the specified tree sequence in which all
    coordinates x are transformed into L - x.

    Makes a bunch of simplifying assumptions.
    """
    assert ts.num_migrations == 0
    assert ts.discrete_genome
    L = ts.sequence_length
    tables = ts.dump_tables()
    left = tables.edges.left
    right = tables.edges.right
    tables.edges.left = mirror(right, L)
    tables.edges.right = mirror(left, L)
    tables.sites.position = mirror(tables.sites.position, L - 1)
    tables.sort()
    return tables.tree_sequence()


def initial_ts():
    reference = core.get_reference_sequence()
    L = core.REFERENCE_SEQUENCE_LENGTH
    assert L == len(reference)
    problematic_sites = set(core.get_problematic_sites())

    tables = tskit.TableCollection(L)
    tables.time_units = core.TIME_UNITS
    base_schema = tskit.MetadataSchema.permissive_json().schema

    tables.metadata_schema = tskit.MetadataSchema(base_schema)

    # TODO gene annotations to top level
    # TODO add known fields to the schemas and document them.
    tables.nodes.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.sites.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.mutations.metadata_schema = tskit.MetadataSchema(base_schema)

    # 1-based coordinates
    for pos in range(1, L):
        if pos not in problematic_sites:
            tables.sites.add_row(pos, reference[pos], metadata={"masked_samples": 0})
    # TODO should probably make the ultimate ancestor time something less
    # plausible or at least configurable. However, this will be removed
    # in later versions when we remove the dependence on tskit.
    tables.nodes.add_row(time=1, metadata={"strain": "Vestigial_ignore"})
    tables.nodes.add_row(
        time=0, metadata={"strain": core.REFERENCE_STRAIN, "date": core.REFERENCE_DATE}
    )
    tables.edges.add_row(0, L, 0, 1)
    return tables.tree_sequence()


def parse_date(date):
    return datetime.datetime.fromisoformat(date)


def filter_samples(samples, alignment_store, max_submission_delay=None):
    if max_submission_delay is None:
        max_submission_delay = 10**8  # Arbitrary large number of days.
    not_in_store = 0
    num_filtered = 0
    ret = []
    for sample in samples:
        if sample.strain not in alignment_store:
            logger.warn(f"{sample.strain} not in alignment store")
            not_in_store += 1
            continue
        if sample.submission_delay < max_submission_delay:
            ret.append(sample)
        else:
            num_filtered += 1
    if not_in_store == len(samples):
        raise ValueError("All samples for day missing")
    logger.info(
        f"Filtered {num_filtered} samples with "
        f"max_submission_delay >= {max_submission_delay}"
    )
    return ret


def last_date(ts):
    if ts.num_samples == 0:
        # Special case for the initial ts which contains the
        # reference but not as a sample
        u = ts.num_nodes - 1
        node = ts.node(u)
        assert node.time == 0
        return parse_date(node.metadata["date"])
    else:
        samples = ts.samples()
        samples_t0 = samples[ts.nodes_time[samples] == 0]
        return max([parse_date(ts.node(u).metadata["date"]) for u in samples_t0])


def increment_time(date, ts):
    diff = parse_date(date) - last_date(ts)
    increment = diff.days
    if increment <= 0:
        raise ValueError(f"Bad date diff: {diff}")

    tables = ts.dump_tables()
    tables.nodes.time += increment
    tables.mutations.time += increment
    return tables.tree_sequence()


def _validate_samples(ts, samples, alignment_store, show_progress):
    strains = [ts.node(u).metadata["strain"] for u in samples]
    G = np.zeros((ts.num_sites, len(samples)), dtype=np.int8)
    keep_sites = ts.sites_position.astype(int)
    strains_iter = enumerate(strains)
    with tqdm.tqdm(
        strains_iter,
        desc="Read",
        total=len(strains),
        position=1,
        leave=False,
        disable=not show_progress,
    ) as bar:
        for j, strain in bar:
            ma = alignments.encode_and_mask(alignment_store[strain])
            G[:, j] = ma.alignment[keep_sites]

    vars_iter = ts.variants(samples=samples, alleles=tuple(core.ALLELES))
    with tqdm.tqdm(
        vars_iter,
        desc="Check",
        total=ts.num_sites,
        position=1,
        leave=False,
        disable=not show_progress,
    ) as bar:
        for var in bar:
            original = G[var.site.id]
            non_missing = original != -1
            if not np.all(var.genotypes[non_missing] == original[non_missing]):
                raise ValueError("Data mismatch")


def validate(ts, alignment_store, show_progress=False):
    """
    Check that all the samples in the specified tree sequence are correctly
    representing the original alignments.
    """
    samples = ts.samples()
    chunk_size = 10**3
    offset = 0
    num_chunks = ts.num_samples // chunk_size
    for chunk_index in tqdm.tqdm(
        range(num_chunks), position=0, disable=not show_progress
    ):
        chunk = samples[offset : offset + chunk_size]
        offset += chunk_size
        _validate_samples(ts, chunk, alignment_store, show_progress)

    if ts.num_samples % chunk_size != 0:
        chunk = samples[offset:]
        _validate_samples(ts, chunk, alignment_store, show_progress)


@dataclasses.dataclass
class Sample:
    metadata: Dict = dataclasses.field(default_factory=dict)
    path: List = dataclasses.field(default_factory=list)
    mutations: List = dataclasses.field(default_factory=list)
    alignment_qc: Dict = dataclasses.field(default_factory=dict)
    masked_sites: List = dataclasses.field(default_factory=list)

    # def __repr__(self):
    #     return self.strain

    # def __str__(self):
    #     return f"{self.strain}: {self.path} + {self.mutations}"

    @property
    def strain(self):
        return self.metadata["strain"]

    @property
    def breakpoints(self):
        breakpoints = [seg.left for seg in self.path]
        return breakpoints + [self.path[-1].right]

    @property
    def parents(self):
        return [seg.parent for seg in self.path]

    @property
    def date(self):
        return parse_date(self.metadata["date"])

    @property
    def submission_date(self):
        return parse_date(self.metadata["date_submitted"])

    @property
    def submission_delay(self):
        return (self.submission_date - self.date).days

    def get_hmm_cost(self, num_mismatches):
        # Note that Recombinant objects have total_cost.
        # This bit of code is sort of repeated.
        return num_mismatches * (len(self.path) - 1) + len(self.mutations)

    def asdict(self):
        return {
            "strain": self.strain,
            "path": self.path,
            "mutations": self.mutations,
            "masked_sites": self.masked_sites.tolist(),
            "alignment_qc": self.alignment_qc,
        }


# # TODO Factor this into the Samples class so that we can move
# # lists of samples to and from files.
# def write_match_json(samples):
#     data = []
#     for sample in samples:
#         data.append(sample.asdict())
#     s = json.dumps(data, indent=2)


def daily_extend(
    *,
    alignment_store,
    metadata_db,
    base_ts,
    num_mismatches=None,
    max_hmm_cost=None,
    min_group_size=None,
    num_past_days=None,
    show_progress=False,
    max_submission_delay=None,
    max_daily_samples=None,
    num_threads=None,
    precision=None,
    rng=None,
    excluded_sample_dir=None,
):
    start_day = last_date(base_ts)

    reconsidered_samples = collections.deque()
    earliest_date = start_day - datetime.timedelta(days=1)
    if base_ts is not None:
        next_day = start_day + datetime.timedelta(days=1)
        reconsidered_samples.extend(
            fetch_samples_from_pickle_file(
                date=next_day,
                num_past_days=num_past_days,
                in_dir=excluded_sample_dir,
            )
        )
        earliest_date = next_day - datetime.timedelta(days=num_past_days)

    last_ts = base_ts
    for date in metadata_db.get_days(start_day):
        ts, excluded_samples = extend(
            alignment_store=alignment_store,
            metadata_db=metadata_db,
            date=date,
            base_ts=last_ts,
            num_mismatches=num_mismatches,
            max_hmm_cost=max_hmm_cost,
            min_group_size=min_group_size,
            show_progress=show_progress,
            max_submission_delay=max_submission_delay,
            max_daily_samples=max_daily_samples,
            num_threads=num_threads,
            precision=precision,
            rng=rng,
            reconsidered_samples=reconsidered_samples,
        )
        yield ts, excluded_samples, date

        # Update list of reconsidered samples.
        if len(reconsidered_samples) > 0:
            while reconsidered_samples[0].date == earliest_date:
                reconsidered_samples.popleft()
            reconsidered_samples.extend(excluded_samples)

        earliest_date += datetime.timedelta(days=1)

        last_ts = ts


def match(
    samples,
    *,
    alignment_store,
    base_ts,
    num_mismatches=None,
    show_progress=False,
    num_threads=None,
    precision=None,
    mirror_coordinates=False,
):
    logger.info(f"Start match for {len(samples)}")

    # Note: there's not a lot of point in making the G matrix here,
    # we should just pass on the encoded alignments to the matching
    # algorithm directly through the Sample class, and let it
    # do the low-level haplotype storage.
    G = np.zeros((base_ts.num_sites, len(samples)), dtype=np.int8)
    keep_sites = base_ts.sites_position.astype(int)
    problematic_sites = core.get_problematic_sites()

    samples_iter = enumerate(samples)
    with tqdm.tqdm(
        samples_iter,
        desc=f"Fetch",
        total=len(samples),
        disable=not show_progress,
    ) as bar:
        for j, sample in bar:
            logger.debug(f"Getting alignment for {sample.strain}")
            alignment = alignment_store[sample.strain]
            sample.alignment = alignment
            logger.debug(f"Encoding alignment")
            ma = alignments.encode_and_mask(alignment)
            # Always mask the problematic_sites as well. We need to do this
            # for follow-up matching to inspect recombinants, as tsinfer
            # needs us to keep all sites in the table when doing mirrored
            # coordinates.
            ma.alignment[problematic_sites] = -1
            G[:, j] = ma.alignment[keep_sites]
            sample.alignment_qc = ma.qc_summary()
            sample.masked_sites = ma.masked_sites

    masked_per_sample = np.mean([len(sample.masked_sites)])
    logger.info(f"Masked average of {masked_per_sample:.2f} nucleotides per sample")

    match_tsinfer(
        samples=samples,
        ts=base_ts,
        genotypes=G,
        num_mismatches=num_mismatches,
        precision=precision,
        num_threads=num_threads,
        show_progress=show_progress,
        mirror_coordinates=mirror_coordinates,
    )
    return samples


def extend(
    *,
    alignment_store,
    metadata_db,
    date,
    base_ts,
    num_mismatches=None,
    max_hmm_cost=None,
    min_group_size=None,
    show_progress=False,
    max_submission_delay=None,
    max_daily_samples=None,
    num_threads=None,
    precision=None,
    rng=None,
    reconsidered_samples=None,
):
    date_samples = [Sample(md) for md in metadata_db.get(date)]
    samples = filter_samples(date_samples, alignment_store, max_submission_delay)

    if max_daily_samples is not None and len(samples) > max_daily_samples:
        samples = rng.sample(samples, max_daily_samples)
        logger.info(f"Sampled down to {len(samples)} samples")

    if len(samples) == 0:
        logger.warning(f"No samples for {date}")
        return base_ts

    logger.info(f"Got {len(samples)} samples")

    # Note num_mismatches is assigned a default value in match_tsinfer.
    samples = match(
        samples,
        alignment_store=alignment_store,
        base_ts=base_ts,
        num_mismatches=num_mismatches,
        show_progress=show_progress,
        num_threads=num_threads,
        precision=precision,
    )
    ts = increment_time(date, base_ts)

    ts, excluded_samples = add_matching_results(
        samples=samples,
        ts=ts,
        date=date,
        num_mismatches=num_mismatches,
        max_hmm_cost=max_hmm_cost,
        min_group_size=None,
        show_progress=show_progress,
    )

    ts, _ = add_matching_results(
        samples=reconsidered_samples,
        ts=ts,
        date=date,
        num_mismatches=num_mismatches,
        max_hmm_cost=None,
        min_group_size=min_group_size,
        show_progress=show_progress,
    )

    return ts, excluded_samples


def match_path_ts(samples, ts, path, reversions):
    """
    Given the specified list of samples with equal copying paths,
    return the tree sequence rooted at zero representing the data.
    """
    tables = tskit.TableCollection(ts.sequence_length)
    tables.nodes.metadata_schema = ts.table_metadata_schemas.node
    # Zero is the attach node
    tables.nodes.add_row(time=1)
    path = samples[0].path
    site_id_map = {}
    first_sample = len(tables.nodes)
    for sample in samples:
        assert sample.path == path
        metadata = {
            **sample.metadata,
            "sc2ts": {
                "qc": sample.alignment_qc,
                "path": [x.asdict() for x in sample.path],
                "mutations": [x.asdict() for x in sample.mutations],
            },
        }
        node_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE, time=0, metadata=metadata
        )
        tables.edges.add_row(0, ts.sequence_length, parent=0, child=node_id)
        for mut in sample.mutations:
            if mut.site_id not in site_id_map:
                new_id = tables.sites.add_row(mut.site_position, mut.inherited_state)
                site_id_map[mut.site_id] = new_id

    # Now add the mutations
    for node_id, sample in enumerate(samples, first_sample):
        # metadata = {**sample.metadata, "sc2ts_qc": sample.alignment_qc}
        for mut in sample.mutations:
            tables.mutations.add_row(
                site=site_id_map[mut.site_id],
                node=node_id,
                time=0,
                derived_state=mut.derived_state,
            )
    tables.sort()
    return tables.tree_sequence()
    # print(tables)


def add_matching_results(
    samples,
    ts,
    date,
    num_mismatches,
    max_hmm_cost,
    min_group_size,
    show_progress=False,
):
    if num_mismatches is None:
        # Note that this is the default assigned in match_tsinfer.
        num_mismatches = 1e3

    if max_hmm_cost is None:
        # By default, arbitraily high.
        max_hmm_cost = 1e6

    if min_group_size is None:
        min_group_size = 1

    # Group matches by path and set of immediate reversions.
    grouped_matches = collections.defaultdict(list)
    excluded_samples = []
    site_masked_samples = np.zeros(int(ts.sequence_length), dtype=int)
    for sample in samples:
        if sample.get_hmm_cost(num_mismatches) > max_hmm_cost:
            excluded_samples.append(sample)
            continue
        site_masked_samples[sample.masked_sites] += 1
        path = tuple(sample.path)
        reversions = tuple(
            (mut.site_id, mut.derived_state)
            for mut in sample.mutations
            if mut.is_immediate_reversion
        )
        grouped_matches[(path, reversions)].append(sample)

    tables = ts.dump_tables()
    logger.info(f"Got {len(grouped_matches)} distinct paths")

    attach_nodes = []

    with tqdm.tqdm(
        grouped_matches.items(),
        desc="Build",
        total=len(grouped_matches),
        disable=not show_progress,
    ) as bar:
        for (path, reversions), match_samples in bar:
            if len(match_samples) < min_group_size:
                continue

            # print(path, reversions, len(match_samples))
            # Delete the reversions from these samples so that we don't
            # build them into the trees
            if len(reversions) > 0:
                for sample in match_samples:
                    new_muts = [
                        mut
                        for mut in sample.mutations
                        if (mut.site_id, mut.derived_state) not in reversions
                    ]
                    assert len(new_muts) == len(sample.mutations) - len(reversions)
                    sample.mutations = new_muts

            flat_ts = match_path_ts(match_samples, ts, path, reversions)
            if flat_ts.num_mutations == 0 or flat_ts.num_samples == 1:
                poly_ts = flat_ts
            else:
                binary_ts = infer_binary(flat_ts)
                # print(binary_ts.draw_text())
                # print(binary_ts.tables.mutations)
                poly_ts = trim_branches(binary_ts)
                # print(poly_ts.draw_text())
                # print(poly_ts.tables.mutations)
                # print("----")
            assert poly_ts.num_samples == flat_ts.num_samples
            tree = poly_ts.first()
            attach_depth = max(tree.depth(u) for u in poly_ts.samples())
            nodes = attach_tree(ts, tables, path, reversions, poly_ts, date)
            # print(nodes)
            logger.debug(
                f"Path {path}: samples={poly_ts.num_samples} "
                f"depth={attach_depth} mutations={poly_ts.num_mutations} "
                f"reversions={reversions} attach_nodes={nodes}"
            )
            attach_nodes.extend(nodes)

    # Update the sites with metadata for these newly added samples.
    tables.sites.clear()
    for site in ts.sites():
        md = site.metadata
        md["masked_samples"] += int(site_masked_samples[int(site.position)])
        tables.sites.append(site.replace(metadata=md))

    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    # print("START")
    # print(ts.draw_text())

    ts = tables.tree_sequence()

    # ts = insert_recombinants(ts)
    # print("BEFORE", attach_nodes)
    # print(ts.draw_text())
    ts = push_up_reversions(ts, attach_nodes)
    # print("AFTER")
    # print(ts.draw_text())
    ts = coalesce_mutations(ts, attach_nodes)

    return ts, excluded_samples


def fetch_samples_from_pickle_file(date, num_past_days=None, in_dir=None):
    if in_dir is None:
        return []
    if num_past_days is None:
        num_past_days = 0
    file_suffix = ".excluded_samples.pickle"
    samples = []
    for i in range(num_past_days, 0, -1):
        past_date = date - datetime.timedelta(days=i)
        pickle_file = in_dir + "/"
        pickle_file += past_date.strftime('%Y-%m-%d') + file_suffix
        if os.path.exists(pickle_file):
            samples += parse_pickle_file(pickle_file)
    return samples


def parse_pickle_file(pickle_file):
    """Return a list of Sample objects."""
    with open(pickle_file, 'rb') as f:
        samples = pickle.load(f)
    return samples


def solve_num_mismatches(ts, k):
    """
    Return the low-level LS parameters corresponding to accepting
    k mismatches in favour of a single recombination.

    NOTE! This is NOT taking into account the spatial distance along
    the genome, and so is not a very good model in some ways.
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

    # Add a tiny bit of extra mass for recombination so that we deterministically
    # chose to recombine over k mutations
    r += r * 0.01
    ls_recomb = np.full(m - 1, r)
    ls_mismatch = np.full(m, mu)
    return ls_recomb, ls_mismatch


# Work around tsinfer's reshuffling of the allele indexes
def unshuffle_allele_index(index, ancestral_state):
    A = core.ALLELES
    as_index = A.index(ancestral_state)
    alleles = ancestral_state + A[:as_index] + A[as_index + 1 :]
    return alleles[index]


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


def process_recombinant(ts, tables, path, children):
    """
    Given a path of (left, right, parent) values and the set of children
    that share this path, insert a recombinant node representing that
    shared path, and push mutations shared by all of those children
    above the recombinant.
    """
    logger.info(f"Adding recombinant for {path} with {len(children)} children")
    min_parent_time = ts.nodes_time[0]
    for _, _, parent in path:
        min_parent_time = min(ts.nodes_time[parent], min_parent_time)

    child_mutations = {}
    # Store a list of tuples here rather than a mapping because JSON
    # only supports string keys.
    mutations_md = []
    for child in children:
        mutations = node_mutation_descriptors(ts, child)
        child_mutations[child] = mutations
        mutations_md.append(
            (
                int(child),
                sorted(
                    [
                        (desc.site, desc.inherited_state, desc.derived_state)
                        for desc in mutations
                    ]
                ),
            )
        )
    recomb_node_time = min_parent_time / 2
    recomb_node = tables.nodes.add_row(
        time=recomb_node_time,
        flags=core.NODE_IS_RECOMBINANT,
        metadata={"path": path, "mutations": mutations_md},
    )

    for left, right, parent in path:
        tables.edges.add_row(left, right, parent, recomb_node)
    for child in children:
        tables.edges.add_row(0, ts.sequence_length, recomb_node, child)

    # Push any mutations shared by *all* children over the recombinant.
    # child_mutations is a mapping from child node to the mapping of
    # mutation descriptors to their original mutation IDs
    child_mutation_sets = [set(mapping.keys()) for mapping in child_mutations.values()]
    shared_mutations = set.intersection(*child_mutation_sets)
    mutations_to_delete = []
    for child in children:
        for desc in shared_mutations:
            mutations_to_delete.append(child_mutations[child][desc])
    for desc in shared_mutations:
        tables.mutations.add_row(
            site=desc.site,
            node=recomb_node,
            time=recomb_node_time,
            derived_state=desc.derived_state,
            metadata={"type": "recomb_overlap"},
        )
    return mutations_to_delete


def insert_recombinants(ts):
    """
    Examine all time-0 samples and see if there are any recombinants.
    For each unique recombinant (copying path) insert a new node.
    """
    recombinants = collections.defaultdict(list)
    edges_to_delete = []
    for u in ts.samples(time=0):
        edges = np.where(ts.edges_child == u)[0]
        if len(edges) > 1:
            path = []
            for eid in edges:
                edge = ts.edge(eid)
                path.append((edge.left, edge.right, edge.parent))
                edges_to_delete.append(eid)
            path = tuple(sorted(path))
            recombinants[path].append(u)

    if len(recombinants) == 0:
        return ts

    tables = ts.dump_tables()
    mutations_to_delete = []
    for path, nodes in recombinants.items():
        mutations_to_delete.extend(process_recombinant(ts, tables, path, nodes))

    return update_tables(tables, edges_to_delete, mutations_to_delete)


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
            metadata={"overlap": md_overlap, "sibs": md_sibs},
        )
        for mut_desc in overlap:
            tables.mutations.add_row(
                site=mut_desc.site,
                derived_state=mut_desc.derived_state,
                node=group_parent,
                time=group_parent_time,
                metadata={"type": "overlap"},
            )

    num_del_mutations = len(mutations_to_delete)
    num_new_nodes = len(tables.nodes) - ts.num_nodes
    logger.info(
        f"Coalescing mutations: delete {num_del_mutations} mutations; "
        f"add {num_new_nodes} new nodes"
    )
    return update_tables(tables, edges_to_delete, mutations_to_delete)


def push_up_reversions(ts, samples):
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
                "sample": int(sample),
                "sites": [int(x) for x in sites],
            },
        )
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


# NOTE: could definitely do better here by using int encoding instead of
# strings, and then njit
@numba.jit(forceobj=True)
def get_indexes_of(array, values):
    n = array.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for j in range(n):
        out[j] = values.index(array[j])
    return out


def convert_tsinfer_sample_data(ts, genotypes):
    """
    Doing this directly using using tsinfer's APIs was very slow because
    of all the error checking etc going on. This circumvents the process.
    """
    alleles = tuple(core.ALLELES)
    sd = tsinfer.SampleData(sequence_length=ts.sequence_length, compressor=None)
    # for site, site_genotypes in zip(ts.sites(), genotypes):
    #     sd.add_site(
    #         site.position,
    #         site_genotypes,
    #         alleles=alleles,
    #         ancestral_allele=alleles.index(site.ancestral_state),
    #     )

    # Let the API add one site to get the basic stuff in there.
    sd.add_site(
        0,
        genotypes[0],
        alleles=alleles,
    )
    sd.finalise()

    ancestral_state = ts.tables.sites.ancestral_state.view("S1").astype(str)
    ancestral_allele = get_indexes_of(ancestral_state, alleles)

    def resize_copy(array, new_size):
        x = array[0]
        array.resize(new_size)
        array[:] = [x] * new_size

    data = zarr.open(store=sd.data.store)
    data["sites/position"] = ts.sites_position
    data["sites/time"] = np.zeros_like(ts.sites_position)
    data["sites/genotypes"] = genotypes
    data["sites/alleles"] = [alleles] * ts.num_sites
    data["sites/ancestral_allele"] = ancestral_allele
    resize_copy(data["sites/metadata"], ts.num_sites)
    return sd


def match_tsinfer(
    samples,
    ts,
    genotypes,
    *,
    num_mismatches=None,
    precision=None,
    num_threads=0,
    show_progress=False,
    mirror_coordinates=False,
):
    # TODO: Should this default be assigned elsewhere?
    if num_mismatches is None:
        # Default to no recombination
        num_mismatches = 1000

    input_ts = ts
    if mirror_coordinates:
        ts = mirror_ts_coordinates(ts)
        genotypes = genotypes[::-1]

    sd = convert_tsinfer_sample_data(ts, genotypes)

    L = int(ts.sequence_length)
    ls_recomb, ls_mismatch = solve_num_mismatches(ts, num_mismatches)
    pm = tsinfer.inference._get_progress_monitor(
        show_progress,
        generate_ancestors=False,
        match_ancestors=False,
        match_samples=False,
    )

    # This is just working around tsinfer's input checking logic. The actual value
    # we're incrementing by has no effect.
    tables = ts.dump_tables()
    tables.nodes.time += 1
    tables.mutations.time += 1
    ancestral_state = tables.sites.ancestral_state.view("S1").astype(str)
    ts = tables.tree_sequence()
    del tables

    manager = Matcher(
        sd,
        ts,
        allow_multiallele=True,
        recombination=ls_recomb,
        mismatch=ls_mismatch,
        progress_monitor=pm,
        num_threads=num_threads,
        precision=precision,
    )
    results = manager.run_match(np.arange(sd.num_samples))

    coord_map = np.append(ts.sites_position, [L]).astype(int)
    coord_map[0] = 0

    sample_paths = []
    sample_mutations = []
    # Update the Sample objects with their paths and sets of mutations.
    for node_id, sample in enumerate(samples, ts.num_nodes):
        path = []
        for left, right, parent in zip(*results.get_path(node_id)):
            if mirror_coordinates:
                left_pos = mirror(int(coord_map[right]), L)
                right_pos = mirror(int(coord_map[left]), L)
            else:
                left_pos = int(coord_map[left])
                right_pos = int(coord_map[right])
            path.append((left_pos, right_pos, int(parent)))
        path.sort()
        sample_paths.append(path)

        mutations = []
        for site_id, derived_state in zip(*results.get_mutations(node_id)):
            site_pos = ts.sites_position[site_id]
            if mirror_coordinates:
                site_pos = mirror(site_pos, L - 1)
            derived_state = unshuffle_allele_index(
                derived_state, ancestral_state[site_id]
            )
            mutations.append((site_pos, derived_state))
        mutations.sort()
        sample_mutations.append(mutations)

    update_path_info(samples, input_ts, sample_paths, sample_mutations)


@dataclasses.dataclass(frozen=True)
class PathSegment:
    left: int
    right: int
    parent: int

    def contains(self, position):
        return self.left <= position < self.right

    def asdict(self):
        return {
            "left": self.left,
            "right": self.right,
            "parent": self.parent,
        }


@dataclasses.dataclass(frozen=True)
class MatchMutation:
    site_id: int
    site_position: int
    derived_state: str
    inherited_state: str
    is_reversion: bool
    is_immediate_reversion: bool

    def __str__(self):
        return f"{int(self.site_position)}{self.inherited_state}>{self.derived_state}"

    def asdict(self):
        return {
            "site_id": int(self.site_id),
            "site_position": int(self.site_position),
            "derived_state": self.derived_state,
            "inherited_state": self.inherited_state,
            "is_reversion": self.is_reversion,
            "is_immediate_reversion": self.is_immediate_reversion,
        }


def update_path_info(samples, ts, sample_paths, sample_mutations):
    tables = ts.tables
    assert np.all(tables.sites.ancestral_state_offset == np.arange(ts.num_sites + 1))
    ancestral_state = tables.sites.ancestral_state.view("S1").astype(str)
    del tables

    tree = ts.first()
    cache = {}

    def get_closest_mutation(node, site_id):
        if (node, site_id) not in cache:
            site = ts.site(site_id)
            mutations = {mutation.node: mutation for mutation in site.mutations}
            tree.seek(site.position)
            u = node
            while u not in mutations and u != -1:
                u = tree.parent(u)
            closest = None
            if u != -1:
                closest = mutations[u]
            cache[(node, site_id)] = closest

        return cache[(node, site_id)]

    for sample, path, mutations in zip(samples, sample_paths, sample_mutations):
        sample.path = [PathSegment(*seg) for seg in path]
        for site_pos, derived_state in mutations:
            site_id = np.searchsorted(ts.sites_position, site_pos)
            assert ts.sites_position[site_id] == site_pos
            seg = [seg for seg in sample.path if seg.contains(site_pos)][0]
            closest_mutation = get_closest_mutation(seg.parent, site_id)
            inherited_state = ancestral_state[site_id]
            is_reversion = False
            is_immediate_reversion = False
            if closest_mutation is not None:
                inherited_state = closest_mutation.derived_state
                parent_inherited_state = ancestral_state[site_id]
                if closest_mutation.parent != -1:
                    grandparent_mutation = ts.mutation(closest_mutation.parent)
                    parent_inherited_state = grandparent_mutation.derived_state
                is_reversion = parent_inherited_state == derived_state
                if is_reversion:
                    is_immediate_reversion = closest_mutation.node == seg.parent

            # TODO it would be nice to assert this here, but it interferes
            # with the testing code. Another sign that the current interface
            # really smells.
            # if derived_state != sample.alignment[site_pos]:
            #     assert site_pos in sample.masked_sites
            assert inherited_state != derived_state

            sample.mutations.append(
                MatchMutation(
                    site_id=site_id,
                    site_position=site_pos,
                    derived_state=derived_state,
                    inherited_state=inherited_state,
                    is_reversion=is_reversion,
                    is_immediate_reversion=is_immediate_reversion,
                )
            )


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


def _linkage_matrix_to_tskit(Z):
    n = Z.shape[0] + 1
    N = 2 * n
    parent = np.full(N, -1, dtype=np.int32)
    time = np.full(N, 0, dtype=np.float64)
    for j, row in enumerate(Z):
        u = n + j
        time[u] = j + 1
        lc = int(row[0])
        rc = int(row[1])
        parent[lc] = u
        parent[rc] = u
    return parent[:-1], time[:-1]


def infer_binary(ts):
    """
    Infer a strictly binary tree from the variation data in the
    specified tree sequence.
    """
    assert ts.num_trees == 1
    tables = ts.dump_tables()
    # Don't clear popualtions for simplicity
    tables.nodes.clear()
    tables.edges.clear()
    tables.sites.clear()
    tables.mutations.clear()

    G = ts.genotype_matrix()
    # Hamming distance should be suitable here because it's giving the overall
    # number of differences between the observations. Euclidean is definitely
    # not because of the allele encoding (difference between 0 and 4 is not
    # greater than 0 and 1).
    Y = scipy.spatial.distance.pdist(G.T, "hamming")
    # This is the UPGMA algorithm
    Z = scipy.cluster.hierarchy.average(Y)
    parent, time = _linkage_matrix_to_tskit(Z)
    # Rescale time to be from 0 to 1
    time /= np.max(time)

    # Add the samples in first
    u = 0
    for v in ts.samples():
        node = ts.node(v)
        assert node.time == 0
        assert time[u] == 0
        assert u == len(tables.nodes)
        tables.nodes.append(node)
        tables.edges.add_row(0, ts.sequence_length, parent=parent[u], child=u)
        u += 1
    while u < len(parent):
        assert u == len(tables.nodes)
        tables.nodes.add_row(time=time[u], flags=0)
        if parent[u] != tskit.NULL:
            tables.edges.add_row(0, ts.sequence_length, parent=parent[u], child=u)
        u += 1

    tables.sort()
    ts_binary = tables.tree_sequence()

    tree = ts_binary.first()
    for var in ts.variants():
        anc, muts = tree.map_mutations(
            var.genotypes, var.alleles, ancestral_state=var.site.ancestral_state
        )
        assert anc == var.site.ancestral_state
        site = tables.sites.add_row(var.site.position, anc)
        for mut in muts:
            tables.mutations.add_row(
                site=site, node=mut.node, derived_state=mut.derived_state
            )
    return tables.tree_sequence()


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


def attach_tree(parent_ts, parent_tables, attach_path, reversions, child_ts, date):

    root_time = min(parent_ts.nodes_time[seg.parent] for seg in attach_path)
    if root_time == 0:
        raise ValueError("Cannot attach at time-zero node")
    if child_ts.num_trees != 1:
        raise ValueError("Can only attach single trees")
    if child_ts.sequence_length != parent_ts.sequence_length:
        raise ValueError("Incompatible sequence length")

    tree = child_ts.first()
    condition = (
        np.any(child_ts.mutations_node == tree.root)
        or len(reversions) > 0
        or len(attach_path) > 1
    )
    if condition:
        child_ts = add_root_edge(child_ts)
        tree = child_ts.first()

    node_id_map = {}
    if child_ts.nodes_time[tree.root] != 1.0:
        raise ValueError("Time must be scaled from 0 to 1.")
    node_time = {}
    for u in tree.postorder()[:-1]:
        node = child_ts.node(u)
        # Tree branch length is scaled from 0 to 1.
        time = node.time * root_time
        node_time[u] = time
        metadata = node.metadata
        if tree.is_internal(u):
            metadata = {"date_added": date}
        new_id = parent_tables.nodes.append(node.replace(time=time, metadata=metadata))
        node_id_map[node.id] = new_id
        for v in tree.children(u):
            parent_tables.edges.add_row(
                0,
                parent_ts.sequence_length,
                child=node_id_map[v],
                parent=node_id_map[u],
            )
    # Attach the children of the root to the input path.
    for child in tree.children(tree.root):
        for seg in attach_path:
            parent_tables.edges.add_row(
                seg.left, seg.right, parent=seg.parent, child=node_id_map[child]
            )

    # Add the mutations.
    for site in child_ts.sites():
        parent_site_id = parent_ts.site(position=site.position).id
        for mutation in site.mutations:
            assert mutation.node != tree.root
            parent_tables.mutations.add_row(
                site=parent_site_id,
                node=node_id_map[mutation.node],
                derived_state=mutation.derived_state,
                time=node_time[mutation.node],
            )
    if len(reversions) > 0:
        # FIXME we should either flag these nodes with a specific value
        # or update the reversion push code below to remove them. I think
        # they'll end up as useless internal nodes? Need to check.
        # Add the reversions back on over the unary root.
        node = tree.children(tree.root)[0]
        assert tree.num_children(tree.root) == 1
        # print("attaching reversions at ", node, node_id_map[node])
        # print(child_ts.draw_text())
        for site_id, derived_state in reversions:
            parent_tables.mutations.add_row(
                site=site_id,
                node=node_id_map[node],
                derived_state=derived_state,
                time=node_time[node],
            )
    if len(attach_path) > 1:
        # Update the recombinant flags also.
        u = node_id_map[tree.children(tree.root)[0]]
        assert tree.num_children(tree.root) == 1
        node = parent_tables.nodes[u]
        parent_tables.nodes[u] = node.replace(flags=core.NODE_IS_RECOMBINANT)
    return [node_id_map[u] for u in tree.children(tree.root)]


def add_root_edge(ts):
    """
    Add another node and edge above the root and rescale time back to
    0-1.
    """
    assert ts.num_trees == 1
    tables = ts.dump_tables()
    root = ts.first().root
    # FIXME this is bogus. We should be doing all the time scaling by numbers
    # of mutations.
    new_root = tables.nodes.add_row(time=1.25)
    tables.edges.add_row(0, ts.sequence_length, parent=new_root, child=root)
    tables.nodes.time /= np.max(tables.nodes.time)
    return tables.tree_sequence()
