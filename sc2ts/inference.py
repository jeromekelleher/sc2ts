import logging
import datetime
import dataclasses
import collections

import tqdm
import tskit
import tsinfer
import numpy as np

from . import core
from . import alignments

logger = logging.getLogger(__name__)


def initial_ts():
    reference = core.get_reference_sequence()
    L = len(reference)
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
    else:
        u = ts.samples()[-1]
    node = ts.node(u)
    assert node.time == 0
    return parse_date(node.metadata["date"])


def increment_time(date, ts):
    diff = parse_date(date) - last_date(ts)
    increment = diff.days
    if increment <= 0:
        raise ValueError(f"Bad date diff: {diff}")

    tables = ts.dump_tables()
    tables.nodes.time += increment
    tables.mutations.time += increment
    return tables.tree_sequence()


def validate(ts, alignment_store, show_progress=False):
    """
    Check that all the samples in the specified tree sequence are correctly
    representing the original alignments.
    """
    samples = ts.samples()
    strains = [ts.node(u).metadata["strain"] for u in samples]
    G = np.zeros((ts.num_sites, len(samples)), dtype=np.int8)
    keep_sites = ts.sites_position.astype(int)
    strains_iter = enumerate(strains)
    with tqdm.tqdm(
        strains_iter, desc="Read", total=len(strains), disable=not show_progress
    ) as bar:
        for j, strain in bar:
            ma = alignments.encode_and_mask(alignment_store[strain])
            G[:, j] = ma.alignment[keep_sites]

    vars_iter = ts.variants(samples=samples, alleles=tuple(core.ALLELES))
    with tqdm.tqdm(
        vars_iter, desc="Check", total=ts.num_sites, disable=not show_progress
    ) as bar:
        for var in bar:
            original = G[var.site.id]
            non_missing = original != -1
            if not np.all(var.genotypes[non_missing] == original[non_missing]):
                raise ValueError("Data mismatch")


class Sample:
    def __init__(self, metadata):
        self.metadata = metadata
        self.alignment_qc = {}
        self.path = None
        self.mutations = None

    def __repr__(self):
        return self.strain

    def __str__(self):
        return f"{self.strain}: {self.path} + {self.mutations}"

    @property
    def strain(self):
        return self.metadata["strain"]

    @property
    def date(self):
        return parse_date(self.metadata["date"])

    @property
    def submission_date(self):
        return parse_date(self.metadata["date_submitted"])

    @property
    def submission_delay(self):
        return (self.submission_date - self.date).days

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
    show_progress=False,
    max_submission_delay=None,
    max_daily_samples=None,
    num_threads=None,
    precision=None,
    rng=None,
):
    start_day = last_date(base_ts)
    last_ts = base_ts
    for date in metadata_db.get_days(start_day):
        ts = extend(
            alignment_store=alignment_store,
            metadata_db=metadata_db,
            date=date,
            base_ts=last_ts,
            num_mismatches=num_mismatches,
            show_progress=show_progress,
            max_submission_delay=max_submission_delay,
            max_daily_samples=max_daily_samples,
            num_threads=num_threads,
            precision=precision,
            rng=rng,
        )
        yield ts, date
        last_ts = ts


def match(
    *,
    alignment_store,
    metadata_db,
    date,
    base_ts,
    num_mismatches=None,
    show_progress=False,
    max_submission_delay=None,
    max_daily_samples=None,
    num_threads=None,
    precision=None,
    rng=None,
):
    logger.info(f"Start match for {date}")
    date_samples = [Sample(md) for md in metadata_db.get(date)]
    samples = filter_samples(date_samples, alignment_store, max_submission_delay)
    if len(samples) == 0:
        logger.warning(f"No samples for {date}")
        return []
    logger.info(f"Got {len(samples)} samples")

    if max_daily_samples is not None and len(samples) > max_daily_samples:
        samples = rng.sample(samples, max_daily_samples)
        logger.info(f"Sampled down to {len(samples)} samples")

    G = np.zeros((base_ts.num_sites, len(samples)), dtype=np.int8)
    keep_sites = base_ts.sites_position.astype(int)

    samples_iter = enumerate(samples)
    with tqdm.tqdm(
        samples_iter,
        desc=f"Fetch {date}",
        total=len(samples),
        disable=not show_progress,
    ) as bar:
        for j, sample in bar:
            logger.debug(f"Getting alignment for {sample.strain}")
            alignment = alignment_store[sample.strain]
            logger.debug(f"Encoding alignment")
            ma = alignments.encode_and_mask(alignment)
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
    )
    return samples


def extend(
    *,
    alignment_store,
    metadata_db,
    date,
    base_ts,
    num_mismatches=None,
    show_progress=False,
    max_submission_delay=None,
    max_daily_samples=None,
    num_threads=None,
    precision=None,
    rng=None,
):
    samples = match(
        alignment_store=alignment_store,
        metadata_db=metadata_db,
        date=date,
        base_ts=base_ts,
        num_mismatches=num_mismatches,
        show_progress=show_progress,
        max_submission_delay=max_submission_delay,
        max_daily_samples=max_daily_samples,
        num_threads=num_threads,
        precision=precision,
        rng=rng,
    )
    if len(samples) == 0:
        return base_ts
    ts = increment_time(date, base_ts)
    return add_matching_results(samples, ts)


def add_matching_results(samples, ts):

    tables = ts.dump_tables()

    site_masked_samples = np.zeros(int(ts.sequence_length), dtype=int)

    for sample in samples:
        site_masked_samples[sample.masked_sites] += 1
        metadata = {**sample.metadata, "sc2ts_qc": sample.alignment_qc}
        node_id = tables.nodes.add_row(
            flags=tskit.NODE_IS_SAMPLE, time=0, metadata=metadata
        )
        for left, right, parent in sample.path:
            tables.edges.add_row(left, right, parent=parent, child=node_id)
        for site, derived_state in sample.mutations:
            tables.mutations.add_row(
                site=site,
                node=node_id,
                time=0,
                derived_state=derived_state,
                metadata={"type": "mismatch"},
            )

    # Update the sites with metadata for these newly added samples.
    tables.sites.clear()
    for site in ts.sites():
        md = site.metadata
        md["masked_samples"] += int(site_masked_samples[int(site.position)])
        tables.sites.append(site.replace(metadata=md))

    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()

    # FIXME: doing these as three separate steps on different tree sequences
    # is definitely suboptimal - we should probably have some algorithm that
    # works directly on the result objects plus the tree sequence, adding
    # all the stuff in afterwards when then new trees have been built.
    ts = insert_recombinants(ts)
    ts = coalesce_mutations(ts)
    ts = push_up_reversions(ts)
    return ts


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
            if ts.nodes_flags[u] == core.NODE_IS_MUTATION_OVERLAP:
                # Not strictly a sample, but represents some time-0 samples
                pass
                # FIXME Getting rid of this for now because we're getting
                # some obscure errors
                # samples.add(u)
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


def match_tsinfer(
    samples,
    ts,
    genotypes,
    *,
    num_mismatches=None,
    precision=None,
    num_threads=None,
    show_progress=False,
):
    if num_mismatches is None:
        # Default to no recombination
        num_mismatches = 1000

    reference = core.get_reference_sequence()
    with tsinfer.SampleData(sequence_length=ts.sequence_length) as sd:
        alleles = tuple(core.ALLELES)
        for pos, site_genotypes in zip(ts.sites_position.astype(int), genotypes):
            sd.add_site(
                pos,
                site_genotypes,
                alleles=alleles,
                ancestral_allele=alleles.index(reference[pos]),
            )

    logger.info(f"Built temporary sample data file")

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
    ts = tables.tree_sequence()

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
    ancestral_state = core.get_reference_sequence()[ts.sites_position.astype(int)]

    coord_map = np.append(ts.sites_position, [ts.sequence_length]).astype(int)
    coord_map[0] = 0

    # Update the Sample objects with their paths and sets of mutations.
    for node_id, sample in enumerate(samples, ts.num_nodes):
        path = []
        for left, right, parent in zip(*results.get_path(node_id)):
            path.append((int(coord_map[left]), int(coord_map[right]), int(parent)))
        path.sort()
        sample.path = path

        mutations = []
        for site, derived_state in zip(*results.get_mutations(node_id)):
            derived_state = unshuffle_allele_index(derived_state, ancestral_state[site])
            mutations.append((int(site), derived_state))
        mutations.sort()
        sample.mutations = mutations


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
