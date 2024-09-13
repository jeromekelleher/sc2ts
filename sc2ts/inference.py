from __future__ import annotations
import bz2
import logging
import datetime
import dataclasses
import collections
import pickle
import hashlib
import os
import sqlite3
import pathlib
import random

import tqdm
import tskit
import tsinfer
import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy
import zarr
import numba
import pandas as pd

from . import core
from . import alignments
from . import metadata

logger = logging.getLogger(__name__)


def get_progress(iterable, date, phase, show_progress, total=None):
    bar_format = (
        "{desc:<22}{percentage:3.0f}%|{bar}"
        "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
    )
    return tqdm.tqdm(
        iterable,
        total=total,
        desc=f"{date}:{phase}",
        disable=not show_progress,
        bar_format=bar_format,
        dynamic_ncols=True,
        smoothing=0.01,
        unit_scale=True,
    )


class TsinferProgressMonitor(tsinfer.progress.ProgressMonitor):
    def __init__(self, date, phase, *args, **kwargs):
        self.date = date
        self.phase = phase
        super().__init__(*args, **kwargs)

    def get(self, key, total):
        self.current_instance = get_progress(
            None, self.date, phase=self.phase, show_progress=self.enabled, total=total
        )
        return self.current_instance


class MatchDb:
    def __init__(self, path):
        uri = f"file:{path}"
        self.uri = uri
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = metadata.dict_factory
        logger.debug(f"Opened MatchDb at {path} mode=rw")

    def __len__(self):
        sql = "SELECT COUNT(*) FROM samples"
        with self.conn:
            row = self.conn.execute(sql).fetchone()
            return row["COUNT(*)"]

    def as_dataframe(self):
        with self.conn:
            cursor = self.conn.execute(
                "SELECT strain, match_date, hmm_cost FROM samples"
            )
            return pd.DataFrame(cursor.fetchall())

    def last_date(self):
        sql = "SELECT MAX(match_date) FROM samples"
        with self.conn:
            row = self.conn.execute(sql).fetchone()
            return row["MAX(match_date)"]

    def count_newer(self, date):
        with self.conn:
            sql = "SELECT COUNT(*) FROM samples WHERE match_date >= ?"
            row = self.conn.execute(sql, (date,)).fetchone()
            return row["COUNT(*)"]

    def delete_newer(self, date):
        sql = "DELETE FROM samples WHERE match_date >= ?"
        with self.conn:
            self.conn.execute(sql, (date,))

    def __str__(self):
        return f"MatchDb at {self.uri} has {len(self)} samples"

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.conn.close()

    def add(self, samples, date, num_mismatches):
        """
        Adds the specified matched samples to this MatchDb.
        """
        sql = """\
            INSERT INTO samples (
            strain, match_date, hmm_cost, pickle)
            VALUES (?, ?, ?, ?)
            """
        data = []
        hmm_cost = np.zeros(len(samples))
        for j, sample in enumerate(samples):
            assert sample.date == date
            # FIXME we want to be more selective about what we're storing
            # here, as we're including the alignment too.
            pkl = pickle.dumps(sample)
            # BZ2 compressing drops this by ~10X, so worth it.
            pkl_compressed = bz2.compress(pkl)
            hmm_cost[j] = sample.get_hmm_cost(num_mismatches)
            args = (
                sample.strain,
                date,
                hmm_cost[j],
                pkl_compressed,
            )
            data.append(args)
            logger.debug(f"MatchDB insert: hmm_cost={hmm_cost[j]} {sample.summary()}")
        # Batch insert, for efficiency.
        with self.conn:
            self.conn.executemany(sql, data)
        logger.info(
            f"Added {len(samples)} samples to match DB for {date}; "
            f"hmm_cost:min={hmm_cost.min()},max={hmm_cost.max()},"
            f"mean={hmm_cost.mean()},median={np.median(hmm_cost)}"
        )

    def create_mask_table(self, ts):
        # NOTE perhaps a better way to do this would be to simply delete
        # the rows in the DB that *are* in the ts, as a separate
        # transaction once we know that the trees have been saved to disk.
        logger.info("Loading used samples into DB")
        samples = [(strain,) for strain in ts.metadata["sc2ts"]["samples_strain"]]
        logger.debug(f"Got {len(samples)} from ts")
        with self.conn:
            self.conn.execute("DROP TABLE IF EXISTS used_samples")
            self.conn.execute(
                "CREATE TABLE used_samples (strain TEXT, PRIMARY KEY (strain))"
            )
            self.conn.executemany("INSERT INTO used_samples VALUES (?)", samples)
        logger.debug("Built temporary table")
        with self.conn:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM samples LEFT JOIN used_samples "
                "ON samples.strain = used_samples.strain "
                "WHERE used_samples.strain IS NULL"
            )
            row = cursor.fetchone()
            samples_not_in_ts = row["COUNT(*)"]
        logger.info(f"DB contains {samples_not_in_ts} samples not in ARG")

    def get(self, where_clause):
        sql = (
            "SELECT * FROM samples LEFT JOIN used_samples "
            "ON samples.strain = used_samples.strain "
            f"WHERE used_samples.strain IS NULL AND {where_clause}"
        )
        with self.conn:
            logger.debug(f"MatchDb run: {sql}")
            for row in self.conn.execute(sql):
                pkl = row.pop("pickle")
                sample = pickle.loads(bz2.decompress(pkl))
                logger.debug(
                    f"MatchDb got: {sample.summary()} hmm_cost={row['hmm_cost']}"
                )
                # print(row)
                yield sample

    @staticmethod
    def initialise(db_path):
        db_path = pathlib.Path(db_path)
        if db_path.exists():
            db_path.unlink()
        sql = """\
            CREATE TABLE samples (
            strain TEXT,
            match_date TEXT,
            hmm_cost REAL,
            pickle BLOB,
            PRIMARY KEY (strain))
            """

        with sqlite3.connect(db_path) as conn:
            conn.execute(sql)
            conn.execute(
                "CREATE INDEX [ix_samples_match_date] on 'samples' " "([match_date]);"
            )
        logger.info(f"Created new MatchDb at {db_path}")
        return MatchDb(db_path)

    def print_all(self):
        """
        Debug method to print out full state of the DB.
        """
        import pandas as pd

        data = []
        with self.conn:
            for row in self.conn.execute("SELECT * from samples"):
                data.append(row)
        df = pd.DataFrame(row, index=["strain"])
        print(df)


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


def initial_ts(additional_problematic_sites=list()):
    reference = core.get_reference_sequence()
    L = core.REFERENCE_SEQUENCE_LENGTH
    assert L == len(reference)
    problematic_sites = set(core.get_problematic_sites()) | set(
        additional_problematic_sites
    )

    tables = tskit.TableCollection(L)
    tables.time_units = core.TIME_UNITS

    # TODO add known fields to the schemas and document them.

    base_schema = tskit.MetadataSchema.permissive_json().schema
    tables.reference_sequence.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.reference_sequence.metadata = {
        "genbank_id": core.REFERENCE_GENBANK,
        "notes": "X prepended to alignment to map from 1-based to 0-based coordinates",
    }
    tables.reference_sequence.data = reference

    tables.metadata_schema = tskit.MetadataSchema(base_schema)
    # TODO gene annotations to top level
    tables.metadata = {
        "sc2ts": {
            "date": core.REFERENCE_DATE,
            "samples_strain": [core.REFERENCE_STRAIN],
            "additional_problematic_sites": additional_problematic_sites,
        }
    }

    tables.nodes.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.sites.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.mutations.metadata_schema = tskit.MetadataSchema(base_schema)

    # 1-based coordinates
    for pos in range(1, L):
        if pos not in problematic_sites:
            tables.sites.add_row(pos, reference[pos], metadata={"masked_samples": 0})
    # TODO should probably make the ultimate ancestor time something less
    # plausible or at least configurable. However, this will be removed
    # in later versions when we remove the dependence on tsinfer.
    tables.nodes.add_row(time=1, metadata={"strain": "Vestigial_ignore"})
    tables.nodes.add_row(
        flags=tskit.NODE_IS_SAMPLE,
        time=0,
        metadata={"strain": core.REFERENCE_STRAIN, "date": core.REFERENCE_DATE},
    )
    tables.edges.add_row(0, L, 0, 1)
    return tables.tree_sequence()


def parse_date(date):
    return datetime.datetime.fromisoformat(date)


def last_date(ts):
    return parse_date(ts.metadata["sc2ts"]["date"])


def increment_time(date, ts):
    diff = parse_date(date) - last_date(ts)
    increment = diff.days
    if increment <= 0:
        raise ValueError(f"Bad date diff: {diff}")

    tables = ts.dump_tables()
    tables.nodes.time += increment
    tables.mutations.time += increment
    return tables.tree_sequence()


@dataclasses.dataclass
class Sample:
    strain: str
    date: str
    metadata: Dict = dataclasses.field(default_factory=dict)
    forward_path: List = dataclasses.field(default_factory=list)
    forward_mutations: List = dataclasses.field(default_factory=list)
    reverse_path: List = dataclasses.field(default_factory=list)
    reverse_mutations: List = dataclasses.field(default_factory=list)
    alignment_qc: Dict = dataclasses.field(default_factory=dict)
    masked_sites: List = dataclasses.field(default_factory=list)
    # FIXME need a better name for this, as it's a different thing
    # the original alignment. Haplotype is probably good, as it's
    # what it would be in the tskit/tsinfer world.
    alignment: List = None
    is_recombinant: bool = False

    @property
    def path(self):
        return self.forward_path

    @property
    def mutations(self):
        return self.forward_mutations

    def path_summary(self, path):
        return ",".join(f"({seg.left}:{seg.right}, {seg.parent})" for seg in path)

    def mutation_summary(self, mutations):
        return "[" + ",".join(str(mutation) for mutation in mutations) + "]"

    def summary(self):
        pango = self.metadata.get("Viridian_pangolin", "Unknown")
        s = f"{self.strain} {self.date} {pango} "
        if self.is_recombinant:
            s += (
                f"forward_path={self.path_summary(self.forward_path)} "
                f"forward_mutations({len(self.forward_mutations)})"
                f"={self.mutation_summary(self.forward_mutations)} "
            )
            s += (
                f"reverse_path={self.path_summary(self.reverse_path)} "
                f"reverse_mutations({len(self.reverse_mutations)})"
                f"={self.mutation_summary(self.reverse_mutations)}"
            )
        else:
            s += (
                f"path={self.path_summary(self.forward_path)} "
                f"mutations({len(self.forward_mutations)})"
                f"={self.mutation_summary(self.forward_mutations)}"
            )
        return s

    @property
    def breakpoints(self):
        breakpoints = [seg.left for seg in self.path]
        return breakpoints + [self.path[-1].right]

    @property
    def parents(self):
        return [seg.parent for seg in self.path]

    def get_hmm_cost(self, num_mismatches):
        # Note that Recombinant objects have total_cost.
        # This bit of code is sort of repeated.
        return num_mismatches * (len(self.path) - 1) + len(self.mutations)


def match_samples(
    date,
    samples,
    *,
    base_ts,
    num_mismatches=None,
    show_progress=False,
    num_threads=None,
):
    run_batch = samples

    mu, rho = solve_num_mismatches(num_mismatches)

    for k in range(2):
        # To catch k mismatches we need a likelihood threshold of mu**k
        likelihood_threshold = mu**k - 1e-15
        # print(k, likelihood_threshold)
        logger.info(
            f"Running match={k} batch of {len(run_batch)} at threshold={likelihood_threshold}"
        )
        match_tsinfer(
            samples=run_batch,
            ts=base_ts,
            mu=mu,
            rho=rho,
            likelihood_threshold=likelihood_threshold,
            num_threads=num_threads,
            show_progress=show_progress,
            date=date,
            phase=f"match({k})",
        )

        exceeding_threshold = []
        for sample in run_batch:
            cost = sample.get_hmm_cost(num_mismatches)
            logger.debug(f"HMM@k={k}: hmm_cost={cost} {sample.summary()}")
            if cost > k + 1:
                exceeding_threshold.append(sample)

        num_matches_found = len(run_batch) - len(exceeding_threshold)
        logger.info(
            f"{num_matches_found} final matches found at k={k}; "
            f"{len(exceeding_threshold)} remain"
        )
        run_batch = exceeding_threshold

    logger.info(f"Running final batch of {len(run_batch)} at high precision")
    match_tsinfer(
        samples=run_batch,
        ts=base_ts,
        mu=mu,
        rho=rho,
        num_threads=num_threads,
        show_progress=show_progress,
        date=date,
        phase=f"match(F)",
    )
    recombinants = []
    for sample in run_batch:
        cost = sample.get_hmm_cost(num_mismatches)
        # print(f"Final HMM pass:{sample.strain} hmm_cost={cost} {sample.summary()}")
        logger.debug(f"Final HMM pass hmm_cost={cost} {sample.summary()}")
        # if len(sample.path) > 1:
        #     sample.is_recombinant = True
        #     recombinants.append(sample)

    # if len(recombinants) > 0:
    # for mirror in [False, True]:
    #     logger.info(
    #         f"Running {len(run_batch)} recombinants at maximum precision in"
    #         f"{['forward', 'backward'][int(mirror)]} direction."
    #     )
    #     match_tsinfer(
    #         samples=recombinants,
    #         ts=base_ts,
    #         mu=mu,
    #         rho=rho,
    #         num_threads=num_threads,
    #         show_progress=show_progress,
    #         # Maximum possible precision
    #         likelihood_threshold=1e-200,
    #         mirror_coordinates=mirror,
    #     )
    # for sample in recombinants:
    #     # We may want to try to improve the location of the breakpoints
    #     # later. For now, just log the info.
    #     logger.info(f"Recombinant: {sample.summary()}")

    return samples


def check_base_ts(ts):
    md = ts.metadata
    assert "sc2ts" in md
    sc2ts_md = md["sc2ts"]
    assert "date" in sc2ts_md
    assert len(sc2ts_md["samples_strain"]) == ts.num_samples


def preprocess(samples_md, base_ts, date, alignment_store, show_progress=False):
    keep_sites = base_ts.sites_position.astype(int)
    problematic_sites = core.get_problematic_sites()

    samples = []
    with get_progress(samples_md, date, "preprocess", show_progress) as bar:
        for md in bar:
            strain = md["strain"]
            try:
                alignment = alignment_store[strain]
            except KeyError:
                logger.debug(f"No alignment stored for {strain}")
                continue
            sample = Sample(strain, date, metadata=md)
            ma = alignments.encode_and_mask(alignment)
            # Always mask the problematic_sites as well. We need to do this
            # for follow-up matching to inspect recombinants, as tsinfer
            # needs us to keep all sites in the table when doing mirrored
            # coordinates.
            ma.alignment[problematic_sites] = -1
            sample.alignment_qc = ma.qc_summary()
            sample.masked_sites = ma.masked_sites
            sample.alignment = ma.alignment[keep_sites]
            samples.append(sample)
            num_Ns = ma.original_base_composition.get("N", 0)
            non_nuc_counts = dict(ma.original_base_composition)
            for nuc in "ACGT":
                del non_nuc_counts[nuc]
                counts = ",".join(
                    f"{key}={count}" for key, count in sorted(non_nuc_counts.items())
                )
            num_masked = len(ma.masked_sites)
            logger.debug(f"Mask {strain}: masked={num_masked} {counts}")

    return samples


def extend(
    *,
    alignment_store,
    metadata_db,
    date,
    base_ts,
    match_db,
    num_mismatches=None,
    hmm_cost_threshold=None,
    min_group_size=None,
    max_daily_samples=None,
    show_progress=False,
    retrospective_window=None,
    random_seed=42,
    num_threads=0,
):
    if num_mismatches is None:
        num_mismatches = 3
    if hmm_cost_threshold is None:
        hmm_cost_threshold = 5
    if min_group_size is None:
        min_group_size = 10
    if retrospective_window is None:
        retrospective_window = 30

    check_base_ts(base_ts)
    logger.info(
        f"Extend {date}; ts:nodes={base_ts.num_nodes};samples={base_ts.num_samples};"
        f"mutations={base_ts.num_mutations};date={base_ts.metadata['sc2ts']['date']}"
    )

    metadata_matches = list(metadata_db.get(date))

    logger.info(f"Got {len(metadata_matches)} metadata matches")
    # first check for samples that are in the alignment_store
    samples_with_aligments = []
    for md in metadata_matches:
        if md["strain"] in alignment_store:
            samples_with_aligments.append(md)

    logger.info(f"Verified {len(samples_with_aligments)} have alignments")
    # metadata_matches = list(
    #     metadata_db.query("SELECT * FROM samples WHERE strain=='SRR19463295'")
    # )
    # TODO implement this.
    if max_daily_samples is not None:
        if max_daily_samples < len(samples_with_aligments):
            seed_prefix = bytes(np.array([random_seed], dtype=int).data)
            seed_suffix = hashlib.sha256(date.encode()).digest()
            rng = random.Random(seed_prefix + seed_suffix)
            samples_with_aligments = rng.sample(
                samples_with_aligments, max_daily_samples
            )
            logger.info(f"Subset to {len(metadata_matches)} samples")

    samples = preprocess(
        samples_with_aligments,
        base_ts,
        date,
        alignment_store,
        show_progress=show_progress,
    )

    if len(samples) == 0:
        logger.warning(f"Nothing to do for {date}")
        return base_ts

    logger.info(
        f"Got alignments for {len(samples)} of {len(metadata_matches)} in metadata"
    )

    samples = match_samples(
        date,
        samples,
        base_ts=base_ts,
        num_mismatches=num_mismatches,
        show_progress=show_progress,
        num_threads=num_threads,
    )

    match_db.add(samples, date, num_mismatches)
    match_db.create_mask_table(base_ts)
    ts = increment_time(date, base_ts)

    ts = add_exact_matches(ts=ts, match_db=match_db, date=date)

    logger.info(f"Update ARG with low-cost samples for {date}")
    ts = add_matching_results(
        f"match_date=='{date}' and hmm_cost>0 and hmm_cost<={hmm_cost_threshold}",
        ts=ts,
        match_db=match_db,
        date=date,
        min_group_size=1,
        show_progress=show_progress,
        phase="add(close)",
    )

    logger.info("Looking for retrospective matches")
    assert min_group_size is not None
    earliest_date = parse_date(date) - datetime.timedelta(days=retrospective_window)
    ts = add_matching_results(
        f"match_date<'{date}' AND match_date>'{earliest_date}'",
        ts=ts,
        match_db=match_db,
        date=date,
        min_group_size=min_group_size,
        min_different_dates=3,  # TODO parametrize
        show_progress=show_progress,
        phase="add(retro)",
    )
    return update_top_level_metadata(ts, date)


def update_top_level_metadata(ts, date):
    tables = ts.dump_tables()
    md = tables.metadata
    md["sc2ts"]["date"] = date
    samples_strain = md["sc2ts"]["samples_strain"]
    new_samples = ts.samples()[len(samples_strain) :]
    for u in new_samples:
        node = ts.node(u)
        samples_strain.append(node.metadata["strain"])
    md["sc2ts"]["samples_strain"] = samples_strain
    tables.metadata = md
    return tables.tree_sequence()


def add_sample_to_tables(sample, tables, flags=tskit.NODE_IS_SAMPLE, time=0):
    hmm_md = [
        {
            "direction": "forward",
            "path": [x.asdict() for x in sample.forward_path],
            "mutations": [x.asdict() for x in sample.forward_mutations],
        }
    ]
    if sample.is_recombinant:
        hmm_md.append(
            {
                "direction": "reverse",
                "path": [x.asdict() for x in sample.forward_path],
                "mutations": [x.asdict() for x in sample.forward_mutations],
            }
        )
    metadata = {
        **sample.metadata,
        "sc2ts": {
            "qc": sample.alignment_qc,
            "hmm": hmm_md,
        },
    }
    return tables.nodes.add_row(flags=flags, time=time, metadata=metadata)


def match_path_ts(samples, ts, path, reversions):
    """
    Given the specified list of samples with equal copying paths,
    return the tree sequence rooted at zero representing the data.
    """
    tables = tskit.TableCollection(ts.sequence_length)
    tables.nodes.metadata_schema = ts.table_metadata_schemas.node
    # Zero is the attach node
    md = {
        "samples": [sample.strain for sample in samples],
        "path": [seg.asdict() for seg in path],
    }
    # FIXME this doesn't work because we don't actually
    # use this node in the output trees. What we probably want to
    # do is create a unique ID that we put into the metadata of
    # every node we ultimately attach from this tree. That could
    # be a hash of the sample IDs, I guess. It could record
    # the number of samples also, usefully. Basically we want
    # to enable tracing around the larger tree later, and to
    # figure out which nodes were added at as part of a particular
    # group, on a particular day.
    tables.nodes.add_row(time=1, metadata={"sc2ts": md})
    path = samples[0].path
    site_id_map = {}
    first_sample = len(tables.nodes)
    logger.debug(
        f"Adding group of {len(samples)} with path={path} and "
        f"reversions={reversions}"
    )
    for sample in samples:
        assert sample.path == path
        node_id = add_sample_to_tables(sample, tables)
        tables.edges.add_row(0, ts.sequence_length, parent=0, child=node_id)
        for mut in sample.mutations:
            if mut.site_id not in site_id_map:
                new_id = tables.sites.add_row(mut.site_position, mut.inherited_state)
                site_id_map[mut.site_id] = new_id

    # Now add the mutations
    for node_id, sample in enumerate(samples, first_sample):
        logger.debug(
            f"Adding {sample.strain}:{sample.date} with "
            f"{len(sample.mutations)} mutations"
        )
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


def add_exact_matches(match_db, ts, date):
    where_clause = f"match_date=='{date}' AND hmm_cost==0"
    logger.info(f"Querying match DB WHERE: {where_clause}")
    samples = list(match_db.get(where_clause))
    if len(samples) == 0:
        logger.info(f"No exact matches on {date}")
        return ts
    logger.info(f"Update ARG with {len(samples)} exact matches for {date}")
    tables = ts.dump_tables()
    for sample in samples:
        assert len(sample.path) == 1
        assert len(sample.mutations) == 0
        node_id = add_sample_to_tables(
            sample,
            tables,
            flags=tskit.NODE_IS_SAMPLE | core.NODE_IS_EXACT_MATCH,
        )
        parent = sample.path[0].parent
        logger.debug(f"ARG add exact match {sample.strain}:{node_id}->{parent}")
        tables.edges.add_row(0, ts.sequence_length, parent=parent, child=node_id)
    tables.sort()
    tables.build_index()
    return tables.tree_sequence()


def add_matching_results(
    where_clause,
    match_db,
    ts,
    date,
    min_group_size=1,
    min_different_dates=1,
    show_progress=False,
    phase=None,
):
    logger.info(f"Querying match DB WHERE: {where_clause}")
    samples = match_db.get(where_clause)

    # Group matches by path and set of immediate reversions.
    grouped_matches = collections.defaultdict(list)
    excluded_samples = []
    site_masked_samples = np.zeros(int(ts.sequence_length), dtype=int)
    num_samples = 0
    for sample in samples:
        site_masked_samples[sample.masked_sites] += 1
        path = tuple(sample.path)
        reversions = tuple(
            (mut.site_id, mut.derived_state)
            for mut in sample.mutations
            if mut.is_immediate_reversion
        )
        grouped_matches[(path, reversions)].append(sample)
        num_samples += 1

    if num_samples == 0:
        logger.info("No candidate samples found in MatchDb")
        return ts

    tables = ts.dump_tables()
    logger.info(f"Got {len(grouped_matches)} distinct paths for {num_samples} samples")

    attach_nodes = []
    added_samples = []
    with get_progress(list(grouped_matches.items()), date, phase, show_progress) as bar:
        for (path, reversions), match_samples in bar:
            different_dates = set(sample.date for sample in match_samples)
            # TODO (1) add group ID from hash of samples (2) better logging of path
            logger.debug(
                f"Group of {len(match_samples)} has {len(different_dates)} different dates"
                f" at {path}, {reversions} "
            )
            if (
                len(match_samples) < min_group_size
                or len(different_dates) < min_different_dates
            ):
                continue

            added_samples.extend(match_samples)

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
                    # FIXME this is quick hack to get things working, but we
                    # shouldn't be assuming that we're always using the
                    # forward mutations.
                    sample.forward_mutations = new_muts

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

    if len(added_samples) == 0:
        logger.info("No samples passing group size requirements found")
        return ts

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
    ts = push_up_reversions(ts, attach_nodes)
    ts = coalesce_mutations(ts, attach_nodes)
    return ts


def solve_num_mismatches(k):
    """
    Return the low-level LS parameters corresponding to accepting
    k mismatches in favour of a single recombination.

    The equation is

    r(1 - r)^(m-1) (1 - 4 mu)^m = (1 - r)^m (1 - 4 mu)^(m - k) mu^k

    Solving for r gives

    r = mu^k / (mu^k + (1 - 4 mu)^k

    The LHS is
    1. P[one recombination]
    2. P[not recombining m - 1 times]
    3. P[not mutating m times]

    The RHS is
    1. P[not recombining m times]
    2. P[not mutating m - k times]
    3. P[mutating k times]

    The 1 - 4mu terms come from having 5 alleles.

    """
    # values of k <= 1 are not relevant for SC2 and lead to awkward corner cases
    assert k > 1
    mu = 0.0125

    denom = mu**k + (1 - 4 * mu) ** k
    rho = mu**k / denom
    # print("r before", r)
    # Add a tiny bit of extra mass so that we deterministically recombine
    rho += rho * 0.1
    return mu, rho


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
    mu,
    rho,
    *,
    likelihood_threshold=None,
    num_threads=0,
    show_progress=False,
    date=None,
    phase=None,
    mirror_coordinates=False,
):
    if len(samples) == 0:
        return
    genotypes = np.array([sample.alignment for sample in samples], dtype=np.int8).T
    input_ts = ts
    if mirror_coordinates:
        ts = mirror_ts_coordinates(ts)
        genotypes = genotypes[::-1]

    sd = convert_tsinfer_sample_data(ts, genotypes)

    L = int(ts.sequence_length)
    ls_recomb = np.full(ts.num_sites - 1, rho)
    ls_mismatch = np.full(ts.num_sites, mu)
    if likelihood_threshold is None:
        # Let's say a double break with 5 mutations is the most unlikely thing
        # we're interested in solving for exactly.
        likelihood_threshold = rho**2 * mu**5

    pm = TsinferProgressMonitor(date, phase, enabled=show_progress)

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
        likelihood_threshold=likelihood_threshold,
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

    update_path_info(
        samples,
        input_ts,
        sample_paths,
        sample_mutations,
        forward=not mirror_coordinates,
    )


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
            "site_position": int(self.site_position),
            "derived_state": self.derived_state,
            "inherited_state": self.inherited_state,
        }


def update_path_info(samples, ts, sample_paths, sample_mutations, forward=True):
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
        sample_path = [PathSegment(*seg) for seg in path]
        sample_mutations = []
        for site_pos, derived_state in mutations:
            site_id = np.searchsorted(ts.sites_position, site_pos)
            assert ts.sites_position[site_id] == site_pos
            seg = [seg for seg in sample_path if seg.contains(site_pos)][0]
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
            sample_mutations.append(
                MatchMutation(
                    site_id=site_id,
                    site_position=int(site_pos),
                    derived_state=derived_state,
                    inherited_state=inherited_state,
                    is_reversion=is_reversion,
                    is_immediate_reversion=is_immediate_reversion,
                )
            )
        if forward:
            sample.forward_path = sample_path
            sample.forward_mutations = sample_mutations
        else:
            sample.reverse_path = sample_path
            sample.reverse_mutations = sample_mutations


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


def attach_tree(
    parent_ts,
    parent_tables,
    attach_path,
    reversions,
    child_ts,
    date,
    epsilon=None,
):
    if epsilon is None:
        epsilon = 1e-6  # In time units of days ago

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

    # Add sample node times
    current_date = parse_date(date)
    node_time = {}  # In time units of days ago
    for u in tree.postorder():
        if tree.is_sample(u):
            node = child_ts.node(u)
            sample_date = parse_date(node.metadata["date"])
            node_time[u] = (current_date - sample_date).days
            assert node_time[u] >= 0.0
    max_sample_time = max(node_time.values())

    node_id_map = {}
    if child_ts.nodes_time[tree.root] != 1.0:
        raise ValueError("Time must be scaled from 0 to 1.")

    num_internal_nodes_visited = 0
    for u in tree.postorder()[:-1]:
        node = child_ts.node(u)
        if tree.is_sample(u):
            # All sample nodes are terminal
            time = node_time[u]
        else:
            num_internal_nodes_visited += 1
            time = max_sample_time + num_internal_nodes_visited * epsilon
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
