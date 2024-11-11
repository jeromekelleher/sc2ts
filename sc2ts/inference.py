from __future__ import annotations
import bz2
import logging
import datetime
import dataclasses
import collections
import concurrent.futures as cf
import time
import json
import pickle
import hashlib
import sqlite3
import pathlib
import random
import threading

import tqdm
import tskit
import _tsinfer
import numpy as np
import zarr
import numba
import humanize
import pandas as pd

from . import core
from . import alignments
from . import metadata
from . import tree_ops

logger = logging.getLogger(__name__)

MISSING = -1
DELETION = core.ALLELES.index("-")


def get_progress(iterable, title, phase, show_progress, total=None):
    bar_format = (
        "{desc:<22}{percentage:3.0f}%|{bar}"
        "| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
    )
    return tqdm.tqdm(
        iterable,
        total=total,
        desc=f"{title}:{phase}",
        disable=not show_progress,
        bar_format=bar_format,
        dynamic_ncols=True,
        smoothing=0.01,
        unit_scale=True,
    )


class MatchDb:
    def __init__(self, path):
        uri = f"file:{path}"
        self.path = path
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

    def add(self, samples, date, show_progress=False):
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
        bar = get_progress(
            enumerate(samples), date, "update mdb", show_progress, total=len(samples)
        )
        for j, sample in bar:
            assert sample.date == date
            pkl = pickle.dumps(sample)
            # BZ2 compressing drops this by ~10X, so worth it.
            pkl_compressed = bz2.compress(pkl)
            hmm_cost[j] = sample.hmm_match.cost
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


def initial_ts(problematic_sites=list()):
    reference = core.get_reference_sequence()
    L = core.REFERENCE_SEQUENCE_LENGTH
    assert L == len(reference)
    problematic_sites = set(problematic_sites)

    logger.info(f"Masking out {len(problematic_sites)} sites")
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
    tables.metadata = {
        "sc2ts": {
            "date": core.REFERENCE_DATE,
            "samples_strain": [core.REFERENCE_STRAIN],
            "exact_matches": {
                "pango": {},
                "date": {},
                "node": {},
            },
            "num_samples_processed": {},
            "retro_groups": [],
        }
    }

    tables.nodes.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.sites.metadata_schema = tskit.MetadataSchema(base_schema)
    tables.mutations.metadata_schema = tskit.MetadataSchema(base_schema)

    # 1-based coordinates
    for pos in range(1, L):
        if pos not in problematic_sites:
            tables.sites.add_row(
                pos,
                reference[pos],
                metadata={"sc2ts": {"missing_samples": 0, "deletion_samples": 0}},
            )
    # TODO should probably make the ultimate ancestor time something less
    # plausible or at least configurable. However, this will be removed
    # in later versions when we remove the dependence on tsinfer.
    tables.nodes.add_row(
        time=1,
        metadata={
            "strain": "Vestigial_ignore",
            "sc2ts": {"notes": "Vestigial root required for technical reasons"},
        },
    )
    tables.nodes.add_row(
        flags=tskit.NODE_IS_SAMPLE,
        time=0,
        metadata={
            "strain": core.REFERENCE_STRAIN,
            "date": core.REFERENCE_DATE,
            "sc2ts": {"notes": "Reference sequence"},
        },
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
    date: str = "1999-01-01"
    pango: str = "Unknown"
    metadata: Dict = dataclasses.field(default_factory=dict)
    alignment_composition: Dict = None
    haplotype: List = None
    hmm_match: HmmMatch = None
    hmm_reruns: Dict = dataclasses.field(default_factory=dict)

    @property
    def is_recombinant(self):
        return len(self.hmm_match.path) > 1

    @property
    def num_missing_sites(self):
        return int(np.sum(self.haplotype == MISSING))

    @property
    def num_deletion_sites(self):
        return int(np.sum(self.haplotype == DELETION))

    def summary(self):
        hmm_match = "No match" if self.hmm_match is None else self.hmm_match.summary()
        s = f"{self.strain} {self.date} {self.pango} {hmm_match}"
        for name, hmm_match in self.hmm_reruns.items():
            s += f"; {name}: {hmm_match.summary()}"
        return s


# TODO not clear if we still need this as mirroring is done differently now.
# Remove if we don't have any issues with running the HMM in reverse
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


# TODO remove this
def match_recombinants(
    samples, base_ts, num_mismatches, show_progress=False, num_threads=None
):
    for hmm_pass in ["forward", "reverse", "no_recombination"]:
        logger.info(f"Running {hmm_pass} pass for {len(samples)} recombinants")
        match_tsinfer(
            samples=samples,
            ts=base_ts,
            num_mismatches=1000 if hmm_pass == "no_recombination" else num_mismatches,
            mismatch_threshold=100,
            num_threads=num_threads,
            show_progress=show_progress,
            mirror_coordinates=hmm_pass == "reverse",
        )

        for sample in samples:
            sample.hmm_reruns[hmm_pass] = sample.hmm_match


def match_samples(
    date,
    samples,
    *,
    base_ts,
    deletions_as_missing=False,
    num_mismatches=None,
    show_progress=False,
    num_threads=0,
    memory_limit=-1,
):
    run_batch = list(samples)

    for k in range(2):
        logger.info(f"Running match={k} batch of {len(run_batch)}")
        match_tsinfer(
            samples=run_batch,
            ts=base_ts,
            num_mismatches=num_mismatches,
            mismatch_threshold=k,
            deletions_as_missing=deletions_as_missing,
            num_threads=num_threads,
            memory_limit=memory_limit,
            show_progress=show_progress,
            progress_title=date,
            progress_phase=f"match({k})",
        )

        exceeding_threshold = []
        for sample in run_batch:
            cost = sample.hmm_match.cost
            if cost > k + 1:
                exceeding_threshold.append(sample)

        num_matches_found = len(run_batch) - len(exceeding_threshold)
        logger.info(
            f"{num_matches_found} final matches found at k={k}; "
            f"{len(exceeding_threshold)} remain"
        )
        run_batch = exceeding_threshold

    # Order the run_batch by the likelihood of their matches so that
    # we don't run out of memory because of an initial glut of
    # difficult matches.
    # NOTE: we should probably do this type of thing withing the
    # MatchManager, where we preferentially choose high-likelihood
    # sequences when we're under memory pressure? This is quite
    # complicated though, and hard to test.
    run_batch.sort(key=lambda s: -s.hmm_match.likelihood)
    likelihoods = collections.Counter([s.hmm_match.likelihood for s in run_batch])
    logger.debug(f"L dist: {dict(likelihoods)}")
    if len(run_batch) > num_threads > 0:
        start_batch = run_batch[:num_threads]
        rest = run_batch[num_threads:]
        rng = random.Random(42)  # Seed doesn't matter here
        rng.shuffle(rest)
        run_batch = start_batch + rest

    logger.info(f"Running final batch of {len(run_batch)} at high precision")
    match_tsinfer(
        samples=run_batch,
        ts=base_ts,
        num_mismatches=num_mismatches,
        num_threads=num_threads,
        memory_limit=memory_limit,
        deletions_as_missing=deletions_as_missing,
        show_progress=show_progress,
        progress_title=date,
        progress_phase=f"match(F)",
    )


def check_base_ts(ts):
    md = ts.metadata
    assert "sc2ts" in md
    sc2ts_md = md["sc2ts"]
    assert len(sc2ts_md["samples_strain"]) == ts.num_samples
    # Avoid parsing the metadata again to get the date.
    return sc2ts_md["date"]


def preprocess_worker(strains, alignment_store_path, keep_sites):
    assert keep_sites is not None
    with alignments.AlignmentStore(alignment_store_path) as alignment_store:
        samples = []
        for strain in strains:
            alignment = alignment_store.get(strain, None)
            sample = Sample(strain)
            if alignment is not None:
                a = alignment[keep_sites]
                sample.haplotype = alignments.encode_alignment(a)
                # Need to do this here because encoding gets rid of
                # ambiguous bases etc.
                sample.alignment_composition = collections.Counter(a)
            samples.append(sample)
    return samples


def preprocess(
    strains,
    alignment_store_path,
    *,
    keep_sites,
    progress_title="",
    show_progress=False,
    num_workers=0,
):
    if len(strains) == 0:
        return []
    num_workers = max(1, num_workers)
    splits = min(len(strains), 2 * num_workers)
    work = np.array_split(strains, splits)
    samples = []
    bar = get_progress(strains, progress_title, "preprocess", show_progress)
    with cf.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(preprocess_worker, w, alignment_store_path, keep_sites)
            for w in work
        ]
        for future in cf.as_completed(futures):
            for s in future.result():
                bar.update()
                samples.append(s)
    bar.close()
    return samples


def extend(
    *,
    alignment_store,
    metadata_db,
    date,
    base_ts,
    match_db,
    include_samples=None,
    num_mismatches=None,
    hmm_cost_threshold=None,
    min_group_size=None,
    min_root_mutations=None,
    min_different_dates=None,
    max_mutations_per_sample=None,
    max_recurrent_mutations=None,
    deletions_as_missing=None,
    max_daily_samples=None,
    show_progress=False,
    retrospective_window=None,
    max_missing_sites=None,
    random_seed=42,
    num_threads=0,
    memory_limit=0,
):
    if num_mismatches is None:
        num_mismatches = 3
    if hmm_cost_threshold is None:
        hmm_cost_threshold = 5
    if min_group_size is None:
        min_group_size = 10
    if min_root_mutations is None:
        min_root_mutations = 2
    if max_mutations_per_sample is None:
        max_mutations_per_sample = 100
    if max_recurrent_mutations is None:
        max_recurrent_mutations = 100
    if min_different_dates is None:
        min_different_dates = 3
    if retrospective_window is None:
        retrospective_window = 30
    if max_missing_sites is None:
        max_missing_sites = np.inf
    if deletions_as_missing is None:
        deletions_as_missing = False
    if include_samples is None:
        include_samples = []

    previous_date = check_base_ts(base_ts)
    logger.info(
        f"Extend {date}; ts:nodes={base_ts.num_nodes};samples={base_ts.num_samples};"
        f"mutations={base_ts.num_mutations};date={previous_date}"
    )

    metadata_matches = {md["strain"]: md for md in metadata_db.get(date)}

    logger.info(f"Got {len(metadata_matches)} metadata matches")

    preprocessed_samples = preprocess(
        strains=list(metadata_matches.keys()),
        alignment_store_path=alignment_store.path,
        keep_sites=base_ts.sites_position.astype(int),
        progress_title=date,
        show_progress=show_progress,
        num_workers=num_threads,
    )
    # FIXME parametrise
    pango_lineage_key = "Viridian_pangolin"

    include_strains = set(include_samples)
    unconditional_include_samples = []
    samples = []
    for s in preprocessed_samples:
        if s.haplotype is None:
            logger.debug(f"No alignment stored for {s.strain}")
            continue
        md = metadata_matches[s.strain]
        s.metadata = md
        s.pango = md.get(pango_lineage_key, "Unknown")
        s.date = date
        num_missing_sites = s.num_missing_sites
        num_deletion_sites = s.num_deletion_sites
        logger.debug(
            f"Encoded {s.strain} {s.pango} missing={num_missing_sites} "
            f"deletions={num_deletion_sites}"
        )
        if s.strain in include_strains:
            unconditional_include_samples.append(s)
        elif num_missing_sites <= max_missing_sites:
            samples.append(s)
        else:
            logger.debug(
                f"Filter {s.strain}: missing={num_missing_sites} > {max_missing_sites}"
            )

    if max_daily_samples is not None:
        if max_daily_samples < len(samples):
            seed_prefix = bytes(np.array([random_seed], dtype=int).data)
            seed_suffix = hashlib.sha256(date.encode()).digest()
            rng = random.Random(seed_prefix + seed_suffix)
            logger.info(f"Subset from {len(samples)} to {max_daily_samples}")
            samples = rng.sample(samples, max_daily_samples)

    samples = samples + unconditional_include_samples
    samples.sort(key=lambda s: s.strain)

    ts = increment_time(date, base_ts)
    if len(samples) > 0:
        logger.info(
            f"Got alignments for {len(samples)} of {len(metadata_matches)} in metadata"
        )

        match_samples(
            date,
            samples,
            base_ts=base_ts,
            num_mismatches=num_mismatches,
            deletions_as_missing=deletions_as_missing,
            show_progress=show_progress,
            num_threads=num_threads,
            memory_limit=memory_limit,
        )

        characterise_match_mutations(base_ts, samples)
        for sample in unconditional_include_samples:
            # We want this sample to included unconditionally, so we set the
            # hmm cost to 0 < hmm_cost < hmm_cost_threshold. We use 0.5
            # arbitrarily here to distinguish it from real one-mutation
            sample.hmm_match.cost = 0.5
            logger.warning(f"Unconditionally including {sample.summary()}")

        match_db.add(samples, date, show_progress)
        match_db.create_mask_table(base_ts)

        ts = add_exact_matches(ts=ts, match_db=match_db, date=date)

        logger.info(f"Update ARG with low-cost samples for {date}")
        ts, _ = add_matching_results(
            f"match_date=='{date}' and hmm_cost>0 and hmm_cost<={hmm_cost_threshold}",
            ts=ts,
            match_db=match_db,
            date=date,
            min_group_size=1,
            additional_node_flags=core.NODE_IN_SAMPLE_GROUP,
            show_progress=show_progress,
            phase="close",
        )

    logger.info("Looking for retrospective matches")
    assert min_group_size is not None
    earliest_date = parse_date(date) - datetime.timedelta(days=retrospective_window)
    ts, groups = add_matching_results(
        f"hmm_cost>0 AND match_date<'{date}' AND match_date>'{earliest_date}'",
        ts=ts,
        match_db=match_db,
        date=date,
        min_group_size=min_group_size,
        min_different_dates=min_different_dates,
        min_root_mutations=min_root_mutations,
        max_mutations_per_sample=max_mutations_per_sample,
        max_recurrent_mutations=max_recurrent_mutations,
        additional_node_flags=core.NODE_IN_RETROSPECTIVE_SAMPLE_GROUP,
        show_progress=show_progress,
        phase="retro",
    )
    for group in groups:
        logger.warning(
            f"Add retro group {group.summary()}:"
            f"{group.tree_quality_metrics.summary()}"
        )
    return update_top_level_metadata(ts, date, groups, len(samples))


def update_top_level_metadata(ts, date, retro_groups, num_samples):
    tables = ts.dump_tables()
    md = tables.metadata
    md["sc2ts"]["date"] = date
    samples_strain = md["sc2ts"]["samples_strain"]
    new_samples = ts.samples()[len(samples_strain) :]
    for u in new_samples:
        node = ts.node(u)
        samples_strain.append(node.metadata["strain"])
    md["sc2ts"]["samples_strain"] = samples_strain
    md["sc2ts"]["num_samples_processed"][date] = num_samples
    existing_retro_groups = md["sc2ts"].get("retro_groups", [])
    if isinstance(existing_retro_groups, dict):
        # Hack to implement metadata format change
        existing_retro_groups = []
    for group in retro_groups:
        d = group.tree_quality_metrics.asdict()
        d["group_id"] = group.sample_hash
        existing_retro_groups.append(d)
    md["sc2ts"]["retro_groups"] = existing_retro_groups
    tables.metadata = md
    return tables.tree_sequence()


def add_sample_to_tables(sample, tables, flags=tskit.NODE_IS_SAMPLE, group_id=None):
    sc2ts_md = {
        "hmm_match": sample.hmm_match.asdict(),
        "hmm_reruns": {k: m.asdict() for k, m in sample.hmm_reruns.items()},
        "alignment_composition": dict(sample.alignment_composition),
        "num_missing_sites": sample.num_missing_sites,
    }
    if group_id is not None:
        sc2ts_md["group_id"] = group_id
    metadata = {**sample.metadata, "sc2ts": sc2ts_md}
    return tables.nodes.add_row(flags=flags, metadata=metadata)


def match_path_ts(group):
    """
    Given the specified SampleGroup return the tree sequence rooted at
    zero representing the data.
    """
    tables = tskit.TableCollection(core.REFERENCE_SEQUENCE_LENGTH)
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    site_id_map = {}
    first_sample = len(tables.nodes)
    root = len(group)
    for sample in group:
        assert sample.hmm_match.path == list(group.path)
        node_id = add_sample_to_tables(sample, tables, group_id=group.sample_hash)
        tables.edges.add_row(0, tables.sequence_length, parent=root, child=node_id)
        for mut in sample.hmm_match.mutations:
            if (mut.site_id, mut.derived_state) in group.immediate_reversions:
                # We don't include any of the marked reversions so that they
                # aren't used in tree building.
                continue

            if mut.site_id not in site_id_map:
                new_id = tables.sites.add_row(mut.site_position, mut.inherited_state)
                site_id_map[mut.site_id] = new_id

            tables.mutations.add_row(
                site=site_id_map[mut.site_id],
                node=node_id,
                time=0,
                derived_state=mut.derived_state,
            )
    # add the root
    tables.nodes.add_row(time=1)
    tables.sort()
    return tables.tree_sequence()


def add_exact_matches(match_db, ts, date):
    where_clause = f"match_date=='{date}' AND hmm_cost==0"
    logger.info(f"Querying match DB WHERE: {where_clause}")
    samples = list(match_db.get(where_clause))
    if len(samples) == 0:
        logger.info(f"No exact matches on {date}")
        return ts
    logger.info(f"Update ARG with {len(samples)} exact matches for {date}")
    pango_counts = collections.Counter()
    node_counts = collections.Counter()
    for sample in samples:
        assert len(sample.hmm_match.path) == 1
        assert len(sample.hmm_match.mutations) == 0
        parent = sample.hmm_match.path[0].parent
        logger.debug(f"Increment exact match {sample.strain}->{parent}")
        # JSON treats dictionary keys as strings
        node_counts[str(parent)] += 1
        pango_counts[sample.pango] += 1
        # node_id = add_sample_to_tables(
        #     sample,
        #     tables,
        #     flags=tskit.NODE_IS_SAMPLE | core.NODE_IS_EXACT_MATCH,
        # )
        # logger.debug(f"ARG add exact match {sample.strain}:{node_id}->{parent}")
        # tables.edges.add_row(0, ts.sequence_length, parent=parent, child=node_id)
    tables = ts.dump_tables()
    md = tables.metadata
    exact_matches_md = md["sc2ts"]["exact_matches"]
    exact_matches_md["date"][date] = sum(pango_counts.values())
    pango_counts.update(exact_matches_md["pango"])
    exact_matches_md["pango"] = dict(pango_counts)
    node_counts.update(exact_matches_md["node"])
    exact_matches_md["node"] = dict(node_counts)
    tables.metadata = md
    logger.info(f"Updated exact match counts: {dict(pango_counts)}")
    return tables.tree_sequence()


@dataclasses.dataclass
class GroupTreeQualityMetrics:
    """
    Set of metrics used to assess the quality of an in inferred sample group tree.
    """

    strains: List[str]
    pango_lineages: List[str]
    dates: List[str]
    num_nodes: int
    num_root_mutations: int
    num_mutations: int
    num_recurrent_mutations: int
    depth: int
    date_added: str

    def asdict(self):
        return dataclasses.asdict(self)

    @property
    def num_samples(self):
        return len(self.strains)

    @property
    def mean_mutations_per_sample(self):
        return self.num_mutations / self.num_samples

    def summary(self):
        return (
            f"samples={self.num_samples} "
            f"depth={self.depth} total_muts={self.num_mutations} "
            f"root_muts={self.num_root_mutations} "
            f"muts_per_sample={self.mean_mutations_per_sample} "
            f"recurrent_muts={self.num_recurrent_mutations} "
        )


@dataclasses.dataclass
class SampleGroup:
    """
    A Group of samples that get added into the overall ARG in as
    a "local" tree.
    """

    samples: List = None
    path: List = None
    immediate_reversions: List = None
    additional_keys: Dict = None
    sample_hash: str = None
    tree_quality_metrics: GroupTreeQualityMetrics = None

    def __post_init__(self):
        m = hashlib.md5()
        for strain in sorted(self.strains):
            m.update(strain.encode())
        self.sample_hash = m.hexdigest()

    @property
    def strains(self):
        return [s.strain for s in self.samples]

    @property
    def date_count(self):
        return collections.Counter([s.date for s in self.samples])

    @property
    def pango_count(self):
        return collections.Counter([s.pango for s in self.samples])

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def summary(self):
        return (
            f"{self.sample_hash} n={len(self.samples)} "
            f"{dict(self.date_count)} "
            f"{dict(self.pango_count)} "
            f"immediate_reversions={self.immediate_reversions} "
            f"additional_keys={self.additional_keys} "
            f"path={path_summary(self.path)} "
            f"strains={self.strains}"
        )

    def add_tree_quality_metrics(self, ts, date):
        tree = ts.first()
        assert ts.num_trees == 1
        self.tree_quality_metrics = GroupTreeQualityMetrics(
            strains=self.strains,
            pango_lineages=[s.pango for s in self.samples],
            dates=[s.date for s in self.samples],
            num_nodes=ts.num_nodes,
            num_mutations=ts.num_mutations,
            num_root_mutations=int(np.sum(ts.mutations_node == tree.root)),
            num_recurrent_mutations=int(np.sum(ts.mutations_parent != -1)),
            depth=max(tree.depth(u) for u in ts.samples()),
            date_added=date,
        )
        return self.tree_quality_metrics


def add_matching_results(
    where_clause,
    match_db,
    ts,
    date,
    min_group_size=1,
    min_different_dates=1,
    min_root_mutations=0,
    max_mutations_per_sample=np.inf,
    max_recurrent_mutations=np.inf,
    additional_node_flags=None,
    show_progress=False,
    additional_group_metadata_keys=list(),
    phase=None,
):
    logger.info(f"Querying match DB WHERE: {where_clause}")

    # Group matches by path and set of immediate reversions.
    grouped_matches = collections.defaultdict(list)
    site_missing_samples = np.zeros(ts.num_sites, dtype=int)
    site_deletion_samples = np.zeros(ts.num_sites, dtype=int)
    num_samples = 0
    for sample in match_db.get(where_clause):
        assert all(mut.is_reversion is not None for mut in sample.hmm_match.mutations)
        assert all(
            mut.is_immediate_reversion is not None for mut in sample.hmm_match.mutations
        )
        path = tuple(sample.hmm_match.path)
        immediate_reversions = tuple(
            (mut.site_id, mut.derived_state)
            for mut in sample.hmm_match.mutations
            if mut.is_immediate_reversion
        )
        additional_metadata = [
            sample.metadata.get(k, None) for k in additional_group_metadata_keys
        ]
        key = (path, immediate_reversions, *additional_metadata)
        grouped_matches[key].append(sample)
        num_samples += 1

    if num_samples == 0:
        logger.info("No candidate samples found in MatchDb")
        return ts, []

    groups = [
        SampleGroup(
            samples,
            key[0],
            key[1],
            {k: v for k, v in zip(additional_group_metadata_keys, key[2:])},
        )
        for key, samples in grouped_matches.items()
    ]
    logger.info(f"Got {len(groups)} groups for {num_samples} samples")

    tables = ts.dump_tables()

    attach_nodes = []
    added_groups = []
    with get_progress(groups, date, f"add({phase})", show_progress) as bar:
        for group in bar:
            if (
                len(group) < min_group_size
                or len(group.date_count) < min_different_dates
            ):
                logger.debug(
                    f"Skipping size={len(group)} dates={len(group.date_count)}: "
                    f"{group.summary()}"
                )
                continue
            flat_ts = match_path_ts(group)
            if flat_ts.num_mutations == 0 or flat_ts.num_samples == 1:
                poly_ts = flat_ts
            else:
                binary_ts = tree_ops.infer_binary(flat_ts)
                poly_ts = tree_ops.trim_branches(binary_ts)
            assert poly_ts.num_samples == flat_ts.num_samples
            tqm = group.add_tree_quality_metrics(poly_ts, date)
            if tqm.num_root_mutations < min_root_mutations:
                logger.debug(
                    f"Skipping root_mutations={tqm.num_root_mutations} < threshold "
                    f"{group.summary()}"
                )
                continue
            if tqm.mean_mutations_per_sample > max_mutations_per_sample:
                logger.debug(
                    f"Skipping mean_mutations_per_sample={tqm.mean_mutations_per_sample} "
                    f"exceeds threshold {group.summary()}"
                )
                continue
            if tqm.num_recurrent_mutations > max_recurrent_mutations:
                logger.debug(
                    f"Skipping num_recurrent_mutations={tqm.num_recurrent_mutations} "
                    f"exceeds threshold: {group.summary()}"
                )
                continue
            nodes = attach_tree(ts, tables, group, poly_ts, date, additional_node_flags)
            logger.debug(
                f"Attach {phase} metrics:{tqm.summary()}"
                f"attach_nodes={len(nodes)} "
                f"group={group.summary()}"
            )
            attach_nodes.extend(nodes)
            added_groups.append(group)

            # Update the metadata
            for sample in group:
                missing_sites = np.where(sample.haplotype == MISSING)[0]
                site_missing_samples[missing_sites] += 1
                deletion_sites = np.where(sample.haplotype == DELETION)
                site_deletion_samples[deletion_sites] += 1

    # Update the sites with metadata for these newly added samples.
    tables.sites.clear()
    for site in ts.sites():
        md = site.metadata
        md["sc2ts"]["missing_samples"] += int(site_missing_samples[site.id])
        md["sc2ts"]["deletion_samples"] += int(site_deletion_samples[site.id])
        tables.sites.append(site.replace(metadata=md))

    # NOTE: Doing the parsimony hueristic updates really is complicated a lot
    # by doing all of group batches together. It should be simpler if we reason
    # about *one* tree being added at a time. We might get hit by a high-cost
    # in terms of creating tree sequence objects over and again, but maybe we
    # can do less sorting by thinking more clearly about how to add the edges.
    # If we only add edges pointing to one parent at a time, we should be able
    # to just insert them into the middle of the table?
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ts = tables.tree_sequence()
    ts = tree_ops.push_up_reversions(ts, attach_nodes, date)
    ts = tree_ops.coalesce_mutations(ts, attach_nodes)
    ts = delete_immediate_reversion_nodes(ts, attach_nodes)
    return ts, added_groups


def solve_num_mismatches(k, num_alleles=5):
    """
    Return the low-level LS parameters corresponding to accepting
    k mismatches in favour of a single recombination.

    mu is scaled by the number of distinct alleles.

    The equation is

    r(1 - r)^(m-1) (1 - [a-1] mu)^m = (1 - r)^m (1 - [a-1] mu)^(m - k) mu^k

    Solving for r gives

    r = mu^k / (mu^k + (1 - [a-1] mu)^k

    The LHS is
    1. P[one recombination]
    2. P[not recombining m - 1 times]
    3. P[not mutating m times]

    The RHS is
    1. P[not recombining m times]
    2. P[not mutating m - k times]
    3. P[mutating k times]
    """
    # values of k <= 1 are not relevant for SC2 and lead to awkward corner cases
    assert k > 1
    assert num_alleles > 1

    mu = 0.0125

    denom = mu**k + (1 - (num_alleles - 1) * mu) ** k
    rho = mu**k / denom
    # print("r before", r)
    # Add a tiny bit of extra mass so that we deterministically recombine
    rho += rho * 0.1
    # Don't let rho actually go to zero
    rho = max(rho, 1e-200)
    return mu, rho


def is_full_span(tree, u):
    """
    Returns true if the edge in which the specified node is a child
    covers the full span of the tree sequence.
    """
    ts = tree.tree_sequence
    e = tree.edge(u)
    assert e != -1
    edge = ts.edge(e)
    return edge.left == 0 and edge.right == ts.sequence_length


def delete_immediate_reversion_nodes(ts, attach_nodes):
    tree = ts.first()
    nodes_to_delete = []
    for u in attach_nodes:
        # If a node is a node inserted to track the immediate reversions
        # shared by all the samples in a group, and it covers the full
        # span (because it's easier), and it has no mutations, delete it.
        condition = (
            ts.nodes_flags[u] == core.NODE_IS_IMMEDIATE_REVERSION_MARKER
            and is_full_span(tree, u)
            and all(is_full_span(tree, v) for v in tree.children(u))
            and np.sum(ts.mutations_node == u) == 0
        )
        if condition:
            nodes_to_delete.append(u)

    if len(nodes_to_delete) == 0:
        return ts

    # This is all quite a roundabout way of removing a node from the
    # tree we shouldn't be adding in the first place. There must be a
    # better way.
    tables = ts.dump_tables()
    edges_to_delete = []
    for u in nodes_to_delete:
        logger.debug(f"Deleting immediate reversion node {u}")
        edges_to_delete.append(tree.edge(u))
        parent = tree.parent(u)
        assert tree.num_children(u) > 0
        for v in tree.children(u):
            e = tree.edge(v)
            tables.edges[e] = ts.edge(e).replace(parent=parent)

    keep_edges = np.ones(ts.num_edges, dtype=bool)
    keep_edges[edges_to_delete] = 0
    tables.edges.keep_rows(keep_edges)
    keep_nodes = np.ones(ts.num_nodes, dtype=bool)
    keep_nodes[nodes_to_delete] = 0
    node_map = tables.nodes.keep_rows(keep_nodes)
    tables.edges.child = node_map[tables.edges.child]
    tables.edges.parent = node_map[tables.edges.parent]
    tables.mutations.node = node_map[tables.mutations.node]
    tables.sort()
    tables.build_index()
    logger.debug(f"Deleted {len(nodes_to_delete)} immediate reversion nodes")
    return tables.tree_sequence()


def make_tsb(ts, num_alleles, mirror_coordinates=False):
    if mirror_coordinates:
        # TODO inline this conversion here because we're doing an additional
        # sort
        ts = mirror_ts_coordinates(ts)

    tables = ts.tables
    assert np.all(tables.sites.ancestral_state_offset == np.arange(ts.num_sites + 1))
    ancestral_state = alignments.encode_alignment(
        tables.sites.ancestral_state.view("S1").astype(str)
    )
    assert np.all(
        tables.mutations.derived_state_offset == np.arange(ts.num_mutations + 1)
    )
    derived_state = alignments.encode_alignment(
        tables.mutations.derived_state.view("S1").astype(str)
    )
    del tables

    tsb = _tsinfer.TreeSequenceBuilder(
        num_alleles=np.full(ts.num_sites, num_alleles, dtype=np.uint64),
        max_nodes=ts.num_nodes,
        max_edges=ts.num_edges,
        ancestral_state=ancestral_state,
    )

    position_map = np.hstack([ts.sites_position, [ts.sequence_length]])
    # bracketing by 0 on the left here while we're translating edge locations.
    position_map[0] = 0
    # Get the indexes into the position array.
    left = np.searchsorted(position_map, ts.edges_left)
    if np.any(position_map[left] != ts.edges_left):
        raise ValueError("Invalid left coordinates")
    right = np.searchsorted(position_map, ts.edges_right)
    if np.any(position_map[right] != ts.edges_right):
        raise ValueError("Invalid right coordinates")

    position_map[0] = ts.sites_position[0]
    # Need to sort by child ID here and left so that we can efficiently
    # insert the child paths.
    index = np.lexsort((left, ts.edges_child))
    tsb.restore_nodes(ts.nodes_time, ts.nodes_flags)
    tsb.restore_edges(
        left[index].astype(np.int32),
        right[index].astype(np.int32),
        ts.edges_parent[index],
        ts.edges_child[index],
    )
    assert tsb.num_match_nodes == ts.num_nodes

    tsb.restore_mutations(
        ts.mutations_site, ts.mutations_node, derived_state, ts.mutations_parent
    )
    return tsb, position_map.astype(int)


class MatchingManager:
    def __init__(self, tsb, work, num_threads, progress_bar, memory_limit):
        self.tsb = tsb
        self.num_threads = max(num_threads, 0)
        self.matchers = [None for _ in range(max(num_threads, 1))]
        self.matchers_lock = threading.Lock()
        # This is a thread-safe operation, so we don't need locks on the
        # work and results lists.
        self.work = collections.deque(work)
        self.results = collections.deque()
        self.progress_bar = progress_bar
        self.memory_limit = 2**64 if memory_limit <= 0 else memory_limit
        if num_threads > 0:
            self.threads = [
                threading.Thread(target=self.match_worker, args=(j,))
                for j in range(num_threads)
            ]

    def run(self):
        if self.num_threads == 0:
            logger.debug(f"Running {len(self.work)} matches synchronously")
            for w in self.work:
                self.run_match(w, 0)
        else:
            for thread in self.threads:
                thread.start()
            logger.debug(
                f"Running {len(self.work)} matches in {len(self.threads)} threads"
            )
            for j in range(self.num_threads):
                self.threads[j].join()
                self.threads[j] = None
            logger.debug("Match worker threads completed")
        self.progress_bar.close()

    def run_match(self, work, thread_index):

        h = work.haplotype
        num_sites = len(h)
        matcher = _tsinfer.AncestorMatcher(
            self.tsb,
            recombination=np.full(num_sites, work.rho),
            mismatch=np.full(num_sites, work.mu),
            likelihood_threshold=work.likelihood_threshold,
        )
        with self.matchers_lock:
            assert self.matchers[thread_index] is None
            self.matchers[thread_index] = matcher

        is_missing = h == MISSING
        m = np.full(num_sites, MISSING, dtype=np.int8)

        before = time.thread_time()
        match_path = matcher.find_path(h, 0, num_sites, m)
        duration = time.thread_time() - before

        path = []
        for left, right, parent in zip(*match_path):
            path.append(PathSegment(int(left), int(right), int(parent)))
        mutations = []
        # Mask out the imputed sites
        m[is_missing] = MISSING
        for site_id in np.where(h != m)[0]:
            derived_state = core.ALLELES[h[site_id]]
            inherited_state = core.ALLELES[m[site_id]]
            mutations.append(
                MatchMutation(
                    site_id=int(site_id),
                    derived_state=derived_state,
                    inherited_state=inherited_state,
                )
            )

        path_len = len(match_path[0])
        num_muts = np.sum(m != h)
        likelihood = work.rho ** (path_len - 1) * work.mu**num_muts
        cost = work.num_mismatches * (path_len - 1) + num_muts
        hmm_match = HmmMatch(path, mutations, likelihood=likelihood, cost=cost)

        logger.debug(
            f"Found path len={path_len} and muts={num_muts} L={likelihood:.2g} "
            f"(L_t={work.likelihood_threshold:.2g}) "
            f"for {work.strain} in {duration:.3f}s "
            f"mean_tb_size={matcher.mean_traceback_size:.1f} "
            f"match_mem={humanize.naturalsize(matcher.total_memory, binary=True)}"
        )
        with self.matchers_lock:
            self.matchers[thread_index] = None
        self.results.append((work, hmm_match))
        self.progress_bar.update()

    def total_matcher_memory(self):
        total_memory = 0
        active_matchers = 0
        with self.matchers_lock:
            for matcher in self.matchers:
                if matcher is not None:
                    active_matchers += 1
                    total_memory += matcher.total_memory
        logger.debug(
            f"Total matcher memory (active={active_matchers})="
            f"{humanize.naturalsize(total_memory, binary=True)}"
        )
        return total_memory

    def match_worker(self, thread_index):
        """
        Start the match worker, and read work from the queue until completed.
        """
        logger.debug(f"Starting match worker thread {thread_index}")
        while True:
            if len(self.work) == 0:
                logger.debug(f"Thread {thread_index} work done, exiting")
                return
            work = self.work.popleft()
            wait_count = 0
            while self.total_matcher_memory() > self.memory_limit:
                logger.debug(
                    f"Thread {thread_index} Over memory budget: waiting {wait_count}"
                )
                time.sleep(1)
                wait_count += 1
            logger.debug(f"Starting match {work.strain} in thread {thread_index}")
            self.run_match(work, thread_index)


@dataclasses.dataclass(frozen=True)
class MatchWork:
    strain: str
    haplotype: List
    num_mismatches: int
    mu: float
    rho: float
    likelihood_threshold: float


def match_tsinfer(
    samples,
    ts,
    *,
    num_mismatches,
    mismatch_threshold=None,
    deletions_as_missing=False,
    num_threads=0,
    memory_limit=-1,
    show_progress=False,
    progress_title=None,
    progress_phase=None,
    mirror_coordinates=False,
):

    num_alleles = 4 if deletions_as_missing else 5
    mu, rho = solve_num_mismatches(num_mismatches, num_alleles)

    tsb, coord_map = make_tsb(ts, num_alleles, mirror_coordinates)

    work = []
    for sample in samples:
        h = sample.haplotype.copy()
        if mirror_coordinates:
            h = h[::-1]
        if deletions_as_missing:
            h[h == DELETION] = MISSING
        if mismatch_threshold is not None:
            # Likelihood threshold is slightly less than k mutations
            likelihood_threshold = mu**mismatch_threshold * 0.99
        else:
            assert sample.hmm_match is not None
            likelihood_threshold = sample.hmm_match.likelihood

        work.append(
            MatchWork(
                strain=sample.strain,
                haplotype=h,
                num_mismatches=num_mismatches,
                mu=mu,
                rho=rho,
                likelihood_threshold=likelihood_threshold,
            )
        )

    bar = get_progress(work, progress_title, progress_phase, show_progress)
    manager = MatchingManager(tsb, work, num_threads, bar, memory_limit)
    manager.run()
    results = {work.strain: hmm_match for work, hmm_match in manager.results}

    for sample in samples:
        raw_hmm_match = results[sample.strain]
        sample.hmm_match = raw_hmm_match.translate_coordinates(
            coord_map, mirror_coordinates, ts.sites_position
        )
        logger.debug(
            f"HMM@T={mismatch_threshold}: {sample.strain} "
            f"hmm_cost={sample.hmm_match.cost} match={sample.hmm_match.summary()}"
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


@dataclasses.dataclass
class MatchMutation:
    site_id: int
    derived_state: str
    inherited_state: str
    site_position: int = None
    is_reversion: bool = None
    is_immediate_reversion: bool = None

    def __str__(self):
        return f"{self.inherited_state}{int(self.site_position)}{self.derived_state}"

    def asdict(self):
        return {
            "site_position": int(self.site_position),
            "derived_state": self.derived_state,
            "inherited_state": self.inherited_state,
        }


def path_summary(path):
    return ", ".join(f"({seg.left}:{seg.right}, {seg.parent})" for seg in path)


@dataclasses.dataclass
class HmmMatch:
    path: List[PathSegment]
    mutations: List[MatchMutation]
    likelihood: float = None
    cost: float = None

    def asdict(self):
        return {
            "path": [x.asdict() for x in self.path],
            "mutations": [x.asdict() for x in self.mutations],
        }

    def summary(self):
        return (
            f"path={self.path_summary()} "
            f"mutations({len(self.mutations)})"
            f"={self.mutation_summary()}"
        )

    @property
    def breakpoints(self):
        breakpoints = [seg.left for seg in self.path]
        return breakpoints + [self.path[-1].right]

    @property
    def parents(self):
        return [seg.parent for seg in self.path]

    def compute_cost(self, num_mismatches):
        self.cost = num_mismatches * (len(self.path) - 1) + len(self.mutations)

    def path_summary(self):
        return path_summary(self.path)

    def mutation_summary(self):
        return "[" + ", ".join(str(mutation) for mutation in self.mutations) + "]"

    def translate_coordinates(self, coord_map, mirror_coordinates, sites_position):
        """
        Return a copy of this HmmMatch with coordinates translated from raw site-based
        values to their final positions.
        """
        L = coord_map[-1]
        first_pos = coord_map[0]
        # Set first pos to 0 while we're translating edge coords
        coord_map[0] = 0
        path = []
        for seg in self.path:
            if mirror_coordinates:
                left_pos = int(mirror(coord_map[seg.right], L))
                right_pos = int(mirror(coord_map[seg.left], L))
            else:
                left_pos = int(coord_map[seg.left])
                right_pos = int(coord_map[seg.right])
            path.append(PathSegment(left_pos, right_pos, int(seg.parent)))

        if not mirror_coordinates:
            # tsinfer returns the path right-to-left, which we reverse
            # if matching forwards
            path = path[::-1]

        coord_map[0] = first_pos
        mutations = []
        for mut in self.mutations:
            site_id = mut.site_id
            if mirror_coordinates:
                site_id = mirror(mut.site_id, len(coord_map) - 2)
            mutations.append(
                MatchMutation(
                    site_id=int(site_id),
                    site_position=int(sites_position[site_id]),
                    derived_state=mut.derived_state,
                    inherited_state=mut.inherited_state,
                )
            )
        if mirror_coordinates:
            mutations = mutations[::-1]
        return HmmMatch(path, mutations, self.likelihood, self.cost)


def characterise_match_mutations(ts, samples):
    """
    Update the hmm matches for each of the samples in place so that we characterise
    reversions and immediate_reversions.
    """
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
            cache[(node, site_id)] = site.ancestral_state, closest

        return cache[(node, site_id)]

    # Note: this algorithm is pretty dumb - we should sort all the mutations by
    # left coordinate and do this in a single pass through the trees.
    num_mutations = 0
    for sample in samples:
        for mutation in sample.hmm_match.mutations:
            num_mutations += 1
            node = None
            for seg in sample.hmm_match.path:
                if seg.left <= mutation.site_position < seg.right:
                    node = seg.parent
                    break
            assert node is not None

            ancestral_state, closest_mutation = get_closest_mutation(
                node, mutation.site_id
            )
            mutation.is_reversion = False
            mutation.is_immediate_reversion = False
            if closest_mutation is not None:
                inherited_state = closest_mutation.derived_state
                parent_inherited_state = ancestral_state
                if closest_mutation.parent != -1:
                    grandparent_mutation = ts.mutation(closest_mutation.parent)
                    parent_inherited_state = grandparent_mutation.derived_state
                mutation.is_reversion = parent_inherited_state == mutation.derived_state
                if mutation.is_reversion:
                    mutation.is_immediate_reversion = (
                        closest_mutation.node == seg.parent
                    )
    logger.debug(f"Characterised {num_mutations}")


def attach_tree(
    parent_ts,
    parent_tables,
    group,
    child_ts,
    date,
    additional_node_flags,
    epsilon=None,
):
    attach_path = group.path
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
    has_root_mutations = np.any(child_ts.mutations_node == tree.root)
    condition = (
        has_root_mutations
        or len(attach_path) > 1
        or len(group.immediate_reversions) > 0
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
            metadata = {
                "sc2ts": {
                    "group_id": group.sample_hash,
                    "date_added": date,
                }
            }
        new_id = parent_tables.nodes.append(
            node.replace(
                flags=node.flags | additional_node_flags, time=time, metadata=metadata
            )
        )
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
                metadata={
                    "sc2ts": {"type": "parsimony", "group_id": group.sample_hash}
                },
            )

    if len(group.immediate_reversions) > 0:
        # Flag the node as an NODE_IS_IMMEDIATE_REVERSION_MARKER, which we've
        # added as a unary above-the-root note above.
        # This should be removed, along with the mutations we're adding here by
        # push_up_reversions in all cases except recombinants (which we've wussed
        # out on handling properly).
        # This is all very roundabout, and we're also missing the opportunity
        # to remove any non-immediate reversions if they exist withing
        # the local tree group.
        node = tree.children(tree.root)[0]
        assert tree.num_children(tree.root) == 1
        u = node_id_map[node]
        row = parent_tables.nodes[u]
        if not has_root_mutations:
            # We only really want to remove this if there are no nodes
            logger.debug(f"Flagging reversion at node {u} for {group.summary()}")
            parent_tables.nodes[u] = row.replace(
                flags=core.NODE_IS_IMMEDIATE_REVERSION_MARKER
            )
        # print("attaching reversions at ", node, node_id_map[node])
        # print(child_ts.draw_text())
        for site_id, derived_state in group.immediate_reversions:
            parent_tables.mutations.add_row(
                site=site_id,
                node=u,
                derived_state=derived_state,
                time=node_time[node],
                metadata={
                    "sc2ts": {"type": "match_reversion", "group_id": group.sample_hash}
                },
            )

    if len(attach_path) > 1:
        # Update the recombinant flags also.
        u = node_id_map[tree.children(tree.root)[0]]
        assert tree.num_children(tree.root) == 1
        node = parent_tables.nodes[u]
        parent_tables.nodes[u] = node.replace(flags=core.NODE_IS_RECOMBINANT)
    return [node_id_map[u] for u in tree.children(tree.root)]


def add_root_edge(ts, flags=0):
    """
    Add another node and edge above the root and rescale time back to
    0-1.
    """
    assert ts.num_trees == 1
    tables = ts.dump_tables()
    root = ts.first().root
    # FIXME this is bogus. We should be doing all the time scaling by numbers
    # of mutations.
    new_root = tables.nodes.add_row(time=1.25, flags=flags)
    tables.edges.add_row(0, ts.sequence_length, parent=new_root, child=root)
    tables.nodes.time /= np.max(tables.nodes.time)
    return tables.tree_sequence()


def get_group_strains(ts):
    """
    Returns the strain IDs for samples gathered by sample group ID.
    """
    groups = collections.defaultdict(list)
    for u in ts.samples():
        md = ts.node(u).metadata
        group_id = md["sc2ts"].get("group_id", None)
        if group_id is not None:
            groups[group_id].append(md["strain"])
    return groups


def get_recombinant_strains(ts):
    """
    Returns a map of recombinant node ID to the strains originally associated
    with it.
    """
    groups = get_group_strains(ts)
    recombinants = np.where(ts.nodes_flags & core.NODE_IS_RECOMBINANT > 0)[0]
    ret = {}
    for u in recombinants:
        node = ts.node(u)
        group_id = node.metadata["sc2ts"]["group_id"]
        ret[u] = groups[group_id]
    return ret
