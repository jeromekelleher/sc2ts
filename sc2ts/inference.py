from __future__ import annotations
import bz2
import logging
import datetime
import dataclasses
import collections
import concurrent.futures
import pickle
import hashlib
import sqlite3
import pathlib
import random

import tqdm
import tskit
import tsinfer
import numpy as np
import zarr
import numba
import pandas as pd


from . import core
from . import alignments
from . import metadata
from . import tree_ops

logger = logging.getLogger(__name__)

MISSING = -1
DELETION = core.ALLELES.index("-")


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
            hmm_cost[j] = sample.hmm_match.get_hmm_cost(num_mismatches)
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
    # TODO gene annotations to top level
    tables.metadata = {
        "sc2ts": {
            "date": core.REFERENCE_DATE,
            "samples_strain": [core.REFERENCE_STRAIN],
            "num_exact_matches": {},
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
    date: str = "2020-01-01"
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


def match_recombinants(
    samples, base_ts, num_mismatches, show_progress=False, num_threads=None
):
    mu, rho = solve_num_mismatches(num_mismatches)
    for hmm_pass in ["forward", "reverse", "no_recombination"]:
        logger.info(f"Running {hmm_pass} pass for {len(samples)} recombinants")
        matches = match_tsinfer(
            samples=samples,
            ts=base_ts,
            mu=mu,
            rho=1e-30 if hmm_pass == "no_recombination" else rho,
            num_threads=num_threads,
            show_progress=show_progress,
            # Maximum possible precision
            likelihood_threshold=1e-200,
            mirror_coordinates=hmm_pass == "reverse",
        )
        for hmm_match, sample in zip(matches, samples):
            sample.hmm_reruns[hmm_pass] = hmm_match

    for sample in samples:
        # We may want to try to improve the location of the breakpoints
        # later. For now, just log the info.
        logger.info(f"Recombinant: {sample.summary()}")


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
        hmm_matches = match_tsinfer(
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
        for sample, hmm_match in zip(run_batch, hmm_matches):
            cost = hmm_match.get_hmm_cost(num_mismatches)
            logger.debug(
                f"HMM@k={k}: {sample.strain} hmm_cost={cost} match={hmm_match.summary()}"
            )
            if cost > k + 1:
                exceeding_threshold.append(sample)
            else:
                sample.hmm_match = hmm_match

        num_matches_found = len(run_batch) - len(exceeding_threshold)
        logger.info(
            f"{num_matches_found} final matches found at k={k}; "
            f"{len(exceeding_threshold)} remain"
        )
        run_batch = exceeding_threshold

    logger.info(f"Running final batch of {len(run_batch)} at high precision")
    hmm_matches = match_tsinfer(
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
    for sample, hmm_match in zip(run_batch, hmm_matches):
        sample.hmm_match = hmm_match
        cost = hmm_match.get_hmm_cost(num_mismatches)
        # print(f"Final HMM pass:{sample.strain} hmm_cost={cost} {sample.summary()}")
        logger.debug(f"Final HMM pass hmm_cost={cost} {sample.summary()}")
    return samples


def check_base_ts(ts):
    md = ts.metadata
    assert "sc2ts" in md
    sc2ts_md = md["sc2ts"]
    assert "date" in sc2ts_md
    assert len(sc2ts_md["samples_strain"]) == ts.num_samples


def preprocess_worker(strains, alignment_store_path, keep_sites):
    # print("preprocess worker", samples_md)
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
    date,
    alignment_store_path,
    keep_sites=None,
    show_progress=False,
    num_workers=0,
):
    num_workers = max(1, num_workers)
    splits = min(len(strains), 2 * num_workers)
    work = np.array_split(strains, splits)
    samples = []

    bar = get_progress(strains, date, f"preprocess", show_progress)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(preprocess_worker, w, alignment_store_path, keep_sites)
            for w in work
        ]
        for future in concurrent.futures.as_completed(futures):
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
    num_mismatches=None,
    hmm_cost_threshold=None,
    min_group_size=None,
    min_root_mutations=None,
    max_daily_samples=None,
    show_progress=False,
    retrospective_window=None,
    max_missing_sites=None,
    random_seed=42,
    num_threads=0,
):
    if num_mismatches is None:
        num_mismatches = 3
    if hmm_cost_threshold is None:
        hmm_cost_threshold = 5
    if min_group_size is None:
        min_group_size = 10
    if min_root_mutations is None:
        min_root_mutations = 2
    if retrospective_window is None:
        retrospective_window = 30
    if max_missing_sites is None:
        max_missing_sites = np.inf

    check_base_ts(base_ts)
    logger.info(
        f"Extend {date}; ts:nodes={base_ts.num_nodes};samples={base_ts.num_samples};"
        f"mutations={base_ts.num_mutations};date={base_ts.metadata['sc2ts']['date']}"
    )

    metadata_matches = list(metadata_db.get(date))

    logger.info(f"Got {len(metadata_matches)} metadata matches")

    preprocessed_samples = preprocess(
        strains=[md["strain"] for md in metadata_matches],
        date=date,
        alignment_store_path=alignment_store.path,
        keep_sites=base_ts.sites_position.astype(int),
        show_progress=show_progress,
        num_workers=num_threads,
    )
    # FIXME parametrise
    pango_lineage_key = "Viridian_pangolin"

    samples = []
    for s, md in zip(preprocessed_samples, metadata_matches):
        if s.haplotype is None:
            logger.debug(f"No alignment stored for {s.strain}")
            continue
        s.metadata = md
        s.pango = md.get(pango_lineage_key, "Unknown")
        s.date = date
        num_missing_sites = s.num_missing_sites
        num_deletion_sites = s.num_deletion_sites
        logger.debug(
            f"Encoded {s.strain} {s.pango} missing={num_missing_sites} "
            f"deletions={num_deletion_sites}"
        )
        if num_missing_sites <= max_missing_sites:
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
        min_different_dates=3,  # TODO parametrise
        additional_group_metadata_keys=["Country"],  # TODO parametrise
        min_root_mutations=min_root_mutations,
        additional_node_flags=core.NODE_IN_RETROSPECTIVE_SAMPLE_GROUP,
        show_progress=show_progress,
        phase="retro",
    )
    for group in groups:
        logger.warning(f"Add retro group {dict(group.pango_count)}")
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
    nodes_num_exact_matches = np.zeros(ts.num_nodes, dtype=int)
    pango_counts = collections.Counter()
    for sample in samples:
        assert len(sample.hmm_match.path) == 1
        assert len(sample.hmm_match.mutations) == 0
        parent = sample.hmm_match.path[0].parent
        logger.debug(f"Increment exact match {sample.strain}->{parent}")
        nodes_num_exact_matches[parent] += 1
        pango_counts[sample.pango] += 1
        # node_id = add_sample_to_tables(
        #     sample,
        #     tables,
        #     flags=tskit.NODE_IS_SAMPLE | core.NODE_IS_EXACT_MATCH,
        # )
        # logger.debug(f"ARG add exact match {sample.strain}:{node_id}->{parent}")
        # tables.edges.add_row(0, ts.sequence_length, parent=parent, child=node_id)
    logger.info(f"Updating exact match counts: {dict(pango_counts)}")
    tables = ts.dump_tables()
    for u in np.where(nodes_num_exact_matches > 0)[0]:
        row = tables.nodes[u]
        md = row.metadata
        if "num_exact_matches" not in md["sc2ts"]:
            md["sc2ts"]["num_exact_matches"] = 0
        md["sc2ts"]["num_exact_matches"] += int(nodes_num_exact_matches[u])
        tables.nodes[u] = row.replace(metadata=md)
    md = tables.metadata
    pango_counts.update(md["sc2ts"]["num_exact_matches"])
    md["sc2ts"]["num_exact_matches"] = dict(pango_counts)
    tables.metadata = md
    return tables.tree_sequence()


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


def add_matching_results(
    where_clause,
    match_db,
    ts,
    date,
    min_group_size=1,
    min_different_dates=1,
    min_root_mutations=0,
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
            tree = poly_ts.first()
            num_root_mutations = np.sum(poly_ts.mutations_node == tree.root)
            num_recurrent_mutations = np.sum(poly_ts.mutations_parent != -1)
            if num_root_mutations < min_root_mutations:
                logger.debug(
                    f"Skipping root_mutations={num_root_mutations}: "
                    f"{group.summary()}"
                )
                continue
            attach_depth = max(tree.depth(u) for u in poly_ts.samples())
            nodes = attach_tree(ts, tables, group, poly_ts, date, additional_node_flags)
            logger.debug(
                f"Attach {phase} "
                f"depth={attach_depth} total_muts={poly_ts.num_mutations} "
                f"root_muts={num_root_mutations} "
                f"recurrent_muts={num_recurrent_mutations} attach_nodes={len(nodes)} "
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
        return []
    genotypes = np.array([sample.haplotype for sample in samples], dtype=np.int8).T
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

    return get_match_info(input_ts, sample_paths, sample_mutations)


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


def path_summary(path):
    return ", ".join(f"({seg.left}:{seg.right}, {seg.parent})" for seg in path)


@dataclasses.dataclass(frozen=True)
class HmmMatch:
    path: List[PathSegment]
    mutations: List[MatchMutation]

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

    def get_hmm_cost(self, num_mismatches):
        return num_mismatches * (len(self.path) - 1) + len(self.mutations)

    def path_summary(self):
        return path_summary(self.path)

    def mutation_summary(self):
        return "[" + ", ".join(str(mutation) for mutation in self.mutations) + "]"


def get_match_info(ts, sample_paths, sample_mutations):
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

    matches = []
    for path, mutations in zip(sample_paths, sample_mutations):
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
        matches.append(HmmMatch(sample_path, sample_mutations))
    return matches


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
