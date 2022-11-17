import logging
import sqlite3
import pathlib
import collections

import pyfasta
import tqdm
import tsinfer
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def group_by_date(strains, conn):
    by_date = collections.defaultdict(list)

    def dict_factory(cursor, row):
        col_names = [col[0] for col in cursor.description]
        return {key: value for key, value in zip(col_names, row)}

    conn.row_factory = dict_factory
    for strain in strains:
        res = conn.execute(f"SELECT * FROM samples where strain='{strain}'")
        row = res.fetchone()
        # We get a string back with the hours, split to just date
        date = row["date"].split()[0]
        by_date[date].append(row)
    return by_date


ALLELES = "ACGT-"
# This is the length of the reference, i.e. the last coordinate in 1-based
# TODO move to constants
REFERENCE_LENGTH = 29903


def get_haplotype(fasta, key):
    a = np.array(fasta[key]).astype(str)
    # Map anything that's not ACGT- to N
    b = np.full(a.shape, -1, dtype=np.int8)
    for code, char in enumerate(ALLELES):
        b[a == char] = code
    return np.append([-2], b)


def convert_alignments(reference, fasta, rows, sample_data):

    # TODO package data path
    L = REFERENCE_LENGTH
    data_path = pathlib.Path("sc2ts/data")

    problematic_sites = np.loadtxt(data_path / "problematic_sites.txt", dtype=np.int64)
    assert L in problematic_sites
    keep_sites = np.array(list(set(np.arange(1, L + 1)) - set(problematic_sites)))
    keep_sites.sort()
    keep_mask = np.ones(L + 1, dtype=bool)
    keep_mask[problematic_sites] = False
    keep_mask[0] = False

    num_sites = L - len(problematic_sites)
    assert num_sites == len(keep_sites)
    G = np.zeros((num_sites, len(rows)), dtype=np.int8)

    bar = tqdm.tqdm(
        enumerate(rows), desc="Reading", total=len(rows), position=1, leave=False
    )
    for j, row in bar:
        strain = row["strain"]
        h = get_haplotype(fasta, strain)
        assert h.shape[0] == L + 1
        G[:, j] = h[keep_mask]
        row["num_missing_sites"] = int(np.sum(h[keep_mask] == -1))
        sample_data.add_individual(metadata=row)

    bar = tqdm.tqdm(range(num_sites), desc="Writing", position=1, leave=False)
    for j in bar:
        pos = keep_sites[j]
        ref_allele = reference[pos]
        sample_data.add_site(
            pos,
            genotypes=G[j],
            alleles=ALLELES,
            ancestral_allele=ALLELES.index(ref_allele),
        )


def alignments_to_samples(fasta_path, metadata_path, output_dir, show_progress=False):
    logger.info(f"Loading fasta from {fasta_path}")
    fasta = pyfasta.Fasta(fasta_path, record_class=pyfasta.MemoryRecord)
    output_dir = pathlib.Path(output_dir)

    data_path = pathlib.Path("sc2ts/data")
    ref_fasta = pyfasta.Fasta(
        str(data_path / "reference.fasta"), record_class=pyfasta.MemoryRecord
    )
    a = np.array(ref_fasta["MN908947 (Wuhan-Hu-1/2019)"]).astype(str)
    reference = np.append(["X"], a)
    del ref_fasta

    strains = list(fasta.keys())
    logger.info(f"Grouping {len(strains)} strains by date")
    with sqlite3.connect(metadata_path) as conn:
        strains_by_date = group_by_date(strains, conn)

    bar = tqdm.tqdm(sorted(strains_by_date.keys()))
    for date in bar:
        bar.set_description(date)
        rows = strains_by_date[date]
        logger.info(f"Converting for {len(rows)} strains for {date}")
        samples_path = output_dir / f"{date}.samples"
        with tsinfer.SampleData(
            path=str(samples_path),
            sequence_length=REFERENCE_LENGTH + 1,
            num_flush_threads=4,
        ) as sd:
            convert_alignments(reference, fasta, rows, sd)


def metadata_to_db(csv_path, db_path):

    df = pd.read_csv(
        csv_path,
        sep="\t",
        parse_dates=["date", "date_submitted"],
    )
    db_path = pathlib.Path(db_path)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        df.to_sql("samples", conn, index=False)
        conn.execute("CREATE INDEX [ix_samples_strain] on 'samples' ([strain]);")
        conn.execute("CREATE INDEX [ix_samples_date] on 'samples' ([date]);")
