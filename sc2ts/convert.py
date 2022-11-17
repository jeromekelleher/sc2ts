import logging
import sqlite3
import pathlib
import collections

import pyfasta
import tqdm
import tsinfer
import pandas as pd
import numpy as np

from . import core

logger = logging.getLogger(__name__)


def group_by_date(strains, conn):
    by_date = collections.defaultdict(list)

    def dict_factory(cursor, row):
        col_names = [col[0] for col in cursor.description]
        return {key: value for key, value in zip(col_names, row)}

    conn.row_factory = dict_factory
    for strain in strains:
        logger.debug(f"Getting metadata for {strain}")
        res = conn.execute("SELECT * FROM samples WHERE strain=?", (strain,))
        row = res.fetchone()
        # We get a string back with the hours, split to just date
        if row is None:
            logger.warning(f"No metadata for {strain}; skipping")
        else:
            date = row["date"].split()[0]
            by_date[date].append(row)
    return by_date


ALLELES = "ACGT-"
# This is the length of the reference, i.e. the last coordinate in 1-based
# TODO move to constants
REFERENCE_LENGTH = 29903


def mask_flank_deletions(a):
    """
    Update the to replace flanking deletions ("-") with missing data ("N").
    """
    n = a.shape[0]
    j = 0
    while j < n and a[j] == "-":
        a[j] = "N"
        j += 1
    left = j
    j = n - 1
    while j >= 0 and a[j] == "-":
        a[j] = "N"
        j -= 1
    right = n - j - 1
    return left, right


def get_haplotype(fasta, key):
    a = np.array(fasta[key]).astype(str)
    left_mask, right_mask = mask_flank_deletions(a)
    # Map anything that's not ACGT- to N
    b = np.full(a.shape, -1, dtype=np.int8)
    for code, char in enumerate(ALLELES):
        b[a == char] = code
    return np.append([-2], b), left_mask, right_mask


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
        h, left_mask, right_mask = get_haplotype(fasta, strain)
        assert h.shape[0] == L + 1
        G[:, j] = h[keep_mask]
        num_missing = int(np.sum(h[keep_mask] == -1))
        row["num_missing_sites"] = num_missing
        row["masked_flanks"] = (left_mask, right_mask)
        sample_data.add_individual(metadata=row)
        logger.info(
            f"Add {strain} missing={num_missing} "
            f"masked_flanks={(left_mask, right_mask)}"
        )

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

    reference = core.get_reference_sequence()
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


def verify_sample_data(sample_data, fasta_path):
    """
    Verify that each sample in the specified samples file correcly encodes the
    alignment in the specified file.
    """
    # Load the fasta in a 2D array

    fasta = pyfasta.Fasta(fasta_path, record_class=pyfasta.MemoryRecord)
    H = np.zeros(
        (sample_data.num_samples, int(sample_data.sequence_length)), dtype="U1"
    )
    H[:, 0] = "X"
    for ind in sample_data.individuals():
        strain = ind.metadata["strain"]
        H[ind.id, 1:] = fasta[strain]
    del fasta

    print("Num sequences:", H.shape[0])

    identical = 0
    for var in sample_data.variants():
        g1 = np.array(list(var.alleles) + ["N"])[var.genotypes]
        pos = int(var.site.position)
        g2 = H[:, pos]
        if np.all(g1 == g2):
            identical += 1
        # TODO check the different sites and verify they're not ACGT-

    print("identical", identical, identical / sample_data.num_sites)
