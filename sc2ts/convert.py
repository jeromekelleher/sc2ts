import logging
import sqlite3
import pathlib
import collections
import datetime
import hashlib

import numba
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


# This is the length of the reference, i.e. the last coordinate in 1-based
# TODO move to constants
REFERENCE_LENGTH = 29903

GAP = core.ALLELES.index("-")
MISSING = -1


@numba.njit
def mask_alignment(a, start=0, window_size=7):
    """
    Following the approach in fa2vcf, if any base is has two or more ambiguous
    or gap characters with distance window_size of it, mark it as missing data.
    """
    if window_size < 1:
        raise ValueError("Window must be >= 1")
    b = a.copy()
    n = len(a)
    masked = np.zeros(n, dtype=np.int32)
    for j in range(start, n):
        ambiguous = 0
        k = j - 1
        while k >= start and k >= j - window_size:
            if b[k] == GAP or b[k] == MISSING:
                ambiguous += 1
            k -= 1
        k = j + 1
        while k < n and k <= j + window_size:
            if b[k] == GAP or b[k] == MISSING:
                ambiguous += 1
            k += 1
        if ambiguous > 1:
            a[j] = MISSING
            masked[j] = 1
    return masked


def encode_alignment(h):
    # Map anything that's not ACGT- to N
    a = np.full(h.shape, -1, dtype=np.int8)
    for code, char in enumerate(core.ALLELES):
        a[h == char] = code
    return a


def decode_alignment(a):
    if np.any(a < -1) or np.any(a >= len(core.ALLELES)):
        raise ValueError("Cannot decode alignment")
    alleles = np.array(list(core.ALLELES + "N"), dtype="U1")
    return alleles[a]


def base_composition(haplotype):
    return collections.Counter(haplotype)


def convert_alignments(
    samples, fasta, *, show_progress=False, provenance=None, **kwargs
):
    """
    Convert the alignments for specified list of samples (a list of metadata
    dictionaries) from the specified fasta and add them to a tsinfer SampleData
    file created with the specified (additional) kwargs.
    """
    reference = core.get_reference_sequence()
    problematic_sites = core.get_problematic_sites()

    L = len(reference)
    assert L - 1 in problematic_sites
    keep_sites = np.ones(L, dtype=bool)
    keep_sites[problematic_sites] = False
    keep_sites[0] = False

    num_sites = L - len(problematic_sites) - 1
    num_samples = len(samples)
    G = np.zeros((num_sites, num_samples), dtype=np.int8)

    with tsinfer.SampleData(sequence_length=L, **kwargs) as sd:
        # Sort sequences by strain so we have a well defined hash.
        samples = sorted(samples, key=lambda x: x["strain"])
        bar = tqdm.tqdm(
            enumerate(samples),
            desc="Reading",
            total=num_samples,
            position=1,
            leave=False,
            disable=not show_progress,
        )
        masked_per_site = np.zeros(L, dtype=int)
        for j, sample_info in bar:
            # Take a copy so we're not modifying our parameters
            md = dict(sample_info)
            strain = md["strain"]
            alignment = fasta[strain]

            assert alignment.shape[0] == L
            encoded = encode_alignment(alignment)
            # Update the encoded alignment to mask out regions with uncertainty
            masked = mask_alignment(encoded, start=1, window_size=7)
            masked_per_site += masked
            inference_subset = encoded[keep_sites]
            G[:, j] = inference_subset
            # Add some QC measures to the metadata.
            total_masked = int(np.sum(masked))
            masked_outside_problematic = int(np.sum(masked[keep_sites]))
            num_missing = int(np.sum(inference_subset == -1))
            # We want the base composition of the original alignment pre-filtering
            composition = base_composition(alignment[1:])
            alignment_md5 = hashlib.md5(alignment[1:]).hexdigest()
            md["sc2ts_qc"] = {
                "base_composition": dict(composition),
                "masked_overall": total_masked,
                "masked_within": masked_outside_problematic,
                "missing_within": num_missing,
                "alignment_md5": alignment_md5,
            }
            sd.add_individual(metadata=md)
            logger.info(f"Add {strain} metadata={md}")
        assert masked_per_site[0] == 0

        bar = tqdm.tqdm(
            range(num_sites),
            desc="Writing",
            position=1,
            leave=False,
            disable=not show_progress,
        )
        sites = np.where(keep_sites)[0]
        for j in bar:
            pos = sites[j]
            ref_allele = reference[pos]
            masked_samples = masked_per_site[pos]
            sd.add_site(
                pos,
                genotypes=G[j],
                alleles=core.ALLELES,
                ancestral_allele=core.ALLELES.index(ref_allele),
                metadata={"masked_samples": int(masked_samples)},
            )
        sd.add_provenance(
            timestamp=datetime.datetime.now().isoformat(),
            record=provenance)
    return sd


def alignments_to_samples(
    fasta_path, metadata_path, output_dir, show_progress=False, provenance=None
):
    logger.info(f"Loading fasta from {fasta_path}")
    fasta = core.FastaReader(fasta_path)
    output_dir = pathlib.Path(output_dir)

    strains = list(fasta.keys())
    logger.info(f"Grouping {len(strains)} strains by date")
    with sqlite3.connect(metadata_path) as conn:
        strains_by_date = group_by_date(strains, conn)

    bar = tqdm.tqdm(sorted(strains_by_date.keys()), disable=not show_progress)
    for date in bar:
        bar.set_description(date)
        rows = strains_by_date[date]
        logger.info(f"Converting for {len(rows)} strains for {date}")
        samples_path = output_dir / f"{date}.samples"
        convert_alignments(
            rows,
            fasta,
            show_progress=show_progress,
            provenance=provenance,
            num_flush_threads=4,
            path=str(samples_path),
        )


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
    # FIXME use the FastaReader
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
