import calendar
import logging
import sqlite3
import pathlib
import collections

import pyfasta
import tqdm
import tsinfer
import cyvcf2
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def pad_date(s):
    """
    Takes a partial ISO date description and pads it out to the end
    of the month.
    """
    if len(s) == 10:
        return s
    year = int(s[:4])
    if len(s) > 4:
        month = int(s[5:])
    else:
        month = 12
    day = calendar.monthrange(year, month)[-1]
    return f"{year}-{month:02d}-{day:02d}"


def load_usher_metadata(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype={
            "strain": pd.StringDtype(),
            "genbank_accession": pd.StringDtype(),
            "country": pd.StringDtype(),
            "host": pd.StringDtype(),
            "completeness": pd.StringDtype(),
            "date": pd.StringDtype(),
        },
    )


def prepare_metadata(df):
    """
    Takes the specified metadata dataframe, pads partially specified dates,
    removes samples and returns the resulting dataframe with samples
    sorted by date.
    """
    # remove missing and clearly wrong
    date_col = df["date"]
    keep = np.logical_and(date_col != "?", date_col > "2018")
    df = df[keep].copy()
    df.loc[:, "date"] = df.date.apply(pad_date)
    # Sort by padded date
    df = df.sort_values("date")
    # Hack to get tests working. We rely on this so we should require it.
    if "date_submitted" not in df:
        df["date_submitted"] = df["date"]
    # Replace NAs with None for conversion to JSON
    return df.astype(object).where(pd.notnull(df), None)


def recode_snp_sites_missing_data(alleles, genotypes):
    missing_data_index = alleles.index("*")
    assert missing_data_index >= 0
    genotypes = np.array(genotypes, copy=True)
    genotypes[genotypes == missing_data_index] = -1
    genotypes[genotypes > missing_data_index] -= 1
    return [a for a in alleles if a != "*"], genotypes


def recode_acgt_alleles(alleles, genotypes):
    """
    Recode the specified set of alleles so that the first alleles is
    maintained, but the remainder are sorted subset of ACGT, and
    remap the genotypes accordingly.
    """
    remainder = set("ACGT") - set(alleles[0])
    new_alleles = alleles[:1] + list(sorted(remainder))
    genotypes = np.array(genotypes)
    new_genotypes = genotypes.copy()
    for old_index, allele in enumerate(alleles[1:], 1):
        new_index = new_alleles.index(allele)
        new_genotypes[genotypes == old_index] = new_index
    return new_alleles, new_genotypes


def add_sites(
    vcf,
    sample_data,
    index,
    show_progress=False,
    filter_problematic=True,
    force_four_alleles=True,
):
    pbar = tqdm.tqdm(
        total=sample_data.sequence_length, desc="sites", disable=not show_progress
    )
    # Load the problematic sites
    problematic_sites = set()
    if filter_problematic:
        problematic_sites = set(np.loadtxt("problematic_sites.txt", dtype=np.int64))
    pos = 0
    for variant in vcf:
        pbar.update(variant.POS - pos)
        if pos == variant.POS:
            raise ValueError("Duplicate positions for variant at position", pos)
        else:
            pos = variant.POS
        if pos >= sample_data.sequence_length:
            print("EXITING at pos, skipping remaining variants!!")
            break
        if variant.POS in problematic_sites:
            logger.debug(f"Skipping site {variant.POS}")
            continue
        # Assume REF is the ancestral state.
        alleles = [variant.REF] + variant.ALT
        genotypes = np.array([g[0] for g in variant.genotypes])
        # snp-sites doesn't use the standard way of encoding missing data ".", but
        # instead has an allele value of *
        if "*" in alleles:
            alleles, genotypes = recode_snp_sites_missing_data(alleles, genotypes)
        if force_four_alleles:
            alleles, genotypes = recode_acgt_alleles(alleles, genotypes)
        missing_fraction = np.sum(genotypes == -1) / genotypes.shape[0]
        logging.debug(f"Site {pos} added {missing_fraction * 100:.2f}% missing data")
        sample_data.add_site(pos, genotypes=genotypes[index], alleles=alleles)
    pbar.close()


def to_samples(
    vcf_path,
    metadata_path,
    sample_data_path,
    show_progress=False,
    filter_problematic=True,
    force_four_alleles=True,
):

    vcf = cyvcf2.VCF(vcf_path)
    df_md = load_usher_metadata(metadata_path)
    logger.info(f"Loaded metadata with {len(df_md)} rows")
    df_md = prepare_metadata(df_md)
    logger.info(f"Metadata prepped")
    md_samples = list(df_md["strain"])
    vcf_samples = list(vcf.samples)
    logger.info(f"Creating map")
    index = np.zeros(len(md_samples), dtype=int)
    vcf_sample_index_map = {sample: j for j, sample in enumerate(vcf_samples)}
    keep_samples = []
    j = 0
    for iloc, sample in enumerate(md_samples):
        try:
            index[j] = vcf_sample_index_map[sample]
            assert index[j] >= 0
            j += 1
            keep_samples.append((sample, iloc))
        except KeyError:
            pass

    index = index[:j]
    assert len(index) == len(keep_samples)
    logger.info(f"Keeping {len(index)} from VCF with {len(vcf_samples)}")
    with tsinfer.SampleData(path=sample_data_path, sequence_length=29904) as sd:
        for sample, iloc in keep_samples:
            md = df_md.iloc[iloc].to_dict()
            assert md["strain"] == sample
            sd.add_individual(metadata=md)
        add_sites(
            vcf,
            sd,
            index,
            show_progress=show_progress,
            filter_problematic=filter_problematic,
            force_four_alleles=force_four_alleles,
        )
    return sd


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
    data_path = pathlib.Path("sarscov2ts/data")

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

    bar = tqdm.tqdm(enumerate(rows), desc="Reading", total=len(rows))
    for j, row in bar:
        strain = row["strain"]
        h = get_haplotype(fasta, strain)
        assert h.shape[0] == L + 1
        G[:, j] = h[keep_mask]
        row["num_missing_sites"] = int(np.sum(h[keep_mask] == -1))
        sample_data.add_individual(metadata=row)

    ref_fasta = pyfasta.Fasta(str(data_path / "reference.fasta"))
    bar = tqdm.tqdm(range(num_sites), desc="Writing")
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

    data_path = pathlib.Path("sarscov2ts/data")
    ref_fasta = pyfasta.Fasta(
        str(data_path / "reference.fasta"), record_class=pyfasta.MemoryRecord
    )
    a = np.array(ref_fasta["MN908947 (Wuhan-Hu-1/2019)"]).astype(str)
    reference = np.append(["X"], a)

    strains = list(fasta.keys())
    logger.info(f"Grouping {len(strains)} strains by date")
    with sqlite3.connect(metadata_path) as conn:
        strains_by_date = group_by_date(strains, conn)

    for date in sorted(strains_by_date.keys()):
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
