import calendar
import logging

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


def split_samples(sd, prefix, show_progress=False):
    """
    Returns an iterator over the dates and the Sample data subsets
    for individuals at those sampling dates.
    """
    individual_dates = np.array([ind.metadata["date"] for ind in sd.individuals()])
    unique = np.unique(individual_dates)
    for date in tqdm.tqdm(unique, disable=not show_progress):
        path = f"{prefix}{date}.samples"
        subset = np.where(individual_dates == date)[0]
        yield date, sd.subset(individuals=subset, path=path)
