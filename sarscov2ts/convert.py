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


def add_sites(vcf, sample_data, index, show_progress=False):
    pbar = tqdm.tqdm(
        total=sample_data.sequence_length, desc="sites", disable=not show_progress
    )
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
        # print(pos, samples.sequence_length)
        # Assume REF is the ancestral state.
        alleles = [variant.REF] + variant.ALT
        genotypes = np.array(variant.genotypes).T[0]
        missing_fraction = np.sum(genotypes == -1) / genotypes.shape[0]
        logging.debug(f"Site {pos} added {missing_fraction * 100:.2f}% missing data")
        sample_data.add_site(pos, genotypes=genotypes[index], alleles=alleles)
    pbar.close()


def to_samples(vcf_path, metadata_path, sample_data_path, show_progress=False):

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
    keep_samples = set()
    j = 0
    for sample in md_samples:
        try:
            index[j] = vcf_sample_index_map[sample]
            assert index[j] >= 0
            j += 1
            keep_samples.add(sample)
        except KeyError:
            logger.warning(f"Sample {sample} missing from VCF")

    index = index[:j]
    assert len(index) == len(keep_samples)
    logger.info(f"Keeping {len(index)} from VCF with {len(vcf_samples)}")
    with tsinfer.SampleData(path=sample_data_path, sequence_length=29904) as sd:
        pbar = tqdm.tqdm(total=len(df_md), desc="samples", disable=not show_progress)
        for _, row in df_md.iterrows():
            md = row.to_dict()
            if md["strain"] in keep_samples:
                sd.add_individual(metadata=md)
            pbar.update()
        pbar.close()
        add_sites(vcf, sd, index, show_progress=show_progress)

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
