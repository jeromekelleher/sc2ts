import pathlib

import click
import pandas as pd


@click.command()
@click.argument("prefix")
def import_usher_vcf(prefix):
    vcf_path = pathlib.Path(prefix + "all.masked.vcf.gz")

    metadata_path = pathlib.Path(prefix + "metadata.tsv.gz")
    # metadata_path = "metadata-subset.tsv"
    df = pd.read_csv(metadata_path, sep="\t", dtype={"date": pd.StringDtype()})

    date = df["date"]
    complete_dates = date.str.len() == 10
    df = df[complete_dates]

    df = df.sort_values("date")

    subset = df.head(100)
    subset = subset.reset_index(drop=True)
    subset.to_csv("first_100.csv")




@click.group()
def cli():
    pass


cli.add_command(import_usher_vcf)
