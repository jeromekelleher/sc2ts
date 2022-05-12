import pathlib
import logging

import sgkit.io.vcf
import click

from . import convert

@click.command()
@click.argument("inpath")
@click.argument("outpath")
def convert_to_zarr(inpath, outpath):
    sgkit.io.vcf.vcf_to_zarr(inpath, outpath)


@click.command()
@click.argument("prefix")
def import_usher_vcf(prefix):
    vcf_path = pathlib.Path(prefix + "all.masked.vcf.gz")

    metadata_path = pathlib.Path(prefix + "metadata.tsv.gz")

    # metadata_path = "metadata-subset.tsv"

    df = convert.load_usher_metadata(metadata_path)
    df = convert.prepare_metadata(df)

    subset = df.head(100)
    subset = subset.reset_index(drop=True)
    subset.to_csv("first_100.tsv", sep="\t")




@click.group()
def cli():
    pass


cli.add_command(import_usher_vcf)
cli.add_command(convert_to_zarr)
