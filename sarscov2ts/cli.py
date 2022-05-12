import pathlib
import logging

import click
import daiquiri

from . import convert


def setup_logging(verbosity):
    log_level = "WARN"
    if verbosity > 0:
        log_level = "INFO"
    if verbosity > 1:
        log_level = "DEBUG"
    daiquiri.setup(level=log_level)


@click.command()
@click.argument("vcf")
@click.argument("metadata")
@click.argument("output")
def import_usher_vcf(vcf, metadata, output):

    # TODO add verbosity arg
    setup_logging(1)
    sd = convert.to_samples(vcf, metadata, output, show_progress=True)



@click.group()
def cli():
    pass


cli.add_command(import_usher_vcf)
