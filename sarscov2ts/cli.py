import pathlib
import tempfile
import logging

import tskit
import tsinfer
import click
import daiquiri
import numpy as np

from . import convert
from . import inference


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
@click.option("-v", "--verbose", count=True)
def import_usher_vcf(vcf, metadata, output, verbose):
    setup_logging(verbose)
    sd = convert.to_samples(vcf, metadata, output, show_progress=True)


@click.command()
@click.argument("samples-file")
@click.argument("output-file")
@click.option("--ancestors-ts", "-A", default=None, help="Path base to match against")
@click.option("--num-mismatches", default=None, type=float, help="num-mismatches")
@click.option("--num-threads", default=0, type=int, help="Number of match threads")
@click.option(
    "-d", "--daily-prefix", default=None, help="Prefix to output daily result files"
)
@click.option("-v", "--verbose", count=True)
def infer(
    samples_file,
    output_file,
    ancestors_ts,
    num_mismatches,
    num_threads,
    daily_prefix,
    verbose,
):
    setup_logging(verbose)

    if ancestors_ts is not None:
        ancestors_ts = tskit.load(ancestors_ts)
        logging.info(f"Loaded ancestors ts with {ancestors_ts.num_sites} sites")

    pm = tsinfer.inference._get_progress_monitor(
        True,
        generate_ancestors=True,
        match_ancestors=True,
        match_samples=True,
    )

    with tsinfer.load(samples_file) as sd:
        ts = inference.infer(
            sd,
            ancestors_ts=ancestors_ts,
            progress_monitor=pm,
            num_threads=num_threads,
            num_mismatches=num_mismatches,
            daily_prefix=daily_prefix,
            show_progress=True,
        )
        ts.dump(output_file)


@click.command()
@click.argument("samples-file")
@click.argument("ts-file")
@click.option("-v", "--verbose", count=True)
def validate(samples_file, ts_file, verbose):
    setup_logging(verbose)

    ts = tskit.load(ts_file)
    with tsinfer.load(samples_file) as sd:
        inference.validate(sd, ts, show_progress=True)


@click.group()
def cli():
    pass


cli.add_command(import_usher_vcf)
cli.add_command(infer)
cli.add_command(validate)
