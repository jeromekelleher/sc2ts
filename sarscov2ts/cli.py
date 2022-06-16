import pathlib
import tempfile
import logging

import tsinfer
import click
import daiquiri

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
def import_usher_vcf(vcf, metadata, output):

    # TODO add verbosity arg
    setup_logging(1)
    sd = convert.to_samples(vcf, metadata, output, show_progress=True)


@click.command()
@click.argument("samples-file")
@click.argument("output-prefix")
def split_samples(samples_file, output_prefix):
    # TODO add verbosity arg
    setup_logging(1)
    with tsinfer.load(samples_file) as sd:
        with tempfile.NamedTemporaryFile() as f:
            for date, sd_sub in convert.split_samples(
                sd, show_progress=True, prefix=output_prefix
            ):
                logging.info(f"Wrote {sd_sub.num_individuals} samples to {sd_sub.path}")


@click.command()
@click.argument("samples-file")
@click.argument("output-prefix")
@click.option("--mismatch-ratio", default=None, help="Mismatch ratio")
@click.option("--num-threads", default=0, type=int, help="Number of match threads")
def infer(samples_file, output_prefix, mismatch_ratio, num_threads):
    # TODO add verbosity arg
    setup_logging(1)

    pm = tsinfer.inference._get_progress_monitor(
        True, generate_ancestors=True, match_ancestors=True, match_samples=True,
    )
    with tsinfer.load(samples_file) as sd:
        iterator = inference.infer(
            sd,
            progress_monitor=pm,
            num_threads=num_threads,
            mismatch_ratio=mismatch_ratio,
        )
        for date, ts in iterator:
            path = f"{output_prefix}{date}.ts"
            ts.dump(path)


@click.group()
def cli():
    pass


cli.add_command(import_usher_vcf)
cli.add_command(split_samples)
cli.add_command(infer)
