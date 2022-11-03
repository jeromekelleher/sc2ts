import json
import logging
import pathlib
import platform
import sys
import tempfile

import tskit
import tsinfer
import click
import daiquiri
import numpy as np

from . import convert
from . import inference


def get_environment():
    """
    Returns a dictionary describing the environment in which sarscov2ts
    is currently running.
    """
    env = {
        "os": {
            "system": platform.system(),
            "node": platform.node(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "libraries": {
            "tsinfer": {"version": tsinfer.__version__},
            "tskit": {"version": tskit.__version__},
        },
    }
    return env


def get_provenance_dict():
    """
    Returns a dictionary encoding an execution of stdpopsim conforming to the
    tskit provenance schema.
    """
    document = {
        "schema_version": "1.0.0",
        "software": {"name": "sarscov2ts", "version": "dev"},
        "parameters": {"command": sys.argv[0], "args": sys.argv[1:]},
        "environment": get_environment(),
    }
    return document


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
def import_vcf(vcf, metadata, output, verbose):
    setup_logging(verbose)
    sd = convert.to_samples(vcf, metadata, output, show_progress=True)


@click.command()
@click.argument("samples-file")
@click.argument("output-file")
@click.option("--ancestors-ts", "-A", default=None, help="Path base to match against")
@click.option("--num-mismatches", default=None, type=float, help="num-mismatches")
@click.option(
    "--max-submission-delay",
    default=None,
    type=int,
    help=(
        "The maximum number of days between the sample and its submission date "
        "for it to be included in the inference"
    ),
)
@click.option("--num-threads", default=0, type=int, help="Number of match threads")
@click.option("-p", "--precision", default=None, type=int, help="Match precision")
@click.option(
    "-d", "--daily-prefix", default=None, help="Prefix to output daily result files"
)
@click.option("-v", "--verbose", count=True)
def infer(
    samples_file,
    output_file,
    ancestors_ts,
    num_mismatches,
    max_submission_delay,
    num_threads,
    precision,
    daily_prefix,
    verbose,
):
    setup_logging(verbose)

    if ancestors_ts is not None:
        ancestors_ts = tskit.load(ancestors_ts)
        logging.info(f"Loaded ancestors ts with {ancestors_ts.num_sites} sites")

    pm = tsinfer.inference._get_progress_monitor(
        True,
        generate_ancestors=False,
        match_ancestors=False,
        match_samples=False,
    )
    provenance = get_provenance_dict()
    with tsinfer.load(samples_file) as sd:
        ts = inference.infer(
            sd,
            ancestors_ts=ancestors_ts,
            progress_monitor=pm,
            num_threads=num_threads,
            num_mismatches=num_mismatches,
            max_submission_delay=max_submission_delay,
            daily_prefix=daily_prefix,
            precision=precision,
            show_progress=True,
        )
        tables = ts.dump_tables()
        tables.provenances.add_row(json.dumps(provenance))
        ts = tables.tree_sequence()
        ts.dump(output_file)


@click.command()
@click.argument("samples-file")
@click.argument("ts-file")
@click.option("-v", "--verbose", count=True)
@click.option(
    "--max-submission-delay",
    default=None,
    type=int,
    help=(
        "The maximum number of days between the sample and its submission date "
        "for it to be included in the inference"
    ),
)
def validate(samples_file, ts_file, verbose, max_submission_delay):
    setup_logging(verbose)

    ts = tskit.load(ts_file)
    with tsinfer.load(samples_file) as sd:
        inference.validate(
            sd, ts, max_submission_delay=max_submission_delay, show_progress=True
        )


@click.group()
def cli():
    pass


cli.add_command(import_vcf)
cli.add_command(infer)
cli.add_command(validate)
