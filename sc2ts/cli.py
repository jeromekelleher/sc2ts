import json
import logging
import platform
import pathlib
import sys

import tskit
import tsinfer
import click
import daiquiri

from . import core
from . import convert
from . import inference


def get_environment():
    """
    Returns a dictionary describing the environment in which sc2ts
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
        "software": {"name": "sc2ts", "version": core.__version__},
        "parameters": {"command": sys.argv[0], "args": sys.argv[1:]},
        "environment": get_environment(),
    }
    return document


def setup_logging(verbosity, log_file=None):
    log_level = "WARN"
    if verbosity > 0:
        log_level = "INFO"
    if verbosity > 1:
        log_level = "DEBUG"
    outputs = ["stderr"]
    if log_file is not None:
        outputs = [daiquiri.output.File(log_file)]
    # Note using set_excepthook=False means that we don't write errors
    # to the log, so if something happens we'll only see it if we look
    # at the console output. For development this is better than having
    # to go to the log to see the traceback, but for production it may
    # be better to let daiquiri record the errors as well.
    daiquiri.setup(level=log_level, outputs=outputs, set_excepthook=False)


@click.command()
# FIXME this isn't checking for existing!
@click.argument("store", type=click.Path(dir_okay=True, exists=False, file_okay=False))
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def init_alignment_store(store, verbose, log_file):
    setup_logging(verbose, log_file)
    # provenance = get_provenance_dict()
    convert.AlignmentStore.initialise(store)


@click.command()
@click.argument("store", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("fastas", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option("-i", "--initialise", default=False, type=bool, help="Initialise store")
@click.option("--no-progress", default=False, type=bool, help="Don't show progress")
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def import_alignments(store, fastas, initialise, no_progress, verbose, log_file):
    setup_logging(verbose, log_file)
    if initialise:
        a = convert.AlignmentStore.initialise(store)
    else:
        a = convert.AlignmentStore(store, "a")
    for fasta_path in fastas:
        logging.info(f"Reading fasta {fasta_path}")
        fasta = core.FastaReader(fasta_path)
        a.append(fasta, show_progress=True)
    # print(a.storage_info())
    # for strain, x in a.all_alignments():
    #     print(strain, x)
    #     # print(x)
    a.close()

@click.command()
@click.argument("metadata")
@click.argument("db")
@click.option("-v", "--verbose", count=True)
def import_metadata(metadata, db, verbose):
    """
    Convert a CSV formatted metadata file to a database for later use.
    """
    setup_logging(verbose)
    convert.metadata_to_db(metadata, db)


@click.command()
@click.argument("samples-file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output-file", type=click.Path(dir_okay=False))
@click.option(
    "--ancestors-ts",
    "-A",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path base to match against",
)
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
@click.option("--no-progress", default=False, type=bool, help="Don't show progress")
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def infer(
    samples_file,
    output_file,
    ancestors_ts,
    num_mismatches,
    max_submission_delay,
    num_threads,
    precision,
    no_progress,
    verbose,
    log_file,
):
    setup_logging(verbose, log_file)

    if ancestors_ts is not None:
        ancestors_ts = tskit.load(ancestors_ts)
        logging.info(f"Loaded ancestors ts with {ancestors_ts.num_samples} samples")

    with tsinfer.load(samples_file) as sd:
        ts = inference.infer(
            sd,
            ancestors_ts=ancestors_ts,
            num_mismatches=num_mismatches,
            max_submission_delay=max_submission_delay,
            precision=precision,
            num_threads=num_threads,
            show_progress=not no_progress,
        )
        # Record provenance here because this is where the arguments are provided.
        provenance = get_provenance_dict()
        tables = ts.dump_tables()
        tables.provenances.add_row(json.dumps(provenance))
        tables.dump(output_file)


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


@click.version_option(core.__version__)
@click.group()
def cli():
    pass


cli.add_command(init_alignment_store)
cli.add_command(import_alignments)
cli.add_command(import_metadata)
cli.add_command(infer)
cli.add_command(validate)
