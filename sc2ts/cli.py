import json
import concurrent
import logging
import platform
import random
import pathlib
import sys
import contextlib
import dataclasses
import datetime

import tqdm
import tskit
import tszip
import tsinfer
import click
import daiquiri

import sc2ts
from . import core
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
    sc2ts.AlignmentStore.initialise(store)


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
        a = sc2ts.AlignmentStore.initialise(store)
    else:
        a = sc2ts.AlignmentStore(store, "a")
    for fasta_path in fastas:
        logging.info(f"Reading fasta {fasta_path}")
        fasta = core.FastaReader(fasta_path)
        a.append(fasta, show_progress=True)
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
    sc2ts.MetadataDb.import_csv(metadata, db)


def add_provenance(ts, output_file):
    # Record provenance here because this is where the arguments are provided.
    provenance = get_provenance_dict()
    tables = ts.dump_tables()
    tables.provenances.add_row(json.dumps(provenance))
    tables.dump(output_file)


@click.command()
@click.argument("output")
@click.option("-v", "--verbose", count=True)
def init(output, verbose):
    """
    Creates the initial tree sequence containing the reference sequence.
    """
    setup_logging(verbose)
    ts = inference.initial_ts()
    add_provenance(ts, output)


@click.command()
@click.argument("alignments", type=click.Path(exists=True, dir_okay=False))
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("base", type=click.Path(dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.argument("date")
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
def extend(
    alignments,
    metadata,
    base,
    output,
    date,
    num_mismatches,
    max_submission_delay,
    num_threads,
    precision,
    no_progress,
    verbose,
    log_file,
):
    setup_logging(verbose, log_file)

    with contextlib.ExitStack() as exit_stack:
        alignment_store = exit_stack.enter_context(sc2ts.AlignmentStore(alignments))
        metadata_db = exit_stack.enter_context(sc2ts.MetadataDb(metadata))
        base_ts = tskit.load(base)
        ts = inference.extend(
            alignment_store=alignment_store,
            metadata_db=metadata_db,
            date=date,
            base_ts=base_ts,
            num_mismatches=num_mismatches,
            max_submission_delay=max_submission_delay,
            precision=precision,
            num_threads=num_threads,
            show_progress=not no_progress,
        )
        add_provenance(ts, output)


@click.command()
@click.argument("alignments", type=click.Path(exists=True, dir_okay=False))
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("output-prefix")
@click.option(
    "-b",
    "--base",
    type=click.Path(dir_okay=False, exists=True),
    default=None,
    help=(
        "The base tree sequence to match against. If not specified, create "
        "a new initial base containing the reference. "
    ),
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
@click.option(
    "--max-daily-samples",
    default=None,
    type=int,
    help=(
        "The maximum number of samples to match in a single day. If the total "
        "is greater than this, randomly subsample."
    ),
)
@click.option("--num-threads", default=0, type=int, help="Number of match threads")
@click.option("--random-seed", default=42, type=int, help="Random seed for subsampling")
@click.option("-p", "--precision", default=None, type=int, help="Match precision")
@click.option("--no-progress", default=False, type=bool, help="Don't show progress")
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def daily_extend(
    alignments,
    metadata,
    output_prefix,
    base,
    num_mismatches,
    max_submission_delay,
    max_daily_samples,
    num_threads,
    random_seed,
    precision,
    no_progress,
    verbose,
    log_file,
):
    setup_logging(verbose, log_file)
    rng = random.Random(random_seed)
    if base is None:
        base_ts = inference.initial_ts()
    else:
        base_ts = tskit.load(base)

    with contextlib.ExitStack() as exit_stack:
        alignment_store = exit_stack.enter_context(sc2ts.AlignmentStore(alignments))
        metadata_db = exit_stack.enter_context(sc2ts.MetadataDb(metadata))
        ts_iter = inference.daily_extend(
            alignment_store=alignment_store,
            metadata_db=metadata_db,
            base_ts=base_ts,
            num_mismatches=num_mismatches,
            max_submission_delay=max_submission_delay,
            max_daily_samples=max_daily_samples,
            rng=rng,
            precision=precision,
            num_threads=num_threads,
            show_progress=not no_progress,
        )
        for ts, date in ts_iter:
            output = output_prefix + date + ".ts"
            add_provenance(ts, output)


@click.command()
@click.argument("alignment_db")
@click.argument("ts_file")
@click.option("-v", "--verbose", count=True)
def validate(alignment_db, ts_file, verbose):
    setup_logging(verbose)

    ts = tskit.load(ts_file)
    with sc2ts.AlignmentStore(alignment_db) as alignment_store:
        inference.validate(ts, alignment_store, show_progress=True)


def examine_recombinant(work):
    base_ts = tszip.decompress(work.ts_path)
    with sc2ts.AlignmentStore(work.alignment_db) as a:
        data = sc2ts.utils.examine_recombinant(
            work.strain, base_ts, a, num_mismatches=work.num_mismatches
        )
    return data


@dataclasses.dataclass(frozen=True)
class Work:
    strain: str
    ts_path: str
    num_mismatches: int
    alignment_db: str
    sample: int
    recombinant: int


@click.command()
@click.argument("alignment_db")
@click.argument("tsz_prefix")
@click.argument("base_date")
@click.argument("out_tsz")
@click.option("--num-mismatches", default=3, type=float, help="num-mismatches")
@click.option("-v", "--verbose", count=True)
def annotate_recombinants(
    alignment_db, tsz_prefix, base_date, out_tsz, num_mismatches, verbose
):
    """
    Update recombinant nodes in the specified trees with additional
    information about the matching process.
    """
    setup_logging(verbose)
    ts = tszip.decompress(tsz_prefix + base_date + ".ts.tsz")

    recomb_samples = sc2ts.utils.get_recombinant_samples(ts)

    mismatches = [num_mismatches]

    work = []
    for recombinant, sample in recomb_samples.items():
        md = ts.node(sample).metadata
        date = md["date"]
        previous_date = datetime.date.fromisoformat(date)
        previous_date -= datetime.timedelta(days=1)
        tsz_path = f"{tsz_prefix}{previous_date}.ts.tsz"
        for num_mismatches in mismatches:
            work.append(
                Work(
                    strain=md["strain"],
                    ts_path=tsz_path,
                    num_mismatches=num_mismatches,
                    alignment_db=alignment_db,
                    sample=sample,
                    recombinant=recombinant,
                )
            )

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        future_to_work = {
            executor.submit(examine_recombinant, item): item for item in work
        }

        bar = tqdm.tqdm(
            concurrent.futures.as_completed(future_to_work), total=len(work)
        )
        for future in bar:
            try:
                data = future.result()
            except Exception as exc:
                print(f"Work item: {future_to_work[future]} raised exception!")
                raise exc
            work = future_to_work[future]
            results[work.recombinant] = data

    tables = ts.dump_tables()
    # This is probably very inefficient as we're writing back the metadata column
    # many times
    for recomb_node, metadata in tqdm.tqdm(results.items(), desc="Updating metadata"):
        row = tables.nodes[recomb_node]
        d = row.metadata
        d["match_info"] = json.dumps(metadata)
        tables.nodes[recomb_node] = row.replace(metadata=d)

    ts = tables.tree_sequence()
    logging.info("Compressing output")
    tszip.compress(ts, out_tsz)


@click.version_option(core.__version__)
@click.group()
def cli():
    pass


cli.add_command(init_alignment_store)
cli.add_command(import_alignments)
cli.add_command(import_metadata)

cli.add_command(init)
cli.add_command(extend)
cli.add_command(daily_extend)
cli.add_command(validate)
cli.add_command(annotate_recombinants)
