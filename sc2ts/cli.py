import json
import collections
import concurrent.futures as cf
import logging
import itertools
import pathlib
import sys
import contextlib
import dataclasses
import datetime
import time
from typing import List

import tomli
import numpy as np
import tqdm
import tskit
import tszip
import tsinfer
import click
import humanize
import pandas as pd

import sc2ts
from . import core
from . import utils
from . import info

logger = logging.getLogger(__name__)


# Common arguments/options
dataset = click.argument("dataset", type=click.Path(exists=True, dir_okay=True))

num_mismatches = click.option(
    "-k",
    "--num-mismatches",
    default=3,
    show_default=True,
    type=float,
    help="Number of mismatches to accept in favour of recombination",
)
chunk_cache_size = click.option(
    "-C",
    "--chunk-cache-size",
    default=3,
    show_default=True,
    type=int,
    help="Number of dataset chunks to hold in cache",
)
deletions_as_missing = click.option(
    "--deletions-as-missing/--no-deletions-as-missing",
    default=True,
    help="Treat all deletions as missing data when matching haplotypes",
    show_default=True,
)
memory_limit = click.option(
    "-M",
    "--memory-limit",
    default=0,
    type=float,
    help=(
        "Memory limit in GiB for matching. If active memory usage "
        "exceeds this value during matching, do not start any more "
        "matches until it goes below the threshold. "
        "Defaults to 0 (unlimited)"
    ),
)

progress = click.option("--progress/--no-progress", default=True)
verbose = click.option("-v", "--verbose", count=True)
log_file = click.option(
    "-l", "--log-file", default=None, type=click.Path(dir_okay=False)
)


def summarise_usage(ts):
    record = json.loads(ts.provenance(-1).record)
    d = record["resources"]
    # Report times in minutes
    wall_time = d["elapsed_time"] / 60
    user_time = d["user_time"] / 60
    sys_time = d["sys_time"] / 60
    max_mem = d["max_memory"]
    if max_mem > 0:
        maxmem_str = "; max_memory=" + humanize.naturalsize(max_mem, binary=True)
    return f"elapsed={wall_time:.2f}m; user={user_time:.2f}m; sys={sys_time:.2f}m{maxmem_str}"


def setup_logging(verbosity, log_file=None, date=None):
    log_level = "WARN"
    if verbosity > 0:
        log_level = "INFO"
    if verbosity > 1:
        log_level = "DEBUG"
    handler = logging.StreamHandler()
    if log_file is not None:
        handler = logging.FileHandler(log_file)
    # default time format has millisecond precision which we don't need
    time_format = "%Y-%m-%d %H:%M:%S"
    date = "" if date is None else date
    fmt = logging.Formatter(
        f"%(asctime)s %(levelname)s {date} %(message)s", datefmt=time_format
    )
    handler.setFormatter(fmt)

    # This is mainly used to output messages about major events. Possibly
    # should do this with a separate logger entirely, rather than use
    # the "WARNING" channel.
    warn_handler = logging.StreamHandler()
    warn_handler.setFormatter(logging.Formatter(f"%(levelname)s {date} %(message)s"))
    warn_handler.setLevel(logging.WARN)

    for name in ["sc2ts"]:
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.addHandler(warn_handler)


@click.command()
@click.argument("dataset", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("fastas", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option(
    "-i",
    "--initialise",
    is_flag=True,
    flag_value=True,
    help=(
        "If true, initialise a new dataset. WARNING! This will erase and existing "
        "store"
    ),
)
@progress
@verbose
def import_alignments(dataset, fastas, initialise, progress, verbose):
    """
    Import the alignments from all FASTAS into the dataset
    """
    setup_logging(verbose)
    if initialise:
        sc2ts.Dataset.new(dataset)

    f_bar = tqdm.tqdm(sorted(fastas), desc="Files", disable=not progress, position=0)
    for fasta_path in f_bar:
        reader = core.FastaReader(fasta_path, add_zero_base=False)
        logger.info(f"Reading {len(reader)} alignments from {fasta_path}")
        alignments = {}
        a_bar = tqdm.tqdm(
            reader.items(),
            total=len(reader),
            desc="Extract",
            disable=not progress,
            position=1,
        )
        for k, v in a_bar:
            alignments[k] = sc2ts.encode_alignment(v)
        sc2ts.Dataset.append_alignments(dataset, alignments)


@click.command()
@click.argument("dataset", type=click.Path(dir_okay=True, file_okay=False))
@click.argument("metadata", type=click.Path(dir_okay=False, file_okay=True))
@click.option(
    "--viridian",
    is_flag=True,
    help="Do some preprocessing appropriate for the Viridian metadata "
    "(Available at https://figshare.com/ndownloader/files/49694808)",
)
@verbose
def import_metadata(dataset, metadata, viridian, verbose):
    """
    Import a CSV/TSV metadata file into the dataset.
    """
    setup_logging(verbose)
    logger.info(f"Reading {metadata}")
    dtype = {}
    if viridian:
        dtype = {"Artic_primer_version": str}
    df_in = pd.read_csv(metadata, sep="\t", dtype=dtype)
    date_field = "date"
    index_field = "Run"
    if viridian:
        df_in = sc2ts.massage_viridian_metadata(df_in)
    df = df_in.set_index(index_field)
    sc2ts.Dataset.add_metadata(dataset, df)


@click.command()
@click.argument("match_db", type=click.Path(exists=True, dir_okay=False))
@verbose
def info_matches(match_db, verbose):
    """
    Information about an alignment store
    """
    setup_logging(verbose)
    with sc2ts.MatchDb(match_db) as db:
        print(db)
        print("last date = ", db.last_date())
        print("cost\tpercent\tcount")
        df = db.as_dataframe()
        total = len(db)
        hmm_cost_counter = collections.Counter(df["hmm_cost"].astype(int))
        for cost in sorted(hmm_cost_counter.keys()):
            count = hmm_cost_counter[cost]
            percent = count / total * 100
            print(f"{cost}\t{percent:.1f}\t{count}")


@click.command()
@dataset
@verbose
@click.option(
    "-z", "--zarr-details", is_flag=True, help="Show detailed zarr information"
)
def info_dataset(dataset, verbose, zarr_details):
    """
    Information about a sc2ts Zarr dataset
    """
    setup_logging(verbose)
    ds = sc2ts.Dataset(dataset)
    print(ds)
    if zarr_details:
        for array in ds.root.values():
            print(str(array.info).strip())
            print("----")


@click.command()
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-R", "--recombinants", is_flag=True)
@verbose
def info_ts(ts_path, recombinants, verbose):
    """
    Information about a sc2ts inferred ARG
    """
    setup_logging(verbose)
    ts = tszip.load(ts_path)

    ti = sc2ts.TreeInfo(ts, quick=False)
    # print("info", ti.node_counts())
    # TODO output these as TSVs rather than using pandas display?
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)
    print(ti.summary())
    # TODO more
    if recombinants:
        print(ti.recombinants_summary())


def summarise_base(ts, date, progress):
    ti = sc2ts.TreeInfo(ts, quick=True)
    node_info = "; ".join(f"{k}:{v}" for k, v in ti.node_counts().items())
    logger.info(f"Loaded {node_info}")
    if progress:
        print(f"{date} Start base: {node_info}", file=sys.stderr)


def _run_extend(out_path, verbose, log_file, **params):
    date = params["date"]
    setup_logging(verbose, log_file, date=date)
    ts = sc2ts.extend(show_progress=True, **params)
    ts.dump(out_path)
    resource_usage = summarise_usage(ts)
    logger.info(resource_usage)
    print("resources:", resource_usage, file=sys.stderr)
    data = ts.metadata["sc2ts"]["daily_stats"][date]["samples_processed"]
    if len(data) > 0:
        df = pd.DataFrame(
            ts.metadata["sc2ts"]["daily_stats"][date]["samples_processed"]
        ).set_index("scorpio")
        del df["total_hmm_cost"]
        df = df[list(df.columns)[::-1]].sort_values("total")
        print(df, file=sys.stderr)


@click.command()
@click.argument("config_file", type=click.File(mode="rb"))
@click.option(
    "--start", default=None, help="Start inference at this date (inclusive). "
)
@click.option(
    "--stop",
    default="3000",
    help="Stop and exit at this date (non-inclusive)",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    flag_value=True,
    help="Force destructive updates to Match DB",
)
def infer(config_file, start, stop, force):
    """
    Run the full inference pipeline based on values in the config file.
    """
    config = tomli.load(config_file)
    run_id = config.pop("run_id")
    results_dir = pathlib.Path(config.pop("results_dir")) / run_id
    log_dir = pathlib.Path(config.pop("log_dir"))
    matches_dir = pathlib.Path(config.pop("matches_dir"))
    for path in [matches_dir, results_dir, log_dir]:
        path.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"{run_id}.log"
    match_db = matches_dir / f"{run_id}.matches.db"

    ts_file_pattern = str(results_dir / f"{run_id}_{{date}}.ts")
    exclude_sites = config.pop("exclude_sites", [])

    if start is None:
        if match_db.exists() and not force:
            click.confirm(
                f"Do you want to overwrite MatchDB at {match_db}",
                abort=True,
            )
        init_ts = sc2ts.initial_ts(exclude_sites)
        sc2ts.MatchDb.initialise(match_db)
        base_ts = results_dir / f"{run_id}_init.ts"
        init_ts.dump(base_ts)
        start = "2000"
    else:
        base_ts = find_previous_date_path(start, ts_file_pattern)
        print(f"Starting from {base_ts}")
        with sc2ts.MatchDb(match_db) as mdb:
            newer_matches = mdb.count_newer(start)
            if newer_matches > 0:
                if not force:
                    click.confirm(
                        f"Do you want to remove {newer_matches} newer matches "
                        f"from MatchDB >= {start}?",
                        abort=True,
                    )
                mdb.delete_newer(start)

    log_level = config.pop("log_level", 2)
    exclude_dates = set(config.pop("exclude_dates", []))
    param_overrides = config.pop("override", [])
    dataset = config.pop("dataset")
    extend_parameters = config.pop("extend_parameters")

    if len(config) > 0:
        raise ValueError(f"Unknown keys in config: {list(config.keys())}")

    ds = sc2ts.Dataset(dataset)
    for date in np.unique(ds["sample_date"][:]):
        if date >= stop:
            break
        if date < start or date in exclude_dates:
            continue
        if len(date) < 10 or date < "2020":
            # Imprecise, malformed or ludicrous date
            continue

        params = {
            "dataset": dataset,
            "base_ts": str(base_ts),
            "date": date,
            "match_db": str(match_db),
            **extend_parameters,
        }
        for override_set in param_overrides:
            if override_set["start"] <= date < override_set["stop"]:
                print(f"{date} overriding {override_set}")
                params.update(override_set["parameters"])

        base_ts = ts_file_pattern.format(date=date)
        with cf.ProcessPoolExecutor(1) as executor:
            future = executor.submit(
                _run_extend, base_ts, log_level, log_file, **params
            )
            # Block and wait, raising exception if it occured
            future.result()


@click.command()
@dataset
@click.argument("ts_file")
@deletions_as_missing
@click.option(
    "--genotypes/--no-genotypes",
    default=True,
    help="Validate all genotypes",
    show_default=True,
)
@click.option(
    "--metadata/--no-metadata",
    default=True,
    help="Validate metadata",
    show_default=True,
)
@click.option(
    "-s",
    "--skip",
    default=[],
    help="Skip this metadata field during comparison",
    show_default=True,
    multiple=True,
)
@chunk_cache_size
@verbose
def validate(
    dataset,
    ts_file,
    deletions_as_missing,
    genotypes,
    metadata,
    skip,
    chunk_cache_size,
    verbose,
):
    """
    Check that the specified trees correctly encode data
    """
    setup_logging(verbose)

    ts = tszip.load(ts_file)
    ds = sc2ts.Dataset(dataset, chunk_cache_size=chunk_cache_size)
    if genotypes:
        sc2ts.validate_genotypes(ts, ds, deletions_as_missing, show_progress=True)
    if metadata:
        sc2ts.validate_metadata(ts, ds, skip_fields=set(skip), show_progress=True)


@click.command()
@click.argument("ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--verbose", count=True)
def tally_lineages(ts, metadata, verbose):
    """
    Output a table in TSV format comparing the number of samples associated
    each pango lineage in the ARG along with the corresponding number in
    the metadata DB.
    """
    setup_logging(verbose)
    ts = tszip.load(ts)
    with sc2ts.MetadataDb(metadata) as metadata_db:
        df = info.tally_lineages(ts, metadata_db, show_progress=True)
    df.to_csv(sys.stdout, sep="\t", index=False)


@click.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("strains", nargs=-1)
@num_mismatches
@deletions_as_missing
@click.option(
    "--mismatch-threshold",
    type=int,
    default=100,
    show_default=True,
    help="Set the HMM likelihood threshold to this number of mutations",
)
@click.option(
    "--direction",
    type=click.Choice(["forward", "reverse"]),
    default="forward",
    help="Direction to run HMM in",
)
@click.option(
    "--num-threads",
    default=0,
    type=int,
    help="Number of match threads (default to one)",
)
@click.option("--progress/--no-progress", default=True)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def run_hmm(
    dataset,
    ts_path,
    strains,
    num_mismatches,
    deletions_as_missing,
    mismatch_threshold,
    direction,
    num_threads,
    progress,
    verbose,
    log_file,
):
    """
    Run matches for a specified set of strains, outputing details to stdout as JSON.
    """
    setup_logging(verbose, log_file)

    runs = sc2ts.run_hmm(
        dataset,
        ts_path,
        strains=strains,
        num_mismatches=num_mismatches,
        mismatch_threshold=mismatch_threshold,
        direction=direction,
        num_threads=num_threads,
        show_progress=progress,
    )
    for run in runs:
        print(run.asjson())


def find_previous_date_path(date, path_pattern):
    """
    Find the path with the most-recent date to the specified one
    matching the given pattern.
    """
    date = datetime.date.fromisoformat(date)
    for j in range(1, 30):
        previous_date = date - datetime.timedelta(days=j)
        path = pathlib.Path(path_pattern.format(date=previous_date))
        logger.debug(f"Trying {path}")
        if path.exists():
            break
    else:
        raise ValueError(
            f"No path exists for pattern {path_pattern} starting at {date}"
        )
    return path


@click.version_option(core.__version__)
@click.group()
def cli():
    pass


cli.add_command(import_alignments)
cli.add_command(import_metadata)

cli.add_command(info_dataset)
cli.add_command(info_matches)
cli.add_command(info_ts)

cli.add_command(infer)
cli.add_command(validate)
cli.add_command(run_hmm)

cli.add_command(tally_lineages)
