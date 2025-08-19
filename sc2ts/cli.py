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
    default=4,
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
    "--field-descriptions",
    type=click.File(mode="r"),
    default=None,
    help="JSON formatted file of field descriptions",
)
@click.option(
    "--viridian",
    is_flag=True,
    help="Do some preprocessing appropriate for the Viridian metadata "
    "(Available at https://figshare.com/ndownloader/files/49694808)",
)
@verbose
def import_metadata(dataset, metadata, field_descriptions, viridian, verbose):
    """
    Import a CSV/TSV metadata file into the dataset.
    """
    setup_logging(verbose)
    logger.info(f"Reading {metadata}")
    dtype = {}
    if viridian:
        dtype = {"Artic_primer_version": str}
    df_in = pd.read_csv(metadata, sep="\t", dtype=dtype)
    index_field = "Run"
    if viridian:
        df_in = sc2ts.massage_viridian_metadata(df_in)
    df = df_in.set_index(index_field)
    d = {}
    if field_descriptions is not None:
        d = json.load(field_descriptions)
    sc2ts.Dataset.add_metadata(dataset, df, field_descriptions=d)


def summarise_match_db(db):
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


def list_all_matches(db):
    print("strain", "n_parents", "n_mutations", "parents", "mutations", sep="\t")
    for sample in tqdm.tqdm(db.all_samples(), total=len(db)):
        hmm_match = sample.hmm_match
        print(
            sample.strain,
            len(hmm_match.path),
            len(hmm_match.mutations),
            hmm_match.path_summary(),
            hmm_match.mutation_summary(),
            sep="\t",
        )


@click.command()
@click.argument("match_db", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-A", "--all-matches", is_flag=True, help="Export information about all matches"
)
@verbose
def info_matches(match_db, all_matches, verbose):
    """
    Information about matches in the MatchDB
    """
    setup_logging(verbose)
    with sc2ts.MatchDb(match_db) as db:
        if all_matches:
            list_all_matches(db)
        else:
            summarise_match_db(db)


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
    date_field = config.pop("date_field")
    extend_parameters = config.pop("extend_parameters")

    if len(config) > 0:
        raise ValueError(f"Unknown keys in config: {list(config.keys())}")
    ds = sc2ts.Dataset(dataset, date_field=date_field)

    for date in np.unique(ds.metadata.sample_date):
        if date >= stop:
            break
        if date < start or date in exclude_dates:
            continue
        if len(date) < 10 or date < "2020":
            # Imprecise, malformed or ludicrous date
            continue

        params = {
            "dataset": dataset,
            "date_field": date_field,
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
    "--date-field",
    default=None,
    help="Specify date field to use. Required for metadata.",
)
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
    date_field,
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
    ds = sc2ts.Dataset(
        dataset, date_field=date_field, chunk_cache_size=chunk_cache_size
    )
    if genotypes:
        sc2ts.validate_genotypes(ts, ds, deletions_as_missing, show_progress=True)
    if metadata:
        sc2ts.validate_metadata(ts, ds, skip_fields=set(skip), show_progress=True)


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
        deletions_as_missing=deletions_as_missing,
        mismatch_threshold=mismatch_threshold,
        direction=direction,
        num_threads=num_threads,
        show_progress=progress,
    )
    for run in runs:
        print(run.asjson())


@click.command()
@click.argument("ts_in", type=click.Path(exists=True, dir_okay=False))
@click.argument("ts_out", type=click.Path(exists=False, dir_okay=False))
@click.option("--match-db", type=click.Path(exists=True, dir_okay=False))
@click.option("--progress/--no-progress", default=True)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def postprocess(
    ts_in,
    ts_out,
    match_db,
    progress,
    verbose,
    log_file,
):
    """
    Perform final postprocessing steps to the specified ARG.
    """
    setup_logging(verbose, log_file)
    ts = tszip.load(ts_in)
    if match_db is not None:
        with sc2ts.MatchDb(match_db) as db:
            ts = sc2ts.append_exact_matches(ts, db, show_progress=progress)

    ts = sc2ts.push_up_unary_recombinant_mutations(ts)
    # See if we can remove some of the reversions in a straightforward way.
    mutations_is_reversion = sc2ts.find_reversions(ts)
    mutations_before = ts.num_mutations
    ts = sc2ts.push_up_reversions(
        ts, ts.mutations_node[mutations_is_reversion], date=None
    )
    ts.dump(ts_out)


@click.command()
@click.argument("ts_in", type=click.Path(exists=True, dir_okay=False))
@click.argument("ts_out", type=click.Path(exists=False, dir_okay=False))
@click.option("--field-mapping", "-m", type=(str, str), multiple=True)
@click.option("--progress/--no-progress", default=True)
@click.option("--drop-vestigial-root/--no-drop-vestigial-root", default=False)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def minimise_metadata(
    ts_in,
    ts_out,
    field_mapping,
    progress,
    drop_vestigial_root,
    verbose,
    log_file,
):
    """
    Generate the final "analysis" version of the ARG by dropping all
    metadata other the specified fields, and recoding the minimal information
    using the struct codec.

    By default we only remap the "strain" metadata field to "sample_id". If
    other fields are required, these can be provided with the
    -m [old metadata name] [new metadata name], e.g

    python -m sc2ts minimise-metadata -m strain sample_id -m Viridian_pangolin pango

    The -m option can be provided as many times as we like, but it's important
    to note that the strain/sample ID mapping *must* be provided if so.

    Currently only supports string fields
    """
    if len(field_mapping) == 0:
        field_mapping = None
    else:
        field_mapping = dict(field_mapping)
    setup_logging(verbose, log_file)
    ts = tszip.load(ts_in)
    ts = sc2ts.minimise_metadata(ts, field_mapping, show_progress=progress)
    if drop_vestigial_root:
        ts = sc2ts.drop_vestigial_root_edge(ts)
    ts.dump(ts_out)


@click.command()
@click.argument("dataset", type=click.Path(exists=True, dir_okay=False))
@click.argument("ts_in", type=click.Path(exists=True, dir_okay=False))
@click.argument("ts_out", type=click.Path(exists=False, dir_okay=False))
@click.option("--sites", default=None)
@click.option("--report", default=None)
@click.option("--progress/--no-progress", default=True)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def map_parsimony(
    dataset,
    ts_in,
    ts_out,
    sites,
    report,
    progress,
    verbose,
    log_file,
):
    """
    Map variation at the specified set of sites to the ARG using parsimony.
    """
    setup_logging(verbose, log_file)
    ds = sc2ts.Dataset(dataset)
    ts = tszip.load(ts_in)
    if sites is not None:
        sites = np.loadtxt(sites, dtype=int)
    result = sc2ts.map_parsimony(ts, ds, sites, show_progress=progress)
    if report is not None:
        result.report.to_csv(report)
    result.tree_sequence.dump(ts_out)


@click.command()
@click.argument("ts_in", type=click.Path(exists=True, dir_okay=False))
@click.argument("ts_out", type=click.Path(exists=False, dir_okay=False))
@click.option("--report", default=None)
@click.option("--progress/--no-progress", default=True)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def apply_node_parsimony(
    ts_in,
    ts_out,
    report,
    progress,
    verbose,
    log_file,
):
    """
    Apply the node parsimony hueristics iteratively until convergance
    and save the output.
    """
    setup_logging(verbose, log_file)
    ts = tszip.load(ts_in)

    result = sc2ts.apply_node_parsimony_heuristics(ts, show_progress=progress)
    if report is not None:
        result.report.to_csv(report)
    result.tree_sequence.dump(ts_out)


@click.command()
@click.argument("node_id", type=int)
@click.option("--path-pattern", default=None)
@click.option("--date", default=None)
@click.option("--base-ts", default=None)
@click.option("--recomb-ts", default=None)
@num_mismatches
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def rematch_recombinant(
    path_pattern,
    date,
    base_ts,
    recomb_ts,
    node_id,
    num_mismatches,
    verbose,
    log_file,
):
    """
    Rerun recombinant matching for the specified recombinants and output necessary
    information to validate.
    """

    setup_logging(verbose, log_file)
    if path_pattern is not None:
        recomb_ts = path_pattern.format(date=date)
        base_ts = find_previous_date_path(date, path_pattern)

    base_ts = tszip.load(base_ts)
    recomb_ts = tszip.load(recomb_ts)
    result = sc2ts.rematch_recombinant(
        base_ts, recomb_ts, node_id, num_mismatches=num_mismatches
    )
    print(json.dumps(result.asdict()))


@click.command()
@click.argument("ts")
@click.argument("node_id", type=int)
@num_mismatches
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def rematch_recombinant_lbs(ts, node_id, num_mismatches, verbose, log_file):
    """
    Runs the lbs recombinant rematch on the specified tree sequence
    and return the result as JSON.
    """
    setup_logging(verbose, log_file)

    ts = tszip.load(ts)
    result = sc2ts.rematch_recombinant_lbs(ts, node_id, num_mismatches=num_mismatches)
    print(json.dumps(result.asdict()))


@click.command()
@click.argument("ts_in")
@click.argument("rematch_data")
@click.argument("ts_out")
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def rewire_lbs(ts_in, rematch_data, ts_out, verbose, log_file):
    """
    Rewires the specified tree sequence using information from the specified
    JSON datafile of runs from rematch_recombinant_lbs.
    """
    setup_logging(verbose, log_file)

    ts = tszip.load(ts_in)

    records = []
    with open(rematch_data) as f:
        for d in json.load(f):
            records.append(sc2ts.RematchRecombinantsLbsResult.fromdict(d))

    recombs_to_rewire = []
    rewire_existing = 0
    rewire_lbs = 0
    for r in records:
        lbs = r.long_branch_split
        if lbs is None:
            assert len(r.recomb_match.path) == 1
            rewire_existing += 1
            recombs_to_rewire.append(r)
        elif len(lbs.hmm_match.path) == 1:
            rewire_lbs += 1
            recombs_to_rewire.append(r)

    logger.info(
        f"Rewire {len(recombs_to_rewire)}/{len(records)} recombinants "
        f"(existing={rewire_existing} lbs={rewire_lbs})"
    )

    ts = sc2ts.push_up_unary_recombinant_mutations(ts)
    ts = sc2ts.rewire_long_branch_splits(ts, recombs_to_rewire)
    ts.dump(ts_out)


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
cli.add_command(postprocess)
cli.add_command(minimise_metadata)
cli.add_command(map_parsimony)
cli.add_command(apply_node_parsimony)
cli.add_command(run_hmm)
cli.add_command(rematch_recombinant)
cli.add_command(rematch_recombinant_lbs)
cli.add_command(rewire_lbs)
cli.add_command(run_hmm)
