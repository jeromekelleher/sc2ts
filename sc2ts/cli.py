import json
import collections
import concurrent
import logging
import platform
import sys
import contextlib
import dataclasses
import datetime
import time
import os

import numpy as np
import tqdm
import tskit
import tszip
import tsinfer
import click
import humanize
import pandas as pd

try:
    import resource
except ImportError:
    resource = None  # resource.getrusage absent on windows, so skip outputting max mem

import sc2ts
from . import core
from . import utils
from . import info

logger = logging.getLogger(__name__)

__before = time.time()


def summarise_usage():
    # Measure all times in minutes
    wall_time = (time.time() - __before) / 60
    user_time = os.times().user / 60
    sys_time = os.times().system / 60
    if resource is None:
        # Don't report max memory on Windows. We could do this using the psutil lib, via
        # psutil.Process(os.getpid()).get_ext_memory_info().peak_wset if demand exists
        maxmem_str = "?"
    else:
        max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            max_mem *= 1024  # Linux and other OSs (e.g. freeBSD) report maxrss in kb
        maxmem_str = "; max_memory=" + humanize.naturalsize(max_mem, binary=True)
    return (
        f"elapsed={wall_time:.2f}m; user={user_time:.2f}m; sys={sys_time:.2f}m"
        + maxmem_str
    )


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
    handler = logging.StreamHandler()
    if log_file is not None:
        handler = logging.FileHandler(log_file)
    # default time format has millisecond precision which we don't need
    time_format = "%Y-%m-%d %H:%M:%S"
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s", datefmt=time_format
    )
    handler.setFormatter(fmt)

    # This is mainly used to output messages about major events. Possibly
    # should do this with a separate logger entirely, rather than use
    # the "WARNING" channel.
    warn_handler = logging.StreamHandler()
    warn_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    warn_handler.setLevel(logging.WARN)

    for name in ["sc2ts", "tsinfer.inference"]:
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.addHandler(handler)
        logger.addHandler(warn_handler)


# TODO add options to list keys, dump specific alignments etc
@click.command()
@click.argument("store", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def info_alignments(store, verbose, log_file):
    """
    Information about an alignment store
    """
    setup_logging(verbose, log_file)
    with sc2ts.AlignmentStore(store) as alignment_store:
        print(alignment_store)


@click.command()
@click.argument("store", type=click.Path(dir_okay=False, file_okay=True))
@click.argument("fastas", type=click.Path(exists=True, dir_okay=False), nargs=-1)
@click.option("-i", "--initialise", default=False, type=bool, help="Initialise store")
@click.option("--no-progress", default=False, type=bool, help="Don't show progress")
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def import_alignments(store, fastas, initialise, no_progress, verbose, log_file):
    """
    Import the alignments from all FASTAS into STORE.
    """
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


@click.command()
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def info_metadata(metadata, verbose, log_file):
    """
    Information about a metadata DB
    """
    setup_logging(verbose, log_file)
    with sc2ts.MetadataDb(metadata) as metadata_db:
        print(metadata_db)


@click.command()
@click.argument("match_db", type=click.Path(exists=True, dir_okay=False))
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def info_matches(match_db, verbose, log_file):
    """
    Information about an alignment store
    """
    setup_logging(verbose, log_file)
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
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-R", "--recombinants", is_flag=True)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def info_ts(ts_path, recombinants, verbose, log_file):
    """
    Information about a sc2ts inferred ARG
    """
    setup_logging(verbose, log_file)
    ts = tszip.load(ts_path)

    ti = sc2ts.TreeInfo(ts, quick=False)
    # print("info", ti.node_counts())
    # TODO output these as TSVs rather than using pandas display?
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(ti.summary())
    # TODO more
    if recombinants:
        print(ti.recombinants_summary())


def add_provenance(ts, output_file):
    # Record provenance here because this is where the arguments are provided.
    provenance = get_provenance_dict()
    tables = ts.dump_tables()
    tables.provenances.add_row(json.dumps(provenance))
    tables.dump(output_file)
    logger.info(f"Wrote {output_file}")


@click.command()
@click.argument("ts", type=click.Path(dir_okay=False))
@click.argument("match_db", type=click.Path(dir_okay=False))
@click.option(
    "--additional-problematic-sites",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="File containing the list of additional problematic sites to exclude.",
)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def initialise(ts, match_db, additional_problematic_sites, verbose, log_file):
    """
    Initialise a new base tree sequence to begin inference.
    """
    setup_logging(verbose, log_file)

    additional_problematic = []
    if additional_problematic_sites is not None:
        additional_problematic = (
            np.loadtxt(additional_problematic_sites, ndmin=1).astype(int).tolist()
        )
        logger.info(
            f"Excluding additional {len(additional_problematic)} problematic sites"
        )

    base_ts = sc2ts.initial_ts(additional_problematic)
    base_ts.dump(ts)
    logger.info(f"New base ts at {ts}")
    sc2ts.MatchDb.initialise(match_db)


@click.command()
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.option("--counts/--no-counts", default=False)
@click.option(
    "--after", default="1900-01-01", help="show dates after the specified value"
)
@click.option("-v", "--verbose", count=True)
@click.option("-l", "--log-file", default=None, type=click.Path(dir_okay=False))
def list_dates(metadata, counts, after, verbose, log_file):
    """
    List the dates included in specified metadataDB
    """
    setup_logging(verbose, log_file)
    with sc2ts.MetadataDb(metadata) as metadata_db:
        counter = metadata_db.date_sample_counts()
        for k in counter:
            if k > after:
                if counts:
                    print(k, counter[k], sep="\t")
                else:
                    print(k)


def summarise_base(ts, date, progress):
    ti = sc2ts.TreeInfo(ts, quick=True)
    node_info = "; ".join(f"{k}:{v}" for k, v in ti.node_counts().items())
    logger.info(f"Loaded {node_info}")
    if progress:
        print(f"{date} Start base: {node_info}", file=sys.stderr)


@click.command()
@click.argument("base_ts", type=click.Path(exists=True, dir_okay=False))
@click.argument("date")
@click.argument("alignments", type=click.Path(exists=True, dir_okay=False))
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("matches", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_ts", type=click.Path(dir_okay=False))
@click.option(
    "--num-mismatches",
    default=3,
    show_default=True,
    type=float,
    help="Number of mismatches to accept in favour of recombination",
)
@click.option(
    "--hmm-cost-threshold",
    default=5,
    type=float,
    show_default=True,
    help="The maximum HMM cost for samples to be included unconditionally",
)
@click.option(
    "--min-group-size",
    default=10,
    show_default=True,
    type=int,
    help="Minimum size of groups of reconsidered samples for inclusion",
)
@click.option(
    "--min-root-mutations",
    default=2,
    show_default=True,
    type=int,
    help="Minimum number of shared mutations for reconsidered sample groups",
)
@click.option(
    "--retrospective-window",
    default=30,
    show_default=True,
    type=int,
    help="Number of days in the past to reconsider potential matches",
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
@click.option(
    "--max-missing-sites",
    default=None,
    type=int,
    help=(
        "The maximum number of missing sites in a sample to be accepted for inclusion"
    ),
)
@click.option(
    "--random-seed",
    default=42,
    type=int,
    help="Random seed for subsampling",
    show_default=True,
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
@click.option(
    "-f",
    "--force",
    is_flag=True,
    flag_value=True,
    help="Force clearing newer matches from DB",
)
def extend(
    base_ts,
    date,
    alignments,
    metadata,
    matches,
    output_ts,
    num_mismatches,
    hmm_cost_threshold,
    min_group_size,
    min_root_mutations,
    retrospective_window,
    max_daily_samples,
    max_missing_sites,
    num_threads,
    random_seed,
    progress,
    verbose,
    log_file,
    force,
):
    """
    Extend base_ts with sequences for the specified date, using specified
    alignments and metadata databases, updating the specified matches
    database, and outputting the result to the specified file.
    """
    setup_logging(verbose, log_file)
    base = tskit.load(base_ts)
    summarise_base(base, date, progress)
    with contextlib.ExitStack() as exit_stack:
        alignment_store = exit_stack.enter_context(sc2ts.AlignmentStore(alignments))
        metadata_db = exit_stack.enter_context(sc2ts.MetadataDb(metadata))
        match_db = exit_stack.enter_context(sc2ts.MatchDb(matches))

        newer_matches = match_db.count_newer(date)
        if newer_matches > 0:
            if not force:
                click.confirm(
                    f"Do you want to remove {newer_matches} newer matches "
                    f"from MatchDB > {date}?",
                    abort=True,
                )
                match_db.delete_newer(date)
        ts_out = sc2ts.extend(
            alignment_store=alignment_store,
            metadata_db=metadata_db,
            base_ts=base,
            date=date,
            match_db=match_db,
            num_mismatches=num_mismatches,
            hmm_cost_threshold=hmm_cost_threshold,
            min_group_size=min_group_size,
            min_root_mutations=min_root_mutations,
            retrospective_window=retrospective_window,
            max_daily_samples=max_daily_samples,
            max_missing_sites=max_missing_sites,
            random_seed=random_seed,
            num_threads=num_threads,
            show_progress=progress,
        )
        add_provenance(ts_out, output_ts)
    resource_usage = f"{date}:{summarise_usage()}"
    logger.info(resource_usage)
    if progress:
        print(resource_usage, file=sys.stderr)


@click.command()
@click.argument("alignment_db")
@click.argument("ts_file")
@click.option("-v", "--verbose", count=True)
def validate(alignment_db, ts_file, verbose):
    """
    Check that the specified trees correctly encode alignments for samples.
    """
    setup_logging(verbose)

    ts = tszip.load(ts_file)
    with sc2ts.AlignmentStore(alignment_db) as alignment_store:
        sc2ts.validate(ts, alignment_store, show_progress=True)


@click.command()
@click.argument("ts_file")
@click.option("-v", "--verbose", count=True)
def export_alignments(ts_file, verbose):
    """
    Export alignments from the specified tskit file to FASTA
    """
    setup_logging(verbose)
    ts = tszip.load(ts_file)
    for u, alignment in zip(ts.samples(), ts.alignments(left=1)):
        strain = ts.node(u).metadata["strain"]
        if strain == core.REFERENCE_STRAIN:
            continue
        print(f">{strain}")
        print(alignment)


@click.command()
@click.argument("ts_file")
@click.option("-v", "--verbose", count=True)
def export_metadata(ts_file, verbose):
    """
    Export metadata from the specified tskit file to TSV
    """
    setup_logging(verbose)
    ts = tszip.load(ts_file)
    data = []
    for u in ts.samples():
        md = ts.node(u).metadata
        if md["strain"] == core.REFERENCE_STRAIN:
            continue
        try:
            # FIXME this try/except is needed because of some samples not having full
            # metadata. Can drop when fixed.
            del md["sc2ts"]
        except KeyError:
            pass
        data.append(md)
    df = pd.DataFrame(data)
    df.to_csv(sys.stdout, sep="\t", index=False)


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


def examine_recombinant(work):
    base_ts = tszip.load(work.ts_path)
    # NOTE: this is needed because we have to have all the sites in the trees
    # for tsinfer matching to work in the reverse direction. There is the
    # possibility of subtle differences in the match path because of this.
    # We probably won't offer this interface anyway for long, though, and
    # the forward/backward in the inference
    base_ts = sc2ts.pad_sites(base_ts)
    with contextlib.ExitStack() as exit_stack:
        alignment_store = exit_stack.enter_context(
            sc2ts.AlignmentStore(work.alignments)
        )
        metadata_db = exit_stack.enter_context(sc2ts.MetadataDb(work.metadata))
        metadata_matches = list(
            metadata_db.query(f"SELECT * FROM samples WHERE strain=='{work.strain}'")
        )
        samples = sc2ts.preprocess(
            metadata_matches,
            base_ts,
            metadata_matches[0]["date"],
            alignment_store,
            show_progress=False,
        )
        try:
            sc2ts.match_recombinants(
                samples,
                base_ts,
                num_mismatches=work.num_mismatches,
                show_progress=False,
                num_threads=0,
            )
        except Exception as e:
            print("ERROR in matching", samples[0].strain)
            raise e
    return samples[0]


@dataclasses.dataclass(frozen=True)
class Work:
    strain: str
    ts_path: str
    num_mismatches: int
    alignments: str
    metadata: str
    sample: int
    recombinant: int


@click.command()
@click.argument("alignments", type=click.Path(exists=True, dir_okay=False))
@click.argument("metadata", type=click.Path(exists=True, dir_okay=False))
@click.argument("tsz_prefix")
@click.argument("base_date")
@click.argument("out_tsz")
@click.option("--num-mismatches", default=3, type=float, help="num-mismatches")
@click.option("-v", "--verbose", count=True)
def annotate_recombinants(
    alignments, metadata, tsz_prefix, base_date, out_tsz, num_mismatches, verbose
):
    """
    Update recombinant nodes in the specified trees with additional
    information about the matching process.
    """
    setup_logging(verbose)
    ts = tszip.load(tsz_prefix + base_date + ".ts.tsz")

    recomb_samples = sc2ts.utils.get_recombinant_samples(ts)
    mismatches = [num_mismatches]

    work = []
    for recombinant, sample in recomb_samples.items():
        md = ts.node(sample).metadata
        date = md["date"]
        previous_date = datetime.date.fromisoformat(date)
        previous_date -= datetime.timedelta(days=1)
        tsz_path = f"{tsz_prefix}{previous_date}.ts"
        for num_mismatches in mismatches:
            work.append(
                Work(
                    strain=md["strain"],
                    ts_path=tsz_path,
                    num_mismatches=num_mismatches,
                    alignments=alignments,
                    metadata=metadata,
                    sample=sample,
                    recombinant=recombinant,
                )
            )

    results = {}
    # for item in work:
    #     sample = examine_recombinant(item)
    #     results[item.recombinant] = sample
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
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
                print(exc)
            work = future_to_work[future]
            results[work.recombinant] = data

    tables = ts.dump_tables()
    # This is probably very inefficient as we're writing back the metadata column
    # many times
    for recomb_node, sample in tqdm.tqdm(results.items(), desc="Updating metadata"):
        row = tables.nodes[recomb_node]

        hmm_md = [
            {
                "direction": "forward",
                "path": [x.asdict() for x in sample.forward_path],
                "mutations": [x.asdict() for x in sample.forward_mutations],
            },
            {
                "direction": "reverse",
                "path": [x.asdict() for x in sample.reverse_path],
                "mutations": [x.asdict() for x in sample.reverse_mutations],
            },
        ]
        d = row.metadata
        d["sc2ts"] = {"hmm": hmm_md}
        # print(json.dumps(hmm_md, indent=2))
        tables.nodes[recomb_node] = row.replace(metadata=d)

    ts = tables.tree_sequence()
    logging.info("Compressing output")
    tszip.compress(ts, out_tsz)


@click.version_option(core.__version__)
@click.group()
def cli():
    pass


cli.add_command(import_alignments)
cli.add_command(import_metadata)
cli.add_command(info_alignments)
cli.add_command(info_metadata)
cli.add_command(info_matches)
cli.add_command(info_ts)
cli.add_command(export_alignments)
cli.add_command(export_metadata)

cli.add_command(initialise)
cli.add_command(list_dates)
cli.add_command(extend)
cli.add_command(validate)
cli.add_command(annotate_recombinants)
cli.add_command(tally_lineages)
