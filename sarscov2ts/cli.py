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
@click.option("--num-mismatches", default=None, type=float, help="num-mismatches")
@click.option("--num-threads", default=0, type=int, help="Number of match threads")
@click.option("-v", "--verbose", count=True)
def infer(samples_file, output_prefix, num_mismatches, num_threads, verbose):
    setup_logging(verbose)

    pm = tsinfer.inference._get_progress_monitor(
        True,
        generate_ancestors=True,
        match_ancestors=True,
        match_samples=True,
    )
    with tsinfer.load(samples_file) as sd:
        iterator = inference.infer(
            sd,
            progress_monitor=pm,
            num_threads=num_threads,
            num_mismatches=num_mismatches,
        )
        for date, ts in iterator:
            path = f"{output_prefix}{date}.ts"
            ts.dump(path)


@click.command()
@click.argument("samples-file")
@click.argument("ts-file")
@click.option("-v", "--verbose", count=True)
def validate(samples_file, ts_file, verbose):
    setup_logging(verbose)

    ts = tskit.load(ts_file)
    name_map = {ts.node(u).metadata["strain"]: u for u in ts.samples()}
    with tsinfer.load(samples_file) as sd:
        assert ts.num_sites == sd.num_sites
        ts_samples = np.zeros(sd.num_individuals, dtype=np.int32)
        for j, ind in enumerate(sd.individuals()):
            strain = ind.metadata["strain"]
            if strain not in name_map:
                raise ValueError(f"Strain {strain} not in ts nodes")
            ts_samples[j] = name_map[strain]
            # print(ind.metadata["strain"])
        ts_vars = ts.variants(samples=ts_samples)
        vars_iter = zip(ts_vars, sd.variants())
        with click.progressbar(vars_iter, length=ts.num_sites) as bar:
            for ts_var, sd_var in bar:
                ts_a = np.array(ts_var.alleles)
                sd_a = np.array(sd_var.alleles)
                non_missing = sd_var.genotypes != -1
                # Convert to actual allelic observations here because
                # allele encoding isn't stable
                ts_chars = ts_a[ts_var.genotypes[non_missing]]
                sd_chars = sd_a[sd_var.genotypes[non_missing]]
                if not np.all(ts_chars == sd_chars):
                    print(ts_chars)
                    print(sd_chars)
                    raise ValueError("Data mismatch")


@click.group()
def cli():
    pass


cli.add_command(import_usher_vcf)
cli.add_command(split_samples)
cli.add_command(infer)
cli.add_command(validate)
