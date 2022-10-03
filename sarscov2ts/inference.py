import logging
import datetime

import tqdm
import tsinfer
import numpy as np


def _parse_date(date):
    return datetime.datetime.fromisoformat(date)


def infer(
    sd,
    *,
    ancestors_ts=None,
    num_mismatches=None,
    show_progress=False,
    daily_prefix=None,
    **kwargs,
):
    if num_mismatches is None:
        # Default to no recombination
        num_mismatches = 1000

    dates = np.array([ind.metadata["date"] for ind in sd.individuals()])
    unique_dates = np.unique(dates)
    extender = tsinfer.SequentialExtender(
        sd, ancestors_ts=ancestors_ts, time_units="days_ago"
    )
    base_proba = 1e-3
    ls_recomb = np.zeros(sd.num_sites - 1) + base_proba
    ls_mismatch = np.zeros(sd.num_sites) + base_proba * 10
    ts = ancestors_ts

    previous_date = None
    if ancestors_ts is not None:
        previous_date = _parse_date(ts.node(ts.samples()[-1]).metadata["date"])

    for date in unique_dates:
        current = _parse_date(date)
        if previous_date is None:
            increment = 1
        else:
            diff = current - previous_date
            increment = diff.days

        samples = np.where(dates == date)[0]
        logging.info(f"date={date} {len(samples)} samples")
        ts = extender.extend(
            samples, num_mismatches=num_mismatches, time_increment=increment, **kwargs
        )
        if daily_prefix is not None:
            filename = f"{daily_prefix}{date}.ts"
            ts.dump(filename)
            logging.info(f"Storing daily result to {filename}")
        previous_date = current
    return ts


def _validate_dates(ts):
    """
    Check that the time in the ts is days-ago in sync with the date
    metadata field.
    """
    samples = ts.samples()
    today = _parse_date(ts.node(samples[-1]).metadata["date"])
    for u in samples:
        node = ts.node(u)
        date = _parse_date(node.metadata["date"])
        diff = today - date
        assert diff.seconds == 0
        assert diff.microseconds == 0


def validate(sd, ts, show_progress=False):
    """
    Check that the ts contains all the data in the sample data.
    """
    assert ts.time_units == "days_ago"
    assert ts.num_sites == sd.num_sites
    name_map = {ts.node(u).metadata["strain"]: u for u in ts.samples()}
    ts_samples = np.zeros(sd.num_individuals, dtype=np.int32)
    for j, ind in enumerate(sd.individuals()):
        strain = ind.metadata["strain"]
        if strain not in name_map:
            raise ValueError(f"Strain {strain} not in ts nodes")
        ts_samples[j] = name_map[strain]

    _validate_dates(ts)

    ts_vars = ts.variants(samples=ts_samples)
    vars_iter = zip(ts_vars, sd.variants())
    with tqdm.tqdm(vars_iter, total=ts.num_sites, disable=not show_progress) as bar:
        for ts_var, sd_var in bar:
            ts_a = np.array(ts_var.alleles)
            sd_a = np.array(sd_var.alleles)
            non_missing = sd_var.genotypes != -1
            # Convert to actual allelic observations here because
            # allele encoding isn't stable
            ts_chars = ts_a[ts_var.genotypes[non_missing]]
            sd_chars = sd_a[sd_var.genotypes[non_missing]]
            if not np.all(ts_chars == sd_chars):
                raise ValueError("Data mismatch")
