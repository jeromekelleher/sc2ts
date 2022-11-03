import logging
import datetime

import tqdm
import tsinfer
import numpy as np


def infer(
    sd,
    *,
    ancestors_ts=None,
    num_mismatches=None,
    show_progress=False,
    daily_prefix=None,
    max_submission_delay=None,
    **kwargs,
):
    if num_mismatches is None:
        # Default to no recombination
        num_mismatches = 1000
    if max_submission_delay is None:
        max_submission_delay = 10**8  # Arbitrary large number of days.

    max_submission_delay = np.timedelta64(max_submission_delay, "D")

    date = []
    date_submitted = []
    for ind in sd.individuals():
        date.append(ind.metadata["date"])
        date_submitted.append(ind.metadata["date_submitted"])
    date = np.array(date, dtype=np.datetime64)
    date_submitted = np.array(date_submitted, dtype=np.datetime64)

    submission_delay = date_submitted - date
    submission_delay = submission_delay.astype("timedelta64[D]")

    unique_dates = np.unique(date)
    extender = tsinfer.SequentialExtender(
        sd, ancestors_ts=ancestors_ts, time_units="days_ago"
    )
    base_proba = 1e-3
    ls_recomb = np.zeros(sd.num_sites - 1) + base_proba
    ls_mismatch = np.zeros(sd.num_sites) + base_proba * 10
    ts = ancestors_ts

    previous_date = None
    if ancestors_ts is not None:
        previous_date = np.datetime64(ts.node(ts.samples()[-1]).metadata["date"])

    for current_date in unique_dates:
        if previous_date is None:
            increment = 1
        else:
            diff = current_date - previous_date
            increment = diff.astype("timedelta64[D]").astype("int")
            assert increment > 0

        num_samples_for_date = np.sum(current_date == date)
        condition = np.logical_and(
            current_date == date, submission_delay <= max_submission_delay
        )
        samples = np.where(condition)[0]
        num_samples = len(samples)
        num_rejected = num_samples_for_date - num_samples
        fraction_rejected = num_rejected / num_samples_for_date

        logging.info(
            f"Filtered {num_rejected} ({100 * fraction_rejected:.2f}%) samples "
            f"with submission_delay > {max_submission_delay}"
        )
        logging.info(f"Extending for {current_date} with {len(samples)} samples")
        ts = extender.extend(
            samples, num_mismatches=num_mismatches, time_increment=increment, **kwargs
        )
        if daily_prefix is not None:
            filename = f"{daily_prefix}{current_date}.ts"
            ts.dump(filename)
            logging.info(f"Storing daily result to {filename}")
        previous_date = current_date
    return ts


def _parse_date(date):
    return datetime.datetime.fromisoformat(date)


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


def validate(sd, ts, max_submission_delay=None, show_progress=False):
    """
    Check that the ts contains all the data in the sample data.
    """
    assert ts.time_units == "days_ago"
    assert ts.num_sites == sd.num_sites
    if max_submission_delay is None:
        max_submission_delay = 10**9 - 1
    max_submission_delay = datetime.timedelta(days=max_submission_delay)
    name_map = {ts.node(u).metadata["strain"]: u for u in ts.samples()}
    ts_samples = []
    sd_samples = []
    for j, ind in enumerate(sd.individuals()):
        strain = ind.metadata["strain"]
        submission_delay = (
            _parse_date(ind.metadata["date_submitted"]) -
            _parse_date(ind.metadata["date"]))
        if submission_delay <= max_submission_delay:
            if strain not in name_map:
                raise ValueError(f"Strain {strain} not in ts nodes")
            sd_samples.append(j)
            ts_samples.append(name_map[strain])
        else:
            if strain in name_map:
                raise ValueError(f"Strain {strain} should have been filtered")
    sd_samples = np.array(sd_samples)
    ts_samples = np.array(ts_samples)

    _validate_dates(ts)

    ts_vars = ts.variants(samples=ts_samples)
    vars_iter = zip(ts_vars, sd.variants())
    with tqdm.tqdm(vars_iter, total=ts.num_sites, disable=not show_progress) as bar:
        for ts_var, sd_var in bar:
            ts_a = np.array(ts_var.alleles)
            sd_a = np.array(sd_var.alleles)

            sd_genotypes = sd_var.genotypes[sd_samples]
            non_missing = sd_genotypes != -1
            # Convert to actual allelic observations here because
            # allele encoding isn't stable
            ts_chars = ts_a[ts_var.genotypes[non_missing]]
            sd_chars = sd_a[sd_genotypes[non_missing]]
            if not np.all(ts_chars == sd_chars):
                raise ValueError("Data mismatch")
