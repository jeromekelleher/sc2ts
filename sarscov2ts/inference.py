import logging

import tsinfer
import numpy as np


def infer(sd, *, num_mismatches=None, ancestors_ts=None, **kwargs):
    if num_mismatches is None:
        # Default to no recombination
        num_mismatches = 1000

    dates = np.array([ind.metadata["date"] for ind in sd.individuals()])
    unique_dates = np.unique(dates)
    extender = tsinfer.SequentialExtender(sd, ancestors_ts=ancestors_ts)
    base_proba = 1e-3
    ls_recomb = np.zeros(sd.num_sites - 1) + base_proba
    ls_mismatch = np.zeros(sd.num_sites) + base_proba * 10
    for date in unique_dates:
        samples = np.where(dates == date)[0]
        logging.info(f"date={date} {len(samples)} samples")
        ts = extender.extend(samples, num_mismatches=num_mismatches, **kwargs)
        yield date, ts
