import logging

import tsinfer
import numpy as np

def infer(sd, *, mismatch_ratio=None, **kwargs):
    if mismatch_ratio is None:
        # Default to no recombination
        mismatch_ratio = 1e10

    dates = np.array([ind.metadata["date"] for ind in sd.individuals()])
    unique_dates = np.unique(dates)
    extender = tsinfer.SequentialExtender(sd)
    base_proba = 1e-3
    ls_recomb = np.zeros(sd.num_sites - 1) + base_proba
    ls_mismatch = np.zeros(sd.num_sites) + base_proba * 10
    for date in unique_dates:
        samples = np.where(dates == date)[0]
        logging.info(f"date={date} {len(samples)} samples")
        ts = extender.extend(samples,
                recombination=ls_recomb,
                mismatch=ls_mismatch,
                **kwargs)
        yield date, ts


