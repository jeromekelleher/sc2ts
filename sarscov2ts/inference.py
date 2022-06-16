import logging

import tsinfer
import numpy as np

def infer(sd, *, mismatch_ratio=None, file_pattern=None, **kwargs):
    if mismatch_ratio is None:
        # Default to no recombination
        mismatch_ratio = 1e20
    dates = np.array([ind.metadata["date"] for ind in sd.individuals()])
    unique_dates = np.unique(dates)

    extender = tsinfer.SequentialExtender(sd)
    for date in unique_dates:
        samples = np.where(dates == date)[0]
        logging.info(f"date={date} {len(samples)} samples")
        ts = extender.extend(
            samples, mismatch_ratio=mismatch_ratio, recombination_rate=1e-8, **kwargs)
        if file_pattern is not None:
            path = file_pattern.format(date=date)
            ts.dump(path)
    return ts

