import tsinfer
import numpy as np

def infer(sd, *, file_pattern=None, **kwargs):
    dates = np.array([ind.metadata["date"] for ind in sd.individuals()])
    unique_dates = np.unique(dates)

    extender = tsinfer.SequentialExtender(sd)
    for date in unique_dates:
        samples = np.where(dates == date)[0]
        print(date, len(samples))
        ts = extender.extend(samples, **kwargs)
        if file_pattern is not None:
            path = file_pattern.format(date=date)
            ts.dump(path)
    return ts

