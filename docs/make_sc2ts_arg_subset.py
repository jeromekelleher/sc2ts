import tszip
import numpy as np

ts = tszip.load("sc2ts_viridian_v1.2.trees.tsz")

k = 1000
idx = np.round(np.linspace(0, ts.num_samples - 1, k)).astype(int)

subset = ts.samples()[idx]
print(subset)
tss = ts.simplify(subset, filter_sites=False)

tszip.compress(tss, f"sc2ts_viridian_v1.2_subset_{k}.trees.tsz")
