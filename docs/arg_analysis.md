(sec_arg_analysis)=
# ARG analysis


## ARG analysis API

The sc2ts API provides two convenience functions to compute summary
dataframes for the nodes and mutations in a sc2ts-output ARG.

To see some examples, first download the (31MB) sc2ts inferred ARG
from [Zenodo](https://zenodo.org/records/17558489/):

```
curl -O https://zenodo.org/records/17558489/files/sc2ts_viridian_v1.2.trees.tsz
```

We can then use these like

```python
import sc2ts
import tszip

ts = tszip.load("sc2ts_viridian_v1.2.trees.tsz")

df_node = sc2ts.node_data(ts)
df_mutation = sc2ts.mutation_data(ts)
```

See the [live demo](https://tskit.dev/explore/lab/index.html?path=sc2ts.ipynb)
for a browser based interactive demo of using these dataframes for
real-time pandemic-scale analysis.

## Dataset API

Sc2ts also provides a convenient API for accessing large-scale
alignments and metadata stored in
[VCF Zarr](https://doi.org/10.1093/gigascience/giaf049) format.

Resources:

- See this [notebook](https://github.com/jeromekelleher/sc2ts-paper/blob/main/notebooks/example_data_processing.ipynb)
for an example in which we access the data variant-by-variant and
which explains the low-level data encoding
- See the [VCF Zarr publication](https://doi.org/10.1093/gigascience/giaf049)
for more details on and benchmarks on this dataset.


**TODO** Add some references to API documentation



