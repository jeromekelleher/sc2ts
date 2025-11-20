# sc2ts

Sc2ts stands for "SARS-CoV-2 to tree sequence" (pronounced "scoots" optionally)
and consists of

1. A method to infer Ancestral Recombination Graphs (ARGs) from SARS-CoV-2
data at pandemic scale
2. A lightweight wrapper around [tskit Python APIs](https://tskit.dev/tskit/docs/stable/python-api.html) specialised for the output of sc2ts which enables efficient node metadata
access.
3. A lightweight wrapper around [Zarr Python](https://zarr.dev) which enables
convenient and efficient access to the full Viridian dataset (alignments and metadata)
in a single file using the [VCF Zarr specification](https://doi.org/10.1093/gigascience/giaf049).

Please see the [preprint](https://www.biorxiv.org/content/10.1101/2023.06.08.544212v2)
for details.

## Installation

Install sc2ts from PyPI:

```
python -m pip install sc2ts
```

This installs the minimum requirement to enable the
[ARG analysis](#ARG-analysis-API) and [Dataset](#Dataset-API)s.
To run [inference](#inference), you must install some extra
dependencies using the 'inference' optional extra:

```
python -m pip install sc2ts[inference]
```

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



## Development

To run the unit tests, use

```
python3 -m pytest
```

You may need to regenerate some cached test fixtures occasionaly (particularly
if getting cryptic errors when running the test suite). To do this, run

```
rm -fR tests/data/cache/
```

and rerun tests as above.

### Debug utilities

The tree sequence files output during primary inference have a lot
of debugging metadata, and there are some developer tools for inspecting
this in the ``sc2ts.debug`` package. In particular, the ``ArgInfo``
class has a lot of useful utilities designed to be used in a Jupyter
notebook. Note that ``matplotlib`` is required for these. Use it like:

```python
import sc2ts.debug as sd
import tskit

ts = tskit.load("path_to_daily_inference.ts")
ai = sd.ArgInfo(ts)
ai # view summary in notebook
```


