---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{eval-rst}
.. currentmodule:: sc2ts
```

(sec_arg_analysis)=
# ARG analysis

The sc2ts API provides some convenience functions to compute summary
dataframes for the nodes and mutations in a sc2ts-output ARG.


## Prerequisites

Download a subset of the [sc2ts Viridian ARG](https://zenodo.org/records/17558489/)
with 1000 samples:

```
curl -O https://raw.githubusercontent.com/tskit-dev/sc2ts/refs/heads/main/docs/sc2ts_viridian_v1.2_subset_1000.trees.tsz
```

We'll use this small subset as an example throughout.

## Loading


```{code-cell}
import sc2ts
import tszip

ts = tszip.load("sc2ts_viridian_v1.2_subset_1000.trees.tsz")
```

You can then use the full [tskit](https://tskit.dev/tskit/docs/)
Python API on this ARG.

## Node data

The {func}`node_data` function returns a Pandas dataframe of data for each
node in the ARG.

```{code-cell}
dfn = sc2ts.node_data(ts)
dfn
```


## Mutation data

The {func}`mutation_data` function returns a Pandas dataframe of data for each
mutation_in the ARG.

```{code-cell}
dfm = sc2ts.mutation_data(ts)
dfm
```

