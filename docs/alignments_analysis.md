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


(sec_alignments_analysis)=

# Alignments analysis

## Prerequisites

Download the first 1000 samples of the Viridian dataset (450K):

```
curl -O https://raw.githubusercontent.com/tskit-dev/sc2ts/refs/heads/main/docs/viridian_mafft_subset_1000_v1.vcz.zip
```

We'll use this small subset as an example throughout.


## Loading and getting information

To load up a dataset, we use the {class}`Dataset` constructor:


```{code-cell}
import sc2ts

ds = sc2ts.Dataset("viridian_mafft_subset_1000_v1.vcz.zip")
ds
```

When we return the dataset object from a notebook cell (as here) it prints
out a summary of the contents. Here, we're working with a small subset
of the Viridian dataset consisting of the first 1000 samples at the 29903
sites in the SARS-CoV-2 genome.

The basic information is also available in the {attr}`Dataset.num_samples`
and {attr}`Dataset.num_variants` attributes.

To get information on the metadata fields that are present, we can use

```{code-cell}
ds.metadata.field_descriptors()
```
:::{warning}
The ``description`` column is currently empty because of a bug in the
data ingest pipeline for the Virian data. Later versions will include
this information so that the dataset is self-describing.
See [GitHub issue](https://github.com/tskit-dev/sc2ts/issues/579).
:::



## Accessing per-sample information

The easiest way to get information about a single sample is through the
the ``.metadata`` and ``.haplotypes`` interfaces. First, let's get
the sample IDs for the first 10 samples:

```{code-cell}
ds.sample_id[:10]
```
Then, we can get the metadata for a given sample as a dictionary using
the {attr}`Dataset.metadata` interface:

```{code-cell}
ds.metadata["SRR11597146"]
```

Similarly, we can get the integer encoded alignment for a sample using
the {attr}`Dataset.alignment` interface:

```{code-cell}
ds.alignment["SRR11597146"]
```

:::{seealso}
See the section {ref}`sec_alignments_analysis_data_encoding` for
details on the integer encoding for alignment data used here.
:::

Both the ``.metadata`` and ``.aligments`` interfaces are **cached**
(avoiding repeated decompression of the same underlying Zarr chunks)
and support iteration, and so provide an efficient way of accessing
data in bulk. For example, here we compute the mean number of
gap ("-") characters per sample:

```{code-cell}
import numpy as np

GAP = sc2ts.IUPAC_ALLELES.index("-")

gap_count = np.zeros(ds.num_samples)
for j, a in enumerate(ds.alignment.values()):
    gap_count[j] = np.sum(a == GAP)
np.mean(gap_count)
```


(sec_alignments_analysis_data_encoding)=

## Alignment data encoding

Stuff


