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


