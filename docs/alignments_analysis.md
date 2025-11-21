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

:::{warning}
The arrays returned by the ``alignment`` interface are **zero based** and you
must compensate to use **one-based** coordinates.
:::

If you want to access
specific slices of the array based on **one-based** coordinates, it's important
to take the zero-based nature of this into account. Suppose we wanted to
access the first 10 bases of Spike for a given sample. The first
base of Spike is 21563 in standard one-based coordinates. While we could do
some arithmetic to compensate, the simplest way to translate is to simply
prepend some value to the alignment array:

```{code-cell}
a = np.append([-1], ds.alignment["SRR11597146"])
spike_start = 21_563
a[spike_start: spike_start + 10]
```

(sec_alignments_analysis_data_encoding)=

## Alignment data encoding

A key element of processing data efficiently in [tskit](https://tskit.dev) and VCF
Zarr is to use numpy
arrays of integers to represent allelic states, instead of the classical
approach of using strings. In sc2ts, alleles are given fixed integer
representations, such that A=0, C=1, G=2, and T=3. So, to represent the DNA
string "AACTG" we would use the numpy array [0, 0, 1, 3, 2] instead. This has
many advantages and makes it much easier to write efficient code.

The drawback of this is that it's not as easy to inspect and debug, and we must
always be aware of the translation required.

Sc2ts provides some utilities for doing this. The easiest way to get the string
values is to use {func}`decode_alleles` function:

```{code-cell}
a = sc2ts.decode_alleles(ds.alignment["SRR11597146"])
a
```
This is a numpy string array, which can still be processed quite efficiently.
However, it is best to stay in native integer encoding where possible, as it
is much more efficient.


Sc2ts uses the [IUPAC](https://www.bioinformatics.org/sms/iupac.html)
uncertainty codes to encode ambiguous bases, and the {attr}`sc2ts.IUPAC_ALLELES`
variable stores the mapping from these values to their integer indexes.

```{code-cell}
sc2ts.IUPAC_ALLELES
```

Thus, "A" corresponds to 0, "-" to 4 and so on.


### Missing data

Missing data is an important element of the data model. Usually, missing data is
encoded as an "N" character in the alignments. Howevever, there is no "N"
in the ``IUPAC_ALLELES`` list above. This is because missing data is handled specially
in VCF Zarr by mapping to the reserved ``-1`` value. Missing data can therefore be flagged
easily and handled correctly by downstream utilities.

:::{warning}
It is important to take this into account when translating the integer encoded data into
strings, because -1 is interpreted as the last element of the list in Python. Please
use the {func}`decode_alleles` function to avoid this tripwire.
:::


## Accessing by variant

A unique feature of the VCF Zarr encoding used here is that we can efficiently access
the alignment data by sample **and** by site. The best way to access data by site
is to use the {meth}`Dataset.variants` method.

:::{note}
The {meth}`Dataset.variants` method is deliberately designed to follow the semantics
of the corresponding [tskit](https://tskit.dev) function
({meth}`tskit.TreeSequence.variants`), enabling straightforward joint analysis of the
ARG and alignments.
:::

Here we use this interface to count the number of samples that carry the gap characters
at each site along the genome:

```{code-cell}
GAP = sc2ts.IUPAC_ALLELES.index("-")

gap_count = np.zeros(ds.num_variants)
for j, var in enumerate(ds.variants()):
    gap_count[j] = np.sum(var.genotypes == GAP)

gap_count
```

Here, we can see that all 1000 samples in our small subset have flanking deletions called.

We can use the ``position`` argument to supply a list of (**one-based**) site positions
of interest:

```{code-cell}
spike_pos = np.arange(21_563, 25_385)
gap_count = np.zeros_like(spike_pos)
for j, var in enumerate(ds.variants(position=spike_pos)):
    gap_count[j] = np.sum(var.genotypes == GAP)

gap_count
```

We can also use the ``sample_id`` argument to specify subsets of samples.

## Bulk metadata analysis

Accessing the metadata row-by-row using the ``.metadata`` mapping above is
inefficient when we want to look at large numbers of samples. In this case,
it is much more convenient to export the metadata to a Pandas dataframe
using the {meth}`Dataset.metadata_dataframe` and then work with this.

```{code-cell}
df = ds.metadata_dataframe()
df
```

Then, suppose we want to find all samples from the USA:

```{code-cell}
usa_samples = df[df["Country"] == "USA"].index
usa_samples
```

:::{important}
For performance reasons it's a good idea to use the ``fields`` parameter to
{meth}`Dataset.metadata_dataframe` to limit the amount of metadata decoded
to what you actually need.
:::


## Getting FASTA output

Getting FASTA output is straightforward using the {meth}`Dataset.write_fasta`
method. Here, we use the ``sample_id`` argument to write the FASTA aligments
of the USA samples found in the last example:

```{code-cell}
with open("/tmp/usa.fa", "w") as f:
    ds.write_fasta(f, sample_id=usa_samples)
```


