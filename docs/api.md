# Python API

This page documents the public Python API exposed by ``sc2ts``.
Inference is driven via the command line interface (see the
{ref}`CLI documentation <sc2ts_sec_cli>`); the functions and classes
listed here are intended for working with tree sequences and datasets
that have already been generated.

The reference documentation is concise and exhaustive; for higher level
discussion and worked examples, see the project README and example
notebooks.

```{eval-rst}
.. currentmodule:: sc2ts
```

## ARG analysis

```{eval-rst}
.. autosummary::
   node_data
   mutation_data
```

```{eval-rst}
.. autofunction:: node_data

.. autofunction:: mutation_data
```

## Dataset access

```{eval-rst}
.. autosummary::
   Dataset
   decode_alignment
   mask_ambiguous
   mask_flanking_deletions
```

```{eval-rst}
.. autoclass:: Dataset
   :members:

.. autofunction:: decode_alignment

.. autofunction:: mask_ambiguous

.. autofunction:: mask_flanking_deletions
```

## Core constants and helpers

```{eval-rst}
.. autosummary::
   REFERENCE_STRAIN
   REFERENCE_DATE
   REFERENCE_GENBANK
   REFERENCE_SEQUENCE_LENGTH
   IUPAC_ALLELES
   decode_flags
   flags_summary
```

```{eval-rst}
.. autodata:: REFERENCE_STRAIN

.. autodata:: REFERENCE_DATE

.. autodata:: REFERENCE_GENBANK

.. autodata:: REFERENCE_SEQUENCE_LENGTH

.. autodata:: IUPAC_ALLELES

.. autofunction:: decode_flags

.. autofunction:: flags_summary
```
