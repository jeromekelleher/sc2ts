(sec_python_api)=

# Python API

This page documents the public Python API exposed by ``sc2ts``.
Inference is driven via the command line interface (see the
{ref}`CLI documentation <sc2ts_sec_cli>`); the functions and classes
listed here are intended for working with tree sequences and datasets
that have already been generated.


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

## Alignment and metadata analysis

```{eval-rst}
.. autosummary::
   Dataset
   decode_alleles
   mask_ambiguous
   mask_flanking_deletions
```

```{eval-rst}
.. autoclass:: Dataset
   :members:

.. autoclass:: Variant
   :members:

.. autofunction:: decode_alleles

.. autofunction:: mask_ambiguous

.. autofunction:: mask_flanking_deletions
```

## Core constants and helpers

```{eval-rst}
.. autosummary::
   decode_flags
   flags_summary
```

```{eval-rst}
.. autofunction:: decode_flags

.. autofunction:: flags_summary
```
