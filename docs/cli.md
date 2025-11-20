
(sc2ts_sec_cli)=

# Command line interface

The ``sc2ts`` package provides a command line interface for running
inference and working with sc2ts datasets. After installation, the
``sc2ts`` entry point should be available

```
$ sc2ts --help
```

You can also invoke the CLI via the module::
```
$ python -m sc2ts --help
```

## Order of high-level commands

In a typical end-to-end workflow, the main subcommands are used in the
following order:

1. ``import-alignments`` and ``import-metadata`` to build a VCF Zarr
   dataset from raw alignments and metadata.
2. ``infer`` to run primary inference over the dataset and produce a
   series of tree sequence files and a match database.
3. ``postprocess`` to apply housekeeping steps and incorporate exact
   matches, outputting a cleaned ARG.
4. ``minimise-metadata`` to generate an analysis-ready ARG with compact
   metadata suitable for use with the Python analysis APIs.


## CLI reference

<!-- Below we list all subcommands and options provided by the CLI. This -->
<!-- output is generated directly from the Click definitions in -->
<!-- ``sc2ts.cli`` using the ``sphinx-click`` extension, and so stays in -->
<!-- sync with the implementation. -->

:::{todo}
Add the sphinx-click output here somehow.
:::

<!-- ```{eval-rst} -->
<!-- .. click:: sc2ts.cli:cli -->
<!--    :prog: sc2ts infer -->
<!--    :nested: full -->
<!-- ``` -->


