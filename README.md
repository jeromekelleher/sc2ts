# sc2ts
Infer a succinct tree sequence from SARS-COV-2 variation data

**This is an early alpha version not intended for production use!!**

If you are interested in helping to develop sc2ts or would like to
work with the inferred ARGS, please get in touch.

## Installation

** TODO document local install **

## Inference workflow

### Command line inference

Inference is intended to be run from the command-line primarily,
and most likely orchestrated via a shell script or Snakemake file, etc.

The CLI is split into subcommands. Get help by running the CLI without arguments:

```
python3 -m sc2ts
```

**TODO document the process of getting a Zarr dataset and using it**


## Licensing

The code is marked as licensed under the MIT license,
but because the current implementation is used the matching
engine from tsinfer (which is GPL licensed) this code is
therefore also GPL.

However, we plan to switch out the matching engine for an
implementation provided by tskit, which is MIT licensed.
This will be done before the first official release.


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


