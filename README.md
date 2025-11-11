# sc2ts
Infer a succinct tree sequence from SARS-COV-2 variation data


If you are interested in helping to develop sc2ts or would like to
work with the inferred ARGS, please get in touch.


Then, download the ARG in tszip format from
[Zenodo](https://zenodo.org/records/17558489/):

```
curl -O https://zenodo.org/records/17558489/files/sc2ts_viridian_v1.2.trees.tsz
```



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


## Inference

Here we'll run through a quick example of how to get inference running
on a local machine using an example config file, using the Viridian data downloaded
from Zenodo.

### Prerequisites

First, install the "inference" version of sc2ts from pypi:

```
python -m pip install sc2ts[inference]
```

**This is essential! The base install of sc2ts contains the minimal
dependencies required to access the analysis utilities outlined above.**

Then, download the Viridian dataset in
[VCF Zarr format](https://doi.org/10.1093/gigascience/giaf049) from
[Zenodo](https://zenodo.org/records/16314739):

```
curl -O https://zenodo.org/records/16314739/files/viridian_mafft_2024-10-14_v1.vcz.zip
```
### CLI

Inference is performed using the CLI, which is composed of number of subcommands.
See the online help for more information:

```
python -m sc2ts --help
```

### Primary inference

Primary inference is performed using the ``infer`` subcommand of the CLI,
and all parameters are specified using a toml file.

Then inference under the [example config](example_config.toml)
for little while to see how things work:

```
python3 -m sc2ts infer example_config.toml --stop=2020-02-02
```

Once this finishes (it should take a few minutes), the results of the
inference will be in the ``example_inference`` directory (as specified in the
config file) and look something like this:

```
$ tree example_inference
example_inference
├── ex1
│   ├── ex1_2020-01-01.ts
│   ├── ex1_2020-01-10.ts
│   ├── ex1_2020-01-12.ts
│   ├── ex1_2020-01-19.ts
│   ├── ex1_2020-01-24.ts
│   ├── ex1_2020-01-25.ts
│   ├── ex1_2020-01-28.ts
│   ├── ex1_2020-01-29.ts
│   ├── ex1_2020-01-30.ts
│   ├── ex1_2020-01-31.ts
│   ├── ex1_2020-02-01.ts
│   └── ex1_init.ts
├── ex1.log
└── ex1.matches.db
```

Here we've run inference for all dates in January 2020 for which we have data
and Feb 01. The results of inference for each day is stored in the
``example_inference/ex1`` directory as a tskit file representing the ARG
inferred up to that day. There is a lot of redundancy in keeping all these
daily files lying around, but it is useful to be able to go back to the
state of the ARG at a particular date and they don't take up much space.

The file ``ex1.log`` contains the log file. The config file set the log-level
to 2, which is full debug output. There is a lot of useful information in there,
and it can be very helpful when debugging, so we recommend keeping the logs.

The ``ex1.matches.db`` is the "match DB" which stores information about the
HMM match for each sample. This is mainly used to store exact matches
found during inference.

The ARGs output during primary inference (this step here) have a lot of
debugging metadata included (see the section on the Debug utilities below)

Primary inference can be stopped and picked up again at any point using
the ``--start`` option.


### Postprocessing

Once we've finished primary inference we can run postprocessing to perform
 a few housekeeping tasks. Continuing the example above:

```
$ python3 -m sc2ts postprocess -vv \
    --match-db example_inference/ex1.matches.db \
    example_inference/ex1/ex1_2020-02-01.ts     \
    example_inference/ex1_2020-02-01_pp.ts
```

Among other things, this incorporates the exact matches in the match DB
into the final ARG.

### Generating final analysis file

To generate the final analysis ready file (used as input to the analysis
APIs above) we need to run ``minimise-metadata``. This removes all but
the most necessary metadata from the ARG, and recodes node metadata
using the [struct codec](https://tskit.dev/tskit/docs/stable/metadata.html#structured-array-metadata)
for efficiency. On our example above:

```
$ python -m sc2ts minimise-metadata \
    -m strain sample_id \
    -m Viridian_pangolin pango \
    example_inference/ex1_2020-02-01_pp.ts \
    example_inference/ex1_2020-02-01_pp_mm.ts
```

This recodes the metadata in the input tree sequence such that
the existing ``strain`` field is renamed to ``sample_id``
(for compatibility with VCF Zarr) and the ``Viridian_pangolin``
field (extracted from the Viridian metadata) is renamed to ``pango``.

We can then use the analysis APIs on this file:
```python
import sc2ts
import tskit

ts = tskit.load("example_inference/ex1_2020-02-01_pp_mm.ts")
dfn = sc2ts.node_data(ts)
print(dfn)
```

giving something like:

```
   pango         sample_id  node_id  is_sample  is_recombinant  num_mutations       date
0         Vestigial_ignore        0      False           False              0 2019-12-25
1          Wuhan/Hu-1/2019        1      False           False              0 2019-12-26
2      A       SRR11772659        2       True           False              1 2020-01-19
3      B       SRR11397727        3       True           False              0 2020-01-24
4      B       SRR11397730        4       True           False              0 2020-01-24
..   ...               ...      ...        ...             ...            ...        ...
60     A       SRR11597177       60       True           False              0 2020-01-30
61     A       SRR11597197       61       True           False              0 2020-01-30
62     B       SRR11597144       62       True           False              0 2020-02-01
63     B       SRR11597148       63       True           False              0 2020-02-01
64     B       SRR25229386       64       True           False              0 2020-02-01
```

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

The tree sequences files output during primary inference have a lot
of debugging metadata, and there are some developer tools for inspecting
this in the ``sc2ts.debug`` package. In particular, the ``ArgInfo``
class has a lot of useful utilities designed to be used in a Jupyter
notebook. Use it like

```python
import sc2ts.debug as sd
import tskit

ts = tskit.load("path_to_daily_inference.ts")
ai = sd.ArgInfo(ts)
ai # view summary in notebook
```


