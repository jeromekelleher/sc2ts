# sc2ts
Infer a succinct tree sequence from SARS-COV-2 variation data

**This is an early version not intended for production use!!**

If you are interested in helping to develop sc2ts or would like to 
work with the inferred ARGS, please get in touch.

## Installation

To run the downstream analysis utilties, install from pip using

```
python3 -m pip install sc2ts[analysis]
```

This installs matplotlib and some other heavyweight dependencies.

For just running the inference tools, use

```
python3 -m pip install sc2ts
```

## Inference workflow

### Command line inference

Inference is intended to be run from the command-line primarily,
and most likely orchestrated via a shell script or Snakemake file, etc.

The CLI is split into subcommands. Get help by running the CLI without arguments:

```
python3 -m sc2ts
```

### Import metadata to local database

Metadata for all samples must be available, and provided in a tab-separated
file. We need to convert from a standard text file to a SQLite database
so that we can quickly search for strains collected on  a given day, without
loading the entire set each time.

```
python3 -m sc2ts import-metadata data/metadata.tsv data/metadata.db
```

**TODO: Document required fields**

### Import alignments

To provide fast access to the individual alignments, we store them in a local
database file. These must be imported before inference can be performed.

The basic approach is to use the ``import-alignments`` command, with a
path to a ``alignments.db`` file which we are creating, and one or more
FASTA files that we are importing into it.

```bash
python3 -m sc2ts import-alignments data/alignments.db data/alignments/.fasta
```

By default the database file is updated each time, so this can be done
in stages.

**TODO discuss the storage and time requirements for this step!**


### Run the inference

The basic approach is to run the ``daily-extend`` command which runs the
basic extension operation day-by-day using the information
in the metadata DB.

```
python3 -m sc2ts daily-extend data/alignments.db data/metadata.db results/output-prefix
```

### Example run script

Here is a script used to run the inference for the Long ARG
in the preprint:

```
#!/bin/bash
set -e

precision=12
mismatches=3
max_submission_delay=30
max_daily_samples=1000
num_threads=40

datadir=data
run_id=upgma-mds-$max_daily_samples-md-$max_submission_delay-mm-$mismatches
resultsdir=results/$run_id
results_prefix=$resultsdir/$run_id-
logfile=logs/$run_id.log

options="--num-threads $num_threads -vv -l $logfile "
options+="--max-submission-delay $max_submission_delay "
options+="--max-daily-samples $max_daily_samples "
options+="--precision $precision --num-mismatches $mismatches"

mkdir -p $resultsdir

alignments=$datadir/alignments2.db
metadata=$datadir/metadata.filtered.db
# NOTE: we can start from a given data also with the -b option
# basets="$results_prefix"2022-01-24.ts
# options+=" -b $basets"

python3 -m sc2ts daily-extend $alignments $metadata $results_prefix $options
```
