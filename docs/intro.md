# sc2ts

`sc2ts` (SARS-CoV-2 to tree sequence, pronounced "scoots") provides tools
to infer and analyse tskit ancestral recombination graphs (ARGs) for SARS-CoV-2
at pandemic scale.
It consists of:

1. A CLI-driven method to infer ARGs from SARS-CoV-2 data.
2. A lightweight wrapper around the :mod:`tskit` Python APIs, specialised
   for the output of sc2ts and enabling efficient node metadata access.
3. A lightweight wrapper around :mod:`zarr` for convenient access to the
   Viridian dataset (alignments and metadata) in VCF Zarr format.

The underlying methods are described in the sc2ts pre-print:
<https://www.biorxiv.org/content/10.1101/2023.06.08.544212v2>.

Most users will run sc2ts via the command line interface,
which drives inference and postprocessing steps (see the
{ref}`CLI documentation <sc2ts_sec_cli>`). The Python API is intended for
working with tree sequences and datasets produced by sc2ts (see the
{ref}`Python API reference <api>`).

For an overview and examples, see the project README and associated
notebooks in the repository root.

## Installation

Install sc2ts from PyPI:

```sh
python -m pip install sc2ts
```

This installs the minimal requirements for the analysis and dataset APIs.
To run inference from the command line, install the optional inference
dependencies:

```sh
python -m pip install 'sc2ts[inference]'
```

## Quick start: ARG analysis

To compute summary dataframes for nodes and mutations in an inferred ARG,
you can load an sc2ts tree sequence and call the analysis helpers. For
example, download the sc2ts paper ARG from Zenodo:

```sh
curl -O https://zenodo.org/records/17558489/files/sc2ts_viridian_v1.2.trees.tsz
```

and then:

```python
import sc2ts
import tszip

ts = tszip.load("sc2ts_viridian_v1.2.trees.tsz")
df_node = sc2ts.node_data(ts)
df_mutation = sc2ts.mutation_data(ts)
```

See the {ref}`Python API reference <api>` for full details of these
functions.

## Quick start: CLI inference

To run inference locally using the example Viridian dataset and config:

1. Install the inference extras (if you have not already):

   ```sh
   python -m pip install 'sc2ts[inference]'
   ```

2. Download the Viridian dataset in VCF Zarr format:

   ```sh
   curl -O https://zenodo.org/records/16314739/files/viridian_mafft_2024-10-14_v1.vcz.zip
   ```

3. Run primary inference using the CLI and the example config in this repo:

   ```sh
   python -m sc2ts infer example_config.toml --stop=2020-02-02
   ```

   This will produce a series of `.ts` files and a match database in the
   output directory specified by the config (see the README for details).

4. Postprocess and generate an analysis-ready ARG:

   ```sh
   python -m sc2ts postprocess -vv \
       --match-db example_inference/ex1.matches.db \
       example_inference/ex1/ex1_2020-02-01.ts \
       example_inference/ex1_2020-02-01_pp.ts

   python -m sc2ts minimise-metadata \
       -m strain sample_id \
       -m Viridian_pangolin pango \
       example_inference/ex1_2020-02-01_pp.ts \
       example_inference/ex1_2020-02-01_pp_mm.ts
   ```

   The file `example_inference/ex1_2020-02-01_pp_mm.ts` can then be used
   with the Python analysis APIs shown above.

See the {ref}`CLI documentation <sc2ts_sec_cli>` for a complete listing of
subcommands and options.
