# sc2ts

`sc2ts` (SARS-CoV-2 to tree sequence, pronounced "scoots") provides tools
to infer and analyse tskit ancestral recombination graphs (ARGs) for SARS-CoV-2
at pandemic scale.
It consists of:

1. A CLI-driven method to infer ARGs from SARS-CoV-2 data.
2. A lightweight wrapper around the {mod}`tskit` Python APIs, specialised
   for the output of sc2ts and enabling efficient node metadata access.
3. A lightweight wrapper around [Zarr](https://zarr.dev) for convenient access to the
   Viridian dataset (alignments and metadata) in VCF Zarr format.

The underlying methods are described in the sc2ts [preprint](
<https://www.biorxiv.org/content/10.1101/2023.06.08.544212v2>).

Most users will use the {ref}`sec_python_api` to perform {ref}`sec_arg_analysis`
on the sc2ts inferred ARG or {ref}`sec_alignments_analysis` on the
Zarr-formatted Viridian dataset distributed on Zenodo.

Uses who wish to perform {ref}`sec_inference` use the
{ref}`sc2ts_sec_cli`.
