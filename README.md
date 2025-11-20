# sc2ts

Sc2ts stands for "SARS-CoV-2 to tree sequence" (pronounced "scoots" optionally)
and consists of

1. A method to infer Ancestral Recombination Graphs (ARGs) from SARS-CoV-2
data at pandemic scale
2. A lightweight wrapper around [tskit Python APIs](https://tskit.dev/tskit/docs/stable/python-api.html) specialised for the output of sc2ts which enables efficient node metadata
access.
3. A lightweight wrapper around [Zarr Python](https://zarr.dev) which enables
convenient and efficient access to the full Viridian dataset (alignments and metadata)
in a single file using the [VCF Zarr specification](https://doi.org/10.1093/gigascience/giaf049).

Please see the online [documentation](https://tskit.dev/sc2ts/docs) for details
on the software
and the [preprint](https://www.biorxiv.org/content/10.1101/2023.06.08.544212v2)
for information on the method and the inferred ARG.

