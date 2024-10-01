import numpy as np

import tqdm

from . import alignments
from . import core


def _validate_samples(ts, samples, alignment_store, show_progress):
    strains = [ts.node(u).metadata["strain"] for u in samples]
    G = np.zeros((ts.num_sites, len(samples)), dtype=np.int8)
    keep_sites = ts.sites_position.astype(int)
    strains_iter = enumerate(strains)
    with tqdm.tqdm(
        strains_iter,
        desc="Read",
        total=len(strains),
        position=1,
        leave=False,
        disable=not show_progress,
    ) as bar:
        for j, strain in bar:
            a = alignments.encode_alignment(alignment_store[strain])
            G[:, j] = a[keep_sites]

    vars_iter = ts.variants(samples=samples, alleles=tuple(core.ALLELES))
    with tqdm.tqdm(
        vars_iter,
        desc="Check",
        total=ts.num_sites,
        position=1,
        leave=False,
        disable=not show_progress,
    ) as bar:
        for var in bar:
            original = G[var.site.id]
            non_missing = original != -1
            if not np.all(var.genotypes[non_missing] == original[non_missing]):
                raise ValueError("Data mismatch")


def validate(ts, alignment_store, show_progress=False):
    """
    Check that all the samples in the specified tree sequence are correctly
    representing the original alignments.
    """
    samples = ts.samples()[1:]
    chunk_size = 10**3
    offset = 0
    num_chunks = ts.num_samples // chunk_size
    for chunk_index in tqdm.tqdm(
        range(num_chunks), position=0, disable=not show_progress
    ):
        chunk = samples[offset : offset + chunk_size]
        offset += chunk_size
        _validate_samples(ts, chunk, alignment_store, show_progress)

    if ts.num_samples % chunk_size != 0:
        chunk = samples[offset:]
        _validate_samples(ts, chunk, alignment_store, show_progress)
