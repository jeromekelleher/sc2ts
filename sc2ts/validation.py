import logging
import unittest

import numpy as np
import numpy.testing as nt
import tqdm

logger = logging.getLogger(__name__)

import sc2ts

MISSING = -1
DELETION = sc2ts.IUPAC_ALLELES.index("-")


def validate_genotypes(ts, dataset, deletions_as_missing=False, show_progress=False):
    sample_id = ts.metadata["sc2ts"]["samples_strain"]
    logger.info(f"Validating ARG for with {len(sample_id)} samples")
    bar = tqdm.tqdm(total=ts.num_sites, desc="Genotypes", disable=not show_progress)
    for var1, var2 in zip(
        ts.variants(alleles=tuple(sc2ts.IUPAC_ALLELES)),
        dataset.variants(sample_id, ts.sites_position),
    ):
        assert var1.site.position == var2.position
        g2 = sc2ts.mask_ambiguous(var2.genotypes)
        if deletions_as_missing:
            g2[g2 == DELETION] = -1
        select = g2 > 0
        nt.assert_array_equal(var1.genotypes[select], g2[select])
        bar.update()
    bar.close()


def validate_metadata(ts, dataset, show_progress=False, skip_fields=set()):

    samples = ts.samples()
    bar = tqdm.tqdm(samples, desc="Metadata", disable=not show_progress)
    for u in bar:
        md1 = ts.node(u).metadata
        keys = set(md1.keys()) - ({"sc2ts"} | skip_fields)
        md2 = dataset.metadata[md1["strain"]]
        md1 = {k: md1[k] for k in keys}
        md2 = {k: md2[k] for k in keys}
        if u == samples[0]:
            logger.info(f"Comparing {len(keys)} keys, e.g.: {md1})")
        unittest.TestCase().assertDictEqual(md1, md2)


def validate(ts, dataset, deletions_as_missing=False, show_progress=False):
    """
    Check that all the samples in the specified tree sequence are correctly
    representing the original alignments.
    """
    validate_genotypes(
        ts,
        dataset,
        deletions_as_missing=deletions_as_missing,
        show_progress=show_progress,
    )
    validate_metadata(ts, dataset, show_progress=show_progress)
