import logging 

import numpy as np
import numpy.testing as nt
import tqdm

logger = logging.getLogger(__name__)

import sc2ts

MISSING = -1
DELETION = sc2ts.IUPAC_ALLELES.index("-")


def validate(ts, dataset, deletions_as_missing=False, show_progress=False):
    """
    Check that all the samples in the specified tree sequence are correctly
    representing the original alignments.
    """
    sample_id = ts.metadata["sc2ts"]["samples_strain"][1:]
    logger.info(f"Validating ARG for with {len(sample_id)} samples")
    bar = tqdm.tqdm(total=ts.num_sites, disable=not show_progress)
    for var1, var2 in zip(
        ts.variants(samples=ts.samples()[1:], alleles=tuple(sc2ts.IUPAC_ALLELES)),
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
