import datetime

import numpy as np
import pytest
import tsinfer
import tskit

import sarscov2ts as sc2ts
import sarscov2ts.convert as convert


@pytest.fixture
def sd_fixture(tmp_path):
    return convert.to_samples(
        "tests/data/100-samplesx100-sites.vcf",
        "tests/data/100-samplesx100-sites.metadata.tsv",
        str(tmp_path / "tmp.samples"),
    )


@pytest.fixture
def ts_fixture(sd_fixture):
    return sc2ts.infer(sd_fixture)


class TestSubsetInferenceDefaults:
    def test_metadata(self, ts_fixture):
        for node in ts_fixture.nodes():
            if node.flags == 0:
                assert node.metadata == {}
            elif node.flags == tskit.NODE_IS_SAMPLE:
                assert "strain" in node.metadata
            else:
                assert node.flags == tsinfer.NODE_IS_IDENTICAL_SAMPLE_ANCESTOR
                assert node.metadata == {}

    # def test_stuff(self, ts_fixture):
    #     print(ts_fixture.draw_text())

    #     # for haplotype in ts_fixture.haplotypes():
    #     #     print(haplotype)
    #     # print(ts_fixture.genotype_matrix())
    #     for var in ts_fixture.variants():
    #         print(var)

    #     print(ts_fixture.tables.mutations)
