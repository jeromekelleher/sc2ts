import datetime

import numpy as np
import pytest
import tsinfer

import sarscov2ts as sc2ts
import sarscov2ts.convert as convert

@pytest.fixture
def sd_fixture(tmp_path):
    return convert.to_samples(
        "tests/data/100-samplesx100-sites.vcf",
        "tests/data/100-samplesx100-sites.metadata.tsv",
        str(tmp_path / "tmp.samples"),
    )


class TestSubsetInference:

    def test_defaults(self, sd_fixture):

        ts = sc2ts.infer(sd_fixture)
        print(ts.draw_text())
