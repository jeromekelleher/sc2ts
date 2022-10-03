import datetime

import numpy as np
import pytest
import tsinfer
import tskit

import sarscov2ts as sc2ts
import sarscov2ts.convert as convert
import sarscov2ts.inference as inference


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


class TestInference:
    def test_daily_prefix(self, tmp_path, sd_fixture):
        prefix = str(tmp_path) + "/x"
        ts = sc2ts.infer(sd_fixture, daily_prefix=prefix)
        paths = sorted(list(tmp_path.glob("x*")))
        dailies = [tskit.load(x) for x in paths]
        assert len(dailies) > 0
        ts.tables.assert_equals(dailies[-1].tables)

    @pytest.mark.parametrize("num_mismatches", [1, 2, 4, 1000])
    def test_integrity(self, sd_fixture, num_mismatches):
        ts = sc2ts.infer(sd_fixture, num_mismatches=num_mismatches)
        assert ts.sequence_length == 29904
        inference.validate(sd_fixture, ts)


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

    def test_integrity(sd_fixture):
        inference.validate(sd_fixture, ts_fixture)
