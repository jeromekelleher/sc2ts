import numpy as np
import pytest
import tsinfer
import tskit

import sc2ts
from sc2ts import inference
from sc2ts import core
from sc2ts import convert



# @pytest.fixture
def small_sd_fixture():
    reference = core.get_reference_sequence()
    print(reference)
    fasta = {"REF": reference}
    rows = [{"strain": "REF"}]
    sd = convert.convert_alignments(rows, fasta)

    return sd

    # ref = core.get_reference_sequence()
    # with tsinfer.SampleData(sequence_length=len(ref)) as sd:
    #     sd.add_individual(
    #         metadata={
    #             "strain": "A",
    #             "date": "2019-12-30",
    #             "date_submitted": "2020-01-02",
    #         }
    #     )
    #     sd.add_individual(
    #         metadata={
    #             "strain": "B",
    #             "date": "2020-01-01",
    #             "date_submitted": "2020-02-02",
    #         }
    #     )
    #     sd.add_individual(
    #         metadata={
    #             "strain": "C",
    #             "date": "2020-01-01",
    #             "date_submitted": "2020-02-02",
    #         }
    #     )
    #     sd.add_individual(
    #         metadata={
    #             "strain": "D",
    #             "date": "2020-01-02",
    #             "date_submitted": "2022-02-02",
    #         }
    #     )
    #     sd.add_individual(
    #         metadata={
    #             "strain": "E",
    #             "date": "2020-01-06",
    #             "date_submitted": "2020-02-02",
    #         }
    #     )
    # for
    # return sd

class TestInitialTables:
    def test_site_schema(self):
        sd = small_sd_fixture()
        pass



@pytest.mark.skip()
class TestInference:
    def test_small_sd_times(self, small_sd_fixture):
        ts = sc2ts.infer(small_sd_fixture)
        inference.validate(small_sd_fixture, ts)
        # Day 0 is Jan 6, and ultimate ancestor is one day older than the
        # real root (reference)
        np.testing.assert_array_equal(ts.nodes_time, [9, 8, 7, 5, 5, 4, 0])

    def test_small_sd_submission_delay(self, small_sd_fixture):
        ts = sc2ts.infer(small_sd_fixture, max_submission_delay=100)
        strains = [ts.node(u).metadata["strain"] for u in ts.samples()]
        # Strain D should be filtered out.
        assert strains == ["A", "B", "C", "E"]
        with pytest.raises(ValueError):
            inference.validate(small_sd_fixture, ts)
        inference.validate(small_sd_fixture, ts, max_submission_delay=100)

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


@pytest.mark.skip()
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

    def test_integrity(self, sd_fixture, ts_fixture):
        inference.validate(sd_fixture, ts_fixture)
