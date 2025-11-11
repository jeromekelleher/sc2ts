import pytest
import tszip

import sc2ts


@pytest.fixture
def fx_v1_ex():
    return tszip.load("tests/data/v0.1_output_ex.ts.tsz")


class TestBasics:

    def test_top_level_md(self, fx_v1_ex):
        ts = fx_v1_ex
        assert ts.metadata == {"time_zero_date": "2020-02-15"}

    def test_node_data(self, fx_v1_ex):
        ts = fx_v1_ex
        df = sc2ts.node_data(ts)
        assert df.shape[0] == ts.num_nodes

    def test_mutation_data(self, fx_v1_ex):
        ts = fx_v1_ex
        df = sc2ts.mutation_data(ts)
        assert df.shape[0] == ts.num_mutations
