import pytest
import tszip

import sc2ts


@pytest.fixture
def fx_ts_v1():
    return tszip.load("tests/data/v0.1_output_ex.ts.tsz")


@pytest.fixture
def fx_ds_v1():
    return sc2ts.Dataset("tests/data/v0.1_dataset_ex.vcz.zip")


class TestStatFuncs:

    def test_top_level_md(self, fx_ts_v1):
        ts = fx_ts_v1
        assert ts.metadata == {"time_zero_date": "2020-02-15"}

    def test_node_data(self, fx_ts_v1):
        ts = fx_ts_v1
        df = sc2ts.node_data(ts)
        assert df.shape[0] == ts.num_nodes

    def test_mutation_data(self, fx_ts_v1):
        ts = fx_ts_v1
        df = sc2ts.mutation_data(ts)
        assert df.shape[0] == ts.num_mutations


class TestDataset:

    def test_basics(self, fx_ds_v1):
        ds = fx_ds_v1
        G = ds["call_genotype"][:].squeeze()
        pos = ds["variant_position"][:]
        assert G.shape == (29903, 55)
        assert pos.shape == (29903,)
        assert pos[0] == 1
        assert pos[-1] == 29903
