import datetime
import inspect

import numpy as np
import numpy.testing as nt
import pandas as pd
import matplotlib
import pytest

import msprime
import tskit

import sc2ts
from sc2ts import info


@pytest.fixture
def fx_ti_2020_02_13(fx_ts_map):
    ts = fx_ts_map["2020-02-13"]
    return info.TreeInfo(ts, show_progress=False)


@pytest.fixture
def fx_ti_2020_02_15(fx_ts_map):
    ts = fx_ts_map["2020-02-15"]
    return info.TreeInfo(ts, show_progress=False)


@pytest.fixture
def fx_ts_min_2020_02_15(fx_ts_map):
    ts = fx_ts_map["2020-02-15"]
    field_mapping = {
        "strain": "sample_id",
        "Viridian_pangolin": "pango",
        "Viridian_scorpio": "scorpio",
    }
    return sc2ts.minimise_metadata(ts, field_mapping)


@pytest.fixture
def fx_ti_recombinant_example_1(fx_recombinant_example_1):
    return info.TreeInfo(fx_recombinant_example_1, show_progress=False)


def test_get_gene_coordinates():
    d = sc2ts.get_gene_coordinates()
    assert len(d) == 11
    assert d["S"] == (21563, 25385)


class TestCopyingTable:
    def test_copying_table(self, fx_ti_recombinant_example_1):
        # should work with a plain tree sequence
        ti = fx_ti_recombinant_example_1
        ts = ti.ts
        recombinants = np.where(ts.nodes_flags & sc2ts.NODE_IS_RECOMBINANT)[0]
        re_node = recombinants[0]
        ct_via_ti = ti.copying_table(re_node, child_label="TestChild")
        ct_via_ts = info.CopyingTable(ts, re_node).html(child_label="TestChild")
        assert "TestChild" in ct_via_ti
        assert ct_via_ti == ct_via_ts


class TestTreeInfo:
    def test_tree_info_values(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        assert list(ti.nodes_num_missing_sites[:5]) == [0, 0, 121, 693, 667]
        assert list(ti.sites_num_missing_samples[:5]) == [39] * 5
        assert list(ti.sites_num_deletion_samples[:5]) == [0] * 5

    @pytest.mark.parametrize(
        "method",
        [
            func
            for (name, func) in inspect.getmembers(info.TreeInfo)
            if name.startswith("plot")
        ],
    )
    def test_plots(self, fx_ti_2020_02_13, method):
        fig, axes = method(fx_ti_2020_02_13)
        assert isinstance(fig, matplotlib.figure.Figure)
        for ax in axes:
            assert isinstance(ax, matplotlib.axes.Axes)

    def test_exact_match_counts(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        counts = ti.ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["node"]
        for j in range(ti.ts.num_nodes):
            c = counts.get(str(j), 0)
            assert ti.nodes_num_exact_matches[j] == c

    def test_draw_pango_lineage_subtree(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        svg = ti.draw_pango_lineage_subtree("A")
        svg2 = ti.draw_subtree(tracked_pango=["A"])
        assert svg == svg2
        assert svg.startswith("<svg")
        for u in ti.pango_lineage_samples["A"]:
            assert f"node n{u}" in svg

    def test_draw_subtree(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        samples = ti.ts.samples()[0:5]
        svg = ti.draw_subtree(tracked_samples=samples)
        assert svg.startswith("<svg")
        for u in samples:
            assert f"node n{u}" in svg

    def test_resources_summary(self, fx_ti_2020_02_13):
        df = fx_ti_2020_02_13.resources_summary()
        assert df.shape[0] == 20
        assert np.all(df.index.astype(str).str.startswith("2020"))

    def test_samples_summary(self, fx_ti_2020_02_13):
        df = fx_ti_2020_02_13.samples_summary()
        assert np.all(df["total"] >= (df["inserted"] + df["exact_matches"]))
        assert df.shape[0] > 0

    def test_node_type_summary(self, fx_ti_2020_02_13):
        df = fx_ti_2020_02_13.node_type_summary()
        assert df.loc["S"].total == 39

    def test_sample_group_summary(self, fx_ti_2020_02_13):
        df = fx_ti_2020_02_13.sample_groups_summary()
        assert df.shape[0] == 26
        assert np.all(df["nodes"] >= df["samples"])
        assert np.all(df["nodes"] > 0)
        assert np.all(~df["is_retro"])

    def test_sample_group_summary_with_retro(self, fx_ti_2020_02_15):
        df = fx_ti_2020_02_15.sample_groups_summary()
        assert df.shape[0] == 27
        assert np.all(df["nodes"] >= df["samples"])
        assert np.all(df["nodes"] > 0)
        assert np.all(~df["is_retro"][:-1])
        assert df["is_retro"].iloc[-1]

    def test_retro_sample_group_summary(self, fx_ti_2020_02_15):
        df1 = fx_ti_2020_02_15.sample_groups_summary()
        df1 = df1[df1.is_retro]
        df2 = fx_ti_2020_02_15.retro_sample_groups_summary()
        assert df1.shape[0] == 1
        assert df2.shape[0] == 1
        assert np.all(df1.index == df2.index)
        row1 = df1.iloc[0]
        row2 = df2.iloc[0]
        assert row1.samples == row2.samples
        # Mutations may be deleted later through parsimony hueristics
        assert row1.mutations <= row2.num_mutations

    def test_node_summary(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        for u in range(ti.ts.num_nodes):
            d = ti._node_summary(u)
            assert d["node"] == u
            assert len(d["flags"]) == 8

    def test_node_report(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        report = ti.node_report(strain="SRR11597190")
        assert len(report) > 0

    def test_summary(self, fx_ti_2020_02_15):
        df = fx_ti_2020_02_15.summary()
        assert df.loc["samples"].value == 43
        assert df.loc["sample_groups"].value == 27
        assert df.loc["retro_sample_groups"].value == 1

    def test_summary(self, fx_ti_2020_02_15):
        df = fx_ti_2020_02_15.summary()
        assert df.loc["samples"].value == 43
        assert df.loc["sample_groups"].value == 27
        assert df.loc["retro_sample_groups"].value == 1

    def test_recombinants_summary_example_1(self, fx_ti_recombinant_example_1):
        df = fx_ti_recombinant_example_1.recombinants_summary()
        assert df.shape[0] == 1
        row = df.iloc[0]
        assert row.num_descendant_samples == 2
        assert row["sample"] == 53
        assert row.sample_id == "recombinant_example_1_0"
        assert row.num_samples == 2
        assert row.group_size == 3
        assert row.distinct_sample_pango == 1
        assert row.sample_pango == "Unknown"
        assert row.interval_left == 3788
        assert row.interval_right == 11083
        assert row.parent_left == 31
        assert row.parent_left_pango == "B"
        assert row.parent_right == 46
        assert row.parent_right_pango == "Unknown"
        assert row.oldest_child == 53
        assert row.oldest_child_time == 0
        assert str(row.oldest_child_date).split()[0] == "2020-02-15"
        assert row.num_mutations == 0
        assert row.parent_mrca == 1
        assert row.parent_mrca_time == 51
        assert "diffs" not in df
        df2 = fx_ti_recombinant_example_1.recombinants_summary(
            characterise_copying=True, show_progress=False
        )
        assert set(df) < set(df2)
        row2 = df2.iloc[0]
        assert row2.diffs == 6
        assert row2.max_run_length == 0

    def test_recombinants_summary_example_2(self, fx_recombinant_example_2):
        ti = info.TreeInfo(fx_recombinant_example_2, show_progress=False)
        df = ti.recombinants_summary(characterise_copying=True, show_progress=False)
        assert df.shape[0] == 1
        row = df.iloc[0]
        assert row.num_descendant_samples == 1
        assert row["sample"] == 55
        assert row["sample_id"] == "recombinant_114:29825"
        assert row["distinct_sample_pango"] == 1
        assert row["recombinant"] == 56
        assert row["recombinant_pango"] == "Unknown"
        assert row["recombinant_time"] == 0.000001
        assert row["sample_pango"] == "Unknown"
        assert row["num_mutations"] == 0
        assert row["parent_left"] == 53
        assert row["parent_left_pango"] == "Unknown"
        assert row["parent_right"] == 54
        assert row["parent_right_pango"] == "Unknown"
        assert row["parent_mrca"] == 48
        assert row["group_size"] == 2
        assert row["diffs"] == 6
        assert row["max_run_length"] == 2

    def test_recombinants_summary_example_2_bp_shift(self, fx_recombinant_example_2):
        tables = fx_recombinant_example_2.dump_tables()
        sample = 55
        row = tables.nodes[sample]
        md = row.metadata
        # Shift the bp one base right
        md["sc2ts"]["breakpoint_intervals"] = [[114, 29825]]
        tables.nodes[sample] = row.replace(metadata=md)
        ts = tables.tree_sequence()
        ti = info.TreeInfo(ts, show_progress=False)
        df1 = ti.recombinants_summary(characterise_copying=True, show_progress=False)
        ti = info.TreeInfo(fx_recombinant_example_2, show_progress=False)
        df2 = ti.recombinants_summary(characterise_copying=True, show_progress=False)
        pd.testing.assert_frame_equal(df1, df2)

    def test_recombinants_summary_example_2_bp_shift_past_left(
        self, fx_recombinant_example_2
    ):
        tables = fx_recombinant_example_2.dump_tables()
        sample = 55
        row = tables.nodes[sample]
        md = row.metadata
        md["sc2ts"]["breakpoint_intervals"] = [[29827, 29825]]
        tables.nodes[sample] = row.replace(metadata=md)
        ts = tables.tree_sequence()
        ti = info.TreeInfo(ts, show_progress=False)
        df1 = ti.recombinants_summary(show_progress=False)
        nt.assert_array_equal(df1.interval_left.values, [29824])
        nt.assert_array_equal(df1.interval_right.values, [29825])


class TestSampleGroupInfo:
    def test_draw_svg(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        sg = list(ti.sample_group_nodes.keys())[0]
        sg_info = ti.get_sample_group_info(sg)
        svg = sg_info.draw_svg()
        assert svg.startswith("<svg")


class TestDataFuncs:

    def test_example_node(self, fx_ts_min_2020_02_15, fx_ti_2020_02_15):
        ts = fx_ts_min_2020_02_15
        ti = fx_ti_2020_02_15
        df = sc2ts.node_data(ts)
        assert df.shape[0] == ti.ts.num_nodes
        nt.assert_array_equal(ti.nodes_num_mutations, df["num_mutations"])
        nt.assert_array_equal(np.arange(ti.ts.num_nodes), df["node_id"])
        nt.assert_array_equal(
            ti.nodes_max_descendant_samples, df["max_descendant_samples"]
        )
        nt.assert_array_equal(ti.nodes_date, df["date"])
        assert list(np.where(df["is_recombinant"])[0]) == list(ti.recombinants)
        assert list(np.where(df["is_sample"])[0]) == list(ts.samples())

    def test_example_node_no_date(self, fx_ts_min_2020_02_15, fx_ti_2020_02_15):
        ts = fx_ts_min_2020_02_15
        df1 = sc2ts.node_data(ts)
        tables = ts.dump_tables()
        tables.metadata = {}
        df2 = sc2ts.node_data(tables.tree_sequence())
        assert set(df1) == set(df2) | {"date"}
        for col in df2:
            nt.assert_array_equal(df1[col].values, df2[col].values)

    def test_example_mutation(self, fx_ts_min_2020_02_15, fx_ti_2020_02_15):
        ts = fx_ts_min_2020_02_15
        df = sc2ts.mutation_data(fx_ts_min_2020_02_15, parsimony_stats=True)
        ti = fx_ti_2020_02_15
        assert df.shape[0] == ti.ts.num_mutations
        nt.assert_array_equal(np.arange(ti.ts.num_mutations), df["mutation_id"])
        nt.assert_array_equal(ti.mutations_num_descendants, df["num_descendants"])
        nt.assert_array_equal(ti.mutations_num_inheritors, df["num_inheritors"])
        nt.assert_array_equal(ti.mutations_derived_state, df["derived_state"])
        nt.assert_array_equal(ti.mutations_inherited_state, df["inherited_state"])
        nt.assert_array_equal(ts.mutations_node, df["node"])
        nt.assert_array_equal(ts.mutations_parent, df["parent"])
        nt.assert_array_equal(
            ti.mutations_is_immediate_reversion, df["is_immediate_reversion"]
        )

    def test_example_mutation_no_date(self, fx_ts_min_2020_02_15, fx_ti_2020_02_15):
        ts = fx_ts_min_2020_02_15
        df1 = sc2ts.mutation_data(ts)
        tables = ts.dump_tables()
        tables.metadata = {}
        df2 = sc2ts.mutation_data(tables.tree_sequence())
        assert set(df1) == set(df2) | {"date"}
        for col in df2:
            nt.assert_array_equal(df1[col].values, df2[col].values)
