import datetime
import inspect

import pytest
import numpy as np
import pandas as pd
import matplotlib

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
def fx_ti_recombinant_example_1(fx_recombinant_example_1):
    return info.TreeInfo(fx_recombinant_example_1, show_progress=False)


def test_get_gene_coordinates():
    d = sc2ts.get_gene_coordinates()
    assert len(d) == 11
    assert d["S"] == (21563, 25385)


# This functionality should be removed and kept track of online in the metadata.
@pytest.mark.skip("Broken by dataset")
class TestTallyLineages:

    def test_last_date(self, fx_ts_map, fx_metadata_db):
        date = "2020-02-13"
        df = info.tally_lineages(fx_ts_map[date], fx_metadata_db)
        assert list(df["pango"]) == [
            "B",
            "A",
            "B.1",
            "B.40",
            "B.33",
            "B.4",
            "A.5",
            "B.1.177",
            "B.1.36.29",
        ]
        assert list(df["db_count"]) == [26, 15, 4, 4, 1, 3, 1, 1, 1]
        assert list(df["arg_count"]) == [23, 15, 4, 3, 1, 1, 0, 0, 0]


class TestCountMutations:
    def test_1tree_0mut(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_1tree_1mut_below_root(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        ts = tables.tree_sequence()
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 1
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_1tree_1mut_above_root(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=6, derived_state="T")
        ts = tables.tree_sequence()
        expected = np.ones(ts.num_nodes, dtype=np.int32)
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_1tree_2mut_homoplasies(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=3, derived_state="T")
        ts = tables.tree_sequence()
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 1
        expected[3] = 1
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_1tree_2mut_reversion(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="A")
        tables.mutations.add_row(site=0, node=4, derived_state="T")
        ts = tables.tree_sequence()
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 2
        expected[1] = 1
        expected[4] = 1
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_2trees_0mut(self):
        ts = msprime.sim_ancestry(
            2,
            recombination_rate=1e6,  # Nearly guarantee recomb.
            sequence_length=2,
        )
        assert ts.num_trees == 2
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_2trees_1mut(self):
        ts = msprime.sim_ancestry(
            4,
            ploidy=1,
            recombination_rate=1e6,  # Nearly guarantee recomb.
            sequence_length=2,
        )
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        ts = tables.tree_sequence()
        assert ts.num_trees == 2
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 1
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_2trees_2mut_diff_trees(self):
        ts = msprime.sim_ancestry(
            4,
            ploidy=1,
            recombination_rate=1e6,  # Nearly guarantee recomb.
            sequence_length=2,
        )
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(1, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=0, derived_state="T")
        ts = tables.tree_sequence()
        assert ts.num_trees == 2
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 2
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_2trees_2mut_same_tree(self):
        ts = msprime.sim_ancestry(
            4,
            ploidy=1,
            recombination_rate=1e6,  # Nearly guarantee recomb.
            sequence_length=2,
        )
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(1, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=1, node=3, derived_state="T")
        ts = tables.tree_sequence()
        assert ts.num_trees == 2
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 1
        expected[3] = 1
        actual = info.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)


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
        assert row.descendants == 2
        assert row["sample"] == 53
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
        assert row.num_mutations == 0
        assert row.mrca == 1
        assert row.t_mrca == 51
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
        assert row.descendants == 1
        assert row["sample"] == 55
        assert row["distinct_sample_pango"] == 1
        assert row["recombinant"] == 56
        assert row["sample_pango"] == "Unknown"
        assert row["num_mutations"] == 0
        assert row["parent_left"] == 53
        assert row["parent_left_pango"] == "Unknown"
        assert row["parent_right"] == 54
        assert row["parent_right_pango"] == "Unknown"
        assert row["mrca"] == 48
        assert row["group_size"] == 2
        assert row["diffs"] == 6
        assert row["max_run_length"] == 2


class TestSampleGroupInfo:
    def test_draw_svg(self, fx_ti_2020_02_13):
        ti = fx_ti_2020_02_13
        sg = list(ti.sample_group_nodes.keys())[0]
        sg_info = ti.get_sample_group_info(sg)
        svg = sg_info.draw_svg()
        assert svg.startswith("<svg")
