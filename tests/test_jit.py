import numpy as np
import msprime
import tskit

from sc2ts import jit


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
        actual = jit.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_1tree_1mut_below_root(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        ts = tables.tree_sequence()
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        expected[0] = 1
        actual = jit.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_1tree_1mut_above_root(self):
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=6, derived_state="T")
        ts = tables.tree_sequence()
        expected = np.ones(ts.num_nodes, dtype=np.int32)
        actual = jit.get_num_muts(ts)
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
        actual = jit.get_num_muts(ts)
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
        actual = jit.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)

    def test_2trees_0mut(self):
        ts = msprime.sim_ancestry(
            2,
            recombination_rate=1e6,  # Nearly guarantee recomb.
            sequence_length=2,
        )
        assert ts.num_trees == 2
        expected = np.zeros(ts.num_nodes, dtype=np.int32)
        actual = jit.get_num_muts(ts)
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
        actual = jit.get_num_muts(ts)
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
        actual = jit.get_num_muts(ts)
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
        actual = jit.get_num_muts(ts)
        np.testing.assert_equal(expected, actual)
