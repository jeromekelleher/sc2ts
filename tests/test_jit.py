import numpy as np
import msprime
import tskit
import numpy.testing as nt

from sc2ts import jit


def single_tree_example_ts():
    # 2.00┊    6    ┊
    #     ┊  ┏━┻━┓  ┊
    # 1.00┊  4   5  ┊
    #     ┊ ┏┻┓ ┏┻┓ ┊
    # 0.00┊ 0 1 2 3 ┊
    #     0         10
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    for j in range(6):
        tables.sites.add_row(position=j + 1, ancestral_state="A")
        tables.mutations.add_row(site=j, derived_state="T", node=j)
    tables.sites.add_row(position=7, ancestral_state="F")
    tables.mutations.add_row(site=6, derived_state="D", node=6)
    tables.compute_mutation_times()
    return tables.tree_sequence()


def single_tree_recurrent_mutation_example_ts():
    # 2.00 ┊                    6                    ┊
    #      ┊            ┏━━━━━━━┻━━━━━━━┓            ┊
    #      ┊      4:A→T x               x 5:A→T      ┊
    #      ┊            |               x 6:A→G      ┊
    # 1.00 ┊            4               5            ┊
    #      ┊       ┏━━━━┻━━━━┓     ┏━━━━┻━━━━┓       ┊
    #      ┊ 0:A→T x   1:A→T x     x 2:A→T   x 3:A→T ┊
    #      ┊       |         |     |         |       ┊
    # 0.00 ┊       0         1     2         3       ┊
    #      0                                        10
    ts = tskit.Tree.generate_balanced(4, span=10).tree_sequence
    tables = ts.dump_tables()
    for j in range(6):
        tables.sites.add_row(position=j + 1, ancestral_state="A")
        tables.mutations.add_row(site=j, derived_state="T", node=j)
    tables.mutations.add_row(site=j, derived_state="G", node=j, parent=j)
    ts = tables.tree_sequence()
    return tables.tree_sequence()


def multiple_trees_example_ts():
    # 2.00┊   4   ┊   4   ┊
    #     ┊ ┏━┻┓  ┊  ┏┻━┓ ┊
    # 1.00┊ ┃  3  ┊  3  ┃ ┊
    #     ┊ ┃ ┏┻┓ ┊ ┏┻┓ ┃ ┊
    # 0.00┊ 0 1 2 ┊ 0 1 2 ┊
    #     0       5      10
    ts = tskit.Tree.generate_balanced(3, span=10).tree_sequence
    tables = ts.dump_tables()
    tables.edges[1] = tables.edges[1].replace(right=5)
    tables.edges[2] = tables.edges[2].replace(right=5)
    tables.edges.add_row(5, 10, 3, 0)
    tables.edges.add_row(5, 10, 4, 2)
    tables.sort()
    return tables.tree_sequence()


class TestArgCounts:

    def test_single_tree_example(self):
        ts = single_tree_example_ts()
        c = jit.count(ts)
        nt.assert_array_equal(c.nodes_max_descendant_samples, [1, 1, 1, 1, 2, 2, 4])
        nt.assert_array_equal(c.mutations_num_parents, [0] * 7)
        nt.assert_array_equal(c.mutations_num_descendants, [1] * 4 + [2] * 2 + [4])
        nt.assert_array_equal(c.mutations_num_inheritors, [1] * 4 + [2] * 2 + [4])

    def test_single_tree_recurrent_mutation_example(self):
        ts = single_tree_recurrent_mutation_example_ts()
        c = jit.count(ts)
        nt.assert_array_equal(c.nodes_max_descendant_samples, [1, 1, 1, 1, 2, 2, 4])
        nt.assert_array_equal(c.mutations_num_parents, [0] * 6 + [1])
        nt.assert_array_equal(c.mutations_num_descendants, [1] * 4 + [2] * 3)
        nt.assert_array_equal(c.mutations_num_inheritors, [1] * 4 + [2, 0, 2])

    def test_multiple_tree_example(self):
        ts = multiple_trees_example_ts()
        c = jit.count(ts)
        nt.assert_array_equal(c.nodes_max_descendant_samples, [1, 1, 1, 2, 3])
        nt.assert_array_equal(c.mutations_num_parents, [])
        nt.assert_array_equal(c.mutations_num_descendants, [])
        nt.assert_array_equal(c.mutations_num_inheritors, [])


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
        tables.mutations.add_row(site=0, node=4, derived_state="T")
        tables.mutations.add_row(site=0, node=0, derived_state="A", parent=0)
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
