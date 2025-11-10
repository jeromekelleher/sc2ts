import numpy as np
import pytest
import tskit
import msprime
import biotite.sequence.phylo as bsp
import numpy.testing as nt

import sc2ts


def all_trees_ts(n):
    """
    Generate a tree sequence that corresponds to the lexicographic listing
    of all trees with n leaves (i.e. from tskit.all_trees(n)).

    Copied from tskit's tsutil testing module

    """
    tables = tskit.TableCollection(0)
    for _ in range(n):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for j in range(1, n):
        tables.nodes.add_row(flags=0, time=j)

    L = 0
    for tree in tskit.all_trees(n):
        for u in tree.preorder()[1:]:
            tables.edges.add_row(L, L + 1, tree.parent(u), u)
        L += 1
    tables.sequence_length = L
    tables.sort()
    tables.simplify()
    return tables.tree_sequence()


def assert_variants_equal(vars1, vars2, allele_shuffle=False):
    assert vars1.num_sites == vars2.num_sites
    assert vars1.num_samples == vars2.num_samples
    for var1, var2 in zip(vars1.variants(), vars2.variants()):
        if allele_shuffle:
            h1 = np.array(var1.alleles)[var1.genotypes]
            h2 = np.array(var2.alleles)[var2.genotypes]
            assert np.all(h1 == h2)
        else:
            assert var1.alleles == var2.alleles
            assert np.all(var1.genotypes == var2.genotypes)


def assert_sequences_equal(ts1, ts2):
    """
    Check that the variation data for the specifed tree sequences
    is identical.
    """
    ts1.tables.sites.assert_equals(ts2.tables.sites)
    for var1, var2 in zip(ts1.variants(), ts2.variants()):
        states1 = np.array(var1.alleles)[var1.genotypes]
        states2 = np.array(var2.alleles)[var2.genotypes]
        np.testing.assert_array_equal(states1, states2)


def prepare(tables):
    """
    Make changes needed for generic table collection to be used.
    """
    tables.mutations.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.nodes.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    return tables.tree_sequence()


class TestSplitBranch:

    def test_root(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        with pytest.raises(ValueError, match="root"):
            sc2ts.split_branch(ts1, 6, [])

    def test_wrong_mutations(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=1, time=0, derived_state="T")
        ts = prepare(tables)
        with pytest.raises(ValueError, match="must be associated with"):
            sc2ts.split_branch(ts, 0, [0])

    def test_no_mutations(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        ts2 = sc2ts.split_branch(prepare(ts1.dump_tables()), 0, [])
        assert ts2.num_nodes == 8
        assert ts2.nodes_time[7] == 0.5
        assert ts2.first().parent_dict == {0: 7, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 7: 4}

    def test_two_mutations(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4, arity=2, span=100).tree_sequence
        tables = ts1.dump_tables()
        for j in range(2):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=0, time=0, derived_state="T")
        ts = prepare(tables)

        ts2 = sc2ts.split_branch(ts, 0, [0])
        assert ts2.num_nodes == 8
        assert ts2.nodes_time[7] == 0.5
        assert ts2.first().parent_dict == {0: 7, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 7: 4}
        assert ts2.mutations_time[0] == 0.5
        assert ts2.mutations_node[0] == 7
        assert ts2.mutations_time[1] == 0
        assert ts2.mutations_node[1] == 0

    @pytest.mark.parametrize("n", range(5))
    def test_all_mutations(self, n):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4, arity=2, span=100).tree_sequence
        tables = ts1.dump_tables()
        for j in range(n):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=0, time=0, derived_state="T")
        ts = prepare(tables)

        ts2 = sc2ts.split_branch(ts, 0, range(n))
        assert ts2.num_nodes == 8
        assert ts2.nodes_time[7] == 0.5
        assert ts2.first().parent_dict == {0: 7, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6, 7: 4}
        assert np.all(ts2.mutations_time == 0.5)
        assert np.all(ts2.mutations_node == 7)

    def test_all_trees_ts(self):
        # 2.00┊       ┊   4   ┊   4   ┊   4   ┊
        #     ┊       ┊ ┏━┻┓  ┊  ┏┻━┓ ┊  ┏┻━┓ ┊
        # 1.00┊   3   ┊ ┃  3  ┊  3  ┃ ┊  3  ┃ ┊
        #     ┊ ┏━╋━┓ ┊ ┃ ┏┻┓ ┊ ┏┻┓ ┃ ┊ ┏┻┓ ┃ ┊
        # 0.00┊ 0 1 2 ┊ 0 1 2 ┊ 0 2 1 ┊ 0 1 2 ┊
        #     0       1       2       3       4
        # index   0       1       2       3
        tables = all_trees_ts(3).dump_tables()
        tables.sites.add_row(0, "A")
        for j in range(3):
            tables.mutations.add_row(site=0, node=j, time=0, derived_state="T")
        ts1 = prepare(tables)
        ts2 = sc2ts.split_branch(ts1, 0, [0])

        assert ts2.num_nodes == 6
        assert ts2.nodes_time[5] == 0.5
        assert ts2.at_index(0).parent_dict == {0: 5, 5: 3, 1: 3, 2: 3}
        assert ts2.at_index(1).parent_dict == {0: 5, 5: 4, 3: 4, 1: 3, 2: 3}
        assert ts2.at_index(2).parent_dict == {0: 5, 5: 3, 3: 4, 1: 4, 2: 3}
        assert ts2.at_index(3).parent_dict == {0: 5, 5: 3, 3: 4, 1: 3, 2: 4}
        assert ts1.num_mutations == ts2.num_mutations
        # This is a case where we only need one edge joining 0->5 along the
        # full sequence.
        assert ts2.num_edges == ts1.num_edges + 1


class TestCoalesceMutations:
    def test_no_mutations(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        ts2 = sc2ts.coalesce_mutations(ts1)
        ts1.tables.assert_equals(ts2.tables)

    def test_two_mutation_groups_one_parent(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=1, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=2, time=0, derived_state="G")
        tables.mutations.add_row(site=0, node=3, time=0, derived_state="G")
        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == 7

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    def test_two_mutation_groups_two_parents(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=1, derived_state="T")
        tables.mutations.add_row(site=0, node=2, derived_state="G")
        tables.mutations.add_row(site=0, node=3, derived_state="G")
        tables.compute_mutation_times()
        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == 9

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    def test_internal_sib(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        ts = tskit.Tree.generate_balanced(3, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, derived_state="T")
        tables.mutations.add_row(site=0, node=3, derived_state="T")
        tables.compute_mutation_times()
        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 1
        assert ts2.num_nodes == 6

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    def test_nested_mutation(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(site=0, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=1, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=2, time=0, derived_state="T")
        tables.mutations.add_row(site=1, node=2, time=0, derived_state="G")
        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == 6

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    def test_conflicting_nested_mutations(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(site=0, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=1, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=2, time=0, derived_state="G")
        tables.mutations.add_row(site=1, node=1, time=0, derived_state="T")
        tables.mutations.add_row(site=1, node=2, time=0, derived_state="G")
        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 4
        assert ts2.num_nodes == 6

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    def test_node_in_multiple_mutation_sets(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        # Node 0 particpates in 3 different maximum sets.
        ts = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(0.25, "A")
        tables.sites.add_row(0.75, "A")
        tables.mutations.add_row(site=0, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=1, time=0, derived_state="T")
        tables.mutations.add_row(site=1, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=1, node=2, time=0, derived_state="T")
        tables.mutations.add_row(site=2, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=2, node=2, time=0, derived_state="T")
        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 4
        assert ts2.num_nodes == 6

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    # This test was broken as part of making the parsimony ops more scalable in #526
    @pytest.mark.skip("Not implemented")
    def test_mutations_on_same_branch(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=0, node=0, time=0, derived_state="C", parent=0)
        ts = prepare(tables)

        with pytest.raises(ValueError, match="Multiple mutations"):
            sc2ts.coalesce_mutations(ts)

    def test_mutation_parent(self):
        # 2.00┊   4   ┊
        #     ┊ ┏━┻┓  ┊
        # 1.00┊ ┃  3  ┊
        #     ┊ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 ┊
        #     0       1
        ts = tskit.Tree.generate_balanced(3, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(0.1, "A")
        tables.mutations.add_row(site=0, node=3, time=1, derived_state="T")
        tables.mutations.add_row(site=0, node=1, time=0, derived_state="G", parent=0)
        tables.mutations.add_row(site=0, node=2, time=0, derived_state="G", parent=0)
        # Site 1 has a complicated mutation pattern and no coalesceable mutations
        tables.mutations.add_row(site=1, node=3, time=1, derived_state="G")
        tables.mutations.add_row(site=1, node=0, time=0, derived_state="T")
        tables.mutations.add_row(site=1, node=1, time=0, derived_state="A", parent=4)
        tables.mutations.add_row(site=1, node=2, time=0, derived_state="C", parent=4)

        ts = prepare(tables)

        ts2 = sc2ts.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 6
        assert ts2.num_nodes == 6

        ts3 = sc2ts.apply_node_parsimony_heuristics(
            ts, push_reversions=False
        ).tree_sequence
        ts3.tables.assert_equals(ts2.tables, ignore_provenance=True)

    def test_time_bug(self):
        # We rely on the mutation time being the time of the node at the
        # bottom of the edge.
        ts = tskit.load("tests/data/coalesce_mutations_time_bug.trees")
        tables = ts.dump_tables()
        # Only the first three mutations are necessary.
        tables.mutations.truncate(3)
        ts2 = sc2ts.coalesce_mutations(tables.tree_sequence(), [0])
        assert ts2.num_mutations == 2


class TestPushUpReversions:
    def test_no_mutations(self):
        ts1 = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        ts2 = sc2ts.push_up_reversions(ts1, [0, 1, 2, 3])
        ts1.tables.assert_equals(ts2.tables)
        ts3 = sc2ts.apply_node_parsimony_heuristics(ts1).tree_sequence
        ts1.tables.assert_equals(ts3.tables, ignore_provenance=True)

    def test_one_site_simple_reversion(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=4, time=1, derived_state="T")
        tables.mutations.add_row(site=0, node=3, time=0, derived_state="A")
        ts = prepare(tables)

        ts2 = sc2ts.push_up_reversions(ts, [0, 1, 2, 3])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts2.tables.assert_equals(ts3.tables, ignore_provenance=True)

    def test_one_site_simple_reversion_internal(self):
        # 4.00┊   8       ┊
        #     ┊ ┏━┻━┓     ┊
        # 3.00┊ ┃   7     ┊
        #     ┊ ┃ ┏━┻━┓   ┊
        # 2.00┊ ┃ ┃   6   ┊
        #     ┊ ┃ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃ ┃  5  ┊
        #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 ┊
        #     0           1
        ts = tskit.Tree.generate_comb(5).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=6, time=2, derived_state="T")
        tables.mutations.add_row(site=0, node=5, time=1, derived_state="A")
        ts = prepare(tables)
        ts2 = sc2ts.push_up_reversions(ts, [5])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts2.tables.assert_equals(ts3.tables, ignore_provenance=True)

    def test_multiple_reversions_same_node(self):
        # 4.00┊   8       ┊
        #     ┊ ┏━┻━┓     ┊
        # 3.00┊ ┃   7     ┊
        #     ┊ ┃ ┏━┻━┓   ┊
        # 2.00┊ ┃ ┃   6   ┊
        #     ┊ ┃ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃ ┃  5  ┊
        #     ┊ ┃ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 ┊
        #     0           1
        ts = tskit.Tree.generate_comb(5).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=7, time=3, derived_state="T")
        tables.mutations.add_row(site=0, node=6, time=2, derived_state="A")
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(site=1, node=6, time=2, derived_state="T")
        tables.mutations.add_row(site=1, node=5, time=1, derived_state="A")
        ts = prepare(tables)
        ts2 = sc2ts.push_up_reversions(ts, [5, 6])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1
        ts3 = sc2ts.push_up_reversions(ts2, [9])
        assert_sequences_equal(ts, ts3)
        assert ts3.num_mutations == ts2.num_mutations - 1
        assert ts3.num_nodes == ts2.num_nodes + 1

        ts4 = sc2ts.apply_node_parsimony_heuristics(
            ts, coalesce_mutations=False
        ).tree_sequence
        ts3.tables.assert_equals(ts4.tables, ignore_provenance=True)

    def test_two_sites_reversion_and_shared(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(0.5, "A")
        tables.mutations.add_row(site=0, node=4, time=1, derived_state="T")
        tables.mutations.add_row(site=0, node=3, time=0, derived_state="A")
        # Shared mutation over 4
        tables.mutations.add_row(site=1, node=4, time=1, derived_state="T")

        ts = prepare(tables)

        ts2 = sc2ts.push_up_reversions(ts, [0, 1, 2, 3])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1

        ts3 = sc2ts.apply_node_parsimony_heuristics(ts).tree_sequence
        ts2.tables.assert_equals(ts3.tables, ignore_provenance=True)


class TestTrimBranches:
    def test_one_mutation_three_children(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5 x ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=5, derived_state="T")
        ts1 = tables.tree_sequence()

        ts2 = sc2ts.trim_branches(ts1)
        # 3.00┊   5     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   4   ┊
        #     ┊ ┃ ┏━╋━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        assert ts2.num_trees == 1
        assert ts2.first().parent_dict == {0: 5, 1: 4, 2: 4, 3: 4, 4: 5}
        assert_variants_equal(ts1, ts2)

    def test_no_mutations(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5 x ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        ts1 = tables.tree_sequence()

        ts2 = sc2ts.trim_branches(ts1)
        assert ts2.num_trees == 1
        assert ts2.first().parent_dict == {0: 4, 1: 4, 2: 4, 3: 4}
        assert_variants_equal(ts1, ts2)

    def test_mutation_over_root(self):
        # 3.00┊   6 x   ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=6, derived_state="T")
        ts1 = tables.tree_sequence()

        ts2 = sc2ts.trim_branches(ts1)
        assert ts2.num_trees == 1
        assert ts2.first().parent_dict == {0: 4, 1: 4, 2: 4, 3: 4}
        assert_variants_equal(ts1, ts2)

    def test_one_leaf_mutation(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3x┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(site=0, node=3, derived_state="T")
        ts1 = tables.tree_sequence()

        ts2 = sc2ts.trim_branches(ts1)
        # Tree is also flat because this mutation is private
        assert ts2.num_trees == 1
        assert ts2.first().parent_dict == {0: 4, 1: 4, 2: 4, 3: 4}
        assert_variants_equal(ts1, ts2)

    def test_n_leaf_mutations(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0x1x2x3x┊
        #     0         1
        ts = tskit.Tree.generate_comb(4).tree_sequence
        tables = ts.dump_tables()
        tables.sites.add_row(0, "A")
        for j in range(4):
            tables.mutations.add_row(site=0, node=j, derived_state="T")
        ts1 = tables.tree_sequence()

        ts2 = sc2ts.trim_branches(ts1)
        # Tree is also flat because all mutations are private
        assert ts2.num_trees == 1
        assert ts2.first().parent_dict == {0: 4, 1: 4, 2: 4, 3: 4}
        assert_variants_equal(ts1, ts2)

    def test_mutations_each_branch(self):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4, span=10).tree_sequence
        tables = ts.dump_tables()
        for j in range(6):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=j, derived_state="T")
        ts1 = tables.tree_sequence()

        ts2 = sc2ts.trim_branches(ts1)
        assert ts2.num_trees == 1
        assert ts2.first().parent_dict == ts1.first().parent_dict
        assert_variants_equal(ts1, ts2)

    @pytest.mark.parametrize("n", [2, 10, 100])
    @pytest.mark.parametrize("mutation_rate", [0.1, 0.5, 1.5])
    def test_simulation(self, n, mutation_rate):
        ts1 = msprime.sim_ancestry(n, sequence_length=100, ploidy=1, random_seed=3)
        ts1 = msprime.sim_mutations(ts1, rate=mutation_rate, random_seed=3234)
        ts2 = sc2ts.trim_branches(ts1)
        assert_variants_equal(ts1, ts2)


class TestFullSpanSibs:

    @pytest.mark.parametrize(
        ["nodes", "sibs"],
        [
            ([0], [0, 5]),
            ([1], [1, 4]),
            ([2], [2, 3]),
            ([0, 1], [0, 1, 4, 5]),
            ([6], []),
        ],
    )
    def test_single_tree(self, nodes, sibs):
        # 3.00┊   6     ┊
        #     ┊ ┏━┻━┓   ┊
        # 2.00┊ ┃   5   ┊
        #     ┊ ┃ ┏━┻┓  ┊
        # 1.00┊ ┃ ┃  4  ┊
        #     ┊ ┃ ┃ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_comb(4, span=10).tree_sequence
        result = sc2ts.full_span_sibs(ts, nodes)
        nt.assert_array_equal(result, sibs)

    @pytest.mark.parametrize(
        "nodes",
        [
            [0],
            [1],
            [2],
            [3],
            [4],
            [0, 1, 2],
        ],
    )
    def test_all_trees_ts(self, nodes):
        # 2.00┊       ┊   4   ┊   4   ┊   4   ┊
        #     ┊       ┊ ┏━┻┓  ┊  ┏┻━┓ ┊  ┏┻━┓ ┊
        # 1.00┊   3   ┊ ┃  3  ┊  3  ┃ ┊  3  ┃ ┊
        #     ┊ ┏━╋━┓ ┊ ┃ ┏┻┓ ┊ ┏┻┓ ┃ ┊ ┏┻┓ ┃ ┊
        # 0.00┊ 0 1 2 ┊ 0 1 2 ┊ 0 2 1 ┊ 0 1 2 ┊
        #     0       1       2       3       4
        # index   0       1       2       3
        ts = all_trees_ts(3)
        result = sc2ts.full_span_sibs(ts, nodes)
        assert len(result) == 0


class TestInferBinary:

    def check_properties(self, ts):
        assert ts.num_trees == 1
        tree = ts.first()
        if ts.num_samples > 1:
            assert ts.nodes_time[tree.root] == 1
            # for u in tree.nodes():
            #     assert len(tree.children(u)) in (0, 2)

    @pytest.mark.parametrize("n", range(1, 5))
    def test_flat_one_site_unique_mutations(self, n):
        L = n + 1
        tables = tskit.TableCollection(L)
        tables.sites.add_row(0, "A")
        root = n
        for j in range(n):
            u = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            tables.edges.add_row(0, L, root, u)
            tables.mutations.add_row(0, derived_state=f"{j}", node=u)
        tables.nodes.add_row(time=1)
        ts1 = tables.tree_sequence()
        ts2 = sc2ts.infer_binary(ts1)
        assert ts2.num_mutations == n
        assert_sequences_equal(ts1, ts2)
        self.check_properties(ts2)

    @pytest.mark.parametrize("n", range(1, 5))
    def test_flat_one_site_one_mutation(self, n):
        L = n + 1
        tables = tskit.TableCollection(L)
        root = n
        for j in range(n):
            u = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            tables.edges.add_row(0, L, root, u)
        tables.nodes.add_row(time=1)
        tables.sites.add_row(0, "A")
        tables.mutations.add_row(0, derived_state="T", node=0)
        ts1 = tables.tree_sequence()
        ts2 = sc2ts.infer_binary(ts1)
        assert ts2.num_mutations == 1
        assert_sequences_equal(ts1, ts2)
        self.check_properties(ts2)

    @pytest.mark.parametrize("n", [2, 10, 15])
    @pytest.mark.parametrize("mutation_rate", [0.1, 0.5, 1.5])
    def test_simulation(self, n, mutation_rate):
        ts1 = msprime.sim_ancestry(n, sequence_length=100, ploidy=1, random_seed=3)
        ts1 = msprime.sim_mutations(ts1, rate=mutation_rate, random_seed=3234)
        ts2 = sc2ts.infer_binary(ts1)
        assert_variants_equal(ts1, ts2, allele_shuffle=True)
        self.check_properties(ts2)

    @pytest.mark.parametrize("n", [2, 10])
    @pytest.mark.parametrize("num_mutations", [1, 2, 10])
    def test_simulation_root_mutations(self, n, num_mutations):
        ts1 = msprime.sim_ancestry(n, sequence_length=100, ploidy=1, random_seed=3)
        root = ts1.first().root
        tables = ts1.dump_tables()
        for j in range(num_mutations):
            tables.sites.add_row(j, "A")
            tables.mutations.add_row(site=j, node=root, derived_state="T")
        ts1 = tables.tree_sequence()
        ts2 = sc2ts.infer_binary(ts1)
        assert_variants_equal(ts1, ts2)
        root = ts2.first().root
        assert np.all(ts2.mutations_node == root)
        self.check_properties(ts2)


class TestFromBiotite:

    def check_round_trip(self, tsk_tree):
        node_labels = {u: f"{u}" for u in tsk_tree.tree_sequence.samples()}
        nwk = tsk_tree.as_newick(node_labels=node_labels)
        biotite_tree = bsp.Tree.from_newick(nwk)
        converted = sc2ts.biotite_to_tskit(biotite_tree)
        assert converted.tree_sequence.num_trees == 1
        assert converted.rank() == tsk_tree.rank()

    @pytest.mark.parametrize("n", range(1, 5))
    def test_balanced_binary(self, n):
        tsk_tree = tskit.Tree.generate_balanced(n)
        self.check_round_trip(tsk_tree)

    @pytest.mark.parametrize("n", range(2, 5))
    def test_comb(self, n):
        tsk_tree = tskit.Tree.generate_comb(n)
        self.check_round_trip(tsk_tree)


class TestRerooting:

    def check_properties(self, before, after, root):
        assert after.num_trees == 1
        assert before.sequence_length == after.sequence_length
        after_tree = after.first()
        assert after_tree.root == root
        # Node tables should be identical other than time.
        before_nodes = before.dump_tables().nodes
        after_nodes = before.dump_tables().nodes
        before_nodes.time = np.zeros_like(before_nodes.time)
        after_nodes.time = np.zeros_like(after_nodes.time)
        before_nodes.assert_equals(after_nodes)

    def test_example_n2(self):
        #  1.00┊  2  ┊
        #      ┊ ┏┻┓ ┊
        #  0.00┊ 0 1 ┊
        #      0     1
        # ->
        # 2.00┊ 1 ┊
        #     ┊ ┃ ┊
        # 1.00┊ 2 ┊
        #     ┊ ┃ ┊
        # 0.00┊ 0 ┊
        #     0   1
        ts1 = tskit.Tree.generate_balanced(2, arity=2).tree_sequence
        ts2 = sc2ts.reroot_ts(ts1, 1)
        self.check_properties(ts1, ts2, 1)
        tree = ts2.first()
        nt.assert_array_equal(tree.parent_array, [2, -1, 1, -1])
        nt.assert_array_equal(ts2.nodes_time, [0, 2, 1])

    def test_binary_example_n4_internal(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        # ->
        # 3.00┊     5   ┊
        #     ┊  ┏━━╋━┓ ┊
        # 2.00┊  6  ┃ ┃ ┊
        #     ┊  ┃  ┃ ┃ ┊
        # 1.00┊  4  ┃ ┃ ┊
        #     ┊ ┏┻┓ ┃ ┃ ┊
        #    0┊ 0 1 2 3 ┊
        #     0         1

        ts1 = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        root = 5
        ts2 = sc2ts.reroot_ts(ts1, root)
        self.check_properties(ts1, ts2, root)
        tree = ts2.first()
        nt.assert_array_equal(tree.parent_array, [4, 4, 5, 5, 6, -1, 5, -1])
        nt.assert_array_equal(ts2.nodes_time, [0, 0, 0, 0, 1, 3, 2])

    def test_binary_example_n4_leaf(self):
        # 2.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 1.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        # ->
        # 4.00┊   2   ┊
        #     ┊   ┃   ┊
        # 3.00┊   5   ┊
        #     ┊  ┏┻━┓ ┊
        # 2.00┊  6  ┃ ┊
        #     ┊  ┃  ┃ ┊
        # 1.00┊  4  ┃ ┊
        #     ┊ ┏┻┓ ┃ ┊
        # 0.00┊ 0 1 3 ┊
        #     0       1
        ts1 = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        root = 2
        ts2 = sc2ts.reroot_ts(ts1, root)
        self.check_properties(ts1, ts2, root)
        tree = ts2.first()
        nt.assert_array_equal(tree.parent_array, [4, 4, -1, 5, 6, 2, 5, -1])
        nt.assert_array_equal(ts2.nodes_time, [0, 0, 4, 0, 1, 3, 2])

    def test_ternary_example_n6_leaf(self):
        # 2.00┊      9      ┊
        #     ┊  ┏━━━╋━━━┓  ┊
        # 1.00┊  6   7   8  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 5 ┊
        #     0             1
        # ->
        # 4.00┊       3   ┊
        #     ┊       ┃   ┊
        # 3.00┊       7   ┊
        #     ┊    ┏━━┻━┓ ┊
        # 2.00┊    9    ┃ ┊
        #     ┊  ┏━┻━┓  ┃ ┊
        # 1.00┊  6   8  ┃ ┊
        #     ┊ ┏┻┓ ┏┻┓ ┃ ┊
        # 0.00┊ 0 1 4 5 2 ┊
        #     0           1
        ts1 = tskit.Tree.generate_balanced(6, arity=3).tree_sequence
        root = 3
        ts2 = sc2ts.reroot_ts(ts1, root)
        self.check_properties(ts1, ts2, root)
        tree = ts2.first()
        nt.assert_array_equal(tree.parent_array, [6, 6, 7, -1, 8, 8, 9, 3, 9, 7, -1])
        nt.assert_array_equal(ts2.nodes_time, [0, 0, 0, 4, 0, 0, 1, 3, 1, 2])

    def test_example_same_root(self):
        #  1.00┊  2  ┊
        #      ┊ ┏┻┓ ┊
        #  0.00┊ 0 1 ┊
        #      0     1
        # ->
        #  1.00┊  2  ┊
        #      ┊ ┏┻┓ ┊
        #  0.00┊ 0 1 ┊
        #      0     1
        ts1 = tskit.Tree.generate_balanced(2, arity=2).tree_sequence
        ts2 = sc2ts.reroot_ts(ts1, new_root=2)
        self.check_properties(before=ts1, after=ts2, root=2)
        ts1.tables.assert_equals(ts2.tables)
