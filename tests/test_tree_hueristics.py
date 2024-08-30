import numpy as np
import pytest
import tskit
import msprime

import sc2ts


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


class TestCoalesceMutations:
    def test_no_mutations(self):
        # 1.00┊    4    ┊
        #     ┊ ┏━┳┻┳━┓ ┊
        # 0.00┊ 0 1 2 3 ┊
        #     0         1
        ts1 = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        ts2 = sc2ts.inference.coalesce_mutations(ts1)
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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == 7

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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == 9

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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 1
        assert ts2.num_nodes == 6

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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == 6

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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 4
        assert ts2.num_nodes == 6

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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 4
        assert ts2.num_nodes == 6

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
            sc2ts.inference.coalesce_mutations(ts)

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

        ts2 = sc2ts.inference.coalesce_mutations(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 6
        assert ts2.num_nodes == 6


class TestPushUpReversions:
    def test_no_mutations(self):
        ts1 = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        ts2 = sc2ts.inference.push_up_reversions(ts1, [0, 1, 2, 3])
        ts1.tables.assert_equals(ts2.tables)

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

        ts2 = sc2ts.inference.push_up_reversions(ts, [0, 1, 2, 3])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1

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
        ts2 = sc2ts.inference.push_up_reversions(ts, [5])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1

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

        ts2 = sc2ts.inference.push_up_reversions(ts, [0, 1, 2, 3])
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == ts.num_mutations - 1
        assert ts2.num_nodes == ts.num_nodes + 1


class TestInsertRecombinants:
    def test_no_recombination(self):
        ts1 = tskit.Tree.generate_balanced(4, arity=4).tree_sequence
        ts2 = sc2ts.inference.insert_recombinants(ts1)
        ts1.tables.assert_equals(ts2.tables)

    def test_single_breakpoint_single_recombinant_no_mutations(self):
        tables = tskit.TableCollection(10)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=1, time=0)
        tables.edges.add_row(0, 5, parent=0, child=2)
        tables.edges.add_row(5, 10, parent=1, child=2)
        ts = prepare(tables)

        ts2 = sc2ts.inference.insert_recombinants(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 0
        assert ts2.num_nodes == ts.num_nodes + 1
        assert ts2.num_edges == ts.num_edges + 1
        assert_sequences_equal(ts, ts2)

    def test_single_breakpoint_two_recombinants_no_mutations(self):
        tables = tskit.TableCollection(10)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=1, time=0)
        tables.nodes.add_row(flags=1, time=0)
        tables.edges.add_row(0, 5, parent=0, child=2)
        tables.edges.add_row(5, 10, parent=1, child=2)
        tables.edges.add_row(0, 5, parent=0, child=3)
        tables.edges.add_row(5, 10, parent=1, child=3)
        ts = prepare(tables)

        ts2 = sc2ts.inference.insert_recombinants(ts)
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 0
        assert ts2.num_nodes == ts.num_nodes + 1
        assert ts2.num_edges == ts.num_edges
        assert_sequences_equal(ts, ts2)

    def test_single_breakpoint_single_recombinant_one_mutation(self):
        tables = tskit.TableCollection(10)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=1, time=0)
        tables.edges.add_row(0, 5, parent=0, child=2)
        tables.edges.add_row(5, 10, parent=1, child=2)
        tables.sites.add_row(4, "A")
        tables.mutations.add_row(site=0, node=2, derived_state="T")
        ts = prepare(tables)

        ts2 = sc2ts.inference.insert_recombinants(ts)
        md = ts2.node(3).metadata
        assert md["mutations"] == [[2, [[0, "A", "T"]]]]
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 1
        assert ts2.num_nodes == ts.num_nodes + 1
        assert ts2.num_edges == ts.num_edges + 1
        assert np.all(ts2.mutations_node == 3)

    def test_single_breakpoint_single_recombinant_two_mutations(self):
        tables = tskit.TableCollection(10)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=1, time=0)
        tables.edges.add_row(0, 5, parent=0, child=2)
        tables.edges.add_row(5, 10, parent=1, child=2)
        tables.sites.add_row(4, "A")
        tables.sites.add_row(5, "G")
        tables.mutations.add_row(site=0, node=2, derived_state="T")
        tables.mutations.add_row(site=1, node=2, derived_state="C")
        ts = prepare(tables)

        ts2 = sc2ts.inference.insert_recombinants(ts)
        md = ts2.node(3).metadata
        assert md["mutations"] == [[2, [[0, "A", "T"], [1, "G", "C"]]]]
        assert_sequences_equal(ts, ts2)
        assert ts2.num_mutations == 2
        assert ts2.num_nodes == ts.num_nodes + 1
        assert ts2.num_edges == ts.num_edges + 1
        assert np.all(ts2.mutations_node == 3)

    def test_single_breakpoint_two_recombinants_different_mutations(self):
        tables = tskit.TableCollection(10)
        tables.sites.add_row(4, "A")
        tables.sites.add_row(5, "G")
        tables.nodes.add_row(flags=0, time=1)
        tables.nodes.add_row(flags=0, time=1)
        for j in [2, 3]:
            tables.nodes.add_row(flags=1, time=0)
            tables.edges.add_row(0, 5, parent=0, child=j)
            tables.edges.add_row(5, 10, parent=1, child=j)
            # Share the mutation at site 0
            tables.mutations.add_row(site=0, node=j, derived_state="T")
        # Different mutations at site 1
        tables.mutations.add_row(site=1, node=2, derived_state="C")
        tables.mutations.add_row(site=1, node=3, derived_state="T")
        ts = prepare(tables)

        ts2 = sc2ts.inference.insert_recombinants(ts)
        assert_sequences_equal(ts, ts2)
        md = ts2.node(4).metadata
        assert ts2.num_mutations == 3
        assert ts2.num_nodes == ts.num_nodes + 1
        assert ts2.num_edges == ts.num_edges


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


class TestInferBinary:
    @pytest.mark.parametrize("n", [2, 10, 15])
    @pytest.mark.parametrize("mutation_rate", [0.1, 0.5, 1.5])
    def test_simulation(self, n, mutation_rate):
        ts1 = msprime.sim_ancestry(n, sequence_length=100, ploidy=1, random_seed=3)
        ts1 = msprime.sim_mutations(ts1, rate=mutation_rate, random_seed=3234)
        ts2 = sc2ts.infer_binary(ts1)
        assert_variants_equal(ts1, ts2, allele_shuffle=True)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.num_roots == 1
        for u in tree.nodes():
            assert len(tree.children(u)) in (0, 2)

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
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.num_roots == 1
        for u in tree.nodes():
            assert len(tree.children(u)) in (0, 2)
