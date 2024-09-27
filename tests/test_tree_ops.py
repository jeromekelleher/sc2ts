import numpy as np
import pytest
import tskit
import msprime
import biotite.sequence.phylo as bsp
import numpy.testing as nt

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

    def check_properties(self, ts):
        assert ts.num_trees == 1
        tree = ts.first()
        if ts.num_samples > 1:
            assert ts.nodes_time[tree.root] == 1
            for u in tree.nodes():
                assert len(tree.children(u)) in (0, 2)

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


from typing import List
import dataclasses


@dataclasses.dataclass
class QuintuplyLinkedTree:
    parent: List
    left_child: List
    right_child: List
    left_sib: List
    right_sib: List

    def __str__(self):
        s = "id\tparent\tlchild\trchild\tlsib\trsib\n"
        for j in range(len(self.parent)):
            s += (
                f"{j}\t{self.parent[j]}\t"
                f"{self.left_child[j]}\t{self.right_child[j]}\t"
                f"{self.left_sib[j]}\t{self.right_sib[j]}\n"
            )
        return s

    def remove_branch(self, p, c):
        lsib = self.left_sib[c]
        rsib = self.right_sib[c]
        if lsib == -1:
            self.left_child[p] = rsib
        else:
            self.right_sib[lsib] = rsib
        if rsib == -1:
            self.right_child[p] = lsib
        else:
            self.left_sib[rsib] = lsib
        self.parent[c] = -1
        self.left_sib[c] = -1
        self.right_sib[c] = -1

    def insert_branch(self, p, c):
        assert self.parent[c] == -1, "contradictory edges"
        self.parent[c] = p
        u = self.right_child[p]
        if u == -1:
            self.left_child[p] = c
            self.left_sib[c] = -1
            self.right_sib[c] = -1
        else:
            self.right_sib[u] = c
            self.left_sib[c] = u
            self.right_sib[c] = -1
        self.right_child[p] = c

    def push_up(self, u):
        """
        Push the node u one level up the tree
        """
        parent = self.parent[u]
        assert parent != -1
        self.remove_branch(parent, u)
        grandparent = self.parent[parent]
        if grandparent != -1:
            self.remove_branch(grandparent, parent)
            self.insert_branch(grandparent, u)
        self.insert_branch(u, parent)


def reroot(ts, new_root, scale_time=False):
    """
    Reroot the tree around the specified node, keeping node IDs
    the same.
    """
    assert ts.num_trees == 1
    tree = ts.first()
    qlt = QuintuplyLinkedTree(
        left_child=tree.left_child_array.copy(),
        left_sib=tree.left_sib_array.copy(),
        right_child=tree.right_child_array.copy(),
        right_sib=tree.right_sib_array.copy(),
        parent=tree.parent_array.copy(),
    )

    # print()
    # print(qlt)
    while qlt.parent[new_root] != -1:
        qlt.push_up(new_root)
    # print()
    # print(qlt)
    tables = ts.dump_tables()
    tables.edges.clear()
    # NOTE: could be done with numpy so this will work for large trees.
    for u in range(ts.num_nodes):
        if qlt.parent[u] != -1:
            tables.edges.add_row(0, ts.sequence_length, qlt.parent[u], u)
    sc2ts.set_tree_time(tables, unit_scale=scale_time)
    tables.sort()
    return tables.tree_sequence()


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
        ts2 = reroot(ts1, 1)
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
        ts2 = reroot(ts1, root)
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
        # 3.00┊   2   ┊
        #     ┊  ┏┻━┓ ┊
        # 2.00┊  6  ┃ ┊
        #     ┊  ┃  ┃ ┊
        # 1.00┊  4  5 ┊
        #     ┊ ┏┻┓ ┃ ┊
        # 0.00┊ 0 1 3 ┊
        #     0       1
        ts1 = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        root = 2
        ts2 = reroot(ts1, root)
        self.check_properties(ts1, ts2, root)
        tree = ts2.first()
        nt.assert_array_equal(tree.parent_array, [4, 4, -1, 5, 6, 2, 2, -1])
        nt.assert_array_equal(ts2.nodes_time, [0, 0, 3, 0, 1, 1, 2])

    def test_ternary_example_n6_leaf(self):
        # 2.00┊      9      ┊
        #     ┊  ┏━━━╋━━━┓  ┊
        # 1.00┊  6   7   8  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┏┻┓ ┊
        # 0.00┊ 0 1 2 3 4 5 ┊
        #     0             1
        # ->
        # 3.00┊       3   ┊
        #     ┊    ┏━━┻━┓ ┊
        # 2.00┊    9    ┃ ┊
        #     ┊  ┏━┻━┓  ┃ ┊
        # 1.00┊  6   8  7 ┊
        #     ┊ ┏┻┓ ┏┻┓ ┃ ┊
        # 0.00┊ 0 1 4 5 2 ┊
        #     0           1
        ts1 = tskit.Tree.generate_balanced(6, arity=3).tree_sequence
        root = 3
        ts2 = reroot(ts1, root)
        self.check_properties(ts1, ts2, root)
        tree = ts2.first()
        nt.assert_array_equal(tree.parent_array, [6, 6, 7, -1, 8, 8, 9, 3, 9, 3, -1])
        nt.assert_array_equal(ts2.nodes_time, [0, 0, 0, 3, 0, 0, 1, 1, 1, 2])
