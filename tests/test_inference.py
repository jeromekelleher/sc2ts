import numpy as np
import pytest
import tsinfer
import tskit

import sc2ts


class TestAttach:
    def test_binary_tree_example(self):
        # Attach a balance binary tree to each of its leaves.

        # 5.00┊    6    ┊
        #     ┊  ┏━┻━┓  ┊
        # 4.00┊  4   5  ┊
        #     ┊ ┏┻┓ ┏┻┓ ┊
        # 3.00┊ 0 1 2 3 ┊
        #     0         1
        ts = tskit.Tree.generate_balanced(4, arity=2).tree_sequence
        tables = ts.dump_tables()
        tables.nodes.time += 3
        ts = sc2ts.attach(tables.tree_sequence(), {0: ts, 1: ts, 2: ts, 3: ts})



class TestMatchPathTs:
    def match_path_ts(self, samples, ts):
        ts2 = sc2ts.match_path_ts(samples, ts)
        assert ts2.num_samples == len(samples)
        for u, sample in zip(ts.samples(), samples):
            node = ts.node(u)
            assert node.time == 0
            assert node.metadata == sample.metadata
        return ts2

    def test_one_sample(self):
        ts = sc2ts.initial_ts()
        s1 = sc2ts.Sample(path=[(0, ts.sequence_length, 1)])
        ts2 = self.match_path_ts([s1], ts)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0}

    def test_one_sample_match_recombinant(self):
        # 3.00┊  0  ┊  0  ┊
        #     ┊  ┃  ┊  ┃  ┊
        # 2.00┊  1  ┊  1  ┊
        #     ┊ ┏┻┓ ┊ ┏┻┓ ┊
        # 1.00┊ 3 2 ┊ 2 3 ┊
        #     ┊   ┃ ┊   ┃ ┊
        # 0.00┊   4 ┊   4 ┊
        #    0   14952 29904
        # Our target node is 4

        ts = sc2ts.initial_ts()
        L = ts.sequence_length
        tables = ts.dump_tables()
        tables.nodes.time += 2
        u = ts.num_nodes - 1
        a = tables.nodes.add_row(flags=0, time=1)
        b = tables.nodes.add_row(flags=0, time=1)
        c = tables.nodes.add_row(flags=1, time=0)
        tables.edges.add_row(0, L, parent=u, child=a)
        tables.edges.add_row(0, L, parent=u, child=b)
        tables.edges.add_row(0, L // 2, parent=a, child=c)
        tables.edges.add_row(L // 2, L, parent=b, child=c)
        # Redo the sites to make things simpler
        tables.sites.clear()
        tables.sites.add_row(L // 4, "A")
        tables.sites.add_row(3 * L // 4, "A")
        # Put mutations over 3 at both sites. We should only inherit from
        # the second one.
        tables.mutations.add_row(site=0, derived_state="T", node=3)
        tables.mutations.add_row(site=1, derived_state="T", node=3)
        tables.sort()
        ts = tables.tree_sequence()

        s1 = sc2ts.Sample(
            path=[(0, ts.sequence_length, c)], mutations=[(0, "G"), (1, "G")]
        )
        ts2 = self.match_path_ts([s1], ts)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0}
        assert ts2.num_sites == 2
        assert ts2.num_mutations == 2
        assert ts2.site(0).position == ts.site(0).position
        assert ts2.site(0).ancestral_state == "A"
        assert ts2.site(1).position == ts.site(1).position
        assert ts2.site(1).ancestral_state == "T"
        assert list(ts2.haplotypes()) == ["GG"]

    def test_one_sample_one_mutation(self):
        ts = sc2ts.initial_ts()
        s1 = sc2ts.Sample(path=[(0, ts.sequence_length, 1)], mutations=[(100, "X")])
        ts2 = self.match_path_ts([s1], ts)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0}
        assert ts2.num_sites == 1
        assert ts2.site(0).ancestral_state == ts.site(100).ancestral_state
        assert list(ts2.haplotypes()) == ["X"]

    def test_two_sample_one_mutation_each(self):
        ts = sc2ts.initial_ts()
        s1 = sc2ts.Sample(path=[(0, ts.sequence_length, 1)], mutations=[(100, "X")])
        s2 = sc2ts.Sample(path=[(0, ts.sequence_length, 1)], mutations=[(200, "Y")])
        ts2 = self.match_path_ts([s1, s2], ts)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0, 2: 0}
        assert ts2.num_sites == 2
        site0 = ts2.site(0)
        site1 = ts2.site(1)
        assert site0.ancestral_state == ts.site(100).ancestral_state
        assert site1.ancestral_state == ts.site(200).ancestral_state
        assert len(site0.mutations) == 1
        assert len(site1.mutations) == 1
        assert site0.mutations[0].derived_state == "X"
        assert site1.mutations[0].derived_state == "Y"

    @pytest.mark.parametrize("num_mutations", range(1, 6))
    def test_one_sample_k_mutations(self, num_mutations):
        ts = sc2ts.initial_ts()
        s1 = sc2ts.Sample(
            path=[(0, ts.sequence_length, 1)],
            mutations=[(j, f"{j}") for j in range(num_mutations)],
        )
        ts2 = self.match_path_ts([s1], ts)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0}
        assert ts2.num_sites == num_mutations
        for j in range(num_mutations):
            assert ts2.site(j).ancestral_state == ts.site(j).ancestral_state
        assert list(ts2.haplotypes()) == ["".join(f"{j}" for j in range(num_mutations))]

    def test_n_samples_metadata(self):
        ts = sc2ts.initial_ts()
        samples = []
        for j in range(10):
            samples.append(
                sc2ts.Sample(
                    metadata={f"x{j}": j, f"y{j}": list(range(j))},
                    path=[(0, ts.sequence_length, 1)],
                )
            )
        self.match_path_ts(samples, ts)


class TestAddMatchingResults:
    def test_one_sample(self):
        ts = sc2ts.initial_ts()
        ts = sc2ts.increment_time("2021-01-02", ts)
        s1 = sc2ts.Sample(metadata={}, path=[(0, ts.sequence_length, 1)], mutations=[])

        ts2 = sc2ts.add_matching_results([s1], ts)


# # @pytest.fixture
# def small_sd_fixture():
#     reference = core.get_reference_sequence()
#     print(reference)
#     fasta = {"REF": reference}
#     rows = [{"strain": "REF"}]
#     sd = convert.convert_alignments(rows, fasta)

#     return sd

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

# class TestInitialTables:
#     def test_site_schema(self):
#         sd = small_sd_fixture()
#         pass


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
