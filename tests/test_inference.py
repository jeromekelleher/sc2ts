import numpy as np
import pytest
import tsinfer
import tskit
import msprime

import sc2ts
import util


class TestInitialTs:
    def test_reference_sequence(self):
        ts = sc2ts.initial_ts()
        assert ts.reference_sequence.metadata["genbank_id"] == "MN908947"
        assert ts.reference_sequence.data == sc2ts.core.get_reference_sequence()

    def test_reference_sample(self):
        ts = sc2ts.initial_ts()
        assert ts.num_samples == 1
        node = ts.node(ts.samples()[0])
        assert node.time == 0
        assert node.metadata == {"date": "2019-12-26", "strain": "Wuhan/Hu-1/2019"}
        alignment = next(ts.alignments())
        assert alignment == sc2ts.core.get_reference_sequence()


class TestAddMatchingResults:
    def add_matching_results(
        self,
        samples,
        ts,
        db_path,
        date="2020-01-01",
        num_mismatches=1000,
        max_hmm_cost=1e7,
    ):
        # This is pretty ugly, need to figure out how to neatly factor this
        # model of Sample object vs metadata vs alignment QC
        for sample in samples:
            sample.date = date
            sample.metadata["date"] = date
            sample.metadata["strain"] = sample.strain

        match_db = util.get_match_db(ts, db_path, samples, date, num_mismatches)
        # print("Match DB", len(match_db))
        # match_db.print_all()
        ts2 = sc2ts.add_matching_results(
            f"hmm_cost <= {max_hmm_cost}",
            match_db=match_db,
            ts=ts,
            date=date,
        )
        # assert ts2.num_samples == len(samples) + ts.num_samples
        # for u, sample in zip(ts2.samples()[-len(samples) :], samples):
        #     node = ts2.node(u)
        #     assert node.time == 0
        assert ts2.num_sites == ts.num_sites
        return ts2

    def test_one_sample(self, tmp_path):
        # 4.00┊  0  ┊
        #     ┊  ┃  ┊
        # 3.00┊  1  ┊
        #     ┊  ┃  ┊
        # 2.00┊  4  ┊
        #     ┊ ┏┻┓ ┊
        # 1.00┊ 2 3 ┊
        #     0   29904
        ts = util.example_binary(2)
        samples = util.get_samples(ts, [[(0, ts.sequence_length, 1)]])
        ts2 = self.add_matching_results(samples, ts, tmp_path / "match.db")
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0, 4: 1, 2: 4, 3: 4, 5: 1}

    def test_one_sample_recombinant(self, tmp_path):
        # 4.00┊  0  ┊
        #     ┊  ┃  ┊
        # 3.00┊  1  ┊
        #     ┊  ┃  ┊
        # 2.00┊  4  ┊
        #     ┊ ┏┻┓ ┊
        # 1.00┊ 2 3 ┊
        #     0   29904
        ts = util.example_binary(2)
        L = ts.sequence_length
        x = L / 2
        samples = util.get_samples(ts, [[(0, x, 2), (x, L, 3)]])
        date = "2021-01-05"
        ts2 = self.add_matching_results(samples, ts, tmp_path / "match.db", date=date)

        assert ts2.num_trees == 2
        assert ts2.first().parent_dict == {1: 0, 4: 1, 2: 4, 3: 4, 6: 2, 5: 6}
        assert ts2.last().parent_dict == {1: 0, 4: 1, 2: 4, 3: 4, 6: 3, 5: 6}
        assert ts2.node(6).flags == sc2ts.NODE_IS_RECOMBINANT
        assert ts2.node(6).metadata == {"date_added": date}

    def test_one_sample_recombinant_filtered(self, tmp_path):
        # 4.00┊  0  ┊
        #     ┊  ┃  ┊
        # 3.00┊  1  ┊
        #     ┊  ┃  ┊
        # 2.00┊  4  ┊
        #     ┊ ┏┻┓ ┊
        # 1.00┊ 2 3 ┊
        #     0   29904
        ts = util.example_binary(2)
        L = ts.sequence_length
        x = L / 2
        samples = util.get_samples(ts, [[(0, x, 2), (x, L, 3)]])
        ts2 = self.add_matching_results(
            samples, ts, tmp_path / "match.db", num_mismatches=1e3, max_hmm_cost=1e3 - 1
        )
        assert ts2.num_trees == 1
        assert ts2.num_nodes == ts.num_nodes
        assert ts2.num_samples == ts.num_samples

    def test_two_samples_recombinant_one_filtered(self, tmp_path):
        ts = util.example_binary(2)
        L = ts.sequence_length
        x = L / 2
        new_paths = [
            [(0, x, 2), (x, L, 3)],  # Added
            [
                (0, L / 4, 2),
                (L / 4, L / 2, 3),
                (L / 2, 3 / 4 * L, 4),
                (3 / 4 * L, L, 2),
            ],  # Filtered
        ]
        samples = util.get_samples(ts, new_paths)
        ts2 = self.add_matching_results(
            samples, ts, tmp_path / "match.db", num_mismatches=3, max_hmm_cost=4
        )
        assert ts2.num_trees == 2
        assert ts2.num_samples == ts.num_samples + 1

    def test_one_sample_one_mutation(self, tmp_path):
        ts = sc2ts.initial_ts()
        ts = sc2ts.increment_time("2020-01-01", ts)
        samples = util.get_samples(
            ts, [[(0, ts.sequence_length, 1)]], mutations=[[(0, "X")]]
        )
        ts2 = self.add_matching_results(samples, ts, tmp_path / "match.db")
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0, 2: 1}
        assert ts2.site(0).ancestral_state == ts.site(0).ancestral_state
        assert ts2.num_mutations == 1
        var = next(ts2.variants())
        assert var.alleles[var.genotypes[1]] == "X"

    def test_one_sample_one_mutation_filtered(self, tmp_path):
        ts = sc2ts.initial_ts()
        ts = sc2ts.increment_time("2020-01-01", ts)
        samples = util.get_samples(
            ts, [[(0, ts.sequence_length, 1)]], mutations=[[(0, "X")]]
        )
        ts2 = self.add_matching_results(
            samples, ts, tmp_path / "match.db", num_mismatches=0.0, max_hmm_cost=0.0
        )
        assert ts2.num_trees == ts.num_trees
        assert ts2.site(0).ancestral_state == ts.site(0).ancestral_state
        assert ts2.num_mutations == 0

    def test_two_samples_one_mutation_one_filtered(self, tmp_path):
        ts = sc2ts.initial_ts()
        ts = sc2ts.increment_time("2020-01-01", ts)
        x = int(ts.sequence_length / 2)
        new_paths = [
            [(0, ts.sequence_length, 1)],
            [(0, ts.sequence_length, 1)],
        ]
        new_mutations = [
            [(0, "X")],  # Added
            [(0, "X"), (x, "X")],  # Filtered
        ]
        samples = util.get_samples(
            ts,
            paths=new_paths,
            mutations=new_mutations,
        )
        ts2 = self.add_matching_results(
            samples, ts, tmp_path / "match.db", num_mismatches=3, max_hmm_cost=1
        )
        assert ts2.num_trees == ts.num_trees
        assert ts2.site(0).ancestral_state == ts.site(0).ancestral_state
        assert ts2.num_mutations == 1
        var = next(ts2.variants())
        assert var.alleles[var.genotypes[1]] == "X"


class TestMatchTsinfer:
    def match_tsinfer(self, samples, ts, **kwargs):
        sc2ts.inference.match_tsinfer(
            samples=samples, ts=ts, num_mismatches=1000, **kwargs
        )

    @pytest.mark.parametrize("mirror", [False, True])
    def test_match_reference(self, mirror):
        ts = sc2ts.initial_ts()
        tables = ts.dump_tables()
        tables.sites.truncate(20)
        ts = tables.tree_sequence()
        samples = util.get_samples(ts, [[(0, ts.sequence_length, 1)]])
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        ma = sc2ts.alignments.encode_and_mask(alignment)
        h = ma.alignment[ts.sites_position.astype(int)]
        samples[0].alignment = h
        self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert samples[0].breakpoints == [0, ts.sequence_length]
        assert samples[0].parents == [ts.num_nodes - 1]
        assert len(samples[0].mutations) == 0

    @pytest.mark.parametrize("mirror", [False, True])
    @pytest.mark.parametrize("site_id", [0, 10, 19])
    def test_match_reference_one_mutation(self, mirror, site_id):
        ts = sc2ts.initial_ts()
        tables = ts.dump_tables()
        tables.sites.truncate(20)
        ts = tables.tree_sequence()
        samples = util.get_samples(ts, [[(0, ts.sequence_length, 1)]])
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        ma = sc2ts.alignments.encode_and_mask(alignment)
        h = ma.alignment[ts.sites_position.astype(int)]
        # Mutate to gap
        h[site_id] = sc2ts.core.ALLELES.index("-")
        samples[0].alignment = h
        self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert samples[0].breakpoints == [0, ts.sequence_length]
        assert samples[0].parents == [ts.num_nodes - 1]
        assert len(samples[0].mutations) == 1
        mut = samples[0].mutations[0]
        assert mut.site_id == site_id
        assert mut.site_position == ts.sites_position[site_id]
        assert mut.derived_state == "-"
        assert mut.inherited_state == ts.site(site_id).ancestral_state
        assert not mut.is_reversion
        assert not mut.is_immediate_reversion

    @pytest.mark.parametrize("mirror", [False, True])
    @pytest.mark.parametrize("allele", range(5))
    def test_match_reference_all_same(self, mirror, allele):
        ts = sc2ts.initial_ts()
        tables = ts.dump_tables()
        tables.sites.truncate(20)
        ts = tables.tree_sequence()
        samples = util.get_samples(ts, [[(0, ts.sequence_length, 1)]])
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        ma = sc2ts.alignments.encode_and_mask(alignment)
        ref = ma.alignment[ts.sites_position.astype(int)]
        h = np.zeros_like(ref) + allele
        samples[0].alignment = h
        self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert samples[0].breakpoints == [0, ts.sequence_length]
        assert samples[0].parents == [ts.num_nodes - 1]
        muts = samples[0].mutations
        assert len(muts) > 0
        assert len(muts) == np.sum(ref != allele)
        for site_id, mut in zip(np.where(ref != allele)[0], muts):
            assert mut.site_id == site_id
            assert mut.derived_state == sc2ts.core.ALLELES[allele]


class TestMatchPathTs:
    def match_path_ts(self, samples, ts):
        # FIXME this API is terrible
        ts2 = sc2ts.match_path_ts(samples, ts, samples[0].path, [])
        assert ts2.num_samples == len(samples)
        for u, sample in zip(ts.samples()[1:], samples):
            node = ts.node(u)
            assert node.time == 0
            assert node.metadata == sample.metadata
        return ts2

    def test_one_sample(self):
        ts = sc2ts.initial_ts()
        samples = util.get_samples(ts, [[(0, ts.sequence_length, 1)]])
        ts2 = self.match_path_ts(samples, ts)
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

        samples = util.get_samples(
            ts, [[(0, ts.sequence_length, c)]], mutations=[[(0, "G"), (1, "G")]]
        )
        ts2 = self.match_path_ts(samples, ts)
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
        samples = util.get_samples(
            ts, [[(0, ts.sequence_length, 1)]], mutations=[[(100, "X")]]
        )
        ts2 = self.match_path_ts(samples, ts)
        assert ts2.num_trees == 1
        tree = ts2.first()
        assert tree.parent_dict == {1: 0}
        assert ts2.num_sites == 1
        assert ts2.site(0).ancestral_state == ts.site(100).ancestral_state
        assert list(ts2.haplotypes()) == ["X"]

    def test_two_sample_one_mutation_each(self):
        ts = sc2ts.initial_ts()

        samples = util.get_samples(
            ts,
            [[(0, ts.sequence_length, 1)], [(0, ts.sequence_length, 1)]],
            mutations=[[(100, "X")], [(200, "Y")]],
        )
        ts2 = self.match_path_ts(samples, ts)
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
        samples = util.get_samples(
            ts,
            [[(0, ts.sequence_length, 1)]],
            mutations=[[(j, f"{j}") for j in range(num_mutations)]],
        )
        ts2 = self.match_path_ts(samples, ts)
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
            strain = f"x{j}"
            date = "2021-01-01"
            samples.append(
                sc2ts.Sample(
                    strain=strain,
                    date=date,
                    metadata={f"x{j}": j, f"y{j}": list(range(j))},
                    path=[(0, ts.sequence_length, 1)],
                    mutations=[],
                )
            )

        sc2ts.update_path_info(
            samples, ts, [s.path for s in samples], [s.mutations for s in samples]
        )
        self.match_path_ts(samples, ts)


class TestMirrorTsCoords:
    def test_dense_sites_example(self):
        tree = tskit.Tree.generate_balanced(2, span=10)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(0, "A")
        tables.sites.add_row(2, "C")
        tables.sites.add_row(5, "-")
        tables.sites.add_row(8, "G")
        tables.sites.add_row(9, "T")
        ts1 = tables.tree_sequence()
        ts2 = sc2ts.inference.mirror_ts_coordinates(ts1)
        assert ts2.num_sites == ts1.num_sites
        assert list(ts2.sites_position) == [0, 1, 4, 7, 9]
        assert "".join(site.ancestral_state for site in ts2.sites()) == "TG-CA"

    def test_sparse_sites_example(self):
        tree = tskit.Tree.generate_balanced(2, span=100)
        tables = tree.tree_sequence.dump_tables()
        tables.sites.add_row(10, "A")
        tables.sites.add_row(12, "C")
        tables.sites.add_row(15, "-")
        tables.sites.add_row(18, "G")
        tables.sites.add_row(19, "T")
        ts1 = tables.tree_sequence()
        ts2 = sc2ts.inference.mirror_ts_coordinates(ts1)
        assert ts2.num_sites == ts1.num_sites
        assert list(ts2.sites_position) == [80, 81, 84, 87, 89]
        assert "".join(site.ancestral_state for site in ts2.sites()) == "TG-CA"

    def check_double_mirror(self, ts):
        mirror = sc2ts.inference.mirror_ts_coordinates(ts)
        for h1, h2 in zip(ts.haplotypes(), mirror.haplotypes()):
            assert h1 == h2[::-1]
        double_mirror = sc2ts.inference.mirror_ts_coordinates(mirror)
        ts.tables.assert_equals(double_mirror.tables)

    @pytest.mark.parametrize("n", [2, 3, 13, 20])
    def test_single_tree_no_mutations(self, n):
        ts = msprime.sim_ancestry(n, random_seed=42)
        self.check_double_mirror(ts)

    @pytest.mark.parametrize("n", [2, 3, 13, 20])
    def test_multiple_trees_no_mutations(self, n):
        ts = msprime.sim_ancestry(
            n,
            sequence_length=100,
            recombination_rate=1,
            random_seed=420,
        )
        assert ts.num_trees > 1
        self.check_double_mirror(ts)

    @pytest.mark.parametrize("n", [2, 3, 13, 20])
    def test_single_tree_mutations(self, n):
        ts = msprime.sim_ancestry(n, sequence_length=100, random_seed=42234)
        ts = msprime.sim_mutations(ts, rate=0.01, random_seed=32234)
        assert ts.num_sites > 2
        self.check_double_mirror(ts)

    @pytest.mark.parametrize("n", [2, 3, 13, 20])
    def test_multiple_tree_mutations(self, n):
        ts = msprime.sim_ancestry(
            n, sequence_length=100, recombination_rate=0.1, random_seed=1234
        )
        ts = msprime.sim_mutations(ts, rate=0.1, random_seed=334)
        assert ts.num_sites > 2
        assert ts.num_trees > 2
        self.check_double_mirror(ts)

    def test_high_recomb_mutation(self):
        # Example that's saturated for muts and recombs
        ts = msprime.sim_ancestry(
            10, sequence_length=10, recombination_rate=10, random_seed=1
        )
        assert ts.num_trees == 10
        ts = msprime.sim_mutations(ts, rate=1, random_seed=1)
        assert ts.num_sites == 10
        assert ts.num_mutations > 10
        self.check_double_mirror(ts)


class TestRealData:
    def test_first_day(self, tmp_path, fx_alignment_store, fx_metadata_db):
        ts = sc2ts.extend(
            alignment_store=fx_alignment_store,
            metadata_db=fx_metadata_db,
            base_ts=sc2ts.initial_ts(),
            date="2020-01-19",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        # 25.00┊ 0 ┊
        #      ┊ ┃ ┊
        # 24.00┊ 1 ┊
        #      ┊ ┃ ┊
        # 0.00 ┊ 2 ┊
        #     0 29904
        assert ts.num_trees == 1
        assert ts.num_nodes == 3
        assert ts.num_samples == 2
        assert ts.num_mutations == 3
        assert list(ts.nodes_time) == [25, 24, 0]
        assert ts.metadata["sc2ts"]["date"] == "2020-01-19"
        assert ts.metadata["sc2ts"]["samples_strain"] == [
            "Wuhan/Hu-1/2019",
            "SRR11772659",
        ]
        assert list(ts.samples()) == [1, 2]
        assert ts.node(1).metadata["strain"] == "Wuhan/Hu-1/2019"
        assert ts.node(2).metadata["strain"] == "SRR11772659"
        assert list(ts.mutations_node) == [2, 2, 2]
        assert list(ts.mutations_time) == [0, 0, 0]
        assert list(ts.mutations_site) == [8632, 17816, 27786]
        sc2ts_md = ts.node(2).metadata["sc2ts"]
        assert len(sc2ts_md["mutations"]) == 3
        for mut_md, mut in zip(sc2ts_md["mutations"], ts.mutations()):
            assert mut_md["derived_state"] == mut.derived_state
            assert mut_md["site_id"] == mut.site
            assert mut_md["site_position"] == ts.sites_position[mut.site]
            assert mut_md["inherited_state"] == ts.site(mut.site).ancestral_state
        assert sc2ts_md["path"] == [{"left": 0, "parent": 1, "right": 29904}]
        assert sc2ts_md["qc"] == {
            "num_masked_sites": 133,
            "original_base_composition": {
                "A": 8894,
                "C": 5472,
                "G": 5850,
                "N": 121,
                "T": 9566,
            },
            "original_md5": "e96feaa72c4f4baba73c2e147ede7502",
        }

    def test_2020_02_10_metadata(self, fx_ts_2020_02_10):
        ts = fx_ts_2020_02_10
        assert ts.metadata["sc2ts"]["date"] == "2020-02-10"
        samples_strain = [ts.node(u).metadata["strain"] for u in ts.samples()]
        assert ts.metadata["sc2ts"]["samples_strain"] == samples_strain
        # print(ts.tables.mutations)
        # print(ts.draw_text())


class TestMatchingDetails:

    def test_exact_matches(self, fx_ts_2020_02_10, fx_alignment_store, fx_metadata_db):
        print("HERE")

    def test_other_exact_matches(self, tmp_path, fx_ts_2020_02_10, fx_alignment_store, fx_metadata_db):
        print("HERE")
        match_db = sc2ts.MatchDb.initialise(tmp_path / "match.db")
        ts = sc2ts.extend(
            alignment_store=fx_alignment_store,
            metadata_db=fx_metadata_db,
            base_ts=fx_ts_2020_02_10,
            date="2020-02-11",
            match_db=match_db,
            min_group_size=2,
        )



