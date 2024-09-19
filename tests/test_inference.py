import collections

import numpy as np
import numpy.testing as nt
import pytest
import tsinfer
import tskit
import msprime

import sc2ts
import util


class TestSolveNumMismatches:
    @pytest.mark.parametrize(
        ["k", "expected_rho"],
        [(2, 0.0001904), (3, 2.50582e-06), (4, 3.297146e-08), (1000, 0)],
    )
    def test_examples(self, k, expected_rho):
        mu, rho = sc2ts.solve_num_mismatches(k)
        assert mu == 0.0125
        nt.assert_almost_equal(rho, expected_rho)


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
        assert node.metadata == {
            "date": "2019-12-26",
            "strain": "Wuhan/Hu-1/2019",
            "sc2ts": {"notes": "Reference sequence"},
        }
        alignment = next(ts.alignments())
        assert alignment == sc2ts.core.get_reference_sequence()


class TestMatchTsinfer:
    def match_tsinfer(self, samples, ts, mirror_coordinates=False, **kwargs):
        return sc2ts.inference.match_tsinfer(
            samples=samples,
            ts=ts,
            mu=0.125,
            rho=0,
            mirror_coordinates=mirror_coordinates,
            **kwargs,
        )

    @pytest.mark.parametrize("mirror", [False, True])
    def test_match_reference(self, mirror):
        ts = sc2ts.initial_ts()
        tables = ts.dump_tables()
        tables.sites.truncate(20)
        ts = tables.tree_sequence()
        samples = [sc2ts.Sample("test", "2020-01-01")]
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        ma = sc2ts.alignments.encode_and_mask(alignment)
        h = ma.alignment[ts.sites_position.astype(int)]
        samples[0].alignment = h
        matches = self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert matches[0].breakpoints == [0, ts.sequence_length]
        assert matches[0].parents == [ts.num_nodes - 1]
        assert len(matches[0].mutations) == 0

    @pytest.mark.parametrize("mirror", [False, True])
    @pytest.mark.parametrize("site_id", [0, 10, 19])
    def test_match_reference_one_mutation(self, mirror, site_id):
        ts = sc2ts.initial_ts()
        tables = ts.dump_tables()
        tables.sites.truncate(20)
        ts = tables.tree_sequence()
        samples = [sc2ts.Sample("test", "2020-01-01")]
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        ma = sc2ts.alignments.encode_and_mask(alignment)
        h = ma.alignment[ts.sites_position.astype(int)]
        # Mutate to gap
        h[site_id] = sc2ts.core.ALLELES.index("-")
        samples[0].alignment = h
        matches = self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert matches[0].breakpoints == [0, ts.sequence_length]
        assert matches[0].parents == [ts.num_nodes - 1]
        assert len(matches[0].mutations) == 1
        mut = matches[0].mutations[0]
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
        samples = [sc2ts.Sample("test", "2020-01-01")]
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        ma = sc2ts.alignments.encode_and_mask(alignment)
        ref = ma.alignment[ts.sites_position.astype(int)]
        h = np.zeros_like(ref) + allele
        samples[0].alignment = h
        matches = self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert matches[0].breakpoints == [0, ts.sequence_length]
        assert matches[0].parents == [ts.num_nodes - 1]
        muts = matches[0].mutations
        assert len(muts) > 0
        assert len(muts) == np.sum(ref != allele)
        for site_id, mut in zip(np.where(ref != allele)[0], muts):
            assert mut.site_id == site_id
            assert mut.derived_state == sc2ts.core.ALLELES[allele]


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
    dates = [
        "2020-01-01",
        "2020-01-19",
        "2020-01-24",
        "2020-01-25",
        "2020-01-28",
        "2020-01-29",
        "2020-01-30",
        "2020-01-31",
        "2020-02-01",
        "2020-02-02",
        "2020-02-03",
        "2020-02-04",
        "2020-02-05",
        "2020-02-06",
        "2020-02-07",
        "2020-02-08",
        "2020-02-09",
        "2020-02-10",
        "2020-02-11",
        "2020-02-13",
    ]

    def test_first_day(self, tmp_path, fx_ts_map, fx_alignment_store, fx_metadata_db):
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
        hmm_md = sc2ts_md["hmm"][0]
        assert len(hmm_md["mutations"]) == 3
        for mut_md, mut in zip(hmm_md["mutations"], ts.mutations()):
            assert mut_md["derived_state"] == mut.derived_state
            assert mut_md["site_position"] == ts.sites_position[mut.site]
            assert mut_md["inherited_state"] == ts.site(mut.site).ancestral_state
        assert hmm_md["path"] == [{"left": 0, "parent": 1, "right": 29904}]
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

        ts.tables.assert_equals(fx_ts_map["2020-01-19"].tables, ignore_provenance=True)

    def test_2020_02_02(self, tmp_path, fx_ts_map, fx_alignment_store, fx_metadata_db):
        ts = sc2ts.extend(
            alignment_store=fx_alignment_store,
            metadata_db=fx_metadata_db,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        assert ts.num_samples == 26
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == 4
        # print(samples)
        # print(fx_ts_map["2020-02-02"])
        ts.tables.assert_equals(fx_ts_map["2020-02-02"].tables, ignore_provenance=True)

    def test_2020_02_08(self, tmp_path, fx_ts_map, fx_alignment_store, fx_metadata_db):
        ts = sc2ts.extend(
            alignment_store=fx_alignment_store,
            metadata_db=fx_metadata_db,
            base_ts=fx_ts_map["2020-02-07"],
            date="2020-02-08",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )

        # SRR11597163 has a reversion (4923, 'C')
        # Site ID 4923 has position 5025
        for node in ts.nodes():
            if node.is_sample() and node.metadata["strain"] == "SRR11597163":
                break
        assert node.metadata["strain"] == "SRR11597163"
        scmd = node.metadata["sc2ts"]
        # We have a mutation from a mismatch
        assert scmd["hmm"][0]["mutations"] == [
            {"derived_state": "C", "inherited_state": "T", "site_position": 5025}
        ]
        # But no mutations above the node itself.
        assert np.sum(ts.mutations_node == node.id) == 0

        tree = ts.first()
        rp_node = ts.node(tree.parent(node.id))
        assert rp_node.flags == sc2ts.NODE_IS_REVERSION_PUSH
        assert rp_node.metadata["sc2ts"] == {
            "date_added": "2020-02-08",
            "sites": [4923],
        }
        ts.tables.assert_equals(fx_ts_map["2020-02-08"].tables, ignore_provenance=True)

        sib_sample = ts.node(tree.siblings(node.id)[0])
        assert sib_sample.metadata["strain"] == "SRR11597168"

        assert np.sum(ts.mutations_node == sib_sample.id) == 1
        mutation = ts.mutation(np.where(ts.mutations_node == sib_sample.id)[0][0])
        assert mutation.derived_state == "T"
        assert mutation.parent == -1

    @pytest.mark.parametrize("date", dates)
    def test_date_metadata(self, fx_ts_map, date):
        ts = fx_ts_map[date]
        assert ts.metadata["sc2ts"]["date"] == date
        samples_strain = [ts.node(u).metadata["strain"] for u in ts.samples()]
        assert ts.metadata["sc2ts"]["samples_strain"] == samples_strain
        # print(ts.tables.mutations)
        # print(ts.draw_text())

    @pytest.mark.parametrize("date", dates)
    def test_date_validate(self, fx_ts_map, fx_alignment_store, date):
        ts = fx_ts_map[date]
        sc2ts.validate(ts, fx_alignment_store)

    def test_mutation_type_metadata(self, fx_ts_map):
        ts = fx_ts_map[self.dates[-1]]
        for mutation in ts.mutations():
            md = mutation.metadata["sc2ts"]
            assert md["type"] in ["parsimony", "overlap"]

    def test_node_type_metadata(self, fx_ts_map):
        ts = fx_ts_map[self.dates[-1]]
        exact_matches = 0
        for node in list(ts.nodes())[2:]:
            md = node.metadata["sc2ts"]
            if node.is_sample():
                # All samples are either exact matches, or added as part of a group
                assert "hmm" in md
                if node.flags & sc2ts.NODE_IS_EXACT_MATCH:
                    exact_matches += 1
                else:
                    assert "group_id" in md
        assert exact_matches > 0

    @pytest.mark.parametrize(
        ["gid", "date", "internal", "strains"],
        [
            (
                "02984ed831cd3c72d206959449dcf8c9",
                "2020-01-19",
                0,
                ["SRR11772659"],
            ),
            (
                "635b05f53af60d8385226cd0e00e97ab",
                "2020-02-08",
                0,
                ["SRR11597163"],
            ),
            (
                "0c36395a702379413ffc855f847873c6",
                "2020-01-24",
                1,
                ["SRR11397727", "SRR11397730"],
            ),
            (
                "9d00e2a016661caea4c2d9abf83375b8",
                "2020-01-30",
                1,
                ["SRR12162232", "SRR12162233", "SRR12162234", "SRR12162235"],
            ),
            (
                "2f508c7ba05387dec0adbf2db4a7481a",
                "2020-02-04",
                1,
                ["SRR11597174", "SRR11597188", "SRR11597136", "SRR11597175"],
            ),
        ],
    )
    def test_group(self, fx_ts_map, gid, date, internal, strains):
        ts = fx_ts_map[self.dates[-1]]
        samples = []
        num_internal = 0
        got_strains = []
        for node in ts.nodes():
            md = node.metadata
            group = md["sc2ts"].get("group_id", None)
            if group == gid:
                if node.is_sample():
                    got_strains.append(md["strain"])
                    assert md["date"] == date
                else:
                    assert md["sc2ts"]["date_added"] == date
                    num_internal += 1
        assert num_internal == internal
        assert got_strains == strains

    @pytest.mark.parametrize("date", dates[1:])
    def test_node_mutation_counts(self, fx_ts_map, date):
        # Basic check to make sure our fixtures are what we expect.
        # NOTE: this is somewhat fragile as the numbers of nodes does change
        # a little depending on the exact solution that the HMM choses, for
        # example when there are multiple single-mutation matches at different
        # sites.
        ts = fx_ts_map[date]
        expected = {
            "2020-01-19": {"nodes": 3, "mutations": 3},
            "2020-01-24": {"nodes": 6, "mutations": 4},
            "2020-01-25": {"nodes": 10, "mutations": 6},
            "2020-01-28": {"nodes": 12, "mutations": 11},
            "2020-01-29": {"nodes": 15, "mutations": 15},
            "2020-01-30": {"nodes": 21, "mutations": 19},
            "2020-01-31": {"nodes": 22, "mutations": 21},
            "2020-02-01": {"nodes": 27, "mutations": 27},
            "2020-02-02": {"nodes": 32, "mutations": 36},
            "2020-02-03": {"nodes": 35, "mutations": 42},
            "2020-02-04": {"nodes": 40, "mutations": 48},
            "2020-02-05": {"nodes": 41, "mutations": 48},
            "2020-02-06": {"nodes": 46, "mutations": 51},
            "2020-02-07": {"nodes": 48, "mutations": 57},
            "2020-02-08": {"nodes": 53, "mutations": 58},
            "2020-02-09": {"nodes": 55, "mutations": 61},
            "2020-02-10": {"nodes": 56, "mutations": 65},
            "2020-02-11": {"nodes": 58, "mutations": 66},
            "2020-02-13": {"nodes": 62, "mutations": 68},
        }
        assert ts.num_nodes == expected[date]["nodes"]
        assert ts.num_mutations == expected[date]["mutations"]

    @pytest.mark.parametrize(
        ["strain", "parent"],
        [
            ("SRR11397726", 5),
            ("SRR11397729", 5),
            ("SRR11597132", 9),
            ("SRR11597177", 9),
            ("SRR11597156", 9),
            ("SRR11597216", 1),
            ("SRR11597207", 39),
            ("ERR4205570", 54),
        ],
    )
    def test_exact_matches(self, fx_ts_map, strain, parent):
        ts = fx_ts_map[self.dates[-1]]
        node = ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
        x = ts.node(node)
        assert x.flags == (tskit.NODE_IS_SAMPLE | sc2ts.core.NODE_IS_EXACT_MATCH)
        md = x.metadata
        assert md["strain"] == strain
        sc2ts_md = md["sc2ts"]
        hmm_md = sc2ts_md["hmm"][0]
        assert len(hmm_md["path"]) == 1
        assert hmm_md["path"][0] == {
            "parent": parent,
            "left": 0,
            "right": ts.sequence_length,
        }
        edges = np.where(ts.edges_child == node)[0]
        assert len(edges) == 1
        e = edges[0]
        assert ts.edges_parent[e] == parent
        assert ts.edges_left[e] == 0
        assert ts.edges_right[e] == ts.sequence_length
        assert np.sum(ts.mutations_node == node) == 0


class TestMatchingDetails:
    @pytest.mark.parametrize(
        ("strain", "parent"), [("SRR11597207", 39), ("ERR4205570", 54)]
    )
    @pytest.mark.parametrize("num_mismatches", [2, 3, 4])
    def test_exact_matches(
        self,
        fx_ts_map,
        fx_alignment_store,
        fx_metadata_db,
        strain,
        parent,
        num_mismatches,
    ):
        ts = fx_ts_map["2020-02-10"]
        samples = sc2ts.preprocess(
            [fx_metadata_db[strain]], ts, "2020-02-20", fx_alignment_store
        )
        mu, rho = sc2ts.solve_num_mismatches(num_mismatches)
        matches = sc2ts.match_tsinfer(
            samples=samples,
            ts=ts,
            mu=mu,
            rho=rho,
            likelihood_threshold=mu**num_mismatches - 1e-12,
            num_threads=0,
        )
        s = matches[0]
        assert len(s.mutations) == 0
        assert len(s.path) == 1
        assert s.path[0].parent == parent

    @pytest.mark.parametrize(
        ("strain", "parent", "position", "derived_state"),
        [("SRR11597218", 9, 289, "T"), ("ERR4206593", 54, 26994, "T")],
    )
    @pytest.mark.parametrize("num_mismatches", [2, 3, 4])
    # @pytest.mark.parametrize("precision", [0, 1, 2, 12])
    def test_one_mismatch(
        self,
        fx_ts_map,
        fx_alignment_store,
        fx_metadata_db,
        strain,
        parent,
        position,
        derived_state,
        num_mismatches,
    ):
        ts = fx_ts_map["2020-02-10"]
        samples = sc2ts.preprocess(
            [fx_metadata_db[strain]], ts, "2020-02-20", fx_alignment_store
        )
        mu, rho = sc2ts.solve_num_mismatches(num_mismatches)
        matches = sc2ts.match_tsinfer(
            samples=samples,
            ts=ts,
            mu=mu,
            rho=rho,
            likelihood_threshold=mu - 1e-5,
            num_threads=0,
        )
        s = matches[0]
        assert len(s.mutations) == 1
        assert s.mutations[0].site_position == position
        assert s.mutations[0].derived_state == derived_state
        assert len(s.path) == 1
        assert s.path[0].parent == parent

    @pytest.mark.parametrize("num_mismatches", [2, 3, 4])
    def test_two_mismatches(
        self,
        fx_ts_map,
        fx_alignment_store,
        fx_metadata_db,
        num_mismatches,
    ):
        strain = "ERR4204459"
        ts = fx_ts_map["2020-02-10"]
        samples = sc2ts.preprocess(
            [fx_metadata_db[strain]], ts, "2020-02-20", fx_alignment_store
        )
        mu, rho = sc2ts.solve_num_mismatches(num_mismatches)
        matches = sc2ts.match_tsinfer(
            samples=samples,
            ts=ts,
            mu=mu,
            rho=rho,
            likelihood_threshold=mu**2 - 1e-12,
            num_threads=0,
        )
        s = matches[0]
        assert len(s.path) == 1
        assert s.path[0].parent == 5
        assert len(s.mutations) == 2

    def test_match_recombinant(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        strains = ["SRR11597188", "SRR11597163"]
        nodes = [
            ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
            for strain in strains
        ]
        assert nodes == [36, 51]
        # SRR11597188 36  [(801, 'G'), (2943, 'G'), (3694, 'T')]
        # SRR11597163 51  [(15107, 'T'), (28930, 'T')]
        H = ts.genotype_matrix(samples=nodes, alleles=tuple("ACGT-")).T

        bp = 10_000
        h = H[0].copy()
        h[bp:] = H[1][bp:]

        s = sc2ts.Sample("frankentype", "2020-02-14")
        s.alignment = h

        mu, rho = sc2ts.solve_num_mismatches(2)
        matches = sc2ts.match_tsinfer(
            samples=[s],
            ts=ts,
            mu=mu,
            rho=rho,
            num_threads=0,
        )
        interval_right = ts.sites_position[15107]
        m = matches[0]
        assert len(m.mutations) == 0
        assert len(m.path) == 2
        assert m.path[0].parent == nodes[0]
        assert m.path[0].left == 0
        assert m.path[0].right == interval_right
        # 52 is the parent of 51, and sequence identical.
        assert m.path[1].parent == 52
        assert m.path[1].left == interval_right
        assert m.path[1].right == ts.sequence_length


class TestMatchRecombinants:
    def test_match_recombinant(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        strains = ["SRR11597188", "SRR11597163"]
        nodes = [
            ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
            for strain in strains
        ]
        assert nodes == [36, 51]
        # These are *site IDs*
        # SRR11597188 36  [(801, 'G'), (2943, 'G'), (3694, 'T')]
        # SRR11597163 51  [(15107, 'T'), (28930, 'T')]
        H = ts.genotype_matrix(samples=nodes, alleles=tuple("ACGT-")).T
        bp = 10_000
        h = H[0].copy()
        h[bp:] = H[1][bp:]

        s = sc2ts.Sample("frankentype", "2020-02-14")
        s.alignment = h

        sc2ts.match_recombinants(
            samples=[s],
            base_ts=ts,
            num_mismatches=2,
            num_threads=0,
        )
        interval_right = ts.sites_position[15107]
        m = s.hmm_reruns["forward"]
        assert len(m.mutations) == 0
        assert len(m.path) == 2
        assert m.path[0].parent == nodes[0]
        assert m.path[0].left == 0
        assert m.path[0].right == interval_right
        # 52 is the parent of 51, and sequence identical.
        assert m.path[1].parent == 52
        assert m.path[1].left == interval_right
        assert m.path[1].right == ts.sequence_length

        interval_left = ts.sites_position[3694] + 1
        m = s.hmm_reruns["reverse"]
        assert len(m.mutations) == 0
        assert len(m.path) == 2
        assert m.path[0].parent == nodes[0]
        assert m.path[0].left == 0
        assert m.path[0].right == interval_left
        # 52 is the parent of 51, and sequence identical.
        assert m.path[1].parent == 52
        assert m.path[1].left == interval_left
        assert m.path[1].right == ts.sequence_length

        m = s.hmm_reruns["no_recombination"]
        assert len(m.mutations) == 2
        assert m.mutation_summary() == "[15324C>T, 29303C>T]"
        assert len(m.path) == 1
        assert m.path[0].parent == nodes[0]
        assert m.path[0].left == 0
        assert m.path[0].right == ts.sequence_length

        assert "no_recombination" in s.summary()
