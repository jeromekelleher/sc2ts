import dataclasses
import collections
import hashlib
import logging
import json

import numpy as np
import numpy.testing as nt
import pytest
import tsinfer
import tskit
import msprime
import pandas as pd

import sc2ts
import util


def run_extend(dataset, base_ts, date, match_db, **kwargs):
    return sc2ts.extend(
        dataset=dataset.path,
        base_ts=base_ts.path,
        date=date,
        match_db=match_db.path,
        **kwargs,
    )


def recombinant_example_1(ts_map):
    """
    Example recombinant created by cherry picking two samples that differ
    by mutations on either end of the genome, and smushing them together.
    Note there's only two mutations needed, so we need to set num_mismatches=2
    """
    ts = ts_map["2020-02-13"]
    strains = ["SRR11597188", "SRR11597163"]
    nodes = [
        ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
        for strain in strains
    ]
    assert nodes == [31, 45]
    # Site positions
    # SRR11597188 36  [(871, 'G'), (3027, 'G'), (3787, 'T')]
    # SRR11597163 51  [(15324, 'T'), (29303, 'T')]
    H = ts.genotype_matrix(samples=nodes, alleles=tuple("ACGT-")).T
    bp = 10_000
    h = H[0].copy().astype(np.int8)
    h[bp:] = H[1][bp:]

    s = sc2ts.Sample("frankentype", "2020-02-14", haplotype=h)
    return ts, s


def test_get_group_strains(fx_ts_map):
    ts = fx_ts_map["2020-02-13"]
    groups = sc2ts.get_group_strains(ts)
    assert len(groups) > 0
    for group_id, strains in groups.items():
        m = hashlib.md5()
        for strain in sorted(strains):
            m.update(strain.encode())
        assert group_id == m.hexdigest()


class TestRecombinantHandling:

    def test_get_recombinant_strains_ex1(self, fx_recombinant_example_1):
        d = sc2ts.get_recombinant_strains(fx_recombinant_example_1)
        assert d == {55: ["recombinant_example_1_0", "recombinant_example_1_1"]}

    def test_get_recombinant_strains_ex2(self, fx_recombinant_example_2):
        d = sc2ts.get_recombinant_strains(fx_recombinant_example_2)
        assert d == {56: ["recombinant_114:29825"]}

    def test_get_recombinant_strains_ex4(self, fx_recombinant_example_4):
        d = sc2ts.get_recombinant_strains(fx_recombinant_example_4)
        assert d == {56: ["recombinant_114:29825"]}

    def test_recombinant_example_1(self, fx_recombinant_example_1):
        ts = fx_recombinant_example_1
        samples_strain = ts.metadata["sc2ts"]["samples_strain"]
        samples = ts.samples()
        for s in ["recombinant_example_1_0", "recombinant_example_1_1"]:
            u = samples[samples_strain.index(s)]
            node = ts.node(u)
            md = node.metadata["sc2ts"]
            assert md["breakpoint_intervals"] == [[3788, 11083]]
            assert md["hmm_match"]["path"] == [
                {"left": 0, "parent": 31, "right": 11083},
                {"left": 11083, "parent": 46, "right": 29904},
            ]

    def test_recombinant_example_2(self, fx_recombinant_example_2):
        ts = fx_recombinant_example_2
        samples_strain = ts.metadata["sc2ts"]["samples_strain"]
        u = ts.samples()[samples_strain.index("recombinant_114:29825")]
        node = ts.node(u)
        md = node.metadata["sc2ts"]
        assert md["breakpoint_intervals"] == [[114, 29825]]
        assert md["hmm_match"]["path"] == [
            {"left": 0, "parent": 53, "right": 29825},
            {"left": 29825, "parent": 54, "right": 29904},
        ]

    def test_recombinant_example_3(self, fx_recombinant_example_3):
        ts = fx_recombinant_example_3
        samples_strain = ts.metadata["sc2ts"]["samples_strain"]
        u = ts.samples()[samples_strain.index("recombinant_114:15001:15010:29825")]
        node = ts.node(u)
        md = node.metadata["sc2ts"]
        assert md["breakpoint_intervals"] == [[114, 15001], [15010, 29825]]
        assert md["hmm_match"]["path"] == [
            {"left": 0, "parent": 53, "right": 15001},
            {"left": 15001, "parent": 54, "right": 29825},
            {"left": 29825, "parent": 55, "right": 29904},
        ]


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

    def test_reference_node(self):
        ts = sc2ts.initial_ts()
        assert ts.num_samples == 0
        node = ts.node(1)
        assert node.time == 0
        assert node.metadata == {
            "date": "2019-12-26",
            "strain": "Wuhan/Hu-1/2019",
            "sc2ts": {"notes": "Reference sequence"},
        }
        assert node.flags == sc2ts.NODE_IS_REFERENCE


class TestMatchTsinfer:
    def match_tsinfer(self, samples, ts, mirror_coordinates=False, **kwargs):
        sc2ts.inference.match_tsinfer(
            samples=samples,
            ts=ts,
            num_mismatches=3,
            mismatch_threshold=20,
            mirror_coordinates=mirror_coordinates,
            **kwargs,
        )
        return [s.hmm_match for s in samples]

    @pytest.mark.parametrize("mirror", [False, True])
    def test_match_reference(self, mirror):
        ts = sc2ts.initial_ts()
        tables = ts.dump_tables()
        tables.sites.truncate(20)
        ts = tables.tree_sequence()
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        alignment[0] = "A"
        a = sc2ts.encode_alignment(alignment)
        h = a[ts.sites_position.astype(int)]
        samples = [sc2ts.Sample("test", "2020-01-01", haplotype=h)]
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
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        alignment[0] = "A"
        a = sc2ts.encode_alignment(alignment)
        h = a[ts.sites_position.astype(int)]
        samples = [sc2ts.Sample("test", "2020-01-01", haplotype=h)]
        # Mutate to gap
        h[site_id] = sc2ts.IUPAC_ALLELES.index("-")
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
        alignment = sc2ts.core.get_reference_sequence(as_array=True)
        alignment[0] = "A"
        a = sc2ts.encode_alignment(alignment)
        ref = a[ts.sites_position.astype(int)]
        h = np.zeros_like(ref) + allele
        samples = [sc2ts.Sample("test", "2020-01-01", haplotype=h)]
        matches = self.match_tsinfer(samples, ts, mirror_coordinates=mirror)
        assert matches[0].breakpoints == [0, ts.sequence_length]
        assert matches[0].parents == [ts.num_nodes - 1]
        muts = matches[0].mutations
        assert len(muts) > 0
        assert len(muts) == np.sum(ref != allele)
        for site_id, mut in zip(np.where(ref != allele)[0], muts):
            assert mut.site_id == site_id
            assert mut.derived_state == sc2ts.IUPAC_ALLELES[allele]


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
        "2020-02-15",
    ]

    def test_first_day(self, tmp_path, fx_ts_map, fx_dataset):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map[self.dates[0]],
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
        assert ts.num_samples == 1
        assert ts.num_mutations == 3
        assert list(ts.nodes_time) == [25, 24, 0]
        assert ts.metadata["sc2ts"]["date"] == "2020-01-19"
        assert ts.metadata["sc2ts"]["samples_strain"] == ["SRR11772659"]
        assert list(ts.samples()) == [2]
        assert ts.node(1).metadata["strain"] == "Wuhan/Hu-1/2019"
        assert ts.node(2).metadata["strain"] == "SRR11772659"
        assert list(ts.mutations_node) == [2, 2, 2]
        assert list(ts.mutations_time) == [0, 0, 0]
        assert list(ts.sites_position[ts.mutations_site]) == [8782, 18060, 28144]
        sc2ts_md = ts.node(2).metadata["sc2ts"]
        hmm_md = sc2ts_md["hmm_match"]
        assert len(hmm_md["mutations"]) == 3
        for mut_md, mut in zip(hmm_md["mutations"], ts.mutations()):
            assert mut_md["derived_state"] == mut.derived_state
            assert mut_md["site_position"] == ts.sites_position[mut.site]
            assert mut_md["inherited_state"] == ts.site(mut.site).ancestral_state
        assert hmm_md["path"] == [{"left": 0, "parent": 1, "right": 29904}]
        assert sc2ts_md["num_missing_sites"] == 121
        assert sc2ts_md["alignment_composition"] == {
            "A": 8893,
            "C": 5471,
            "G": 5849,
            "T": 9564,
            "N": 121,
        }
        assert sum(sc2ts_md["alignment_composition"].values()) == ts.num_sites
        ts.tables.assert_equals(fx_ts_map["2020-01-19"].tables, ignore_provenance=True)

    def test_2020_01_25(self, tmp_path, fx_ts_map, fx_dataset):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-01-24"],
            date="2020-01-25",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        assert ts.num_samples == 4
        assert ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["pango"] == {
            "B": 2
        }
        assert ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["node"] == {
            "5": 2
        }
        ts.tables.assert_equals(fx_ts_map["2020-01-25"].tables, ignore_provenance=True)

    def test_using_collection_date(self, tmp_path, fx_ts_map, fx_dataset):

        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map[self.dates[0]],
            date="2020-06-16",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
            date_field="Collection_date",
            hmm_cost_threshold=25,
        )
        assert ts.num_samples == 1
        # This has a hmm cost of 21
        assert ts.metadata["sc2ts"]["samples_strain"] == ["SRR15736313"]

    @pytest.mark.parametrize("num_threads", [0, 1, 3, 10])
    def test_2020_02_02(self, tmp_path, fx_ts_map, fx_dataset, num_threads):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
            num_threads=num_threads,
        )
        assert ts.num_samples == 21
        assert ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["pango"] == {
            "A": 2,
            "B": 2,
        }
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == 4
        assert "SRR11597115" not in ts.metadata["sc2ts"]["samples_strain"]
        ts.tables.assert_equals(fx_ts_map["2020-02-02"].tables, ignore_provenance=True)

    @pytest.mark.parametrize(
        "include_samples", (["SRR11597115"], ["SRR11597115", "NOSUCHSTRAIN"])
    )
    def test_2020_02_02_include_samples(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
        include_samples,
    ):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
            include_samples=include_samples,
        )
        assert ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["pango"] == {
            "A": 2,
            "B": 2,
        }
        assert "SRR11597115" in ts.metadata["sc2ts"]["samples_strain"]
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == 5
        assert ts.num_samples == 22
        u = ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index("SRR11597115")]
        assert ts.nodes_flags[u] & tskit.NODE_IS_SAMPLE > 0
        assert ts.nodes_flags[u] & sc2ts.NODE_IS_UNCONDITIONALLY_INCLUDED > 0

    def test_2020_02_02_mutation_overlap(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
    ):
        base_ts = fx_ts_map["2020-02-01"]
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        assert ts.num_samples == 21
        node = ts.node(27)
        assert node.flags == sc2ts.NODE_IS_MUTATION_OVERLAP
        assert node.metadata == {
            "sc2ts": {
                "date_added": "2020-02-02",
                "mutations": ["C17373T"],
                "sibs": [11, 23],
            }
        }

    @pytest.mark.parametrize("max_samples", range(1, 6))
    def test_2020_02_02_max_samples(self, tmp_path, fx_ts_map, fx_dataset, max_samples):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            max_daily_samples=max_samples,
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        new_samples = min(4, max_samples)
        assert ts.num_samples == 17 + new_samples
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == new_samples

    def test_2020_02_02_max_missing_sites(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
    ):
        max_missing_sites = 123
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            max_missing_sites=max_missing_sites,
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        new_samples = 2
        assert ts.num_samples == 17 + new_samples

        assert np.sum(ts.nodes_time[ts.samples()] == 0) == new_samples
        for u in ts.samples()[-new_samples:]:
            assert (
                ts.node(u).metadata["sc2ts"]["num_missing_sites"] <= max_missing_sites
            )

    @pytest.mark.parametrize(
        ["strain", "start", "length"],
        [("SRR11597164", 1547, 1), ("SRR11597190", 3951, 3)],
    )
    @pytest.mark.parametrize("deletions_as_missing", [True, False])
    def test_2020_02_02_deletion_sample(
        self,
        tmp_path,
        fx_dataset,
        fx_ts_map,
        strain,
        start,
        length,
        deletions_as_missing,
    ):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
            deletions_as_missing=deletions_as_missing,
        )
        u = ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
        md = ts.node(u).metadata["sc2ts"]
        assert md["alignment_composition"]["-"] == length
        for j in range(length):
            site = ts.site(position=start + j)
            assert len(site.mutations) == 0 if deletions_as_missing else 1
            for mut in site.mutations:
                assert mut.node == u
                assert mut.derived_state == "-"
            # We pick the site up as somewhere with deletions regardless
            # of deletions_as_missing
            assert site.metadata["sc2ts"]["deletion_samples"] == 1

    @pytest.mark.parametrize("deletions_as_missing", [True, False])
    def test_2020_02_03_deletions_as_missing(
        self,
        tmp_path,
        fx_dataset,
        fx_ts_map,
        deletions_as_missing,
    ):
        base_ts = fx_ts_map["2020-02-02"]
        assert ord("-") in base_ts.tables.mutations.derived_state
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=base_ts,
            date="2020-02-03",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
            deletions_as_missing=deletions_as_missing,
        )
        ts.tables.assert_equals(fx_ts_map["2020-02-03"].tables, ignore_provenance=True)

    @pytest.mark.parametrize(
        ["strain", "num_missing"], [("SRR11597164", 122), ("SRR11597114", 402)]
    )
    def test_2020_02_02_missing_sample(
        self,
        fx_ts_map,
        fx_dataset,
        strain,
        num_missing,
    ):
        a = fx_dataset.haplotypes[strain]
        a = sc2ts.mask_ambiguous(a)

        missing_positions = np.where(a == -1)[0] + 1
        assert len(missing_positions) == num_missing
        ts_prev = fx_ts_map["2020-02-01"]
        ts = fx_ts_map["2020-02-02"]
        u = ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
        md = ts.node(u).metadata["sc2ts"]
        assert md["num_missing_sites"] == num_missing
        for pos in missing_positions:
            site = ts.site(position=pos)
            site_prev = ts_prev.site(position=pos)
            assert (
                site.metadata["sc2ts"]["missing_samples"]
                > site_prev.metadata["sc2ts"]["missing_samples"]
            )

    @pytest.mark.parametrize("deletions_as_missing", [True, False])
    def test_2020_02_02_deletions_as_missing(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
        deletions_as_missing,
    ):
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-01"],
            date="2020-02-02",
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
            deletions_as_missing=deletions_as_missing,
        )
        ti = sc2ts.TreeInfo(ts, show_progress=False)
        expected = 0 if deletions_as_missing else 4
        assert np.sum(ti.mutations_derived_state == "-") == expected

    def test_2020_02_08(self, tmp_path, fx_ts_map, fx_dataset):
        ts = run_extend(
            dataset=fx_dataset,
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
        assert scmd["hmm_match"]["mutations"] == [
            {"derived_state": "C", "inherited_state": "T", "site_position": 5025}
        ]
        # But no mutations above the node itself.
        assert np.sum(ts.mutations_node == node.id) == 0

        tree = ts.first()
        rp_node = ts.node(tree.parent(node.id))
        assert rp_node.flags == sc2ts.NODE_IS_REVERSION_PUSH
        assert rp_node.metadata["sc2ts"] == {
            "date_added": "2020-02-08",
            # The mutation we tried to revert is the inverse
            "mutations": ["C5025T"],
        }
        ts.tables.assert_equals(fx_ts_map["2020-02-08"].tables, ignore_provenance=True)

        sib_sample = ts.node(tree.siblings(node.id)[0])
        assert sib_sample.metadata["strain"] == "SRR11597168"

        assert np.sum(ts.mutations_node == sib_sample.id) == 1
        mutation = ts.mutation(np.where(ts.mutations_node == sib_sample.id)[0][0])
        assert mutation.derived_state == "T"
        assert mutation.parent == -1

    def test_2020_02_14_all_matches(self, tmp_path, fx_ts_map, fx_dataset, fx_match_db):
        date = "2020-02-14"
        assert len(list(fx_dataset.metadata.samples_for_date(date))) == 0
        ts = run_extend(
            dataset=fx_dataset,
            base_ts=fx_ts_map["2020-02-13"],
            date="2020-02-15",
            match_db=fx_match_db,
            # This should allow everything in
            min_root_mutations=0,
            min_group_size=1,
            min_different_dates=1,
        )
        retro_groups = ts.metadata["sc2ts"]["retro_groups"]
        assert len(retro_groups) == 6
        assert retro_groups[0] == {
            "dates": ["2020-01-29"],
            "depth": 1,
            "group_id": "92312b65f8ec1eaf12de8218db67e737",
            "num_mutations": 19,
            "num_nodes": 2,
            "num_recurrent_mutations": 0,
            "num_root_mutations": 0,
            "pango_lineages": ["A.5"],
            "strains": ["SRR15736313"],
            "date_added": "2020-02-15",
        }

    def test_2020_02_14_skip_recurrent(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
        fx_match_db,
        caplog,
    ):
        date = "2020-02-14"
        assert len(list(fx_dataset.metadata.samples_for_date(date))) == 0
        with caplog.at_level("DEBUG", logger="sc2ts.inference"):
            ts = run_extend(
                dataset=fx_dataset,
                base_ts=fx_ts_map["2020-02-13"],
                date="2020-02-15",
                match_db=fx_match_db,
                # This should allow everything in but exclude on max_recurrent
                min_root_mutations=0,
                min_group_size=1,
                min_different_dates=1,
                max_recurrent_mutations=-1,
            )
            retro_groups = ts.metadata["sc2ts"]["retro_groups"]
            assert len(retro_groups) == 0
            assert "Skipping num_recurrent_mutations=0 exceeds threshold" in caplog.text

    def test_2020_02_14_skip_max_mutations(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
        fx_match_db,
        caplog,
    ):
        date = "2020-02-14"
        assert len(list(fx_dataset.metadata.samples_for_date(date))) == 0
        with caplog.at_level("DEBUG", logger="sc2ts.inference"):
            ts = run_extend(
                dataset=fx_dataset,
                base_ts=fx_ts_map["2020-02-13"],
                date="2020-02-15",
                match_db=fx_match_db,
                min_root_mutations=0,
                min_group_size=1,
                min_different_dates=1,
                max_recurrent_mutations=100,
                # This should allow everything in but exclude on max_mtuations
                max_mutations_per_sample=-1,
            )
            retro_groups = ts.metadata["sc2ts"]["retro_groups"]
            assert len(retro_groups) == 0
            assert (
                "Skipping mean_mutations_per_sample=1.0 exceeds threshold"
                in caplog.text
            )

    def test_2020_02_14_skip_root_mutations(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
        fx_match_db,
        caplog,
    ):
        date = "2020-02-14"
        assert len(list(fx_dataset.metadata.samples_for_date(date))) == 0
        with caplog.at_level("DEBUG", logger="sc2ts.inference"):
            ts = run_extend(
                dataset=fx_dataset,
                base_ts=fx_ts_map["2020-02-13"],
                date="2020-02-15",
                match_db=fx_match_db,
                # This should allow everything in but exclude on min_root_mutations
                min_root_mutations=100,
                min_group_size=1,
                min_different_dates=1,
            )
            retro_groups = ts.metadata["sc2ts"]["retro_groups"]
            assert len(retro_groups) == 0
            assert "Skipping root_mutations=0 < threshold" in caplog.text

    def test_2020_02_14_skip_group_size(
        self,
        tmp_path,
        fx_ts_map,
        fx_dataset,
        fx_match_db,
        caplog,
    ):
        date = "2020-02-14"
        assert len(list(fx_dataset.metadata.samples_for_date(date))) == 0
        with caplog.at_level("DEBUG", logger="sc2ts.inference"):
            ts = run_extend(
                dataset=fx_dataset,
                base_ts=fx_ts_map["2020-02-13"],
                date="2020-02-15",
                match_db=fx_match_db,
                min_root_mutations=0,
                # This should allow everything in but exclude on group size
                min_group_size=100,
                min_different_dates=1,
            )
            retro_groups = ts.metadata["sc2ts"]["retro_groups"]
            assert len(retro_groups) == 0
            assert "Skipping size=" in caplog.text

    @pytest.mark.parametrize("date", dates)
    def test_date_metadata(self, fx_ts_map, date):
        ts = fx_ts_map[date]
        assert ts.metadata["sc2ts"]["date"] == date
        samples_strain = [ts.node(u).metadata["strain"] for u in ts.samples()]
        assert ts.metadata["sc2ts"]["samples_strain"] == samples_strain
        # print(ts.tables.mutations)
        # print(ts.draw_text())

    @pytest.mark.parametrize("date", dates)
    def test_date_validate(self, fx_ts_map, fx_dataset, date):
        ts = fx_ts_map[date]
        sc2ts.validate(ts, fx_dataset)

    def test_mutation_type_metadata(self, fx_ts_map):
        ts = fx_ts_map[self.dates[-1]]
        for mutation in ts.mutations():
            md = mutation.metadata["sc2ts"]
            assert md["type"] in ["parsimony", "overlap"]

    def test_node_type_metadata(self, fx_ts_map):
        ts = fx_ts_map[self.dates[-1]]
        for node in list(ts.nodes())[2:]:
            md = node.metadata["sc2ts"]
            if node.is_sample():
                assert "hmm_match" in md

    @pytest.mark.parametrize("date", dates)
    def test_exact_match_count(self, fx_ts_map, date):
        ts = fx_ts_map[date]
        md = ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]
        nodes_num_exact_matches = md["node"]
        by_pango = md["pango"]
        total = sum(nodes_num_exact_matches.values())
        assert total == sum(by_pango.values())
        by_date = 0
        for d in ts.metadata["sc2ts"]["daily_stats"].values():
            date_count = 0
            for record in d["samples_processed"]:
                date_count += record["exact_matches"]
            by_date += date_count
        assert total == by_date
        if date == self.dates[-1]:
            assert total == 8

    @pytest.mark.parametrize(
        ["strain", "num_deletions"],
        [
            ("SRR11597190", 3),
            ("SRR11597164", 1),
            ("SRR11597218", 3),
        ],
    )
    def test_deletion_samples(self, fx_ts_map, strain, num_deletions):
        ts = fx_ts_map[self.dates[-1]]
        u = ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
        md = ts.node(u).metadata["sc2ts"]
        assert md["alignment_composition"]["-"] == num_deletions

    @pytest.mark.parametrize("position", [1547, 3951, 3952, 3953, 29749, 29750, 29751])
    def test_deletion_tracking(self, fx_ts_map, position):
        ts = fx_ts_map[self.dates[-1]]
        site = ts.site(position=position)
        assert site.metadata["sc2ts"]["deletion_samples"] == 1

    @pytest.mark.parametrize(
        ["gid", "date", "internal", "strains"],
        [
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
                # assert node.flags & sc2ts.NODE_IN_SAMPLE_GROUP > 0
                if node.is_sample():
                    got_strains.append(md["strain"])
                    assert md["date"] == date
                else:
                    assert md["sc2ts"]["date_added"] == date
                    num_internal += 1
        assert num_internal == internal
        assert got_strains == strains

    @pytest.mark.parametrize(
        ["date", "strain"],
        [
            (
                "2020-01-19",
                "SRR11772659",
            ),
            (
                "2020-02-08",
                "SRR11597163",
            ),
        ],
    )
    def test_singleton_group(self, fx_ts_map, date, strain):
        ts = fx_ts_map[date]
        u = ts.samples()[ts.metadata["sc2ts"]["samples_strain"].index(strain)]
        node = ts.node(u)
        sample_hash = hashlib.md5(strain.encode()).hexdigest()
        assert node.metadata["sc2ts"]["group_id"] == sample_hash

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
            "2020-01-25": {"nodes": 8, "mutations": 6},
            "2020-01-28": {"nodes": 10, "mutations": 11},
            "2020-01-29": {"nodes": 12, "mutations": 15},
            "2020-01-30": {"nodes": 17, "mutations": 19},
            "2020-01-31": {"nodes": 18, "mutations": 21},
            "2020-02-01": {"nodes": 23, "mutations": 27},
            "2020-02-02": {"nodes": 28, "mutations": 39},
            "2020-02-03": {"nodes": 31, "mutations": 45},
            "2020-02-04": {"nodes": 35, "mutations": 50},
            "2020-02-05": {"nodes": 35, "mutations": 50},
            "2020-02-06": {"nodes": 40, "mutations": 54},
            "2020-02-07": {"nodes": 42, "mutations": 60},
            "2020-02-08": {"nodes": 47, "mutations": 61},
            "2020-02-09": {"nodes": 48, "mutations": 65},
            "2020-02-10": {"nodes": 49, "mutations": 69},
            "2020-02-11": {"nodes": 50, "mutations": 73},
            "2020-02-13": {"nodes": 53, "mutations": 76},
            "2020-02-15": {"nodes": 60, "mutations": 101},
        }
        assert ts.num_nodes == expected[date]["nodes"]
        assert ts.num_mutations == expected[date]["mutations"]

    @pytest.mark.parametrize(
        ["strain", "parent"],
        [
            ("SRR11397726", 5),
            ("SRR11397729", 5),
            ("SRR11597132", 7),
            ("SRR11597177", 7),
            ("SRR11597156", 7),
        ],
    )
    def test_exact_matches(self, fx_ts_map, strain, parent):
        ts = fx_ts_map[self.dates[-1]]
        md = ts.metadata["sc2ts"]
        assert strain not in md["samples_strain"]
        assert md["cumulative_stats"]["exact_matches"]["node"][str(parent)] >= 1


class TestSyntheticAlignments:

    def test_exact_match(self, tmp_path, fx_ts_map, fx_dataset):
        # Pick two unique strains and we should match exactly with them
        strains = ["SRR11597218", "ERR4204459"]
        fake_strains = ["fake" + s for s in strains]
        alignments = {
            name: fx_dataset.haplotypes[s] for name, s in zip(fake_strains, strains)
        }
        date = "2020-03-01"
        ds = sc2ts.tmp_dataset(tmp_path / "tmp.zarr", alignments, date=date)

        base_ts = fx_ts_map["2020-02-13"]
        ts = run_extend(
            dataset=ds,
            base_ts=base_ts,
            date=date,
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        assert ts.num_nodes == base_ts.num_nodes

        assert (
            sum(
                ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"][
                    "pango"
                ].values()
            )
            == sum(
                base_ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"][
                    "pango"
                ].values()
            )
            + 2
        )
        samples = ts.samples()
        samples_strain = ts.metadata["sc2ts"]["samples_strain"]
        node_count = ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["node"]
        for strain, fake_strain in zip(strains, fake_strains):
            node = samples[samples_strain.index(strain)]
            assert node_count[str(node)] == 1

    def test_recombinant_example_1(self, fx_ts_map, fx_recombinant_example_1):
        base_ts = fx_ts_map["2020-02-13"]
        date = "2020-02-15"
        ts = fx_recombinant_example_1

        assert ts.num_nodes == base_ts.num_nodes + 3
        assert ts.num_edges == base_ts.num_edges + 4
        assert ts.num_samples == base_ts.num_samples + 2
        assert ts.num_mutations == base_ts.num_mutations + 1
        assert ts.num_trees == 2
        samples_strain = ts.metadata["sc2ts"]["samples_strain"]
        assert samples_strain[-2:] == [
            "recombinant_example_1_0",
            "recombinant_example_1_1",
        ]

        group_id = "fc5a70591c67c3db84319c811fec2835"

        left_parent = 31
        right_parent = 46
        bp = 11083

        sample = ts.node(ts.samples()[-2])
        smd = sample.metadata["sc2ts"]
        assert smd["group_id"] == group_id
        assert smd["hmm_match"] == {
            "mutations": [],
            "path": [
                {"left": 0, "parent": left_parent, "right": bp},
                {"left": bp, "parent": right_parent, "right": 29904},
            ],
        }

        sample = ts.node(ts.samples()[-1])
        smd = sample.metadata["sc2ts"]
        assert smd["group_id"] == group_id
        assert smd["hmm_match"] == {
            "mutations": [
                {"derived_state": "C", "inherited_state": "A", "site_position": 9900}
            ],
            "path": [
                {"left": 0, "parent": left_parent, "right": bp},
                {"left": bp, "parent": right_parent, "right": 29904},
            ],
        }

        recomb_node = ts.node(ts.num_nodes - 1)
        assert recomb_node.flags == sc2ts.NODE_IS_RECOMBINANT
        smd = recomb_node.metadata["sc2ts"]
        assert smd["date_added"] == date
        assert smd["group_id"] == group_id

        edges = ts.tables.edges[ts.edges_child == recomb_node.id]
        assert len(edges) == 2
        assert edges[0].left == 0
        assert edges[0].right == bp
        assert edges[0].parent == left_parent
        assert edges[1].left == bp
        assert edges[1].right == 29904
        assert edges[1].parent == right_parent

        edges = ts.tables.edges[ts.edges_parent == recomb_node.id]
        assert len(edges) == 2
        assert edges[0].left == 0
        assert edges[0].right == 29904
        assert edges[0].child == ts.samples()[-2]
        assert edges[1].left == 0
        assert edges[1].right == 29904
        assert edges[1].child == ts.samples()[-1]

    def test_recombinant_example_2(self, fx_ts_map, fx_recombinant_example_2):
        base_ts = fx_ts_map["2020-02-13"]
        date = "2020-03-01"
        rts = fx_recombinant_example_2
        samples_strain = rts.metadata["sc2ts"]["samples_strain"]
        assert samples_strain[-3:] == ["left", "right", "recombinant_114:29825"]

        sample = rts.node(rts.samples()[-1])
        smd = sample.metadata["sc2ts"]
        assert smd["hmm_match"] == {
            "mutations": [],
            "path": [
                {"left": 0, "parent": 53, "right": 29825},
                {"left": 29825, "parent": 54, "right": 29904},
            ],
        }

    def test_all_As(self, tmp_path, fx_ts_map, fx_dataset):
        # Same as the recombinant_example_1() function above
        # Just to get something that looks like an alignment easily
        a = fx_dataset.haplotypes["SRR11597188"]
        a[1:] = 0
        alignments = {"crazytype": a}
        date = "2020-03-01"
        base_ts = fx_ts_map["2020-02-13"]
        ts = run_extend(
            dataset=sc2ts.tmp_dataset(tmp_path / "tmp.zarr", alignments, date=date),
            base_ts=base_ts,
            date=date,
            match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
        )
        # Super high HMM cost means we don't add it in.
        assert ts.num_nodes == base_ts.num_nodes


class TestMatchingDetails:
    @pytest.mark.parametrize(
        ("strain", "parent"), [("SRR11597207", 34), ("ERR4205570", 47)]
    )
    @pytest.mark.parametrize("num_mismatches", [2, 3, 4])
    @pytest.mark.parametrize("direction", ["forward", "reverse"])
    def test_exact_matches(
        self,
        fx_ts_map,
        fx_dataset,
        strain,
        parent,
        num_mismatches,
        direction,
    ):
        ts = fx_ts_map["2020-02-10"]

        runs = sc2ts.run_hmm(
            fx_dataset.path,
            ts.path,
            [strain],
            num_mismatches=num_mismatches,
            direction=direction,
        )
        assert len(runs) == 1
        assert runs[0].num_mismatches == num_mismatches
        assert runs[0].direction == direction
        s = runs[0].match
        assert len(s.mutations) == 0
        assert len(s.path) == 1
        assert s.path[0].parent == parent

    @pytest.mark.parametrize(
        ("strain", "parent", "position", "derived_state"),
        [
            ("ERR4206593", 47, 26994, "T"),
        ],
    )
    @pytest.mark.parametrize("num_mismatches", [2, 3, 4])
    def test_one_mismatch(
        self,
        fx_ts_map,
        fx_dataset,
        strain,
        parent,
        position,
        derived_state,
        num_mismatches,
    ):
        ts = fx_ts_map["2020-02-10"]
        runs = sc2ts.run_hmm(
            fx_dataset.path,
            ts.path,
            [strain],
            num_mismatches=num_mismatches,
        )
        assert len(runs) == 1
        assert runs[0].num_mismatches == num_mismatches
        assert runs[0].direction == "forward"
        s = runs[0].match
        assert len(s.mutations) == 1
        assert s.mutations[0].site_position == position
        assert s.mutations[0].derived_state == derived_state
        assert len(s.path) == 1
        assert s.path[0].parent == parent

    @pytest.mark.parametrize("num_mismatches", [2, 3, 4])
    def test_two_mismatches(
        self,
        fx_ts_map,
        fx_dataset,
        num_mismatches,
    ):
        strain = "SRR11597164"
        ts = fx_ts_map["2020-02-01"]
        runs = sc2ts.run_hmm(
            fx_dataset.path,
            ts.path,
            [strain],
            num_mismatches=num_mismatches,
        )
        assert len(runs) == 1
        assert runs[0].num_mismatches == num_mismatches
        assert runs[0].direction == "forward"
        s = runs[0].match
        assert len(s.path) == 1
        assert s.path[0].parent == 1
        assert len(s.mutations) == 2
        for mut in s.mutations:
            assert mut.is_reversion is False
            assert mut.is_immediate_reversion is False
        asjson = runs[0].asjson()
        assert json.loads(asjson) == runs[0].asdict()

    def test_match_recombinant(self, fx_ts_map):
        ts, s = recombinant_example_1(fx_ts_map)
        sc2ts.match_tsinfer(
            samples=[s],
            ts=ts,
            num_mismatches=2,
            mismatch_threshold=10,
        )
        interval_right = 11083
        left_parent = 31
        right_parent = 46

        m = s.hmm_match
        assert len(m.mutations) == 0
        assert len(m.path) == 2
        assert m.path[0].parent == left_parent
        assert m.path[0].left == 0
        assert m.path[0].right == interval_right
        assert m.path[1].parent == right_parent
        assert m.path[1].left == interval_right
        assert m.path[1].right == ts.sequence_length


class TestRunHmm:

    @pytest.mark.parametrize("direction", ["F", "R", "forwards", "backwards", "", None])
    def test_bad_direction(self, fx_dataset, fx_ts_map, direction):
        strain = "SRR11597164"
        ts = fx_ts_map["2020-02-01"]
        with pytest.raises(ValueError, match="Direction must be one of"):
            sc2ts.run_hmm(
                fx_dataset.path,
                ts.path,
                [strain],
                direction=direction,
                num_mismatches=3,
            )

    def test_no_strains(self, fx_dataset, fx_ts_map):
        ts = fx_ts_map["2020-02-01"]
        assert len(sc2ts.run_hmm(fx_dataset.path, ts.path, [], num_mismatches=3)) == 0


class TestCharacteriseRecombinants:

    def test_example_1(self, fx_ts_map):
        ts, s = recombinant_example_1(fx_ts_map)

        interval_left = 3788
        interval_right = 11083
        left_parent = 31
        right_parent = 46

        sc2ts.match_tsinfer(
            samples=[s],
            ts=ts,
            num_mismatches=2,
            mismatch_threshold=10,
        )
        m = s.hmm_match
        assert len(m.mutations) == 0
        assert len(m.path) == 2
        assert m.path[0].parent == left_parent
        assert m.path[0].left == 0
        assert m.path[0].right == interval_right
        assert m.path[1].parent == right_parent
        assert m.path[1].left == interval_right
        assert m.path[1].right == ts.sequence_length

        sc2ts.characterise_recombinants(ts, [s])
        assert s.breakpoint_intervals == [(interval_left, interval_right)]

        sc2ts.match_tsinfer(
            samples=[s],
            ts=ts,
            num_mismatches=2,
            mismatch_threshold=10,
            mirror_coordinates=True,
        )
        m = s.hmm_match
        assert len(m.mutations) == 0
        assert len(m.path) == 2
        assert m.path[0].parent == left_parent
        assert m.path[0].left == 0
        assert m.path[0].right == interval_left
        assert m.path[1].parent == right_parent
        assert m.path[1].left == interval_left
        assert m.path[1].right == ts.sequence_length

    def test_example_3(self, fx_recombinant_example_3):
        ts = fx_recombinant_example_3
        strains = ts.metadata["sc2ts"]["samples_strain"]
        assert strains[-1].startswith("recomb")
        u = ts.samples()[-1]
        h = ts.genotype_matrix(samples=[u], alleles=tuple(sc2ts.IUPAC_ALLELES)).T[0]
        tables = ts.dump_tables()
        keep_edges = ts.edges_child < u
        tables.edges.keep_rows(keep_edges)
        keep_nodes = np.ones(ts.num_nodes, dtype=bool)
        tables.nodes[u] = tables.nodes[u].replace(flags=0)
        tables.sort()
        base_ts = tables.tree_sequence()

        s = sc2ts.Sample("3way", "2020-02-14", haplotype=h.astype(np.int8))
        sc2ts.match_tsinfer(
            samples=[s],
            ts=base_ts,
            num_mismatches=2,
            mismatch_threshold=10,
            mirror_coordinates=False,
        )
        sc2ts.characterise_recombinants(ts, [s])
        m = s.hmm_match
        assert m.parents == [53, 54, 55]
        assert m.breakpoints == [0, 15001, 29825, 29904]
        assert s.breakpoint_intervals == [(114, 15001), (15010, 29825)]
        # Verify that these breakpoints correspond to the reverse-direction HMM
        sc2ts.match_tsinfer(
            samples=[s],
            ts=base_ts,
            num_mismatches=2,
            mismatch_threshold=10,
            mirror_coordinates=True,
        )
        m = s.hmm_match
        assert m.parents == [53, 54, 55]
        assert m.breakpoints == [0, 114, 15010, 29904]

    def test_example_3_way_same_parent(self, fx_recombinant_example_3):
        ts = fx_recombinant_example_3
        strains = ts.metadata["sc2ts"]["samples_strain"]
        assert strains[-1].startswith("recomb")
        u = ts.samples()[-1]
        h = ts.genotype_matrix(samples=[u], alleles=tuple(sc2ts.IUPAC_ALLELES)).T[0]
        tables = ts.dump_tables()
        keep_edges = ts.edges_child < u
        tables.edges.keep_rows(keep_edges)
        keep_nodes = np.ones(ts.num_nodes, dtype=bool)
        tables.nodes[u] = tables.nodes[u].replace(flags=0)
        tables.sort()
        base_ts = tables.tree_sequence()

        s = sc2ts.Sample("3way", "2020-02-14", haplotype=h.astype(np.int8))
        sc2ts.match_tsinfer(
            samples=[s],
            ts=base_ts,
            num_mismatches=2,
            mismatch_threshold=10,
            mirror_coordinates=False,
        )
        # Force back to the same parent so we can check that we're robust to
        # same parent
        s.hmm_match.path[0] = dataclasses.replace(s.hmm_match.path[0], parent=55)
        sc2ts.characterise_recombinants(ts, [s])

        m = s.hmm_match
        assert m.parents == [55, 54, 55]
        assert m.breakpoints == [0, 15001, 29825, 29904]


class TestExtractHaplotypes:

    @pytest.mark.parametrize(
        ["samples", "result"],
        [
            ([0], [[0]]),
            ([0, 1], [[0], [0]]),
            ([0, 3], [[0], [1]]),
            ([3, 0], [[1], [0]]),
            ([0, 1, 2, 3], [[0], [0], [0], [1]]),
            ([3, 1, 2, 3], [[1], [0], [0], [1]]),
            ([3, 3, 3, 3], [[1], [1], [1], [1]]),
        ],
    )
    def test_one_leaf_mutation(self, samples, result):
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
        ts = tables.tree_sequence()
        nt.assert_array_equal(sc2ts.extract_haplotypes(ts, samples), result)


@pytest.fixture
def fx_ts_exact_matches(fx_ts_map, fx_match_db):
    ts = fx_ts_map["2020-02-13"]
    tsp = sc2ts.append_exact_matches(ts, fx_match_db)
    return tsp


class TestMapDeletions:
    def test_example(self, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-02-13"]
        new_ts = sc2ts.map_deletions(ts, fx_dataset, frequency_threshold=0.001)
        remapped_sites = [
            j
            for j in range(ts.num_sites)
            if "original_mutations" in new_ts.site(j).metadata["sc2ts"]
        ]
        assert remapped_sites == [1541, 3945, 3946, 3947]

        for site_id in remapped_sites:
            site = new_ts.site(site_id)
            old_site = ts.site(site_id)
            original_mutations = site.metadata["sc2ts"]["original_mutations"]
            assert original_mutations == [
                {
                    "node": mutation.node,
                    "derived_state": mutation.derived_state,
                    "metadata": mutation.metadata,
                }
                for mutation in old_site.mutations
            ]
            d = site.metadata["sc2ts"]
            del d["original_mutations"]
            assert old_site.metadata["sc2ts"] == d

            for mut in site.mutations:
                assert mut.metadata["sc2ts"]["type"] == "post_parsimony"

    def test_filter_all(self, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-02-13"]
        new_ts = sc2ts.map_deletions(
            ts, fx_dataset, frequency_threshold=0.001, mutations_threshold=0
        )
        remapped_sites = [
            j
            for j in range(ts.num_sites)
            if "original_mutations" in new_ts.site(j).metadata["sc2ts"]
        ]
        assert remapped_sites == []

    def test_example_exact_matches(self, fx_ts_exact_matches, fx_dataset):
        ts = fx_ts_exact_matches
        new_ts = sc2ts.map_deletions(ts, fx_dataset, frequency_threshold=0.001)
        remapped_sites = [
            j
            for j in range(ts.num_sites)
            if "original_mutations" in new_ts.site(j).metadata["sc2ts"]
        ]
        assert remapped_sites == [1541, 3945, 3946, 3947]

    def test_validate(self, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-02-13"]
        new_ts = sc2ts.map_deletions(ts, fx_dataset, frequency_threshold=0.001)
        sc2ts.validate(new_ts, fx_dataset, deletions_as_missing=False)

    def test_provenance(self, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.map_deletions(ts, fx_dataset, frequency_threshold=0.125)
        assert tsp.num_provenances == ts.num_provenances + 1
        prov = tsp.provenance(-1)
        assert json.loads(prov.record)["parameters"] == {
            "command": "map_deletions",
            "dataset": str(fx_dataset.path),
            "frequency_threshold": 0.125,
            "mutations_threshold": 2**64,
        }


class TestAppendExactMatches:
    def test_validate(self, fx_ts_exact_matches, fx_dataset):
        sc2ts.validate(fx_ts_exact_matches, fx_dataset)

    def test_example_properties(self, fx_ts_exact_matches):
        ts = fx_ts_exact_matches
        samples_strain = ts.metadata["sc2ts"]["samples_strain"]
        assert [ts.node(u).metadata["strain"] for u in ts.samples()] == samples_strain
        assert ts.num_nodes == 61
        tree = ts.first()
        assert tree.num_roots == 1

    def test_metadata_flags(self, fx_ts_exact_matches):
        md = fx_ts_exact_matches.metadata["sc2ts"]
        assert md["includes_exact_matches"]

    def test_times_agree(self, fx_ts_exact_matches):
        ts = fx_ts_exact_matches
        date_to_time = {}
        time_to_date = {}
        for u in ts.samples():
            node = ts.node(u)
            time = node.time
            date = node.metadata["date"]
            if date not in date_to_time:
                date_to_time[date] = time
            assert date_to_time[date] == time
            if time not in time_to_date:
                time_to_date[time] = date
            assert time_to_date[time] == date

    def test_flags(self, fx_ts_exact_matches):
        ts = fx_ts_exact_matches
        assert np.all(
            ts.nodes_flags[-8:] == sc2ts.NODE_IS_EXACT_MATCH | tskit.NODE_IS_SAMPLE
        )

    def test_exact_match_counts(self, fx_ts_exact_matches):
        ts = fx_ts_exact_matches
        tree = ts.first()
        node_count = ts.metadata["sc2ts"]["cumulative_stats"]["exact_matches"]["node"]
        for u in tree.nodes():
            num_exact_matches = 0
            for v in tree.children(u):
                if (ts.nodes_flags[v] & sc2ts.NODE_IS_EXACT_MATCH) > 0:
                    num_exact_matches += 1
            assert node_count.get(str(u), 0) == num_exact_matches

    def test_provenance(self, fx_ts_map, fx_match_db):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.append_exact_matches(ts, fx_match_db)
        assert tsp.num_provenances == ts.num_provenances + 1
        prov = tsp.provenance(-1)
        assert json.loads(prov.record)["parameters"] == {
            "command": "append_exact_matches",
            "match_db": str(fx_match_db.path),
        }


class TestMinimiseMetadata:

    def test_equivalent(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.minimise_metadata(ts)
        ts.tables.assert_equals(
            tsp.tables, ignore_metadata=True, ignore_provenance=True
        )

    def test_properties(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tables = sc2ts.minimise_metadata(ts).dump_tables()
        assert tables.metadata == {}
        assert len(tables.sites.metadata) == 0
        assert len(tables.mutations.metadata) == 0

    def test_fields(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.minimise_metadata(ts)
        for u in tsp.samples():
            md_old = ts.node(u).metadata
            md_new = tsp.node(u).metadata
            if "strain" in md_old:
                assert md_old["strain"] == md_new["sample_id"]
                assert md_old["Viridian_pangolin"] == md_new["pango"]
            else:
                assert md_new == {"sample_id": "", pango: ""}

    def test_dataframe_access(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.minimise_metadata(ts)
        data = tsp.nodes_metadata
        cols = {k: data[k].astype(str) for k in data.dtype.names}
        df = pd.DataFrame(cols)
        for u, row in df.iterrows():
            node = ts.node(u)
            md = node.metadata
            if node.is_sample():
                assert row.sample_id == md["strain"]
                assert row.pango == md["Viridian_pangolin"]
            else:
                assert row.sample_id == md.get("strain", "")
                assert row.pango == ""

    def test_provenance(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.minimise_metadata(ts)
        assert tsp.num_provenances == ts.num_provenances + 1
        prov = tsp.provenance(-1)
        assert json.loads(prov.record)["parameters"] == {"command": "minimise_metadata"}


class TestPushUpRecombinantMutations:

    def test_no_recombinants(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.push_up_unary_recombinant_mutations(ts)
        ts.tables.assert_equals(tsp.tables, ignore_provenance=True)

    def test_recombinant_example_1(self, fx_recombinant_example_1):
        ts = fx_recombinant_example_1
        tsp = sc2ts.push_up_unary_recombinant_mutations(ts)
        ts.tables.assert_equals(tsp.tables, ignore_provenance=True)

    def test_recombinant_example_2(self, fx_recombinant_example_2):
        ts = fx_recombinant_example_2
        tsp = sc2ts.push_up_unary_recombinant_mutations(ts)
        ts.tables.assert_equals(tsp.tables, ignore_provenance=True)

    def test_recombinant_example_3(self, fx_recombinant_example_3):
        ts = fx_recombinant_example_3
        tsp = sc2ts.push_up_unary_recombinant_mutations(ts)
        ts.tables.assert_equals(tsp.tables, ignore_provenance=True)

    def test_recombinant_example_4(self, fx_recombinant_example_4):
        ts = fx_recombinant_example_4
        site = 2500
        mut = ts.site(site).mutations[0]
        assert mut.node == 55
        tsp = sc2ts.push_up_unary_recombinant_mutations(ts)
        mut = tsp.site(site).mutations[0]
        assert mut.node == 56

    def test_provenance(self, fx_ts_map):
        ts = fx_ts_map["2020-02-13"]
        tsp = sc2ts.push_up_unary_recombinant_mutations(ts)
        assert tsp.num_provenances == ts.num_provenances + 1
        prov = tsp.provenance(-1)
        assert json.loads(prov.record)["parameters"] == {
            "command": "push_up_unary_recombinant_mutations"
        }
