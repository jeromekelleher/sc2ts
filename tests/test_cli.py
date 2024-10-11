import json
import collections

import numpy as np
import click.testing as ct
import pytest
import tskit

import sc2ts
from sc2ts import __main__ as main
from sc2ts import cli


class TestInitialise:
    def test_defaults(self, tmp_path):
        ts_path = tmp_path / "trees.ts"
        match_db_path = tmp_path / "match.db"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"initialise {ts_path} {match_db_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        other_ts = sc2ts.initial_ts()
        other_ts.tables.assert_equals(ts.tables, ignore_provenance=True)
        match_db = sc2ts.MatchDb(match_db_path)
        assert len(match_db) == 0

    @pytest.mark.parametrize("problematic", [[100], [100, 200]])
    def test_problematic_sites(self, tmp_path, problematic):
        ts_path = tmp_path / "trees.ts"
        match_db_path = tmp_path / "match.db"
        problematic_path = tmp_path / "problematic.txt"
        np.savetxt(problematic_path, problematic)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"initialise {ts_path} {match_db_path} "
            f"--problematic-sites {problematic_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        other_ts = sc2ts.initial_ts(problematic_sites=problematic)
        other_ts.tables.assert_equals(ts.tables, ignore_provenance=True)
        match_db = sc2ts.MatchDb(match_db_path)
        assert len(match_db) == 0

    def test_mask_flanks(self, tmp_path):
        ts_path = tmp_path / "trees.ts"
        match_db_path = tmp_path / "match.db"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"initialise {ts_path} {match_db_path} --mask-flanks",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        sites = sc2ts.get_masked_sites(ts)
        # < 266 (leftmost coordinate of ORF1a)
        # > 29674 (rightmost coordinate of ORF10)
        assert list(sites) == list(range(1, 266)) + list(range(29675, 29904))

    def test_mask_problematic_regions(self, tmp_path):
        ts_path = tmp_path / "trees.ts"
        match_db_path = tmp_path / "match.db"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"initialise {ts_path} {match_db_path} --mask-problematic-regions",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        sites = sc2ts.get_masked_sites(ts)
        # NTD: [21602-22472)
        # ORF8: [27894, 28260)
        assert list(sites) == list(range(21602, 22472)) + list(range(27894, 28260))

    def test_provenance(self, tmp_path):
        ts_path = tmp_path / "trees.ts"
        match_db_path = tmp_path / "match.db"
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"initialise {ts_path} {match_db_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        assert ts.num_provenances == 1
        prov = ts.provenance(0)
        record = json.loads(prov.record)
        assert "software" in record
        assert "parameters" in record
        assert "environment" in record
        assert "resources" in record
        resources = record["resources"]
        assert "elapsed_time" in resources
        assert "user_time" in resources
        assert "sys_time" in resources
        assert "max_memory" in resources


class TestRunMatch:

    def test_single_defaults(self, tmp_path, fx_ts_map, fx_alignment_store):
        strain = "ERR4206593"
        ts = fx_ts_map["2020-02-04"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"run-match {fx_alignment_store.path} {ts_path} {strain}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["strain"] == strain
        assert d["num_mismatches"] == 3
        assert d["direction"] == "forward"
        assert len(d["match"]["path"]) == 1
        assert len(d["match"]["mutations"]) == 5

    def test_multi_defaults(self, tmp_path, fx_ts_map, fx_alignment_store):
        copies = 10
        strains = ["ERR4206593"] * 10
        ts = fx_ts_map["2020-02-13"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"run-match {fx_alignment_store.path} {ts_path} " + " ".join(strains),
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == copies
        d = json.loads(lines[0])
        assert d["strain"] == strains[0]
        assert d["num_mismatches"] == 3
        assert d["direction"] == "forward"
        assert len(d["match"]["path"]) == 1
        assert len(d["match"]["mutations"]) == 0
        for line in lines[1:]:
            d2 = json.loads(line)
            assert d == d2

    def test_single_options(self, tmp_path, fx_ts_map, fx_alignment_store):
        strain = "ERR4206593"
        ts = fx_ts_map["2020-02-04"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"run-match {fx_alignment_store.path} {ts_path} {strain}"
            " --direction=reverse --num-mismatches=5 --num-threads=4",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["strain"] == strain
        assert d["num_mismatches"] == 5
        assert d["direction"] == "reverse"
        assert len(d["match"]["path"]) == 1
        assert len(d["match"]["mutations"]) == 5


class TestRunRematchRecombinants:

    @pytest.mark.parametrize("num_threads", [0, 1, 2])
    def test_defaults(
        self, tmp_path, fx_recombinant_example_1, fx_data_cache, num_threads
    ):
        ts_path = fx_data_cache / "recombinant_ex1.ts"
        as_path = fx_data_cache / "recombinant_ex1_alignments.db"
        pattern = str(fx_data_cache) + "/{}.ts"
        runner = ct.CliRunner(mix_stderr=False)
        cmd = (
            f"run-rematch-recombinants {as_path} {ts_path} {pattern} "
            f"--num-threads={num_threads}"
        )
        result = runner.invoke(
            cli.cli,
            cmd,
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == 4
        results = collections.defaultdict(list)
        for line in lines:
            d = json.loads(line)
            results[d["strain"]].append(result)

        assert len(results) == 2
        assert set(results.keys()) == {
            "recombinant_example_1_0",
            "recombinant_example_1_1",
        }

        assert len(results["recombinant_example_1_0"]) == 2
        assert len(results["recombinant_example_1_1"]) == 2

    def test_multiple_mismatch_values(
        self, tmp_path, fx_recombinant_example_1, fx_data_cache
    ):
        ts_path = fx_data_cache / "recombinant_ex1.ts"
        as_path = fx_data_cache / "recombinant_ex1_alignments.db"
        pattern = str(fx_data_cache) + "/{}.ts"
        runner = ct.CliRunner(mix_stderr=False)
        cmd = (
            f"run-rematch-recombinants {as_path} {ts_path} {pattern} "
            f"-k 3 --num-mismatches 1000"
        )
        result = runner.invoke(
            cli.cli,
            cmd,
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert len(lines) == 8
        results = collections.defaultdict(list)
        for line in lines:
            d = json.loads(line)
            if d["num_mismatches"] == 3:
                assert len(d["match"]["path"]) == 2
            else:
                assert len(d["match"]["path"]) == 1
            results[d["strain"]].append(result)

        assert len(results) == 2
        assert set(results.keys()) == {
            "recombinant_example_1_0",
            "recombinant_example_1_1",
        }

        assert len(results["recombinant_example_1_0"]) == 4
        assert len(results["recombinant_example_1_1"]) == 4


class TestListDates:
    def test_defaults(self, fx_metadata_db):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"list-dates {fx_metadata_db.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
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

    def test_counts(self, fx_metadata_db):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"list-dates {fx_metadata_db.path} --counts",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert result.stdout.splitlines() == [
            "2020-01-01\t1",
            "2020-01-19\t1",
            "2020-01-24\t2",
            "2020-01-25\t3",
            "2020-01-28\t2",
            "2020-01-29\t4",
            "2020-01-30\t5",
            "2020-01-31\t1",
            "2020-02-01\t5",
            "2020-02-02\t5",
            "2020-02-03\t2",
            "2020-02-04\t5",
            "2020-02-05\t1",
            "2020-02-06\t3",
            "2020-02-07\t2",
            "2020-02-08\t4",
            "2020-02-09\t2",
            "2020-02-10\t2",
            "2020-02-11\t2",
            "2020-02-13\t4",
        ]
