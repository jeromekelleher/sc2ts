import io
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


class TestMatch:

    def test_single_defaults(self, tmp_path, fx_ts_map, fx_dataset):
        strain = "ERR4206593"
        ts = fx_ts_map["2020-02-04"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"match {fx_dataset.path} {ts_path} {strain}",
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

    def test_multi_defaults(self, tmp_path, fx_ts_map, fx_dataset):
        copies = 10
        strains = ["ERR4206593"] * 10
        ts = fx_ts_map["2020-02-13"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"match {fx_dataset.path} {ts_path} " + " ".join(strains),
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

    def test_single_options(self, tmp_path, fx_ts_map, fx_dataset):
        strain = "ERR4206593"
        ts = fx_ts_map["2020-02-04"]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"match {fx_dataset.path} {ts_path} {strain}"
            " --direction=reverse --num-mismatches=5 --num-threads=4",
            " --no-deletions-as-missing",
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


class TestExtend:

    def test_first_day(self, tmp_path, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-01-01"]
        ts_path = tmp_path / "ts.ts"
        output_ts_path = tmp_path / "out.ts"
        ts.dump(ts_path)
        match_db = sc2ts.MatchDb.initialise(tmp_path / "match.db")
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"extend {ts_path} 2020-01-19 {fx_dataset.path} "
            f"{match_db.path} {output_ts_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        out_ts = tskit.load(output_ts_path)
        out_ts.tables.assert_equals(
            fx_ts_map["2020-01-19"].tables, ignore_provenance=True
        )

    def test_include_samples(self, tmp_path, fx_ts_map, fx_dataset):
        ts = fx_ts_map["2020-02-01"]
        ts_path = tmp_path / "ts.ts"
        output_ts_path = tmp_path / "out.ts"
        ts.dump(ts_path)
        include_samples_path = tmp_path / "include_samples.txt"
        with open(include_samples_path, "w") as f:
            print("SRR11597115 This is a test strain", file=f)
            print("ABCD this is a strain that doesn't exist", file=f)
        match_db = sc2ts.MatchDb.initialise(tmp_path / "match.db")
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"extend {ts_path} 2020-02-02 {fx_dataset.path} "
            f"{match_db.path} {output_ts_path} "
            f"--include-samples={include_samples_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(output_ts_path)
        assert "SRR11597115" in ts.metadata["sc2ts"]["samples_strain"]
        assert np.sum(ts.nodes_time[ts.samples()] == 0) == 5
        assert ts.num_samples == 23


@pytest.mark.skip("Broken by dataset")
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
            f"rematch-recombinants {as_path} {ts_path} {pattern} "
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


class TestValidate:

    @pytest.mark.parametrize("date", ["2020-01-01", "2020-02-11"])
    def test_date(self, tmp_path, fx_ts_map, fx_dataset, date):
        ts = fx_ts_map[date]
        ts_path = tmp_path / "ts.ts"
        ts.dump(ts_path)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"validate {ts_path} {fx_dataset.path} ",
            catch_exceptions=False,
        )
        assert result.exit_code == 0


class TestInfoMatches:
    def test_defaults(self, fx_match_db):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"info-matches {fx_match_db.path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0


class TestListDates:
    def test_defaults(self, fx_dataset):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"list-dates {fx_dataset.path}",
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

    @pytest.mark.skip("Final date off by one after dataset")
    def test_counts(self, fx_dataset):
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"list-dates {fx_dataset.path} --counts",
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


class TestParseIncludeSamples:
    @pytest.mark.parametrize(
        ["text", "parsed"],
        [
            ("ABCD\n1234\n56", ["ABCD", "1234", "56"]),
            ("   ABCD\n\t1234\n 56", ["ABCD", "1234", "56"]),
            ("ABCD the rest is a comment", ["ABCD"]),
            ("", []),
        ],
    )
    def test_examples(self, text, parsed):
        result = cli.parse_include_samples(io.StringIO(text))
        assert result == parsed
