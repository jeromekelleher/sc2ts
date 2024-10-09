import json

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

    @pytest.mark.parametrize("additional", [[100], [100, 200]])
    def test_additional_problematic_sites(self, tmp_path, additional):
        ts_path = tmp_path / "trees.ts"
        match_db_path = tmp_path / "match.db"
        problematic_path = tmp_path / "additional_problematic.txt"
        np.savetxt(problematic_path, additional)
        runner = ct.CliRunner(mix_stderr=False)
        result = runner.invoke(
            cli.cli,
            f"initialise {ts_path} {match_db_path} "
            f"--additional-problematic-sites {problematic_path}",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        ts = tskit.load(ts_path)
        other_ts = sc2ts.initial_ts(additional_problematic_sites=additional)
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
        sites = ts.metadata["sc2ts"]["additional_problematic_sites"]
        # < 266 (leftmost coordinate of ORF1a)
        # > 29674 (rightmost coordinate of ORF10)
        assert sites == list(range(1, 266)) + list(range(29675, 29904))


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
