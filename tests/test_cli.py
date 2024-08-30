
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
        other_ts.tables.assert_equals(ts.tables)
        match_db = sc2ts.MatchDb(match_db_path)
        assert len(match_db) == 0
