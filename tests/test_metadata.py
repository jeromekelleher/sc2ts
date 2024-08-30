import pytest
import pandas as pd


class TestMetadataDb:
    def test_known(self, fx_metadata_db):
        record = fx_metadata_db["SRR11772659"]
        assert record["strain"] == "SRR11772659"
        assert record["date"] == "2020-01-19"
        assert record["Viridian_pangolin"] == "A"

    def test_missing_sequence(self, fx_metadata_db):
        # We include sequence that's not in the alignments DB
        assert "ERR_MISSING" in fx_metadata_db

    def test_keys(self, fx_metadata_db):
        keys = list(fx_metadata_db.keys())
        assert "SRR11772659" in keys
        assert len(set(keys)) == len(keys)
        df = pd.read_csv("tests/data/metadata.tsv", sep="\t")
        assert set(keys) == set(df["strain"])

    def test_in(self, fx_metadata_db):
        assert "SRR11772659" in fx_metadata_db
        assert "DEFO_NOT_IN_DB" not in fx_metadata_db

    # TODO test count_days. See test_cli.

    def test_get_days_none(self, fx_metadata_db):
        assert fx_metadata_db.get_days("2022-02-06") == []

    def test_get_first(self, fx_metadata_db):
        results = list(fx_metadata_db.get("2020-01-01"))
        assert len(results) == 1
        assert results[0] == fx_metadata_db["SRR14631544"]

    def test_get_multi(self, fx_metadata_db):
        results = list(fx_metadata_db.get("2020-02-11"))
        assert len(results) == 2
        for result in results:
            assert result["date"] == "2020-02-11"
