import pytest
import pandas as pd


class TestMetadataDb:
    def test_known(self, metadata_db):
        record = metadata_db["SRR11772659"]
        assert record["strain"] == "SRR11772659"
        assert record["date"] == "2020-01-19"
        assert record["Viridian_pangolin"] == "A"

    def test_missing_sequence(self, metadata_db):
        # We include sequence that's not in the alignments DB
        assert "ERR_MISSING" in metadata_db

    def test_keys(self, metadata_db):
        keys = list(metadata_db.keys())
        assert "SRR11772659" in keys
        assert len(set(keys)) == len(keys)
        df = pd.read_csv("tests/data/metadata.tsv", sep="\t")
        assert set(keys) == set(df["strain"])

    def test_in(self, metadata_db):
        assert "SRR11772659" in metadata_db
        assert "DEFO_NOT_IN_DB" not in metadata_db

    def test_get_all_days(self, metadata_db):
        results = metadata_db.get_days()
        assert results == [
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

    def test_get_days_greater(self, metadata_db):
        results = metadata_db.get_days("2020-02-06")
        assert results == [
            "2020-02-07",
            "2020-02-08",
            "2020-02-09",
            "2020-02-10",
            "2020-02-11",
            "2020-02-13",
        ]

    def test_get_days_none(self, metadata_db):
        assert metadata_db.get_days("2022-02-06") == []

    def test_get_first(self, metadata_db):
        results = list(metadata_db.get("2020-01-01"))
        assert len(results) == 1
        assert results[0] == metadata_db["SRR14631544"]

    def test_get_multi(self, metadata_db):
        results = list(metadata_db.get("2020-02-11"))
        assert len(results) == 2
        for result in results:
            assert result["date"] == "2020-02-11"
