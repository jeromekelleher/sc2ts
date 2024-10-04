import pytest
import pandas as pd

from sc2ts import info



class TestTallyLineages:

    def test_last_date(self, fx_ts_map, fx_metadata_db):
        date = "2020-02-13"
        df = info.tally_lineages(fx_ts_map[date], fx_metadata_db)
        assert list(df["pango"]) == [
            "B",
            "A",
            "B.1",
            "B.40",
            "B.33",
            "B.4",
            "A.5",
            "B.1.177",
            "B.1.36.29",
        ]
        assert list(df["db_count"]) == [26, 15, 4, 4, 1, 3, 1, 1, 1]
        assert list(df["arg_count"]) == [24, 15, 4, 3, 1, 1, 0, 0, 0]
