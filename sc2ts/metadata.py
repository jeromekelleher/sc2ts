import collections
import logging
import sqlite3
import pathlib

import pandas as pd


logger = logging.getLogger(__name__)


def dict_factory(cursor, row):
    col_names = [col[0] for col in cursor.description]
    return {key: value for key, value in zip(col_names, row)}


class MetadataDb(collections.abc.Mapping):
    def __init__(self, path):
        uri = f"file:{path}"
        uri += "?mode=ro"
        self.uri = uri
        self.path = path
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = dict_factory

    @staticmethod
    def import_csv(csv_path, db_path):
        df = pd.read_csv(csv_path, sep="\t")
        db_path = pathlib.Path(db_path)
        if db_path.exists():
            db_path.unlink()
        with sqlite3.connect(db_path) as conn:
            df.to_sql("samples", conn, index=False)
            conn.execute(
                "CREATE UNIQUE INDEX [ix_samples_strain] on 'samples' ([strain]);"
            )
            conn.execute("CREATE INDEX [ix_samples_date] on 'samples' ([date]);")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __str__(self):
        return f"MetadataDb at {self.uri} contains {len(self)} sequences"

    def __len__(self):
        sql = "SELECT COUNT(*) FROM samples"
        with self.conn:
            row = self.conn.execute(sql).fetchone()
            return row["COUNT(*)"]

    def __getitem__(self, key):
        sql = "SELECT * FROM samples WHERE strain==?"
        with self.conn:
            result = self.conn.execute(sql, [key]).fetchone()
            if result is None:
                raise KeyError(f"strain {key} not in DB")
            return result

    def __iter__(self):
        sql = "SELECT strain FROM samples"
        with self.conn:
            for result in self.conn.execute(sql):
                yield result["strain"]

    def close(self):
        self.conn.close()

    def get(self, date):
        sql = "SELECT * FROM samples WHERE date==?"
        with self.conn:
            for row in self.conn.execute(sql, [date]):
                yield row

    def date_sample_counts(self):
        sql = "SELECT date, COUNT(*) FROM samples GROUP BY date ORDER BY date;"
        counts = collections.Counter()
        with self.conn:
            for row in self.conn.execute(sql):
                counts[row["date"]] = row["COUNT(*)"]
        return counts

    def get_days(self, date=None):
        if date is None:
            date = "2000-01-01"
        sql = "SELECT DISTINCT(date) FROM samples WHERE date>? ORDER BY date;"
        with self.conn:
            dates = []
            for row in self.conn.execute(sql, [date]):
                dates.append(row["date"])
        return dates
