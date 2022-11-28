import logging
import sqlite3
import pathlib

import pandas as pd


logger = logging.getLogger(__name__)


def metadata_to_db(csv_path, db_path):

    df = pd.read_csv(
        csv_path,
        sep="\t",
        parse_dates=["date", "date_submitted"],
    )
    db_path = pathlib.Path(db_path)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        df.to_sql("samples", conn, index=False)
        conn.execute("CREATE INDEX [ix_samples_strain] on 'samples' ([strain]);")
        conn.execute("CREATE INDEX [ix_samples_date] on 'samples' ([date]);")


def dict_factory(cursor, row):
    col_names = [col[0] for col in cursor.description]
    return {key: value for key, value in zip(col_names, row)}


class MetadataDb:
    def __init__(self, path):
        uri = f"file:{path}"
        uri += "?mode=ro"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.row_factory = dict_factory

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.conn.close()

    @staticmethod
    def import_csv(csv_path, db_path):
        df = pd.read_csv(
            csv_path,
            sep="\t",
        )
        db_path = pathlib.Path(db_path)
        if db_path.exists():
            db_path.unlink()
        with sqlite3.connect(db_path) as conn:
            df.to_sql("samples", conn, index=False)
            conn.execute("CREATE INDEX [ix_samples_strain] on 'samples' ([strain]);")
            conn.execute("CREATE INDEX [ix_samples_date] on 'samples' ([date]);")

    def get(self, date):
        sql = "SELECT * FROM samples WHERE date==?"
        with self.conn:
            for row in self.conn.execute(sql, [date]):
                yield row

    def get_days(self, date):
        sql = "SELECT DISTINCT(date) FROM samples WHERE date>? ORDER BY date;"
        with self.conn:
            dates = []
            for row in self.conn.execute(sql, [date]):
                dates.append(row["date"])
        return dates
