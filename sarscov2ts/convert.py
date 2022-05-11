import calendar

import cyvcf2
import pandas as pd
import numpy as np


def pad_date(s):
    """
    Takes a partial ISO date description and pads it out to the end
    of the month.
    """
    if len(s) == 1:
        return s
    if len(s) == 10:
        return s
    year = int(s[:4])
    if len(s) > 4:
        month = int(s[5:])
    else:
        month = 12
    day = calendar.monthrange(year, month)[-1]
    return f"{year}-{month:02d}-{day:02d}"


def load_usher_metadata(path):
    return pd.read_csv(path, sep="\t", dtype={"date": pd.StringDtype()})


def prepare_metadata(df):
    """
    Takes the specified metadata dataframe, pads partially specified dates,
    removes samples and returns the resulting dataframe with samples
    sorted by date.
    """
    # remove missing
    df = df[df["date"] != "?"].copy()
    df.loc[:, "date"] = df.date.apply(pad_date)
    df = df.sort_values("date")
    return df





