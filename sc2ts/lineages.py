import json
from collections import defaultdict
import pandas as pd


class MutationContainer:
    def __init__(self):
        self.names = {}
        self.positions = []
        self.alts = []
        self.size = 0
        self.all_positions = set()

    def add_root(self, root_lineage_name):
        self.names[root_lineage_name] = self.size
        self.size += 1
        self.positions.append([])
        self.alts.append([])

    def add_item(self, item, position, alt):
        if item not in self.names:
            self.names[item] = self.size
            self.positions.append([position])
            self.alts.append([alt])
            self.size += 1
        else:
            index = self.names[item]
            self.positions[index].append(position)
            self.alts[index].append(alt)
        if position not in self.all_positions:
            self.all_positions.add(position)

    def get_mutations(self, item):
        index = self.names[item]
        return self.positions[index], self.alts[index]


def read_in_mutations(json_filepath, verbose=False):
    """
    Read in lineage-defining mutations from COVIDCG input json file.
    Assumes root lineage is B.
    """

    with open(json_filepath, "r") as file:
        linmuts = json.load(file)

    # Read in lineage defining mutations
    linmuts_dict = MutationContainer()
    linmuts_dict.add_root("B")
    if verbose:
        check_multiallelic_sites = defaultdict(
            set
        )  # will check how many multi-allelic sites there are

    for item in linmuts:
        if item["alt"] != "-" and item["ref"] != "-":  # ignoring indels
            linmuts_dict.add_item(item["name"], item["pos"], item["alt"])
            if verbose:
                check_multiallelic_sites[item["pos"]].add(item["ref"])
            if verbose:
                check_multiallelic_sites[item["pos"]].add(item["alt"])

    if verbose:
        multiallelic_sites_count = 0
        for value in check_multiallelic_sites.values():
            if len(value) > 2:
                multiallelic_sites_count += 1
        print(
            "Multiallelic sites:",
            multiallelic_sites_count,
            "out of",
            len(check_multiallelic_sites),
        )
        print("Number of lineages:", linmuts_dict.size)

    return linmuts_dict


class OHE_transform:
    """
    One hot encoder using pandas get_dummies() for dealing with categorical data (alleles at each position)
    """

    def __init__(self):
        self.new_colnames = None
        self.old_colnames = None

    def fit(self, X):
        self.old_colnames = X.columns
        X = pd.get_dummies(X, drop_first=True)
        self.new_colnames = X.columns
        return X

    def transform(self, X):
        X = pd.get_dummies(X)
        X = X.reindex(columns=self.new_colnames, fill_value=0)
        return X


def read_in_mutations_json(json_filepath):
    """
    Read in COVIDCG json file of lineage-defining mutations into a pandas data frame
    """
    df = pd.read_json(json_filepath)
    df = df.loc[df["ref"] != "-"]
    df = df.loc[df["alt"] != "-"]
    df = df.pivot_table(
        index="name", columns="pos", values="alt", aggfunc="min", fill_value="."
    )
    idx = df.index.append(pd.Index(["B"]))
    df = df.reindex(idx, fill_value=".")
    ohe = OHE_transform()
    df_ohe = ohe.fit(df)
    return df, df_ohe, ohe


