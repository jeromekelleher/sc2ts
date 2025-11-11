import dataclasses
import json
import pathlib
import collections.abc
import csv

import tskit
import pyfaidx
import numpy as np

__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass

TIME_UNITS = "days"

REFERENCE_STRAIN = "Wuhan/Hu-1/2019"
REFERENCE_DATE = "2019-12-26"
REFERENCE_GENBANK = "MN908947"
REFERENCE_SEQUENCE_LENGTH = 29904

NODE_IS_MUTATION_OVERLAP = 1 << 21
NODE_IS_REVERSION_PUSH = 1 << 22
NODE_IS_RECOMBINANT = 1 << 23
NODE_IS_EXACT_MATCH = 1 << 24
NODE_IS_IMMEDIATE_REVERSION_MARKER = 1 << 25
NODE_IS_REFERENCE = 1 << 26
NODE_IS_UNCONDITIONALLY_INCLUDED = 1 << 27


@dataclasses.dataclass(frozen=True)
class FlagValue:
    value: int
    short: str
    long: str
    description: str


flag_values = [
    FlagValue(tskit.NODE_IS_SAMPLE, "S", "Sample", "Tskit defined sample node"),
    FlagValue(
        NODE_IS_MUTATION_OVERLAP,
        "O",
        "MutationOverlap",
        "Node created by coalescing mutations shared by siblings",
    ),
    FlagValue(
        NODE_IS_REVERSION_PUSH,
        "P",
        "ReversionPush",
        "Node created by pushing immediate reversions upwards",
    ),
    FlagValue(
        NODE_IS_RECOMBINANT,
        "R",
        "Recombinant",
        "Node has two or more parents",
    ),
    FlagValue(
        NODE_IS_EXACT_MATCH,
        "E",
        "ExactMatch",
        "Node is an exact match of its parent",
    ),
    FlagValue(
        NODE_IS_IMMEDIATE_REVERSION_MARKER,
        "I",
        "ImmediateReversion",
        "Node is marking the existance of an immediate reversion which "
        "has not been removed for technical reasons",
    ),
    FlagValue(
        NODE_IS_REFERENCE,
        "F",
        "Reference",
        "Node is a reference sequence",
    ),
    FlagValue(
        NODE_IS_UNCONDITIONALLY_INCLUDED,
        "U",
        "UnconditionalInclude",
        "A sample that was flagged for unconditional inclusion",
    ),
]


def decode_flags(f):
    return [v for v in flag_values if (v.value & f) > 0]


def flags_summary(f):
    return "".join([v.short if (v.value & f) > 0 else "_" for v in flag_values])


class FastaReader(collections.abc.Mapping):
    def __init__(self, path, add_zero_base=True):
        self.reader = pyfaidx.Fasta(str(path))
        self._keys = list(self.reader.keys())
        self.add_zero_base = add_zero_base

    def __getitem__(self, key):
        x = self.reader[key]
        h = np.array(x).astype(str)
        h = np.char.upper(h)
        if self.add_zero_base:
            return np.append(["X"], h)
        return h

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)


data_path = pathlib.Path(__file__).parent / "data"


def get_problematic_regions():
    """
    These regions have been reported to have highly recurrent or unusual
    patterns of deletions.

    https://github.com/jeromekelleher/sc2ts/issues/231#issuecomment-2401405355

    Region: NTD domain
    Coords: [21602-22472)
    Multiple highly recurrent deleted regions in NTD domain in Spike
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7971772/

    Region: ORF8
    https://virological.org/t/repeated-loss-of-orf8-expression-in-circulating-sars-cov-2-lineages/931/1

    The 1-based (half-open) coordinates were taken from the UCSC Genome Browser.
    """
    orf8 = get_gene_coordinates()["ORF8"]
    return np.concatenate(
        [
            np.arange(21602, 22472, dtype=np.int64),  # NTD domain in S
            np.arange(*orf8, dtype=np.int64),
        ]
    )


def get_flank_coordinates():
    """
    Return the coordinates at either end of the genome for masking out.
    """
    genes = get_gene_coordinates()
    start = genes["ORF1ab"][0]
    end = genes["ORF10"][1]
    return np.concatenate(
        (np.arange(1, start), np.arange(end, REFERENCE_SEQUENCE_LENGTH))
    )


def get_masked_sites(ts):
    """
    Return the set of sites not used in the sequence.
    """
    unused = np.ones(int(ts.sequence_length), dtype=bool)
    unused[ts.sites_position.astype(int)] = False
    unused[0] = False
    return np.where(unused)[0]


@dataclasses.dataclass
class CovLineage:
    name: str
    earliest_date: str
    latest_date: str
    description: str


def get_cov_lineages_data():
    with open(data_path / "lineages.json") as f:
        data = json.load(f)
    ret = {}
    for record in data:
        lineage = CovLineage(
            record["Lineage"],
            record["Earliest date"],
            record["Latest date"],
            record["Description"],
        )
        assert lineage.name not in ret
        ret[lineage.name] = lineage
    return ret


__cached_reference = None


def get_reference_sequence(as_array=False):
    global __cached_reference
    if __cached_reference is None:
        reader = pyfaidx.Fasta(str(data_path / "reference.fasta"))
        __cached_reference = reader[REFERENCE_GENBANK]
    if as_array:
        h = np.array(__cached_reference).astype(str)
        return np.append(["X"], h)
    else:
        return "X" + str(__cached_reference)


__cached_genes = None


def get_gene_coordinates():
    """
    Returns a map of gene name to interval, (start, stop). These are
    half-open, left-inclusive, right-exclusive.
    """
    global __cached_genes
    if __cached_genes is None:
        d = {}
        with open(data_path / "annotation.csv") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                d[row["gene"]] = (int(row["start"]), int(row["end"]))
        __cached_genes = d
    return __cached_genes


# We omit N here as it's mapped to -1. Make "-" the 5th allele
# as this is a valid allele for us.
IUPAC_ALLELES = "ACGT-RYSWKMBDHV."
