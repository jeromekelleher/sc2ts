import pathlib
import collections.abc

import pyfaidx
import numpy as np

ALLELES = "ACGT-"

TIME_UNITS="days_ago"

REFERENCE_STRAIN = "Wuhan/Hu-1/2019"
REFERENCE_DATE = "2019-12-26"
REFERENCE_GENBANK = "MN908947"

NODE_IS_MUTATION_OVERLAP = 1 << 21
NODE_IS_REVERSION_PUSH = 1 << 22


__version__ = "undefined"
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    pass


class FastaReader(collections.abc.Mapping):
    def __init__(self, path):
        self.reader = pyfaidx.Fasta(str(path))
        self.keys = list(self.reader.keys())

    def __getitem__(self, key):
        x = self.reader[key]
        h = np.array(x).astype(str)
        return np.append(["X"], h)

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)


__cached_reference = None


def get_reference_sequence():
    global __cached_reference
    if __cached_reference is None:
        # NEED packagedata etc.
        data_path = pathlib.Path("sc2ts/data")
        reader = FastaReader(data_path / "reference.fasta")
        __cached_reference = reader[REFERENCE_GENBANK]
    return __cached_reference


def get_problematic_sites():
    data_path = pathlib.Path("sc2ts/data")
    return np.loadtxt(data_path / "problematic_sites.txt", dtype=np.int64)
