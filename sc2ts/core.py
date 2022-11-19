import pathlib
import collections.abc

import pyfasta
import numpy as np

ALLELES = "ACGT-"

TIME_UNITS="days_ago"

NODE_IS_MUTATION_OVERLAP = 1 << 21


class FastaReader(collections.abc.Mapping):
    def __init__(self, path):
        self.reader = pyfasta.Fasta(str(path), record_class=pyfasta.MemoryRecord)

    def __getitem__(self, key):
        x = self.reader[key]
        h = np.array(x).astype(str)
        return np.append(["X"], h)

    def __iter__(self):
        return iter(self.reader)

    def __len__(self):
        return len(self.reader)


__cached_reference = None


def get_reference_sequence():
    global __cached_reference
    if __cached_reference is None:
        # NEED packagedata etc.
        data_path = pathlib.Path("sc2ts/data")
        reader = FastaReader(data_path / "reference.fasta")
        __cached_reference = reader["MN908947 (Wuhan-Hu-1/2019)"]
    return __cached_reference


def get_problematic_sites():
    data_path = pathlib.Path("sc2ts/data")
    return np.loadtxt(data_path / "problematic_sites.txt", dtype=np.int64)
