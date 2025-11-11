import collections.abc
import pathlib
import csv

import numpy as np
import pyfaidx

from . import core


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

__cached_reference = None


def get_reference_sequence(as_array=False):
    global __cached_reference
    if __cached_reference is None:
        reader = pyfaidx.Fasta(str(data_path / "reference.fasta"))
        __cached_reference = reader[core.REFERENCE_GENBANK]
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
