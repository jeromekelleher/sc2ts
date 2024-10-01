import logging
import pathlib
import dataclasses
import collections.abc
import hashlib
import bz2

import lmdb
import numba
import tqdm
import numpy as np

from . import core

logger = logging.getLogger(__name__)

GAP = core.ALLELES.index("-")
MISSING = -1


@numba.njit
def mask_alignment(a, start=0, window_size=7):
    """
    Following the approach in fa2vcf, if any base is has two or more ambiguous
    or gap characters with distance window_size of it, mark it as missing data.
    """
    if window_size < 1:
        raise ValueError("Window must be >= 1")
    b = a.copy()
    n = len(a)
    masked_sites = []
    for j in range(start, n):
        ambiguous = 0
        k = j - 1
        while k >= start and k >= j - window_size:
            if b[k] == GAP or b[k] == MISSING:
                ambiguous += 1
            k -= 1
        k = j + 1
        while k < n and k <= j + window_size:
            if b[k] == GAP or b[k] == MISSING:
                ambiguous += 1
            k += 1
        if ambiguous > 1:
            a[j] = MISSING
            masked_sites.append(j)
    return masked_sites


def encode_alignment(h):
    # Map anything that's not ACGT- to N
    a = np.full(h.shape, -1, dtype=np.int8)
    for code, char in enumerate(core.ALLELES):
        a[h == char] = code
    return a


def decode_alignment(a):
    if np.any(a < -1) or np.any(a >= len(core.ALLELES)):
        raise ValueError("Cannot decode alignment")
    alleles = np.array(list(core.ALLELES + "N"), dtype="U1")
    return alleles[a]


def base_composition(haplotype, excluded_sites=None):
    """
    Haplotype includes an arbitrary character at the start.
    Also, excluded site positions are 1-based.
    """
    if excluded_sites is not None:
        mask = np.zeros(len(haplotype), dtype=bool)
        mask[excluded_sites] = True
        # Remove the first site from both haplotype and mask.
        masked_haplotype = haplotype[1:][~mask[1:]]
        return collections.Counter(masked_haplotype)
    return collections.Counter(haplotype[1:])


def compress_alignment(a):
    return bz2.compress(a.astype("S"))


def decompress_alignment(b):
    buff = bz2.decompress(b)
    x = np.frombuffer(buff, dtype="S1")
    return x.astype(str)


class AlignmentStore(collections.abc.Mapping):
    def __init__(self, path, mode="r"):
        map_size = 1024**4
        self.env = lmdb.Environment(
            str(path), subdir=False, readonly=mode == "r", map_size=map_size
        )
        logger.debug(f"Opened AlignmentStore at {path} mode={mode}")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self.env.close()

    def __str__(self):
        return f"AlignmentStore at {self.env.path()} contains {len(self)} alignments"

    def _flush(self, chunk):
        logger.debug(f"Flushing {len(chunk)} sequences")
        with self.env.begin(write=True) as txn:
            for k, v in chunk:
                txn.put(k.encode(), v)
        logger.debug("Done")

    def append(self, alignments, show_progress=False):
        n = len(alignments)
        chunk_size = 100
        num_chunks = n // chunk_size
        logger.info(f"Appending {n} alignments in {num_chunks} chunks")
        bar = tqdm.tqdm(total=num_chunks, disable=not show_progress)
        chunk = []
        for k, v in alignments.items():
            v = np.char.upper(v)
            chunk.append((k, compress_alignment(v)))
            if len(chunk) == chunk_size:
                self._flush(chunk)
                chunk = []
                bar.update()
        self._flush(chunk)
        bar.close()

    def __getitem__(self, key):
        with self.env.begin() as txn:
            val = txn.get(key.encode())
            if val is None:
                raise KeyError(f"{key} not found")
            return decompress_alignment(val)

    def __iter__(self):
        with self.env.begin() as txn:
            cursor = txn.cursor()
            with txn.cursor() as cursor:
                for key in cursor.iternext(keys=True, values=False):
                    yield key.decode()

    def __len__(self):
        with self.env.begin() as txn:
            return txn.stat()["entries"]


@dataclasses.dataclass
class MaskedAlignment:
    alignment: np.ndarray
    masked_sites: np.ndarray
    original_base_composition: dict
    original_md5: str
    # masked_base_composition: str

    def qc_summary(self):
        return {
            # "num_masked_sites": self.masked_sites.shape[0],
            "original_base_composition": self.original_base_composition,
            "original_md5": self.original_md5,
            # "masked_base_composition": self.masked_base_composition,
        }


def encode_and_mask(alignment, window_size=7):
    # TODO make window_size param
    a = encode_alignment(alignment)
    # masked_sites = mask_alignment(a, start=1, window_size=window_size)
    masked_sites = []
    return MaskedAlignment(
        alignment=a,
        masked_sites=np.array(masked_sites, dtype=int),
        original_base_composition=base_composition(haplotype=alignment),
        original_md5=hashlib.md5(alignment[1:]).hexdigest(),
        # masked_base_composition=base_composition(
        #     haplotype=alignment,
        #     excluded_sites=masked_sites,
        # ),
    )
