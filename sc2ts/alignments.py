import logging
import collections.abc
import bz2

import lmdb
import tqdm
import numpy as np

from . import core

logger = logging.getLogger(__name__)


def old_encode_alignment(h):
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


def compress_alignment(a):
    return bz2.compress(a.astype("S"))


def decompress_alignment(b):
    buff = bz2.decompress(b)
    x = np.frombuffer(buff, dtype="S1")
    return x.astype(str)

class AlignmentStore(collections.abc.Mapping):
    def __init__(self, path, mode="r"):
        store = zarr.DirectoryStore(path, mode=mode)


class OldAlignmentStore(collections.abc.Mapping):
    def __init__(self, path, mode="r"):
        map_size = 1024**4
        self.path = path
        readonly = mode == "r"
        self.env = lmdb.Environment(
            str(path), subdir=False, readonly=readonly, map_size=map_size, lock=not readonly
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
