"""
Methods for managing a sc2ts Zarr based dataset.
"""

import os.path
import zipfile
import collections
import logging
import pathlib

import zarr
import numcodecs
import numpy as np
import numba

from sc2ts import core

logger = logging.getLogger(__name__)

DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7, shuffle=0)

# We omit N here as it's mapped to -1. Make "-" the 5th allele
# as this is a valid allele for us.
IUPAC_ALLELES = "ACGT-RYSWKMBDHV."


# FIXME make cache optional
@numba.njit(cache=True)
def encode_alignment(h):
    # Just so numba knows this is a constant string
    alleles = "ACGT-RYSWKMBDHV."
    n = h.shape[0]
    a = np.full(n, -1, dtype=np.int8)
    for j in range(n):
        if h[j] == "N":
            a[j] = -1
        else:
            for k, c in enumerate(alleles):
                if c == h[j]:
                    break
            else:
                raise ValueError(f"Allele {h[j]} not recognised")
            a[j] = k
    return a


def massage_viridian_metadata(df):
    """
    Takes a pandas dataframe indexex by sample ID and massages it
    so that the returned dataframe has consistent types:

    - bool T/F columns encoded as booleans
    - integer columns encoded with -1 as N/A
    """
    # print(df)
    bool_cols = [name for name in df if name.startswith("In")]
    N = df.shape[0]
    for name in bool_cols:
        data = df[name]
        assert set(data.unique()) <= set(["F", "T"])
        a = np.zeros(N, dtype=bool)
        a[data == "T"] = 1
        df[name] = a
    int_fields = [
        "Genbank_N",
        "Viridian_N",
        "Run_count",
        "Viridian_cons_len",
        "Viridian_cons_het",
    ]
    for name in int_fields:
        data = df[name]
        if str(data.dtype) == "int64":
            continue
        a = np.zeros(N, dtype=int)
        missing = data == "."
        a[missing] = -1
        a[~missing] = np.array(data[~missing], dtype=int)
        df[name] = a
    return df


class CachedAlignmentMapping(collections.abc.Mapping):
    def __init__(self, root, chunk_cache_size):
        self.call_genotype_array = root["call_genotype"]
        self.chunk_cache_size = chunk_cache_size
        self.chunk_cache = {}
        self.sample_id_map = {
            sample_id: k for k, sample_id in enumerate(root["sample_id"][:])
        }

    def get_alignment(self, j):
        chunk_size = self.call_genotype_array.chunks[1]
        chunk = j // chunk_size
        if chunk not in self.chunk_cache:
            logger.debug(f"Alignment chunk cache miss on {chunk}")
            if len(self.chunk_cache) >= self.chunk_cache_size:
                lru = list(self.chunk_cache.keys())[0]
                del self.chunk_cache[lru]
                logger.debug(f"Evicted LRU {lru} from alignment chunk cache")
            self.chunk_cache[chunk] = self.call_genotype_array.blocks[:, chunk]
        G = self.chunk_cache[chunk]
        return G[:, j % chunk_size].squeeze(1)

    def __getitem__(self, key):
        j = self.sample_id_map[key]
        return self.get_alignment(j)

    def __iter__(self):
        return iter(self.sample_id_map)

    def __len__(self):
        return len(self.sample_id_map)


class Dataset:

    def __init__(self, path, chunk_cache_size=1):
        self.path = pathlib.Path(path)
        if self.path.suffix == ".zip":
            self.store = zarr.ZipStore(path)
        else:
            self.store = zarr.DirectoryStore(path)
        self.root = zarr.open(self.store, mode="r")

        self.alignments = CachedAlignmentMapping(self.root, chunk_cache_size)

    @staticmethod
    def new(path, samples_chunk_size=10_000, variants_chunk_size=100):
        L = core.REFERENCE_SEQUENCE_LENGTH - 1
        N = 0  # Samples must be added
        store = zarr.DirectoryStore(path)
        root = zarr.open(store, mode="w")

        z = root.empty(
            "variant_position",
            shape=(L),
            chunks=(variants_chunk_size),
            dtype=np.int32,
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        z[:] = np.arange(1, L + 1)
        z.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

        z = root.empty(
            name="variant_allele",
            dtype="O",
            shape=(L, len(IUPAC_ALLELES)),
            chunks=(variants_chunk_size, len(IUPAC_ALLELES)),
            object_codec=numcodecs.VLenUTF8(),
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        z[:] = np.tile(tuple(IUPAC_ALLELES), L).reshape(L, len(IUPAC_ALLELES))
        z.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

        z = root.empty(
            "contig_id",
            shape=1,
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        z[0] = core.REFERENCE_STRAIN
        z.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

        z = root.empty(
            "contig_length",
            shape=1,
            dtype=np.int64,
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        z[0] = L + 1
        z.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

        z = root.empty(
            "variant_contig",
            shape=(L),
            chunks=variants_chunk_size,
            dtype=np.int8,
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        z[:] = 0
        z.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

        z = root.empty(
            "sample_id",
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
            shape=(N,),
            chunks=(samples_chunk_size,),
        )
        z.attrs["_ARRAY_DIMENSIONS"] = ["samples"]

        shape = (L, N)
        z = root.empty(
            "call_genotype",
            shape=(L, N, 1),
            chunks=(variants_chunk_size, samples_chunk_size),
            dtype=np.int8,
            compressor=numcodecs.Blosc(
                cname="zstd", clevel=7, shuffle=numcodecs.Blosc.BITSHUFFLE
            ),
            dimension_separator="/",
        )
        z.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

        zarr.consolidate_metadata(store)

    @staticmethod
    def append_alignments(path, alignments):
        """
        Append alignments to the store. If this method fails then the store
        should be considered corrupt.
        """
        store = zarr.DirectoryStore(path)
        root = zarr.open(store, mode="a")

        n = len(alignments)
        gt_array = root["call_genotype"]
        sample_id_array = root["sample_id"]
        if len(set(sample_id_array[:]) & set(alignments.keys())) > 0:
            raise ValueError("Attempting to add duplicate samples")
        L, N = gt_array.shape[:2]
        logger.debug(f"Appending {len(alignments)} to store with {N}")

        G = np.zeros((L, n, 1), dtype=np.int8)
        sample_id = []
        for j, (s, h) in enumerate(alignments.items()):
            sample_id.append(s)
            G[:, j, 0] = h

        sample_id_array.append(sample_id)
        gt_array.append(G, axis=1)

        zarr.consolidate_metadata(store)

    @staticmethod
    def add_metadata(path, df):
        """
        Add metadata from the specified dataframe, indexed by sample ID.
        Each column will be added as a new array with prefix "sample_"
        """
        store = zarr.DirectoryStore(path)
        root = zarr.open(store, mode="a")

        sample_id_array = root["sample_id"]
        samples = sample_id_array[:]
        df = df.loc[samples]
        for colname in df:
            data = df[colname]
            dtype = data.dtype
            if dtype == int:
                max_v = data.max()
                if max_v < 127:
                    dtype = "i1"
                elif max_v < 2**15 - 1:
                    dtype = "i2"
                else:
                    dtype = "i4"
            elif dtype != bool:
                dtype = "str"
            z = root.empty(
                f"sample_{colname}",
                dtype=dtype,
                compressor=sample_id_array.compressor,
                shape=sample_id_array.shape,
                chunks=sample_id_array.chunks,
                overwrite=True,
            )
            z.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
            z[:] = data
            logger.info(f"Wrote metadata array {z.name}")

        zarr.consolidate_metadata(store)

    @staticmethod
    def create_zip(in_path, out_path):

        # Based on https://github.com/python/cpython/blob/3.13/Lib/zipfile/__init__.py
        def add_to_zip(zf, path, zippath):
            if os.path.isfile(path):
                zf.write(path, zippath, zipfile.ZIP_STORED)
            elif os.path.isdir(path):
                for nm in os.listdir(path):
                    add_to_zip(zf, os.path.join(path, nm), os.path.join(zippath, nm))

        with zipfile.ZipFile(out_path, "w", allowZip64=True) as zf:
            add_to_zip(zf, in_path, ".")
