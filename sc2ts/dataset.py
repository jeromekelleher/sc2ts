"""
Methods for managing a sc2ts Zarr based dataset.
"""

import dataclasses
import os.path
import zipfile
import collections
import logging
import pathlib
import concurrent.futures as cf

import tskit
import tqdm
import pandas as pd
import zarr
import numcodecs
import numpy as np

from sc2ts import core

logger = logging.getLogger(__name__)

DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7, shuffle=0)


def decode_alignment(a):
    alleles = np.array(tuple(core.IUPAC_ALLELES + "N"), dtype=str)
    return alleles[a]


def readahead_retrieve(array, blocks):

    if len(blocks) == 0:
        return

    def worker(block):
        return block, array.blocks[block]

    result = worker(blocks[0])
    with cf.ThreadPoolExecutor(1) as executor:
        for block in blocks[1:]:
            future = executor.submit(worker, block)
            yield result
            result = future.result()
    yield result


class CachedHaplotypeMapping(collections.abc.Mapping):
    def __init__(self, root, sample_id_map, chunk_cache_size):
        self.call_genotype_array = root["call_genotype"]
        self.chunk_cache_size = chunk_cache_size
        self.chunk_cache = {}
        self.sample_id_map = sample_id_map

    def get_haplotype(self, j):
        chunk_size = self.call_genotype_array.chunks[1]
        chunk = j // chunk_size
        if chunk not in self.chunk_cache:
            logger.debug(f"Haplotype chunk cache miss on {chunk}")
            if len(self.chunk_cache) >= self.chunk_cache_size:
                lru = list(self.chunk_cache.keys())[0]
                del self.chunk_cache[lru]
                logger.debug(f"Evicted LRU {lru} from alignment chunk cache ")
            self.chunk_cache[chunk] = self.call_genotype_array.blocks[:, chunk]
        G = self.chunk_cache[chunk]
        # NOTE: the copy is needed here to avoid memory growing very
        # rapidly in pathological cases. We can end up storing many copies
        # of the same chunk if we have repeated cache missed on it before
        # flushing.
        return G[:, j % chunk_size].squeeze(1).copy()

    def __getitem__(self, key):
        j = self.sample_id_map[key]
        return self.get_haplotype(j)

    def __iter__(self):
        return iter(self.sample_id_map)

    def __len__(self):
        return len(self.sample_id_map)


class CachedMetadataMapping(collections.abc.Mapping):
    def __init__(self, root, sample_id_map, date_field, chunk_cache_size):
        self.sample_id_map = sample_id_map
        self.sample_id = root["sample_id"][:].astype(str)
        self.sample_id_array = root["sample_id"]
        # Mapping of field name to Zarr array
        self.fields = {}
        prefix = "sample_"
        for k, array in root.items():
            if k.startswith(prefix) and k != "sample_id":
                name = k[len(prefix) :]
                self.fields[name] = array
        self.chunk_cache_size = chunk_cache_size
        self.chunk_cache = {}

        logger.debug(f"Got {self.num_fields} metadata fields")
        self.date_field = date_field
        if date_field is not None:
            self.sample_date = root[f"sample_{date_field}"][:].astype(str)

    @property
    def num_fields(self):
        return len(self.fields)

    def get_metadata(self, j):

        chunk_size = self.sample_id_array.chunks[0]
        chunk = j // chunk_size
        if chunk not in self.chunk_cache:
            logger.debug(f"Metadata chunk cache miss on {chunk}")
            if len(self.chunk_cache) >= self.chunk_cache_size:
                lru = list(self.chunk_cache.keys())[0]
                del self.chunk_cache[lru]
                logger.debug(f"Evicted LRU {lru} from metadata chunk cache")
            cached_chunk = {}
            for field, array in self.fields.items():
                cached_chunk[field] = array.blocks[chunk]
            self.chunk_cache[chunk] = cached_chunk
        cached_chunk = self.chunk_cache[chunk]
        k = j % chunk_size

        d = {}
        for key, np_array in cached_chunk.items():
            d[key] = np_array[k]
            if np_array.dtype.kind == "i":
                d[key] = int(d[key])
            elif np_array.dtype.kind == "b":
                d[key] = bool(d[key])
            else:
                d[key] = str(d[key])
        if self.date_field is None:
            raise ValueError("No date field set, cannot get metadata items")
        # For compatibility in the short term:
        d["date"] = self.sample_date[j]
        d["strain"] = self.sample_id[j]
        return d

    def __getitem__(self, key):
        j = self.sample_id_map[key]
        return self.get_metadata(j)

    def __iter__(self):
        return iter(self.sample_id_map)

    def __len__(self):
        return len(self.sample_id_map)

    def samples_for_date(self, date):
        return self.sample_id[self.sample_date == date]

    def as_dataframe(self, fields=None):
        if fields is None:
            fields = list(self.fields.keys())
        data = {"sample_id": self.sample_id}
        for k in fields:
            data[k] = self.fields[k][:]
        return pd.DataFrame(data).set_index("sample_id")

    def field_descriptors(self):
        data = []
        for k, v in self.fields.items():
            data.append(
                {
                    "field": k,
                    "dtype": v.dtype,
                    "description": v.attrs.get("description", ""),
                }
            )
        return pd.DataFrame(data).set_index("field")


@dataclasses.dataclass
class Variant:
    position: int
    genotypes: np.ndarray
    alleles: list


class Dataset(collections.abc.Mapping):

    def __init__(self, path, chunk_cache_size=1, date_field=None):
        logger.info(f"Loading dataset @{path} using {date_field} as date field")
        self.date_field = date_field
        self.path = pathlib.Path(path)
        if self.path.suffix == ".zip":
            self.store = zarr.ZipStore(path)
        else:
            self.store = zarr.DirectoryStore(path)
        self.root = zarr.open(self.store, mode="r")
        self.sample_id = self.root["sample_id"][:].astype(str)

        # TODO we should be storing this mapping in the Zarr somehow.
        self.sample_id_map = {
            sample_id: k for k, sample_id in enumerate(self.sample_id)
        }
        self.haplotypes = CachedHaplotypeMapping(
            self.root, self.sample_id_map, chunk_cache_size
        )
        self.metadata = CachedMetadataMapping(
            self.root,
            self.sample_id_map,
            date_field,
            chunk_cache_size=chunk_cache_size,
        )

    def __getitem__(self, key):
        return self.root[key]

    def __iter__(self):
        return iter(self.root)

    def __len__(self):
        return len(self.root)

    @property
    def samples_chunk_size(self):
        return self.root["call_genotype"].chunks[1]

    @property
    def variants_chunk_size(self):
        return self.root["call_genotype"].chunks[0]

    @property
    def num_samples(self):
        return self.root.call_genotype.shape[1]

    @property
    def num_variants(self):
        return self.root.call_genotype.shape[0]

    def __str__(self):
        return (
            f"Dataset at {self.path} with {self.num_samples} samples, "
            f"{self.num_variants} variants, "
            f"and {self.metadata.num_fields} metadata fields. See "
            "ds.metadata.field_descriptors() for a description of the fields."
        )

    def _repr_markdown_(self):
        return str(self)

    def variants(self, sample_id=None, position=None):
        variant_position = self["variant_position"][:]
        variant_alleles = self["variant_allele"][:]
        call_genotype = self["call_genotype"]
        if sample_id is None:
            sample_id = self.sample_id
        if position is None:
            position = variant_position

        sample_index = np.array(
            [self.sample_id_map[sid] for sid in sample_id], dtype=int
        )
        index = np.searchsorted(variant_position, position)
        if not np.all(variant_position[index] == position):
            raise ValueError("Unknown position")
        v_chunk_size = call_genotype.chunks[0]
        variant_select = zarr.zeros(
            shape=variant_position.shape, chunks=v_chunk_size, dtype=bool
        )
        variant_select[index] = True

        v_chunks = []
        for v_chunk in range(call_genotype.cdata_shape[0]):
            # NOTE: could improve performance quite a lot for small sample
            # sets by pulling only S chunks that are needed.
            v_select = variant_select.blocks[v_chunk]
            if np.any(v_select):
                v_chunks.append(v_chunk)

        for v_chunk, G in readahead_retrieve(call_genotype, v_chunks):
            v_select = variant_select.blocks[v_chunk]
            for k in range(G.shape[0]):
                if v_select[k]:
                    j = v_chunk * v_chunk_size + k
                    yield Variant(
                        variant_position[j],
                        G[k, sample_index].squeeze(1),
                        variant_alleles[j],
                    )

    def write_fasta(self, out, sample_id=None):
        """
        Writes the alignment data in FASTA format to the specified file.
        """
        if sample_id is None:
            sample_id = self.sample_id

        for sid in sample_id:
            h = self.haplotypes[sid]
            a = decode_alignment(h)
            print(f">{sid}", file=out)
            # FIXME this is probably a terrible way to write a large numpy string to
            # a file
            print("".join(a), file=out)

    def copy(
        self,
        path,
        samples_chunk_size=None,
        variants_chunk_size=None,
        sample_id=None,
        show_progress=False,
    ):
        """
        Copy this dataset to the specified path.

        If sample_id is specified, only include these samples in the specified order.
        """
        if samples_chunk_size is None:
            samples_chunk_size = self.samples_chunk_size
        if variants_chunk_size is None:
            variants_chunk_size = self.variants_chunk_size
        if sample_id is None:
            sample_id = self["sample_id"][:]
        Dataset.new(
            path,
            samples_chunk_size=samples_chunk_size,
            variants_chunk_size=variants_chunk_size,
        )
        alignments = {}
        bar = tqdm.tqdm(sample_id, desc="Samples", disable=not show_progress)
        for s in bar:
            alignments[s] = self.haplotypes[s]
            if len(alignments) == samples_chunk_size:
                Dataset.append_alignments(path, alignments)
                alignments = {}
        Dataset.append_alignments(path, alignments)

        df = self.metadata.as_dataframe()
        Dataset.add_metadata(path, df)

    def reorder(self, path, additional_fields=list(), show_progress=False):
        sample_id = self.metadata.sample_id_array[:]
        sort_key = [self.date_field] + list(additional_fields)
        logger.info(f"Reorder sort key = {sort_key})")
        index = np.lexsort([self.metadata.fields[f] for f in sort_key[::-1]])
        self.copy(path, sample_id=sample_id[index], show_progress=show_progress)

    @staticmethod
    def new(path, samples_chunk_size=None, variants_chunk_size=None):

        if samples_chunk_size is None:
            samples_chunk_size = 10_000
        if variants_chunk_size is None:
            variants_chunk_size = 100

        logger.info(f"Creating new dataset at {path}")
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
            shape=(L, len(core.IUPAC_ALLELES)),
            chunks=(variants_chunk_size, len(core.IUPAC_ALLELES)),
            object_codec=numcodecs.VLenUTF8(),
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        z[:] = np.tile(tuple(core.IUPAC_ALLELES), L).reshape(L, len(core.IUPAC_ALLELES))
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
        if len(alignments) == 0:
            return
        store = zarr.DirectoryStore(path)
        root = zarr.open(store, mode="a")

        n = len(alignments)
        gt_array = root["call_genotype"]
        sample_id_array = root["sample_id"]
        L, N = gt_array.shape[:2]
        logger.info(f"Appending {len(alignments)} alignments to store with {N}")
        if len(set(sample_id_array[:]) & set(alignments.keys())) > 0:
            raise ValueError("Attempting to add duplicate samples")

        G = np.zeros((L, n, 1), dtype=np.int8)
        sample_id = []
        for j, (s, h) in enumerate(alignments.items()):
            sample_id.append(s)
            G[:, j, 0] = h

        sample_id_array.append(sample_id)
        gt_array.append(G, axis=1)

        zarr.consolidate_metadata(store)

    @staticmethod
    def add_metadata(path, df, field_descriptions=dict()):
        """
        Add metadata from the specified dataframe, indexed by sample ID.
        Each column will be added as a new array with prefix "sample_"
        """
        store = zarr.DirectoryStore(path)
        root = zarr.open(store, mode="a")

        sample_id_array = root["sample_id"]
        samples = sample_id_array[:]
        if samples.shape[0] == 0:
            raise ValueError("Cannot add metadata to empty dataset")
        df = df.loc[samples].copy()
        for colname in df:
            data = df[colname].to_numpy()
            dtype = data.dtype
            if dtype.kind == "i":
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
            z.attrs["description"] = field_descriptions.get(colname, "")

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


def tmp_dataset(path, alignments, date="2020-01-01"):
    # Minimal hacky thing for testing. Should refactor into something more useful.
    Dataset.new(path)
    Dataset.append_alignments(path, alignments)
    df = pd.DataFrame({"strain": alignments.keys(), "date": [date] * len(alignments)})
    Dataset.add_metadata(path, df.set_index("strain"))
    return Dataset(path)
