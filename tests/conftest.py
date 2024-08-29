import pathlib
import shutil
import gzip

import pytest

import sc2ts


@pytest.fixture
def data_cache():
    cache_path = pathlib.Path("tests/data/cache")
    if not cache_path.exists():
        cache_path.mkdir()
    return cache_path


@pytest.fixture
def alignments_fasta(data_cache):
    cache_path = data_cache / "alignments.fasta"
    if not cache_path.exists():
        with gzip.open("tests/data/alignments.fasta.gz") as src:
            with open(cache_path, "wb") as dest:
                shutil.copyfileobj(src, dest)
    return cache_path


@pytest.fixture
def alignments_store(data_cache, alignments_fasta):
    cache_path = data_cache / "alignments.db"
    if not cache_path.exists():
        with sc2ts.AlignmentStore(cache_path, "a") as a:
            fasta = sc2ts.core.FastaReader(alignments_fasta)
            a.append(fasta, show_progress=False)
    return sc2ts.AlignmentStore(cache_path)
