import pathlib
import shutil
import gzip
import tskit

import pytest

import sc2ts


@pytest.fixture
def fx_data_cache():
    cache_path = pathlib.Path("tests/data/cache")
    if not cache_path.exists():
        cache_path.mkdir()
    return cache_path


@pytest.fixture
def fx_alignments_fasta(fx_data_cache):
    cache_path = fx_data_cache / "alignments.fasta"
    if not cache_path.exists():
        with gzip.open("tests/data/alignments.fasta.gz") as src:
            with open(cache_path, "wb") as dest:
                shutil.copyfileobj(src, dest)
    return cache_path


@pytest.fixture
def fx_alignment_store(fx_data_cache, fx_alignments_fasta):
    cache_path = fx_data_cache / "alignments.db"
    if not cache_path.exists():
        with sc2ts.AlignmentStore(cache_path, "a") as a:
            fasta = sc2ts.core.FastaReader(fx_alignments_fasta)
            a.append(fasta, show_progress=False)
    return sc2ts.AlignmentStore(cache_path)

@pytest.fixture
def fx_metadata_db(fx_data_cache):
    cache_path = fx_data_cache / "metadata.db"
    tsv_path = "tests/data/metadata.tsv"
    if not cache_path.exists():
        sc2ts.MetadataDb.import_csv(tsv_path, cache_path)
    return sc2ts.MetadataDb(cache_path)


@pytest.fixture
def fx_ts_2020_02_10(tmp_path, fx_data_cache, fx_metadata_db, fx_alignment_store):
    target_date = "2020-02-10"
    cache_path = fx_data_cache / f"{target_date}.ts"
    if not cache_path.exists():
        last_ts = sc2ts.initial_ts()
        match_db = sc2ts.MatchDb.initialise(tmp_path / "match.db")
        for date in fx_metadata_db.date_sample_counts():
            print("INFERRING", date)
            last_ts = sc2ts.extend(
                alignment_store=fx_alignment_store,
                metadata_db=fx_metadata_db,
                base_ts=last_ts,
                date=date,
                match_db=match_db,
                min_group_size=2,
            )
            if date == target_date:
                break
        last_ts.dump(cache_path)
    return tskit.load(cache_path)


