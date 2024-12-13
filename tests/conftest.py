import pathlib
import shutil
import gzip
import tskit

import numpy as np
import pandas as pd
import pytest

import sc2ts
from sc2ts import cli


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


def encoded_alignments(path):
    fr = sc2ts.FastaReader(path)
    alignments = {}
    for k, v in fr.items():
        alignments[k] = sc2ts.encode_alignment(v[1:])
    return alignments


@pytest.fixture
def fx_encoded_alignments(fx_alignments_fasta):
    return encoded_alignments(fx_alignments_fasta)


def read_metadata_df(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", index_col="Run")
    return sc2ts.massage_viridian_metadata(df)


@pytest.fixture
def fx_metadata_tsv():
    return "tests/data/metadata.tsv"


@pytest.fixture
def fx_raw_viridian_metadata_tsv():
    tsv_path = "tests/data/raw_viridian_metadata.tsv.gz"
    # TO generate, uncommment:
    # df = pd.read_csv("viridian_metadata.tsv", sep="\t")
    # date = df["Collection_date"]
    # dfs = df[(date < "2020-03-01") & (date.str.len() >= 6)]
    # # Not clear why this sequence is in the suite metadata, but easiest
    # # to just put it in here
    # dfs = pd.concat([dfs, df[df.Run == "SRR15736313"]])
    # dfs.to_csv(tsv_path, sep="\t", index=False)
    return tsv_path


@pytest.fixture
def fx_metadata_df(fx_metadata_tsv):
    return read_metadata_df(fx_metadata_tsv)


@pytest.fixture
def fx_raw_viridian_metadata_df(fx_raw_viridian_metadata_tsv):
    return read_metadata_df(fx_raw_viridian_metadata_tsv)


@pytest.fixture
def fx_dataset(tmp_path, fx_data_cache, fx_alignments_fasta, fx_metadata_df):
    cache_path = fx_data_cache / "dataset.vcz.zip"
    if not cache_path.exists():
        fs_path = tmp_path / "dataset.vcz"
        # Use an awkward chunk size here to make sure we're hitting across
        # chunk stuff by default
        sc2ts.Dataset.new(fs_path, samples_chunk_size=7)
        sc2ts.Dataset.append_alignments(
            fs_path, encoded_alignments(fx_alignments_fasta)
        )
        sc2ts.Dataset.add_metadata(fs_path, fx_metadata_df)
        sc2ts.Dataset.create_zip(fs_path, cache_path)
    return sc2ts.Dataset(cache_path)


@pytest.fixture
def fx_match_db(fx_data_cache):
    cache_path = fx_data_cache / "match.db"
    if not cache_path.exists():
        sc2ts.MatchDb.initialise(cache_path)
    return sc2ts.MatchDb(cache_path)


# TODO make this a session fixture cacheing the tree sequences.
@pytest.fixture
def fx_ts_map(tmp_path, fx_data_cache, fx_dataset, fx_match_db):
    dates = [
        "2020-01-01",
        "2020-01-19",
        "2020-01-24",
        "2020-01-25",
        "2020-01-28",
        "2020-01-29",
        "2020-01-30",
        "2020-01-31",
        "2020-02-01",
        "2020-02-02",
        "2020-02-03",
        "2020-02-04",
        "2020-02-05",
        "2020-02-06",
        "2020-02-07",
        "2020-02-08",
        "2020-02-09",
        "2020-02-10",
        "2020-02-11",
        "2020-02-13",
        "2020-02-15",
    ]
    cache_path = fx_data_cache / f"{dates[-1]}.ts"
    if not cache_path.exists():
        # These sites are masked out in all alignments in the initial data
        # anyway; https://github.com/jeromekelleher/sc2ts/issues/282
        last_ts = sc2ts.initial_ts([56, 57, 58, 59, 60])
        cache_path = fx_data_cache / "initial.ts"
        last_ts.dump(cache_path)
        for date in dates:
            # Load the ts from file to get the provenance data
            extra_kwargs = {}
            if date == dates[-1]:
                # Force a bunch of retro groups in on the last day
                extra_kwargs = {
                    "min_group_size": 1,
                    "min_root_mutations": 0,
                }

            last_ts = tskit.load(cache_path)
            last_ts = sc2ts.extend(
                dataset=fx_dataset.path,
                base_ts=cache_path,
                date=date,
                match_db=fx_match_db.path,
                **extra_kwargs,
            )
            print(
                f"INFERRED {date} nodes={last_ts.num_nodes} mutations={last_ts.num_mutations}"
            )
            cache_path = fx_data_cache / f"{date}.ts"
            last_ts.dump(cache_path)
    d = {}
    for date in dates:
        path = fx_data_cache / f"{date}.ts"
        ts = tskit.load(path)
        ts.path = path
        d[date] = ts
    return d


def recombinant_alignments(dataset):
    """
    Generate some recombinant alignments from existing haplotypes
    """
    strains = ["SRR11597188", "SRR11597163"]
    left_a = dataset.haplotypes[strains[0]]
    right_a = dataset.haplotypes[strains[1]]
    # Recombine in the middle
    bp = 9_999
    h = left_a.copy()
    h[bp:] = right_a[bp:]
    alignments = {}
    alignments["recombinant_example_1_0"] = h
    h = h.copy()
    mut_site = bp - 100
    C = sc2ts.IUPAC_ALLELES.index("C")
    assert h[mut_site] != C
    h[mut_site] = C
    alignments["recombinant_example_1_1"] = h
    return alignments


def recombinant_example_1(tmp_path, fx_ts_map, fx_dataset, ds_path):
    alignments = recombinant_alignments(fx_dataset)

    date = "2020-02-15"
    ds = sc2ts.tmp_dataset(tmp_path / "tmp.zarr", alignments, date=date)
    base_ts = fx_ts_map["2020-02-13"]
    ts = sc2ts.extend(
        dataset=ds.path,
        base_ts=base_ts.path,
        date=date,
        num_mismatches=2,
        match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db").path,
    )
    return ts


def recombinant_example_2(tmp_path, fx_ts_map, fx_dataset, ds_path):
    # Pick a distinct strain to be the root of our two new haplotypes added
    # on the first day.
    root_strain = "SRR11597116"
    a = fx_dataset.haplotypes[root_strain]
    base_ts = fx_ts_map["2020-02-13"]
    # This sequence has a bunch of Ns at the start, so we have to go inwards
    # from them to make sure we're not masking them out.
    start = np.where(a != -1)[0][1] + 7
    left_a = a.copy()
    left_a[start : start + 3] = 2  # "G"

    end = np.where(a != -1)[0][-1] - 8
    right_a = a.copy()
    right_a[end - 3 : end] = 1  # "C"

    a[start : start + 3] = left_a[start : start + 3]
    a[end - 3 : end] = right_a[end - 3 : end]

    date = "2020-03-01"
    alignments = {"left": left_a, "right": right_a}
    ds = sc2ts.tmp_dataset(tmp_path / "tmp.zarr", alignments, date=date)

    ts = sc2ts.extend(
        dataset=ds.path,
        base_ts=base_ts.path,
        date=date,
        match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db").path,
    )
    samples_strain = ts.metadata["sc2ts"]["samples_strain"]
    assert samples_strain[-2:] == ["left", "right"]
    assert ts.num_nodes == base_ts.num_nodes + 2
    assert ts.num_edges == base_ts.num_edges + 2
    assert ts.num_mutations == base_ts.num_mutations + 6

    left_node = ts.samples()[-2]
    right_node = ts.samples()[-1]

    for j, mut_id in enumerate(np.where(ts.mutations_node == left_node)[0]):
        mut = ts.mutation(mut_id)
        assert mut.derived_state == "G"
        assert ts.sites_position[mut.site] == start + j + 1

    for j, mut_id in enumerate(np.where(ts.mutations_node == right_node)[0]):
        mut = ts.mutation(mut_id)
        assert mut.derived_state == "C"
        assert ts.sites_position[mut.site] == end - 3 + j + 1

    ts_path = tmp_path / "intermediate.ts"
    ts.dump(ts_path)

    # Now run again with the recombinant of these two
    date = "2020-03-02"
    ds = sc2ts.tmp_dataset(tmp_path / "tmp.zarr", {"recombinant": a}, date=date)
    rts = sc2ts.extend(
        dataset=ds.path,
        base_ts=ts_path,
        date=date,
        match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db").path,
    )
    return rts


@pytest.fixture
def fx_recombinant_example_1(tmp_path, fx_data_cache, fx_ts_map, fx_dataset):
    cache_path = fx_data_cache / "recombinant_ex1.ts"
    if not cache_path.exists():
        print(f"Generating {cache_path}")
        ds_cache_path = fx_data_cache / "recombinant_ex1_dataset.zarr"
        ts = recombinant_example_1(tmp_path, fx_ts_map, fx_dataset, ds_cache_path)
        ts.dump(cache_path)
    return tskit.load(cache_path)


@pytest.fixture
def fx_recombinant_example_2(tmp_path, fx_data_cache, fx_ts_map, fx_dataset):
    cache_path = fx_data_cache / "recombinant_ex2.ts"
    if not cache_path.exists():
        print(f"Generating {cache_path}")
        ds_cache_path = fx_data_cache / "recombinant_ex2_dataset.zarr"
        ts = recombinant_example_2(tmp_path, fx_ts_map, fx_dataset, ds_cache_path)
        ts.dump(cache_path)
    return tskit.load(cache_path)
