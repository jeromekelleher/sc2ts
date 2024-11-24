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


# TO GENERATE
# @pytest.fixture
# def fx_metadata_df(fx_alignments_fasta):
#     fr = sc2ts.FastaReader(fx_alignments_fasta)
#     sample_id = list(fr.keys)
#     df = pd.read_csv("viridian_metadata.tsv", sep="\t", index_col="Run")
#     dfs = df.loc[sample_id]
#     tsv_path = "tests/data/metadata.tsv"
#     dfs.to_csv(tsv_path, sep="\t")


def read_metadata_df():
    tsv_path = "tests/data/metadata.tsv"
    df = pd.read_csv(tsv_path, sep="\t", index_col="Run")
    return sc2ts.massage_viridian_metadata(df)


@pytest.fixture
def fx_metadata_df():
    return read_metadata_df()


@pytest.fixture
def fx_dataset(tmp_path, fx_data_cache, fx_alignments_fasta):
    cache_path = fx_data_cache / "dataset.vcz.zip"
    if not cache_path.exists():
        fs_path = tmp_path / "dataset.vcz"
        sc2ts.Dataset.new(fs_path)
        sc2ts.Dataset.append_alignments(
            fs_path, encoded_alignments(fx_alignments_fasta)
        )
        sc2ts.Dataset.add_metadata(
            # The metadata we're using has been slightly massaged 
            # already to add more precise dates. Use the "date"
            # field rather than original Collection_date.
            fs_path, read_metadata_df(), date_field="date"
        )
        sc2ts.Dataset.create_zip(fs_path, cache_path)
    return sc2ts.Dataset(cache_path)


# @pytest.fixture
# def fx_alignment_store(fx_data_cache, fx_alignments_fasta):
#     cache_path = fx_data_cache / "alignments.db"
#     if not cache_path.exists():
#         with sc2ts.AlignmentStore(cache_path, "a") as a:
#             fasta = sc2ts.core.FastaReader(fx_alignments_fasta)
#             a.append(fasta, show_progress=False)
#     return sc2ts.AlignmentStore(cache_path)


# @pytest.fixture
# def fx_metadata_db(fx_data_cache):
#     cache_path = fx_data_cache / "metadata.db"
#     tsv_path = "tests/data/metadata.tsv"
#     if not cache_path.exists():
#         sc2ts.MetadataDb.import_csv(tsv_path, cache_path)
#     return sc2ts.MetadataDb(cache_path)


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
    ]
    cache_path = fx_data_cache / f"{dates[-1]}.ts"
    if not cache_path.exists():
        # These sites are masked out in all alignments in the initial data
        # anyway; https://github.com/jeromekelleher/sc2ts/issues/282
        last_ts = sc2ts.initial_ts([56, 57, 58, 59, 60])
        cache_path = fx_data_cache / "initial.ts"
        cli.add_provenance(last_ts, cache_path)
        for date in dates:
            # Load the ts from file to get the provenance data
            last_ts = tskit.load(cache_path)
            last_ts = sc2ts.extend(
                dataset=fx_dataset,
                base_ts=last_ts,
                date=date,
                match_db=fx_match_db,
            )
            print(
                f"INFERRED {date} nodes={last_ts.num_nodes} mutations={last_ts.num_mutations}"
            )
            cache_path = fx_data_cache / f"{date}.ts"
            # The values recorded for resources are nonsense here, but at least it's
            # something to use for tests
            cli.add_provenance(last_ts, cache_path)
    return {date: tskit.load(fx_data_cache / f"{date}.ts") for date in dates}


def tmp_alignment_store(tmp_path, alignments):
    path = tmp_path / "synthetic_alignments.db"
    alignment_db = sc2ts.AlignmentStore(path, mode="rw")
    alignment_db.append(alignments)
    return alignment_db


def tmp_metadata_db(tmp_path, strains, date):
    data = []
    for strain in strains:
        data.append({"strain": strain, "date": date})
    df = pd.DataFrame(data)
    csv_path = tmp_path / "metadata.csv"
    df.to_csv(csv_path)
    db_path = tmp_path / "metadata.db"
    sc2ts.MetadataDb.import_csv(csv_path, db_path, sep=",")
    return sc2ts.MetadataDb(db_path)


def recombinant_alignments(dataset):
    """
    Generate some recombinant alignments from existing haplotypes
    """
    strains = ["SRR11597188", "SRR11597163"]
    left_a = dataset.alignments[strains[0]]
    right_a = dataset.alignments[strains[1]]
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
        dataset=ds,
        base_ts=base_ts,
        date=date,
        num_mismatches=2,
        match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
    )
    return ts


def recombinant_example_2(tmp_path, fx_ts_map, fx_alignment_store):
    # Pick a distinct strain to be the root of our two new haplotypes added
    # on the first day.
    root_strain = "SRR11597116"
    a = fx_alignment_store[root_strain]
    base_ts = fx_ts_map["2020-02-13"]
    # This sequence has a bunch of Ns at the start, so we have to go inwards
    # from them to make sure we're not masking them out.
    start = np.where(a != "N")[0][1] + 7
    left_a = a.copy()
    left_a[start : start + 3] = "G"

    end = np.where(a != "N")[0][-1] - 8
    right_a = a.copy()
    right_a[end - 3 : end] = "C"

    a[start : start + 3] = left_a[start : start + 3]
    a[end - 3 : end] = right_a[end - 3 : end]

    alignments = {"left": left_a, "right": right_a, "recombinant": a}
    local_as = tmp_alignment_store(tmp_path, alignments)

    date = "2020-03-01"
    metadata_db = tmp_metadata_db(tmp_path, ["left", "right"], date)
    ts = sc2ts.extend(
        alignment_store=local_as,
        metadata_db=metadata_db,
        base_ts=base_ts,
        date=date,
        match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
    )
    samples_strain = ts.metadata["sc2ts"]["samples_strain"]
    assert samples_strain[-2:] == ["left", "right"]
    assert ts.num_mutations == base_ts.num_mutations + 6
    assert ts.num_nodes == base_ts.num_nodes + 2
    assert ts.num_edges == base_ts.num_edges + 2

    left_node = ts.samples()[-2]
    right_node = ts.samples()[-1]

    for j, mut_id in enumerate(np.where(ts.mutations_node == left_node)[0]):
        mut = ts.mutation(mut_id)
        assert mut.derived_state == "G"
        assert ts.sites_position[mut.site] == start + j

    for j, mut_id in enumerate(np.where(ts.mutations_node == right_node)[0]):
        mut = ts.mutation(mut_id)
        assert mut.derived_state == "C"
        assert ts.sites_position[mut.site] == end - 3 + j

    # Now run again with the recombinant of these two
    date = "2020-03-02"
    metadata_db = tmp_metadata_db(tmp_path, ["recombinant"], date)
    rts = sc2ts.extend(
        alignment_store=local_as,
        metadata_db=metadata_db,
        base_ts=ts,
        date=date,
        match_db=sc2ts.MatchDb.initialise(tmp_path / "match.db"),
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
def fx_recombinant_example_2(tmp_path, fx_data_cache, fx_ts_map, fx_alignment_store):
    cache_path = fx_data_cache / "recombinant_ex2.ts"
    if not cache_path.exists():
        print(f"Generating {cache_path}")
        ts = recombinant_example_2(tmp_path, fx_ts_map, fx_alignment_store)
        ts.dump(cache_path)
    return tskit.load(cache_path)
