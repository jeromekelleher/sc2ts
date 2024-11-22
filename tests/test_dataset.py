import itertools

import numpy as np
import pytest
import numpy.testing as nt
import sgkit

import sc2ts


@pytest.fixture
def fx_encoded_alignments(fx_alignments_fasta):
    fr = sc2ts.FastaReader(fx_alignments_fasta)
    alignments = {}
    for k, v in fr.items():
        alignments[k] = sc2ts.encode_alignment(v[1:])
    return alignments


def test_massaged_viridian_metadata(fx_metadata_df):
    df = fx_metadata_df
    assert df["In_Viridian_tree"].dtype == bool
    assert df["In_intersection"].dtype == bool

    int_fields = [
        "Genbank_N",
        "Viridian_N",
        "Run_count",
        "Viridian_cons_len",
        "Viridian_cons_het",
    ]
    for field in int_fields:
        assert df[field].dtype == int
    # Genbank N has some missing data
    assert np.sum(df["Genbank_N"]) > 0


class TestCreateDataset:

    def test_new(self, tmp_path):
        path = tmp_path / "dataset.vcz"
        sc2ts.Dataset.new(path)
        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.dims) == {
            "variants": 29903,
            "samples": 0,
            "ploidy": 1,
            "contigs": 1,
            "alleles": 16,
        }
        # TODO check various properties of the dataset

    @pytest.mark.parametrize(
        ["num_samples", "chunk_size"],
        [
            (1, 10),
            (2, 10),
            (2, 1),
            (10, 4),
        ],
    )
    def test_single_append_alignments(
        self, tmp_path, fx_encoded_alignments, num_samples, chunk_size
    ):
        path = tmp_path / "dataset.vcz"
        ds = sc2ts.Dataset.new(path, samples_chunk_size=chunk_size)
        alignments = {
            k: fx_encoded_alignments[k]
            for k in itertools.islice(fx_encoded_alignments.keys(), num_samples)
        }

        ds.append_alignments(alignments)

        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.dims) == {
            "variants": 29903,
            "samples": num_samples,
            "ploidy": 1,
            "contigs": 1,
            "alleles": 16,
        }
        nt.assert_array_equal(sg_ds["sample_id"], list(alignments.keys()))
        H = sg_ds["call_genotype"].values.squeeze(2).T
        for j, h in enumerate(alignments.values()):
            nt.assert_array_equal(h, H[j])

    @pytest.mark.parametrize("num_samples", [1, 10, 20])
    def test_append_same_aligments(self, tmp_path, fx_encoded_alignments, num_samples):
        path = tmp_path / "dataset.vcz"
        ds = sc2ts.Dataset.new(path)
        ds.append_alignments(fx_encoded_alignments)
        alignments = {
            k: fx_encoded_alignments[k]
            for k in itertools.islice(fx_encoded_alignments.keys(), num_samples)
        }

        with pytest.raises(ValueError, match="duplicate"):
            ds.append_alignments(alignments)

    @pytest.mark.parametrize(
        ["num_samples", "chunk_size"],
        [
            (10, 2),
            (10, 3),
            (10, 10),
            (10, 100),
        ],
    )
    def test_multi_append_alignments(
        self, tmp_path, fx_encoded_alignments, num_samples, chunk_size
    ):
        path = tmp_path / "dataset.vcz"
        ds = sc2ts.Dataset.new(path, samples_chunk_size=chunk_size)
        alignments = {
            k: fx_encoded_alignments[k]
            for k in itertools.islice(fx_encoded_alignments.keys(), num_samples)
        }

        for k, v in alignments.items():
            ds.append_alignments({k: v})

        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.dims) == {
            "variants": 29903,
            "samples": num_samples,
            "ploidy": 1,
            "contigs": 1,
            "alleles": 16,
        }
        nt.assert_array_equal(sg_ds["sample_id"], list(alignments.keys()))
        H = sg_ds["call_genotype"].values.squeeze(2).T
        for j, h in enumerate(alignments.values()):
            nt.assert_array_equal(h, H[j])

    def test_add_metadata(self, tmp_path, fx_encoded_alignments, fx_metadata_df):

        path = tmp_path / "dataset.vcz"
        ds = sc2ts.Dataset.new(path)
        ds.append_alignments(fx_encoded_alignments)
        ds.add_metadata(fx_metadata_df)

        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.dims) == {
            "variants": 29903,
            "samples": len(fx_encoded_alignments),
            "ploidy": 1,
            "contigs": 1,
            "alleles": 16,
        }
        df = fx_metadata_df.loc[sg_ds["sample_id"].values]
        for col in fx_metadata_df:
            print("check", col)
            nt.assert_array_equal(df[col], sg_ds[f"sample_{col}"])
