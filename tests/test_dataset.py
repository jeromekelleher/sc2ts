import itertools

import numpy as np
import pytest
import numpy.testing as nt
import sgkit

import sc2ts


def test_massaged_viridian_metadata(fx_metadata_df):
    df = fx_metadata_df
    assert df["In_Viridian_tree"].dtype == bool
    assert df["In_intersection"].dtype == bool

    int_fields = [
        "Genbank_N",
        "Viridian_N",
        # "Run_count",
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
        assert dict(sg_ds.sizes) == {
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
        sc2ts.Dataset.new(path, samples_chunk_size=chunk_size)
        alignments = {
            k: fx_encoded_alignments[k]
            for k in itertools.islice(fx_encoded_alignments.keys(), num_samples)
        }

        sc2ts.Dataset.append_alignments(path, alignments)

        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.sizes) == {
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
    def test_append_same_alignments(self, tmp_path, fx_encoded_alignments, num_samples):
        path = tmp_path / "dataset.vcz"
        sc2ts.Dataset.new(path)
        sc2ts.Dataset.append_alignments(path, fx_encoded_alignments)
        alignments = {
            k: fx_encoded_alignments[k]
            for k in itertools.islice(fx_encoded_alignments.keys(), num_samples)
        }

        with pytest.raises(ValueError, match="duplicate"):
            sc2ts.Dataset.append_alignments(path, alignments)

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
        sc2ts.Dataset.new(path, samples_chunk_size=chunk_size)
        alignments = {
            k: fx_encoded_alignments[k]
            for k in itertools.islice(fx_encoded_alignments.keys(), num_samples)
        }

        for k, v in alignments.items():
            sc2ts.Dataset.append_alignments(path, {k: v})

        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.sizes) == {
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
        sc2ts.Dataset.append_alignments(path, fx_encoded_alignments)
        sc2ts.Dataset.add_metadata(path, fx_metadata_df, "date")

        sg_ds = sgkit.load_dataset(path)
        assert dict(sg_ds.sizes) == {
            "variants": 29903,
            "samples": len(fx_encoded_alignments),
            "ploidy": 1,
            "contigs": 1,
            "alleles": 16,
        }
        df = fx_metadata_df.loc[sg_ds["sample_id"].values]
        for col in fx_metadata_df:
            nt.assert_array_equal(df[col], sg_ds[f"sample_{col}"])

    def test_create_zip(self, tmp_path, fx_encoded_alignments, fx_metadata_df):

        path = tmp_path / "dataset.vcz"
        sc2ts.Dataset.new(path)
        sc2ts.Dataset.append_alignments(path, fx_encoded_alignments)
        sc2ts.Dataset.add_metadata(path, fx_metadata_df, "date")
        zip_path = tmp_path / "dataset.vcz.zip"
        sc2ts.Dataset.create_zip(path, zip_path)

        ds1 = sc2ts.Dataset(path)
        ds2 = sc2ts.Dataset(zip_path)
        alignments1 = dict(ds1.alignments)
        alignments2 = dict(ds2.alignments)
        assert alignments1.keys() == alignments2.keys()
        for k in alignments1.keys():
            nt.assert_array_equal(alignments1[k], alignments2[k])


class TestDatasetAlignments:

    def test_fetch_known(self, fx_dataset):
        a = fx_dataset.alignments["SRR11772659"]
        assert a.shape == (sc2ts.REFERENCE_SEQUENCE_LENGTH - 1,)
        assert a[0] == -1
        assert a[-1] == -1

    def test_compare_fasta(self, fx_dataset, fx_alignments_fasta):
        fr = sc2ts.FastaReader(fx_alignments_fasta)
        for k, a1 in fr.items():
            h = fx_dataset.alignments[k]
            a2 = sc2ts.decode_alignment(h)
            nt.assert_array_equal(a1[1:], a2)

    def test_len(self, fx_dataset):
        assert len(fx_dataset.alignments) == 55

    def test_keys(self, fx_dataset):
        keys = list(fx_dataset.alignments.keys())
        assert len(keys) == len(fx_dataset.alignments)
        assert "SRR11772659" in keys

    def test_in(self, fx_dataset):
        assert "SRR11772659" in fx_dataset.alignments
        assert "NOT_IN_STORE" not in fx_dataset.alignments

    @pytest.mark.parametrize(
        ["chunk_size", "cache_size"],
        [
            (1, 10),
            (10, 1),
        ],
    )
    def test_chunk_size_cache_size(
        self,
        tmp_path,
        fx_encoded_alignments,
        fx_metadata_df,
        chunk_size,
        cache_size,
    ):
        path = tmp_path / "dataset.vcz"
        sc2ts.Dataset.new(path, samples_chunk_size=chunk_size)
        sc2ts.Dataset.append_alignments(path, fx_encoded_alignments)
        sc2ts.Dataset.add_metadata(path, fx_metadata_df, "date")
        ds = sc2ts.Dataset(path, chunk_cache_size=cache_size)
        for k, v in fx_encoded_alignments.items():
            nt.assert_array_equal(v, ds.alignments[k])


class TestDatasetMetadata:

    def test_len(self, fx_dataset):
        assert len(fx_dataset.metadata) == 55

    def test_keys(self, fx_dataset):
        assert fx_dataset.metadata.keys() == fx_dataset.alignments.keys()

    def test_known(self, fx_dataset):
        d = fx_dataset.metadata["SRR11772659"]
        assert d["Artic_primer_version"] == "."
        assert d["date"] == "2020-01-19"
        assert d["In_Viridian_tree"]
        assert not d["In_intersection"]
        # assert d["Run_count"] == 4
        assert d["Viridian_cons_len"] == 29836
        assert d["Genbank_N"] == -1
        assert d["Viridian_pangolin"] == "A"

    def test_in(self, fx_dataset):
        assert "SRR11772659" in fx_dataset.metadata
        assert "DEFO_NOT_IN_DB" not in fx_dataset.metadata

    def test_samples_for_date(self, fx_dataset):
        samples = fx_dataset.metadata.samples_for_date("2020-01-19")
        assert samples == ["SRR11772659"]


class TestEncodeAlignment:
    @pytest.mark.parametrize(
        ["hap", "expected"],
        [
            ("A", [0]),
            ("C", [1]),
            ("G", [2]),
            ("T", [3]),
            ("-", [4]),
            ("N", [-1]),
            ("ACGT-N", [0, 1, 2, 3, 4, -1]),
            ("N-TGCA", [-1, 4, 3, 2, 1, 0]),
            ("ACAGTAC-N", [0, 1, 0, 2, 3, 0, 1, 4, -1]),
        ],
    )
    def test_examples(self, hap, expected):
        h = np.array(list(hap), dtype="U1")
        a = sc2ts.encode_alignment(h)
        nt.assert_array_equal(a, expected)

    @pytest.mark.parametrize("hap", "acgtXZxz")
    def test_other_error(self, hap):
        h = np.array(list(hap), dtype="U1")
        with pytest.raises(ValueError, match="not recognised"):
            sc2ts.encode_alignment(h)
