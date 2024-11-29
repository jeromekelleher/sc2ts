import itertools

import numpy as np
import pytest
import numpy.testing as nt
import xarray.testing as xt
import sgkit

import sc2ts


def assert_datasets_equal(ds1, ds2):
    sg_ds1 = sgkit.load_dataset(ds1.path)
    sg_ds2 = sgkit.load_dataset(ds2.path)
    xt.assert_equal(sg_ds1, sg_ds2)


def test_massaged_viridian_metadata(fx_raw_viridian_metadata_df):
    df = fx_raw_viridian_metadata_df
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
        sc2ts.Dataset.add_metadata(path, fx_metadata_df)

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
        sc2ts.Dataset.add_metadata(path, fx_metadata_df)
        zip_path = tmp_path / "dataset.vcz.zip"
        sc2ts.Dataset.create_zip(path, zip_path)

        ds1 = sc2ts.Dataset(path)
        ds2 = sc2ts.Dataset(zip_path)
        alignments1 = dict(ds1.haplotypes)
        alignments2 = dict(ds2.haplotypes)
        assert alignments1.keys() == alignments2.keys()
        for k in alignments1.keys():
            nt.assert_array_equal(alignments1[k], alignments2[k])

    def test_copy(self, tmp_path, fx_dataset):
        path = tmp_path / "dataset.vcz"
        fx_dataset.copy(path)
        ds = sc2ts.Dataset(path)
        assert_datasets_equal(ds, fx_dataset)

    def test_copy_reorder(self, tmp_path, fx_dataset):
        path = tmp_path / "dataset.vcz"
        sample_id = fx_dataset.sample_id[::-1]
        fx_dataset.copy(path, sample_id=sample_id)
        sg_ds2 = sgkit.load_dataset(path).set_index({"samples": "sample_id"})
        sg_ds1 = sgkit.load_dataset(fx_dataset.path).set_index({"samples": "sample_id"})
        permuted = sg_ds1.sel(samples=sample_id)
        xt.assert_equal(permuted, sg_ds2)

    @pytest.mark.parametrize(
        "sample_id",
        [
            [
                "SRR11597146",
                "SRR11597196",
                "SRR11597178",
                "SRR11597168",
                "SRR11597195",
                "SRR11597190",
                "SRR11597164",
                "SRR11597115",
            ],
            [
                "SRR11597115",
                "SRR11597146",
            ],
            [
                "SRR11597115",
                "SRR11597146",
                "SRR11597164",
                "SRR11597168",
                "SRR11597178",
                "SRR11597190",
                "SRR11597195",
                "SRR11597196",
            ],
        ],
    )
    def test_copy_subset(self, tmp_path, fx_dataset, sample_id):
        path = tmp_path / "dataset.vcz"
        fx_dataset.copy(path, sample_id=sample_id)
        sg_ds2 = sgkit.load_dataset(path).set_index({"samples": "sample_id"})
        sg_ds1 = sgkit.load_dataset(fx_dataset.path).set_index({"samples": "sample_id"})
        permuted = sg_ds1.sel(samples=sample_id)
        xt.assert_equal(permuted, sg_ds2)


class TestDatasetVariants:

    def test_all(self, fx_dataset):
        G = fx_dataset["call_genotype"][:].squeeze()
        pos = fx_dataset["variant_position"][:]
        j = 0
        for var in fx_dataset.variants():
            nt.assert_array_equal(var.genotypes, G[j])
            assert var.position == pos[j]
            j += 1
        assert j == fx_dataset.num_variants

    @pytest.mark.parametrize(
        ["start", "stop"],
        [
            [0, 29903],
            [9999, 10002],
            [333, 2900],
        ],
    )
    def test_variant_slice(self, fx_dataset, start, stop):
        G = fx_dataset["call_genotype"][start:stop].squeeze()
        pos = fx_dataset["variant_position"][start:stop]
        alleles = fx_dataset["variant_allele"][start:stop]
        j = 0
        for var in fx_dataset.variants(position=pos):
            nt.assert_array_equal(var.genotypes, G[j])
            assert var.position == pos[j]
            nt.assert_array_equal(var.alleles, alleles[j])
            j += 1
        assert j == stop - start


class TestDatasetMethods:

    def test_zarr_mapping(self, fx_dataset):
        assert len(fx_dataset) == len(fx_dataset.root)
        assert list(fx_dataset) == list(fx_dataset.root)
        assert dict(fx_dataset) == dict(fx_dataset.root)

    def test_examples(self, fx_dataset):
        nt.assert_array_equal(
            fx_dataset["sample_id"][:3],
            [
                "SRR14631544",
                "SRR11772659",
                "SRR11397727",
            ],
        )


class TestDatasetAlignments:

    def test_fetch_known(self, fx_dataset):
        a = fx_dataset.haplotypes["SRR11772659"]
        assert a.shape == (sc2ts.REFERENCE_SEQUENCE_LENGTH - 1,)
        assert a[0] == -1
        assert a[-1] == -1

    def test_compare_fasta(self, fx_dataset, fx_alignments_fasta):
        fr = sc2ts.FastaReader(fx_alignments_fasta)
        for k, a1 in fr.items():
            h = fx_dataset.haplotypes[k]
            a2 = sc2ts.decode_alignment(h)
            nt.assert_array_equal(a1[1:], a2)

    def test_len(self, fx_dataset):
        assert len(fx_dataset.haplotypes) == 55

    def test_keys(self, fx_dataset):
        keys = list(fx_dataset.haplotypes.keys())
        assert len(keys) == len(fx_dataset.haplotypes)
        assert "SRR11772659" in keys

    def test_in(self, fx_dataset):
        assert "SRR11772659" in fx_dataset.haplotypes
        assert "NOT_IN_STORE" not in fx_dataset.haplotypes

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
        sc2ts.Dataset.add_metadata(path, fx_metadata_df)
        ds = sc2ts.Dataset(path, chunk_cache_size=cache_size)
        for k, v in fx_encoded_alignments.items():
            nt.assert_array_equal(v, ds.haplotypes[k])


class TestDatasetMetadata:

    def test_len(self, fx_dataset):
        assert len(fx_dataset.metadata) == 55

    def test_keys(self, fx_dataset):
        assert fx_dataset.metadata.keys() == fx_dataset.haplotypes.keys()

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
        sc2ts.Dataset.add_metadata(path, fx_metadata_df)
        ds = sc2ts.Dataset(path, chunk_cache_size=cache_size)
        for strain in fx_encoded_alignments.keys():
            row = fx_metadata_df.loc[strain]
            d1 = ds.metadata[strain]
            del d1["strain"]
            d2 = dict(row)
            assert d1 == d2

    def test_in(self, fx_dataset):
        assert "SRR11772659" in fx_dataset.metadata
        assert "DEFO_NOT_IN_DB" not in fx_dataset.metadata

    def test_samples_for_date(self, fx_dataset):
        samples = fx_dataset.metadata.samples_for_date("2020-01-19")
        assert samples == ["SRR11772659"]

    def test_as_dataframe(self, fx_dataset, fx_metadata_df):
        df1 = fx_dataset.metadata.as_dataframe()
        df2 = fx_metadata_df.loc[df1.index]
        assert df1.shape[0] == df2.shape[0]
        for col, data1 in df2.items():
            data2 = df2[col]
            nt.assert_array_equal(data1.to_numpy(), data2.to_numpy())


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
