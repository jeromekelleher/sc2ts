import datetime

import cyvcf2
import numpy as np
import pytest

import sarscov2ts.convert as convert


@pytest.mark.parametrize(
    ["value", "result"],
    [
        ("2021", "2021-12-31"),
        ("2020", "2020-12-31"),
        ("2019", "2019-12-31"),
        ("2020-02", "2020-02-29"),
        ("2021-02", "2021-02-28"),
        ("2020-09", "2020-09-30"),
        ("2021-10", "2021-10-31"),
    ],
)
def test_pad_date(value, result):
    assert convert.pad_date(value) == result


def test_prepare_metadata():
    df = convert.load_usher_metadata("tests/data/usher-metadata-date-test.tsv")
    date_col = df["date"]
    num_missing = np.sum(date_col == "?")
    dfp = convert.prepare_metadata(df)
    assert len(dfp) == len(df) - num_missing
    values = list(dfp["date"])
    assert sorted(values) == values


@pytest.mark.parametrize("force_four_alleles", [True, False])
def test_to_samples(force_four_alleles, tmp_path):

    sd = convert.to_samples(
        "tests/data/100-samplesx100-sites.vcf",
        "tests/data/100-samplesx100-sites.metadata.tsv",
        str(tmp_path / "tmp.samples"),
        filter_problematic=False,
        force_four_alleles=force_four_alleles,
    )
    n = 98  # We drop 2 samples
    assert sd.num_samples == 98
    assert sd.num_individuals == sd.num_samples
    assert sd.num_sites == 98

    # Check the variant data is converted correctly.
    vcf = cyvcf2.VCF("tests/data/100-samplesx100-sites.vcf")
    vcf_variants = list(vcf)
    assert len(vcf_variants) == sd.num_sites

    for vcf_var, sd_var in zip(vcf_variants, sd.variants()):
        assert vcf_var.POS == sd_var.site.position
        assert vcf_var.REF == sd_var.alleles[0]
        if force_four_alleles:
            assert len(sd_var.alleles) == 4
            assert set(sd_var.alleles) == set("ACGT")
        else:
            assert vcf_var.ALT == list(sd_var.alleles[1:])

    strains = [ind.metadata["strain"] for ind in sd.individuals()]

    vcf_haplotypes = {strain: "" for strain in strains}
    for vcf_variant in vcf_variants:
        # This emits an error from cyvcf2 because of np.str usage
        for sample, base in zip(vcf.samples, vcf_variant.gt_bases):
            if sample in vcf_haplotypes:
                vcf_haplotypes[sample] += base

    sd_haplotypes = {strain: "" for strain in strains}
    for var in sd.variants():
        for strain, gt in zip(strains, var.genotypes):
            sd_haplotypes[strain] += var.alleles[gt]
    assert set(sd_haplotypes.keys()) == set(vcf_haplotypes.keys())
    assert sd_haplotypes == vcf_haplotypes


@pytest.fixture
def sd_fixture(tmp_path):
    return convert.to_samples(
        "tests/data/100-samplesx100-sites.vcf",
        "tests/data/100-samplesx100-sites.metadata.tsv",
        str(tmp_path / "tmp.samples"),
    )


class TestConvertedData:
    def test_individual_metadata_keys(self, sd_fixture):
        for ind in sd_fixture.individuals():
            md = ind.metadata
            assert set(md.keys()) == {
                "Nextstrain_clade",
                "country",
                "date",
                "genbank_accession",
                "host",
                "length",
                "pangolin_lineage",
                "strain",
                "completeness",
                "Nextstrain_clade_usher",
                "pango_lineage_usher",
            }

    def test_individual_metadata_value_types(self, sd_fixture):
        for ind in sd_fixture.individuals():
            md = ind.metadata
            assert isinstance(md["Nextstrain_clade"], str)
            assert isinstance(md["date"], str)
            assert isinstance(md["host"], str)
            assert isinstance(md["length"], float)
            assert isinstance(md["pangolin_lineage"], str)
            assert isinstance(md["strain"], str)
            assert md["country"] is None or isinstance(md["country"], str)
            assert md["genbank_accession"] is None or isinstance(
                md["genbank_accession"], str
            )

    def test_individual_date_format(self, sd_fixture):
        for ind in sd_fixture.individuals():
            date = ind.metadata["date"]
            assert len(date) == 10
            parsed = datetime.date.fromisoformat(date)
            assert parsed.isoformat() == date

    def test_individuals_sorted_by_date(self, sd_fixture):
        dates = [ind.metadata["date"] for ind in sd_fixture.individuals()]
        assert list(sorted(dates)) == dates

    def test_some_variation_exists(self, sd_fixture):
        all_zeros = True
        for _, hap in sd_fixture.haplotypes():
            all_zeros = all_zeros and np.all(hap == 0)
        assert not all_zeros



@pytest.mark.parametrize(
    ["alleles_in", "genotypes_in", "alleles_out", "genotypes_out"],
    [
        (["A", "*"], [0, 0, 1], ["A"], [0, 0, -1]),
        (["A", "*"], [1, 1, 1], ["A"], [-1, -1, -1]),
        (["A", "*", "T"], [0, 1, 2], ["A", "T"], [0, -1, 1]),
        (["A", "T", "*"], [0, 1, 2], ["A", "T"], [0, 1, -1]),
    ],
)
def test_recode_snp_sites_missing_data(
    alleles_in, genotypes_in, alleles_out, genotypes_out
):
    result = convert.recode_snp_sites_missing_data(alleles_in, genotypes_in)
    assert alleles_out == result[0]
    np.testing.assert_array_equal(genotypes_out, result[1])


@pytest.mark.parametrize(
    ["alleles_in", "genotypes_in", "alleles_out", "genotypes_out"],
    [
        (["A", "C"], [0, 1, -1], ["A", "C", "G", "T"], [0, 1, -1]),
        (["A", "T"], [0, 1, -1], ["A", "C", "G", "T"], [0, 3, -1]),
        (["T", "A"], [0, 1], ["T", "A", "C", "G"], [0, 1]),
        (["T", "A"], [1, 0], ["T", "A", "C", "G"], [1, 0]),
        (["T"], [0], ["T", "A", "C", "G"], [0]),
        (["C", "G", "A"], [2, 1, 0], ["C", "A", "G", "T"], [1, 2, 0]),
    ],
)
def test_recode_acgt_alleles(alleles_in, genotypes_in, alleles_out, genotypes_out):
    result = convert.recode_acgt_alleles(alleles_in, genotypes_in)
    assert alleles_out == result[0]
    np.testing.assert_array_equal(genotypes_out, result[1])
