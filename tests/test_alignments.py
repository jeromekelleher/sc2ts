import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sc2ts import alignments as sa
from sc2ts import core


class TestAlignmentsStore:
    def test_info(self, fx_alignment_store):
        assert "contains" in str(fx_alignment_store)

    def test_len(self, fx_alignment_store):
        assert len(fx_alignment_store) == 55

    def test_fetch_known(self, fx_alignment_store):
        a = fx_alignment_store["SRR11772659"]
        assert a.shape == (core.REFERENCE_SEQUENCE_LENGTH,)
        assert a[0] == "X"
        assert a[1] == "N"
        assert a[-1] == "N"

    def test_keys(self, fx_alignment_store):
        keys = list(fx_alignment_store.keys())
        assert len(keys) == len(fx_alignment_store)
        assert "SRR11772659" in keys

    def test_in(self, fx_alignment_store):
        assert "SRR11772659" in fx_alignment_store
        assert "NOT_IN_STORE" not in fx_alignment_store


def test_get_gene_coordinates():
    d = core.get_gene_coordinates()
    assert len(d) == 11
    assert d["S"] == (21563, 25385)


class TestEncodeAligment:
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
        a = sa.encode_alignment(h)
        assert_array_equal(a, expected)
        assert_array_equal(h, sa.decode_alignment(a))

    @pytest.mark.parametrize("hap", "RYSWKMDHVN.")
    def test_iupac_uncertain_missing(self, hap):
        h = np.array(list(hap), dtype="U1")
        a = sa.encode_alignment(h)
        assert_array_equal(a, [-1])

    @pytest.mark.parametrize("hap", "XZxz")
    def test_other_missing(self, hap):
        h = np.array(list(hap), dtype="U1")
        a = sa.encode_alignment(h)
        assert_array_equal(a, [-1])

    @pytest.mark.parametrize("hap", "acgt")
    def test_lowercase_nucleotide_missing(self, hap):
        h = np.array(list(hap), dtype="U1")
        a = sa.encode_alignment(h)
        assert_array_equal(a, [-1])

    @pytest.mark.parametrize(
        "a",
        [
            [-2],
            [-3],
            [5],
            [6],
            [0, -2],
        ],
    )
    def test_error__examples(self, a):
        with pytest.raises(ValueError):
            sa.decode_alignment(np.array(a))

    def test_encode_real(self, fx_alignment_store):
        h = fx_alignment_store["SRR11772659"]
        a = sa.encode_alignment(h)
        assert a[0] == -1
        assert a[-1] == -1



# class TestEncodeAndMask:
#     def test_known(self, fx_alignment_store):
#         a = fx_alignment_store["SRR11772659"]
#         ma = sa.encode_and_mask(a)
#         assert ma.original_base_composition == {
#             "T": 9566,
#             "A": 8894,
#             "G": 5850,
#             "C": 5472,
#             "N": 121,
#         }
#         assert ma.original_md5 == "e96feaa72c4f4baba73c2e147ede7502"
#         assert len(ma.masked_sites) == 133
#         assert ma.masked_sites[0] == 1
#         assert ma.masked_sites[-1] == 29903
