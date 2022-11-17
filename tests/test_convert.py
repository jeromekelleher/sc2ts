import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sc2ts import convert


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
        a = convert.encode_alignment(h)
        assert_array_equal(a, expected)
        assert_array_equal(h, convert.decode_alignment(a))

    @pytest.mark.parametrize("hap", "RYSWKMDHVN.")
    def test_iupac_uncertain_missing(self, hap):
        h = np.array(list(hap), dtype="U1")
        a = convert.encode_alignment(h)
        assert_array_equal(a, [-1])

    @pytest.mark.parametrize("hap", "XZxz")
    def test_other_missing(self, hap):
        h = np.array(list(hap), dtype="U1")
        a = convert.encode_alignment(h)
        assert_array_equal(a, [-1])

    @pytest.mark.parametrize("hap", "acgt")
    def test_lowercase_nucleotide_missing(self, hap):
        h = np.array(list(hap), dtype="U1")
        a = convert.encode_alignment(h)
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
    def test_examples(self, a):
        with pytest.raises(ValueError):
            convert.decode_alignment(np.array(a))


class TestMasking:

    # Window size of 1 is weird because we have to have two or more
    # ambiguous characters. That means we only filter if something is
    # surrounded.
    @pytest.mark.parametrize(
        ["hap", "expected", "masked"],
        [
            ("A", "A", 0),
            ("-", "-", 0),
            ("-A-", "-N-", 1),
            ("NAN", "NNN", 1),
            ("---AAC---", "-N-AAC-N-", 2),
        ],
    )
    def test_examples_w1(self, hap, expected, masked):
        hap = np.array(list(hap), dtype="U1")
        a = convert.encode_alignment(hap)
        expected = np.array(list(expected), dtype="U1")
        m = convert.mask_alignment(a, 1)
        assert np.sum(m) == masked
        assert_array_equal(expected, convert.decode_alignment(a))

    @pytest.mark.parametrize(
        ["hap", "expected", "masked"],
        [
            ("A", "A", 0),
            ("-", "-", 0),
            ("--A--", "-NNN-", 3),
            ("---AAAA---", "NNNNAANNNN", 8),
            ("NNNAAAANNN", "NNNNAANNNN", 8),
            ("-N-AAAA-N-", "NNNNAANNNN", 8),
        ],
    )
    def test_examples_w2(self, hap, expected, masked):
        hap = np.array(list(hap), dtype="U1")
        a = convert.encode_alignment(hap)
        expected = np.array(list(expected), dtype="U1")
        m = convert.mask_alignment(a, 2)
        assert np.sum(m) == masked
        assert_array_equal(expected, convert.decode_alignment(a))

    @pytest.mark.parametrize("w", [0, -1, -2])
    def test_bad_window_size(self, w):
        a = np.zeros(2, dtype=np.int8)
        with pytest.raises(ValueError):
            convert.mask_alignment(a, w)


