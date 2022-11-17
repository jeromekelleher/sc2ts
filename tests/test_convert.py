import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sc2ts import convert


class TestMaskFlankDeletions:
    @pytest.mark.parametrize(
        ["hap", "expected", "left_masked", "right_masked"],
        [
            ("---AC---", "NNNACNNN", 3, 3),
            ("-GCACC--", "NGCACCNN", 1, 2),
            ("AC---", "ACNNN", 0, 3),
            ("-AC", "NAC", 1, 0),
            ("", "", 0, 0),
            ("-", "N", 1, 0),
            ("A", "A", 0, 0),
            ("CA", "CA", 0, 0),
        ],
    )
    def test_examples(self, hap, expected, left_masked, right_masked):
        hap = np.array(list(hap), dtype="U1")
        expected = np.array(list(expected), dtype="U1")
        left, right = convert.mask_flank_deletions(hap)
        assert left == left_masked
        assert right == right_masked
        assert_array_equal(hap, expected)
