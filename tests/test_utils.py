import pytest
import numpy as np

import sc2ts
import util


class TestPadSites:
    def check_site_padding(self, ts):
        ts = sc2ts.utils.pad_sites(ts)
        ref = sc2ts.core.get_reference_sequence()
        assert ts.num_sites == len(ref) - 1
        ancestral_state = ts.tables.sites.ancestral_state.view("S1").astype(str)
        assert np.all(ancestral_state == ref[1:])

    def test_initial(self):
        self.check_site_padding(sc2ts.initial_ts())
