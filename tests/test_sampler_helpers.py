"""
Provides unit tests for sampler helper routines.
"""

# License: MIT

from __future__ import absolute_import, division


import numpy as np

import reanalysis_dbns.models as rdm


def test_get_indicator_set_neighborhood():
    """Test getting indicator set neighborhood."""

    k = np.array([0, 1, 0])

    nhd = rdm.get_indicator_set_neighborhood(
        k, allow_exchanges=True, max_nonzero=None,
        include_current_set=False)

    expected_nhd = [
        {'increment': [0], 'decrement': []},
        {'increment': [2], 'decrement': []},
        {'increment': [], 'decrement': [1]},
        {'increment': [0], 'decrement': [1]},
        {'increment': [2], 'decrement': [1]}
    ]

    assert len(nhd) == len(expected_nhd)

    for n in nhd:
        assert n in expected_nhd

    nhd = rdm.get_indicator_set_neighborhood(
        k, allow_exchanges=True, max_nonzero=None,
        include_current_set=True)

    expected_nhd = [
        {'increment': [], 'decrement': []},
        {'increment': [0], 'decrement': []},
        {'increment': [2], 'decrement': []},
        {'increment': [], 'decrement': [1]},
        {'increment': [0], 'decrement': [1]},
        {'increment': [2], 'decrement': [1]}
    ]

    assert len(nhd) == len(expected_nhd)

    for n in nhd:
        assert n in expected_nhd

    nhd = rdm.get_indicator_set_neighborhood(
        k, allow_exchanges=False, max_nonzero=None,
        include_current_set=False)

    expected_nhd = [
        {'increment': [0], 'decrement': []},
        {'increment': [2], 'decrement': []},
        {'increment': [], 'decrement': [1]}
    ]

    assert len(nhd) == len(expected_nhd)

    for n in nhd:
        assert n in expected_nhd

    nhd = rdm.get_indicator_set_neighborhood(
        k, allow_exchanges=True, max_nonzero=1,
        include_current_set=True)

    expected_nhd = [
        {'increment': [], 'decrement': []},
        {'increment': [], 'decrement': [1]},
        {'increment': [0], 'decrement': [1]},
        {'increment': [2], 'decrement': [1]}
    ]

    assert len(nhd) == len(expected_nhd)

    for n in nhd:
        assert n in expected_nhd
