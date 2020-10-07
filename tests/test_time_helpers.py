"""
Provides unit tests for time helper routines.
"""

# License: MIT

import numpy as np
import pandas as pd

import reanalysis_dbns.utils as rdu


def test_datetime_to_string():
    """Test converting date-time objects to string."""

    t = np.datetime64('2010-01-01')
    t_str = rdu.datetime_to_string(t, '%Y%m%d')
    assert t_str == '20100101'

    t = pd.Timestamp('2011-03-15')
    t_str = rdu.datetime_to_string(t, '%Y-%m-%d')
    assert t_str == '2011-03-15'
