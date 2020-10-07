"""
Provides helper routines for handling times.
"""

# License: MIT

from __future__ import absolute_import, division


import cftime
import pandas as pd


def datetime_to_string(t, format_string):
    """Convert datetime to string."""

    if isinstance(t, cftime.datetime):
        return t.strftime(format_string)

    return pd.to_datetime(t).strftime(format_string)
