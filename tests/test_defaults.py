"""
Provides unit tests for default coordinate functions.
"""

# License: MIT

from __future__ import absolute_import

import numpy as np
import pytest
import xarray as xr

import reanalysis_dbns.utils as rdu


def test_get_coordinate_standard_name():
    """Test correctly determines coordinate standard name."""

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={'x': np.arange(3), 'y': np.arange(3)},
        dims=['x', 'y'])

    with pytest.raises(ValueError):
        rdu.get_coordinate_standard_name(da, 'unknown_coordinate')


def test_get_lat_name():
    """Test correctly determines latitude coordinate name."""

    supported_names = ['lat', 'latitude', 'g0_lat_1', 'g0_lat_2',
                       'lat_2', 'yt_ocean']

    for lat_name in supported_names:

        da = xr.DataArray(
            np.zeros((3, 3)),
            coords={lat_name: np.arange(3), 'lon': np.arange(3)},
            dims=[lat_name, 'lon'])

        assert rdu.get_lat_name(da) == lat_name

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={'x': np.arange(3), 'y': np.arange(3)},
        dims=['x', 'y'])

    with pytest.raises(ValueError):
        rdu.get_lat_name(da)


def test_get_level_name():
    """Test correctly determines level coordinate name."""

    supported_names = ['level']

    for level_name in supported_names:

        da = xr.DataArray(
            np.zeros((10, 3)),
            coords={'time': np.arange(10), level_name: np.arange(3)},
            dims=['time', level_name])

        assert rdu.get_level_name(da) == level_name

    da = xr.DataArray(
        np.zeros((10, 3)),
        coords={'x': np.arange(10), 'y': np.arange(3)},
        dims=['x', 'y'])

    with pytest.raises(ValueError):
        rdu.get_level_name(da)


def test_get_lon_name():
    """Test correctly determines longitude coordinate name."""

    supported_names = ['lon', 'longitude', 'g0_lon_2', 'g0_lon_3',
                       'lon_2', 'xt_ocean']

    for lon_name in supported_names:

        da = xr.DataArray(
            np.zeros((3, 3)),
            coords={'lat': np.arange(3), lon_name: np.arange(3)},
            dims=['lat', lon_name])

        assert rdu.get_lon_name(da) == lon_name

    da = xr.DataArray(
        np.zeros((3, 3)),
        coords={'x': np.arange(3), 'y': np.arange(3)},
        dims=['x', 'y'])

    with pytest.raises(ValueError):
        rdu.get_lon_name(da)


def test_get_time_name():
    """Test correctly determines time coordinate name."""

    supported_names = ['time', 'initial_time0_hours']

    for time_name in supported_names:

        da = xr.DataArray(
            np.zeros((10, 3)),
            coords={time_name: np.arange(10), 'level': np.arange(3)},
            dims=[time_name, 'level'])

        assert rdu.get_time_name(da) == time_name

    da = xr.DataArray(
        np.zeros((10, 3)),
        coords={'x': np.arange(10), 'y': np.arange(3)},
        dims=['x', 'y'])

    with pytest.raises(ValueError):
        rdu.get_time_name(da)
