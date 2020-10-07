"""
Provides unit tests for helper computational routines.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import xarray as xr

import reanalysis_dbns.utils as rdu

from reanalysis_dbns.utils.computation import _fix_svd_phases


def test_calc_truncated_svd():
    """Test calculation of truncated SVD."""

    x = np.random.uniform(size=(10, 10))
    k = 5

    u, s, vh = rdu.calc_truncated_svd(x, k)

    expected_u, expected_s, expected_vh = np.linalg.svd(x)
    expected_u, expected_vh = _fix_svd_phases(expected_u, expected_vh)

    expected_u = expected_u[:, :k]
    expected_s = expected_s[:k]
    expected_vh = expected_vh[:k, :]

    assert np.allclose(u, expected_u)
    assert np.allclose(s, expected_s)
    assert np.allclose(vh, expected_vh)


def test_downsample_data():
    """Test downsampling of data."""

    t = pd.date_range(start='1979-01-01', freq='1D', periods=100)

    x = np.arange(100)

    da = xr.DataArray(x, coords={'time': t}, dims=['time'])

    downsampled_da = rdu.downsample_data(da, frequency='monthly')

    assert np.allclose(
        downsampled_da.data, da.resample(time='1MS').mean('time').data)

    t = xr.cftime_range(start='1979-01-01', freq='1D', periods=100,
                        calendar='365_day')

    da = xr.DataArray(x, coords={'time': t}, dims=['time'])

    downsampled_da = rdu.downsample_data(da, frequency='monthly')

    assert np.allclose(
        downsampled_da.data, da.resample(time='1MS').mean('time').data)


def test_select_lat_band():
    """Test selection of latitude band."""

    lat = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    x = xr.DataArray(np.zeros((5, 5)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    band = rdu.select_lat_band(x, lat_bounds=[-10, 10])

    expected_lat = np.array([-10.0, 0.0, 10.0])
    expected_lon = lon

    assert np.allclose(band['lat'], expected_lat)
    assert np.allclose(band['lon'], expected_lon)


def test_select_lon_band():
    """Test selection of longitude band."""

    lat = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    x = xr.DataArray(np.zeros((5, 5)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    band = rdu.select_lon_band(x, lon_bounds=[0, 20])

    expected_lat = lat
    expected_lon = np.array([0.0, 10.0, 20.0])

    assert np.allclose(band['lat'], expected_lat)
    assert np.allclose(band['lon'], expected_lon)


def test_select_latlon_box():
    """Test selection of lat-lon box."""

    lat = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    lon = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    x = xr.DataArray(np.zeros((5, 5)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    box = rdu.select_latlon_box(x, lat_bounds=[-10, 10],
                                lon_bounds=[0.0, 20.0])

    expected_lat = np.array([-10.0, 0.0, 10.0])
    expected_lon = np.array([0.0, 10.0, 20.0])

    assert np.allclose(box['lat'], expected_lat)
    assert np.allclose(box['lon'], expected_lon)


def test_meridional_mean():
    """Test calculation of meridional mean."""

    lat = np.linspace(-90.0, 90.0, 10)
    lon = np.linspace(0.0, 359.0, 10)

    x = xr.DataArray(np.ones((10, 10)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    result = rdu.meridional_mean(x)
    expected = xr.DataArray(np.ones(10, dtype='f8'), coords={'lon': lon},
                            dims=['lon'])

    assert np.allclose(result.data, expected.data)


def test_zonal_mean():
    """Test calculation of zonal mean."""

    lat = np.linspace(-90.0, 90.0, 10)
    lon = np.linspace(0.0, 359.0, 10)

    x = xr.DataArray(np.ones((10, 10)), coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])

    result = rdu.zonal_mean(x)
    expected = xr.DataArray(np.ones(10, dtype='f8'), coords={'lat': lat},
                            dims=['lat'])

    assert np.allclose(result.data, expected.data)


def test_pattern_correlation():
    """Test pattern correlation is correctly calculated."""

    random_seed = 0
    random_state = np.random.default_rng(random_seed)

    x = random_state.uniform(size=(100, 100))
    y = x

    assert rdu.pattern_correlation(x, y, correlation_type='pearsonr') == 1.0
    assert rdu.pattern_correlation(x, -y, correlation_type='pearsonr') == -1.0
    assert rdu.pattern_correlation(x, y, correlation_type='spearmanr') == 1.0
    assert rdu.pattern_correlation(x, -y, correlation_type='spearmanr') == -1.0

    x = xr.DataArray(
        x,
        coords={'lat': np.linspace(0.0, 90.0, 100),
                'lon': np.linspace(0.0, 360.0, 100)},
        dims=['lat', 'lon'])

    y = x

    assert rdu.pattern_correlation(x, y, correlation_type='pearsonr') == 1.0
    assert rdu.pattern_correlation(x, -y, correlation_type='pearsonr') == -1.0
    assert rdu.pattern_correlation(x, y, correlation_type='spearmanr') == 1.0
    assert rdu.pattern_correlation(x, -y, correlation_type='spearmanr') == -1.0

    times = pd.date_range('2010-01-01', periods=4, freq='1D')

    x = np.tile(random_state.uniform(size=(100, 100)), (4, 1, 1))

    x = xr.DataArray(
        x,
        coords={'time': times,
                'lat': np.linspace(0.0, 90.0, 100),
                'lon': np.linspace(0.0, 360.0, 100)},
        dims=['time', 'lat', 'lon'])

    y = x.copy()
    y.loc[{'time': times[1]}] = -y.sel({'time': times[1]})
    y.loc[{'time': times[3]}] = -y.sel({'time': times[3]})

    patt_corrs = rdu.pattern_correlation(x, y, correlation_type='pearsonr')

    assert len(patt_corrs.dims) == 1
    assert patt_corrs.dims[0] == 'time'
    assert np.allclose(patt_corrs.data, np.array([1.0, -1.0, 1.0, -1.0]))


def test_standardized_anomalies():
    """Test calculation of standardized anomalies."""

    t = pd.date_range('1956-01-01', freq='1MS', periods=72)

    lat = np.linspace(-90.0, 90.0, 20)
    lon = np.linspace(0.0, 355.0, 20)

    clim = np.random.normal(size=(12, lat.shape[0], lon.shape[0]))

    anom = np.random.normal(size=(t.shape[0], lat.shape[0], lon.shape[0]))
    anom = anom - np.mean(anom, axis=0, keepdims=True)

    field = np.tile(clim, (6, 1, 1)) + anom

    x = xr.DataArray(field, coords={'time': t, 'lat': lat, 'lon': lon},
                     dims=['time', 'lat', 'lon'])

    std_anom = rdu.standardized_anomalies(x)

    expected_std_anom = (x - x.mean('time')) / x.std('time')

    assert np.allclose(std_anom.data, expected_std_anom.data)

    std_anom = rdu.standardized_anomalies(x, standardize_by='month')

    expected_clim = x.groupby('time.month').mean('time')
    expected_std = x.groupby('time.month').std('time')

    expected_std_anom = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s, x.groupby('time.month'),
        expected_clim, expected_std)

    assert np.allclose(std_anom.data, expected_std_anom.data)
