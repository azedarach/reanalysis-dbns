"""
Provides helper routines for calculations.
"""

# License: MIT

from __future__ import absolute_import, division

import itertools

import cftime
import dask.array
import numpy as np
import pandas as pd
import scipy.linalg as sl
import scipy.stats as ss
import scipy.sparse as sp
import xarray as xr

from .defaults import get_lat_name, get_lon_name, get_time_name
from .validation import (check_base_period, detect_frequency,
                         is_dask_array)


def _fix_svd_phases(u, vh):
    """Impose fixed phase convention on left- and right-singular vectors.

    Given a set of left- and right-singular vectors as the columns of u
    and rows of vh, respectively, imposes the phase convention that for
    each left-singular vector, the element with largest absolute value
    is real and positive.

    Parameters
    ----------
    u : array, shape (M, K)
        Unitary array containing the left-singular vectors as columns.

    vh : array, shape (K, N)
        Unitary array containing the right-singular vectors as rows.

    Returns
    -------
    u_fixed : array, shape (M, K)
        Unitary array containing the left-singular vectors as columns,
        conforming to the chosen phase convention.

    vh_fixed : array, shape (K, N)
        Unitary array containing the right-singular vectors as rows,
        conforming to the chosen phase convention.
    """

    n_cols = u.shape[1]
    max_elem_rows = np.argmax(np.abs(u), axis=0)

    if np.any(np.iscomplexobj(u)):
        phases = np.exp(-1j * np.angle(u[max_elem_rows, range(n_cols)]))
    else:
        phases = np.sign(u[max_elem_rows, range(n_cols)])

    u *= phases
    vh *= phases[:, np.newaxis]

    return u, vh


def calc_truncated_svd(X, k):
    """Calculate the truncated SVD of a 2D array.

    Given an array X with shape (M, N), the SVD of X is computed and the
    leading K = min(k, min(M, N)) singular values are retained.

    The singular values are returned as a 1D array in non-increasing
    order, and the singular vectors are defined such that the array X
    is decomposed as ```u @ np.diag(s) @ vh```.

    Parameters
    ----------
    X : array, shape (M, N)
        The matrix to calculate the SVD of.

    k : integer
        Number of singular values to retain in truncated decomposition.
        If k > min(M, N), then all singular values are retained.

    Returns
    -------
    u : array, shape (M, K)
        Unitary array containing the retained left-singular vectors of X
        as columns.

    s : array, shape (K)
        Array containing the leading K singular vectors of X.

    vh : array, shape (K, N)
        Unitary array containing the retained right-singular vectors of
        X as rows.
    """

    max_modes = min(X.shape)

    if is_dask_array(X):
        dsvd = dask.array.linalg.svd(X)

        u, s, vh = (x.compute() for x in dsvd)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vh = vh[:min(k, len(s)), :]
    elif k < max_modes:
        u, s, vh = sp.linalg.svds(X, k=k)

        # Note that svds returns the singular values with the
        # opposite (i.e., non-decreasing) ordering convention.
        u = u[:, ::-1]
        s = s[::-1]
        vh = vh[::-1]
    else:
        u, s, vh = sl.svd(X, full_matrices=False)

        u = u[:, :min(k, len(s))]
        if k < len(s):
            s = s[:k]
        vh = vh[:min(k, len(s)), :]

    # Impose a fixed phase convention on the singular vectors
    # to avoid phase ambiguity.
    u, vh = _fix_svd_phases(u, vh)

    return u, s, vh


def _get_frequency_offset_alias(frequency):
    """Get pandas offset alias for given frequency."""

    if frequency == 'hourly':
        return '1H'

    if frequency == 'daily':
        return '1D'

    if frequency == 'monthly':
        return '1MS'

    if frequency == 'yearly':
        return '1A'

    return 'QS-DEC'


def downsample_data(da, frequency=None, time_name=None):
    """Perform down-sampling of data."""

    if frequency is None:
        return da

    if frequency not in ('daily', 'monthly', 'seasonal'):
        raise ValueError('Unrecognized down-sampling frequency %r' %
                         frequency)

    time_name = time_name if time_name is not None else get_time_name(da)

    current_frequency = detect_frequency(da, time_name=time_name)
    current_frequency = _get_frequency_offset_alias(current_frequency)

    target_frequency = _get_frequency_offset_alias(frequency)

    if isinstance(da[time_name].values[0], cftime.datetime):
        start_time = pd.Timestamp(da[time_name].values[0].strftime('%Y%m%d'))
    else:
        start_time = pd.to_datetime(da[time_name].values[0])

    current_timestep = (start_time +
                        pd.tseries.frequencies.to_offset(current_frequency))
    target_timestep = (start_time +
                       pd.tseries.frequencies.to_offset(target_frequency))

    if target_timestep < current_timestep:
        raise ValueError('Downsampling frequency appears to be higher'
                         ' than current frequency')

    return da.resample({time_name: target_frequency}).mean(time_name)


def _check_correlation_type(correlation_type):
    """Check given correlation method is valid."""

    valid_corr_types = ('pearsonr', 'spearmanr')

    if correlation_type is None:
        correlation_type = valid_corr_types[0]

    if correlation_type not in valid_corr_types:
        raise ValueError(
            "Unrecognized correlation type '%r'; "
            "must be one of %r" % (correlation_type, valid_corr_types))

    return correlation_type


def _check_pattern_correlation_core_dims(x, y, core_dims=None):
    """Get dimension names for individual patterns."""

    if core_dims is None:

        lat_name = get_lat_name(x)
        lon_name = get_lon_name(x)

        if lat_name not in y.dims or lon_name not in y.dims:
            raise ValueError('Could not determine core dimensions')

        core_dims = [lat_name, lon_name]

    else:

        for d in core_dims:

            if d not in x.dims or d not in y.dims:
                raise ValueError(
                    "Core dimension '%s' not found in given datasets" % d)

    return core_dims


def _pattern_correlation_flat(x, y, correlation_type=None, nan_policy='omit'):
    """Get pattern correlation between flat arrays."""

    correlation_type = _check_correlation_type(correlation_type)

    if x.shape != y.shape:
        raise ValueError(
            'Input arrays x and y do not have the same shape '
            '(got x.shape=%r and y.shape=%r)' % (x.shape, y.shape))

    x_flat = x.ravel()
    y_flat = y.ravel()

    if nan_policy == 'omit':
        mask = np.logical_and(np.isfinite(x_flat), np.isfinite(y_flat))
        x_flat = x_flat[mask]
        y_flat = y_flat[mask]
    elif nan_policy == 'raise':
        if np.any(np.logical_or(~np.isfinite(x_flat), ~np.isfinite(y_flat))):
            raise ValueError('Input arrays contain missing values')

    if correlation_type == 'pearsonr':

        patt_corr = ss.pearsonr(x_flat, y_flat)[0]

    elif correlation_type == 'spearmanr':

        patt_corr = ss.spearmanr(x_flat, y_flat)[0]

    else:
        raise NotImplementedError(
            "Pattern correlation with correlation type '%s' "
            "not implemented" % correlation_type)

    return patt_corr


def pattern_correlation(x, y, correlation_type=None, core_dims=None):
    """Calculate pattern correlation between two series of maps."""

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return _pattern_correlation_flat(
            x, y, correlation_type=correlation_type)

    core_dims = _check_pattern_correlation_core_dims(
        x, y, core_dims=core_dims)

    x_non_core_dims = sorted([d for d in x.dims if d not in core_dims])
    y_non_core_dims = sorted([d for d in y.dims if d not in core_dims])

    if x_non_core_dims != y_non_core_dims:
        raise ValueError(
            'Non-core dimensions do not match between input arrays '
            '(got x.dims=%r and y.dims=%r)' % (x.dims, y.dims))

    if not x_non_core_dims and not y_non_core_dims:
        return _pattern_correlation_flat(
            x.data, y.data, correlation_type=correlation_type)

    output_shapes = []
    for d in x_non_core_dims:
        if x.sizes[d] != y.sizes[d]:
            raise ValueError(
                "Size of dimension '%s' does not match in input arrays "
                "(got x.sizes['%s']=%d and y.sizes['%s']=%d)" %
                (d, d, x.sizes[d], d, y.sizes[d]))

        output_shapes.append(x.sizes[d])

    patt_corr_vals = xr.DataArray(
        np.zeros(output_shapes, dtype='f8'),
        coords={d: x[d] for d in x_non_core_dims},
        dims=x_non_core_dims)

    indices = itertools.product(*[range(s) for s in output_shapes])
    for index in indices:
        idx = {d: x[d].isel({d: index[i]})
               for i, d in enumerate(x_non_core_dims)}

        pc = _pattern_correlation_flat(
            x.sel(idx).data, y.sel(idx).data,
            correlation_type=correlation_type)

        patt_corr_vals.loc[idx] = pc

    return patt_corr_vals


def _get_longitude_mask(lon, lon_bounds):
    """Get mask for specified longitudes."""

    # Normalize longitudes to be in the interval [0, 360)
    lon = lon % 360
    lon_bounds = lon_bounds % 360

    if lon_bounds[0] > lon_bounds[1]:
        return (((lon >= lon_bounds[0]) & (lon <= 360)) |
                ((lon >= 0) & (lon <= lon_bounds[1])))

    return (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])


def select_lat_band(data, lat_bounds, lat_name=None):
    """Select data within given latitude bounds."""

    lat_name = lat_name if lat_name is not None else get_lat_name(data)

    if lat_bounds is None or len(lat_bounds) != 2:
        raise ValueError('Latitude bounds must be a list of length 2')

    lat_bounds = np.array(sorted(lat_bounds))

    return data.where(
        (data[lat_name] >= lat_bounds[0]) &
        (data[lat_name] <= lat_bounds[1]), drop=True)


def select_lon_band(data, lon_bounds, lon_name=None):
    """Select data within given longitude bounds."""

    lon_name = lon_name if lon_name is not None else get_lon_name(data)

    if lon_bounds is None or len(lon_bounds) != 2:
        raise ValueError('Longitude bounds must be a list of length 2')

    lon_bounds = np.array(lon_bounds)

    lon_mask = _get_longitude_mask(
        data[lon_name], lon_bounds=lon_bounds)

    return data.where(lon_mask, drop=True)


def select_latlon_box(data, lat_bounds, lon_bounds,
                      lat_name=None, lon_name=None):
    """Select data in given latitude-longitude box."""

    region_data = select_lat_band(data, lat_bounds, lat_name=lat_name)
    return select_lon_band(region_data, lon_bounds, lon_name=lon_name)


def meridional_mean(data, lat_bounds=None, latitude_weight=False,
                    lat_name=None):
    """Calculate meridional mean between latitude bounds."""

    lat_name = lat_name if lat_name is not None else get_lat_name(data)

    if latitude_weight:
        weights = np.cos(np.deg2rad(data[lat_name]))
        data = data * weights

    if lat_bounds is None:
        return data.mean(dim=[lat_name])

    lat_bounds = sorted(lat_bounds)
    if data[lat_name][0] > data[lat_name][-1]:
        lat_bounds = lat_bounds[::-1]

    lat_slice = slice(lat_bounds[0], lat_bounds[1])

    return data.sel({lat_name: lat_slice}).mean(dim=[lat_name])


def zonal_mean(data, lon_bounds=None, lon_name=None):
    """Calculate zonal mean between longitude bounds."""

    lon_name = lon_name if lon_name is not None else get_lon_name(data)

    if lon_bounds is None:
        return data.mean(dim=[lon_name])

    region_data = select_lon_band(data, lon_bounds, lon_name=lon_name)

    return region_data.mean(dim=[lon_name])


def standardized_anomalies(da, base_period=None, standardize_by=None,
                           time_name=None):
    """Calculate standardized anomalies."""

    time_name = time_name if time_name is not None else get_time_name(da)

    base_period = check_base_period(da, base_period=base_period,
                                    time_name=time_name)

    base_period_da = da.where(
        (da[time_name] >= base_period[0]) &
        (da[time_name] <= base_period[1]), drop=True)

    if standardize_by == 'dayofyear':
        base_period_groups = base_period_da[time_name].dt.dayofyear
        groups = da[time_name].dt.dayofyear
    elif standardize_by == 'month':
        base_period_groups = base_period_da[time_name].dt.month
        groups = da[time_name].dt.month
    elif standardize_by == 'season':
        base_period_groups = base_period_da[time_name].dt.season
        groups = da[time_name].dt.season
    else:
        base_period_groups = None
        groups = None

    if base_period_groups is not None:

        clim_mean = base_period_da.groupby(base_period_groups).mean(time_name)
        clim_std = base_period_da.groupby(base_period_groups).std(time_name)

        std_anom = xr.apply_ufunc(
            lambda x, m, s: (x - m) / s, da.groupby(groups),
            clim_mean, clim_std, dask='allowed')

    else:

        clim_mean = base_period_da.mean(time_name)
        clim_std = base_period_da.std(time_name)

        std_anom = ((da - clim_mean) / clim_std)

    return std_anom
