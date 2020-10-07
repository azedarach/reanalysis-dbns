"""
Provides routines for calculating indices associated with the IOD.
"""

# License: MIT

from __future__ import absolute_import, division

from copy import deepcopy
import dask.array as da
import numpy as np
import scipy.signal
import xarray as xr

from scipy.fft import irfft, rfft, rfftfreq

import reanalysis_dbns.utils as rdu


def _apply_fft_high_pass_filter(data, fmin, fs=None, workers=None,
                                detrend=True, time_name=None):
    """Apply high-pass filter to FFT of given data.

    Parameters
    ----------
    data : xarray.DataArray
        Data to filter.

    fmin : float
        Lowest frequency in pass band.

    fs : float
        Sampling frequency.

    workers : int
        Number of parallel jobs to use in computing FFT.

    detrend : bool
        If True, remove linear trend from data before computing FFT.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    filtered : xarray.DataArray
        Array containing the high-pass filtered data.
    """

    data = rdu.ensure_data_array(data)

    time_name = time_name if time_name is not None else rdu.get_time_name(data)

    feature_dims = [d for d in data.dims if d != time_name]

    # Handle case in which data is simply a time-series
    if not feature_dims:
        original_shape = None
    else:
        original_shape = [data.sizes[d] for d in feature_dims]

    time_dim_pos = data.get_axis_num(time_name)
    if time_dim_pos != 0:
        data = data.transpose(*([time_name] + feature_dims))

    # Convert to 2D array
    n_samples = data.sizes[time_name]

    if feature_dims:
        n_features = np.product(original_shape)
    else:
        n_features = 1

    flat_data = data.values.reshape((n_samples, n_features))

    rdu.check_fixed_missing_values(flat_data, axis=0)

    valid_data, missing_features = rdu.remove_missing_features(flat_data)
    valid_features = [d for d in range(n_features)
                      if d not in missing_features]

    valid_data = valid_data.swapaxes(0, 1)

    if detrend:
        valid_data = scipy.signal.detrend(
            valid_data, axis=-1, type='linear')

    # Compute spectrum and apply high-pass filter
    spectrum = rfft(valid_data, axis=-1, workers=workers)
    fft_freqs = rfftfreq(n_samples, d=(1.0 / fs))

    filter_mask = fft_freqs < fmin

    spectrum[..., filter_mask] = 0.0

    filtered_valid_data = irfft(
        spectrum, n=n_samples, axis=-1, workers=workers).swapaxes(0, 1)

    if rdu.is_dask_array(flat_data):
        filtered_cols = [None] * n_features
        pos = 0
        for j in range(n_features):
            if j in valid_features:
                filtered_cols[j] = filtered_valid_data[:, pos].reshape(
                    (n_samples, 1))
                pos += 1
            else:
                filtered_cols[j] = da.full((n_samples, 1), np.NaN)

        filtered_data = da.hstack(filtered_cols)
    else:
        filtered_data = np.full((n_samples, n_features), np.NaN)
        filtered_data[:, valid_features] = filtered_valid_data

    if original_shape:
        filtered_data = filtered_data.reshape([n_samples] + original_shape)
        filtered_dims = [time_name] + feature_dims
    else:
        filtered_data = filtered_data.ravel()
        filtered_dims = [time_name]

    filtered_coords = deepcopy(data.coords)

    result = xr.DataArray(
        filtered_data, coords=filtered_coords, dims=filtered_dims)

    if time_dim_pos != 0:
        result = result.transpose(*data.dims)

    return result


def _calculate_weekly_anomaly(data, apply_filter=False, base_period=None,
                              lat_name=None, lon_name=None, time_name=None):
    """Calculate weekly anomalies at each grid point."""

    # Ensure that the data provided is a data array
    data = rdu.ensure_data_array(data)

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(data)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(data)
    time_name = time_name if time_name is not None else rdu.get_time_name(data)

    # Get subset of data to use for computing anomalies
    base_period = rdu.check_base_period(
        data, base_period=base_period, time_name=time_name)

    input_frequency = rdu.detect_frequency(data, time_name=time_name)

    if input_frequency not in ('daily', 'weekly', 'monthly'):
        raise RuntimeError(
            'Can only calculate anomalies for daily, weekly or monthly data')

    if input_frequency == 'daily':
        data = data.resample({time_name: '1W'}).mean()
    elif input_frequency == 'monthly':
        data = data.resample({time_name: '1W'}).interpolate('linear')

    base_period_data = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    weekly_clim = base_period_data.groupby(
        base_period_data[time_name].dt.weekofyear).mean(time_name)

    weekly_anom = data.groupby(data[time_name].dt.weekofyear) - weekly_clim

    if apply_filter:
        weekly_anom = weekly_anom.rolling(
            {time_name: 12}).mean().dropna(time_name, how='all')

        # Approximate sampling frequency
        seconds_per_day = 60 * 60 * 24.0
        fs = 1.0 / (seconds_per_day * 7)

        # Remove all modes with period greater than 7 years
        fmin = 1.0 / (seconds_per_day * 365.25 * 7)

        weekly_anom = _apply_fft_high_pass_filter(
            weekly_anom, fmin=fmin, fs=fs, detrend=True,
            time_name=time_name)

    return weekly_anom


def _calculate_monthly_anomaly(data, apply_filter=False, base_period=None,
                               lat_name=None, lon_name=None, time_name=None):
    """Calculate monthly anomalies at each grid point."""

    # Ensure that the data provided is a data array
    data = rdu.ensure_data_array(data)

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(data)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(data)
    time_name = time_name if time_name is not None else rdu.get_time_name(data)

    # Get subset of data to use for computing anomalies
    base_period = rdu.check_base_period(
        data, base_period=base_period, time_name=time_name)

    input_frequency = rdu.detect_frequency(data, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate anomalies for daily or monthly data')

    if input_frequency == 'daily':
        data = data.resample({time_name: '1MS'}).mean()

    base_period_data = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    monthly_clim = base_period_data.groupby(
        base_period_data[time_name].dt.month).mean(time_name)

    monthly_anom = data.groupby(data[time_name].dt.month) - monthly_clim

    if apply_filter:
        monthly_anom = monthly_anom.rolling(
            {time_name: 3}).mean().dropna(time_name, how='all')

        # Approximate sampling frequency
        seconds_per_day = 60 * 60 * 24.0
        fs = 1.0 / (seconds_per_day * 30)

        # Remove all modes with period greater than 7 years
        fmin = 1.0 / (seconds_per_day * 365.25 * 7)

        monthly_anom = _apply_fft_high_pass_filter(
            monthly_anom, fmin=fmin, fs=fs, detrend=True,
            time_name=time_name)

    return monthly_anom


def dmi(sst, frequency='monthly', apply_filter=False, base_period=None,
        area_weight=False, lat_name=None, lon_name=None, time_name=None):
    """Calculate dipole mode index.

    The DMI introduced by Saji et al is defined as the
    difference in SST anomalies between the western and
    south-eastern tropical Indian ocean, defined as the
    regions 50E - 70E, 10S - 10N and 90E - 110E, 10S - 0N,
    respectively.

    See Saji, N. H. et al, "A dipole mode in the tropical
    Indian Ocean", Nature 401, 360 - 363 (1999).

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : None | 'daily' | 'weekly' | 'monthly'
        Frequency to calculate index at.

    apply_filter : bool
        If True, filter the data by applying a three month
        running mean and remove harmonics with periods longer
        than seven years.

    base_period : list
        Earliest and latest times to use for standardization.

    area_weight : bool
        If True, multiply by cos latitude weights before taking
        mean over region.

    lat_name : str
        Name of latitude coordinate.

    lon_name : str
        Name of longitude coordinate.

    time_name : str
        Name of time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the dipole mode index.
    """

    # Ensure that the data provided is a data array
    sst = rdu.ensure_data_array(sst)

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(sst)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(sst)
    time_name = time_name if time_name is not None else rdu.get_time_name(sst)

    # Get names of spatial coordinates to average over
    averaging_dims = [d for d in sst.dims if d != time_name]

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'weekly', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    base_period = rdu.check_base_period(
        sst, base_period=base_period, time_name=time_name)

    western_bounds = {'lat': [-10.0, 10.0], 'lon': [50.0, 70.0]}
    eastern_bounds = {'lat': [-10.0, 0.0], 'lon': [90.0, 110.0]}

    western_sst = rdu.select_latlon_box(
        sst, lat_bounds=western_bounds['lat'],
        lon_bounds=western_bounds['lon'],
        lat_name=lat_name, lon_name=lon_name)
    eastern_sst = rdu.select_latlon_box(
        sst, lat_bounds=eastern_bounds['lat'],
        lon_bounds=eastern_bounds['lon'],
        lat_name=lat_name, lon_name=lon_name)

    if frequency == 'weekly':
        western_sst_anom = _calculate_weekly_anomaly(
            western_sst, apply_filter=apply_filter, base_period=base_period,
            lat_name=lat_name, lon_name=lon_name, time_name=time_name)
    else:
        western_sst_anom = _calculate_monthly_anomaly(
            western_sst, apply_filter=apply_filter, base_period=base_period,
            lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if area_weight:
        weights = np.cos(np.deg2rad(western_sst_anom[lat_name]))
        western_sst_anom = (western_sst_anom * weights)

    western_sst_anom = western_sst_anom.mean(dim=averaging_dims)

    if frequency == 'weekly':
        eastern_sst_anom = _calculate_weekly_anomaly(
            eastern_sst, apply_filter=apply_filter, base_period=base_period,
            lat_name=lat_name, lon_name=lon_name, time_name=time_name)
    else:
        eastern_sst_anom = _calculate_monthly_anomaly(
            eastern_sst, apply_filter=apply_filter, base_period=base_period,
            lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if area_weight:
        weights = np.cos(np.deg2rad(eastern_sst_anom[lat_name]))
        eastern_sst_anom = (eastern_sst_anom * weights)

    eastern_sst_anom = eastern_sst_anom.mean(dim=averaging_dims)

    index = (western_sst_anom - eastern_sst_anom).rename('dmi')

    if frequency == 'daily':
        index = index.resample({time_name: '1D'}).interpolate('linear')

    index.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    index.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    index.attrs['apply_filter'] = '{}'.format(apply_filter)
    index.attrs['area_weight'] = '{}'.format(area_weight)

    return index


def zwi(uwnd, frequency='monthly', apply_filter=False, base_period=None,
        area_weight=False, lat_name=None, lon_name=None, time_name=None):
    """Calculate dipole mode index.

    The ZWI introduced by Saji et al is defined as the
    area-averaged surface zonal wind anomalies in the
    central and eastern tropical Indian ocean,
    defined as the region 5S - 5N, 70E - 90E.

    See Saji, N. H. et al, "A dipole mode in the tropical
    Indian Ocean", Nature 401, 360 - 363 (1999).

    Parameters
    ----------
    uwnd : xarray.DataArray
        Array containing zonal wind values.

    frequency : None | 'daily' | 'monthly'
        Frequency to calculate index at.

    apply_filter : bool
        If True, filter the data by applying a three month
        running mean and remove harmonics with periods longer
        than seven years.

    base_period : list
        Earliest and latest times to use for standardization.

    area_weight : bool
        If True, multiply by cos latitude weights before taking
        mean over region.

    lat_name : str
        Name of latitude coordinate.

    lon_name : str
        Name of longitude coordinate.

    time_name : str
        Name of time coordinate.

    Returns
    -------
    index : xarray.DataArray
        Array containing the values of the zonal wind index.
    """

    # Ensure that the data provided is a data array
    uwnd = rdu.ensure_data_array(uwnd)

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(uwnd)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(uwnd)
    time_name = time_name if time_name is not None else rdu.get_time_name(uwnd)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    base_period = rdu.check_base_period(
        uwnd, base_period=base_period, time_name=time_name)

    region = {lat_name: slice(-5.0, 5.0),
              lon_name: slice(70.0, 90.0)}

    if uwnd[lat_name].values[0] > uwnd[lat_name].values[-1]:
        region[lat_name] = slice(5.0, -5.0)

    uwnd = uwnd.sel(region)

    uwnd_anom = _calculate_monthly_anomaly(
        uwnd, apply_filter=apply_filter, base_period=base_period,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    if area_weight:
        weights = np.cos(np.deg2rad(uwnd_anom[lat_name]))
        uwnd_anom = (uwnd_anom * weights)

    index = uwnd_anom.mean(dim=[lat_name, lon_name]).rename('zwi')

    if frequency == 'daily':
        index = index.resample({time_name: '1D'}).interpolate('linear')

    index.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    index.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    index.attrs['apply_filter'] = '{}'.format(apply_filter)
    index.attrs['area_weight'] = '{}'.format(area_weight)

    return index
