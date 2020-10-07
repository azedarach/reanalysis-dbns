"""
Provides routines for computing MJO indices.
"""

# License: MIT

from copy import deepcopy

import cftime
import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


from scipy.fft import irfft, rfft
from scipy.stats import linregress

import reanalysis_dbns.utils as rdu


def _check_consistent_coordinate_names(
        *arrays, lat_name=None, lon_name=None, time_name=None):
    """Check all arrays have consistent coordinate names."""

    lat_name = (lat_name if lat_name is not None else
                rdu.get_lat_name(arrays[0]))
    lon_name = (lon_name if lon_name is not None else
                rdu.get_lon_name(arrays[0]))
    time_name = (time_name if time_name is not None else
                 rdu.get_time_name(arrays[0]))

    for arr in arrays:
        if time_name not in arr.dims:
            raise ValueError(
                "Could not find time coordinate '%s' in input" % time_name)

        if lat_name not in arr.dims:
            raise ValueError(
                "Could not find latitude coordinate '%s' in input" % lat_name)

        if lon_name not in arr.dims:
            raise ValueError(
                "Could not find longitude coordinate '%s' in input" % lon_name)


def _compute_smooth_seasonal_cycle(seasonal_cycle, n_harmonics=4,
                                   workers=None, sample_dim='dayofyear'):
    """Calculate smooth seasonal cycle based on lowest order harmonics."""

    # Convert to flat array for performing FFT
    feature_dims = [d for d in seasonal_cycle.dims if d != sample_dim]

    if not feature_dims:
        original_shape = None
    else:
        original_shape = [seasonal_cycle.sizes[d] for d in feature_dims]

    sample_dim_pos = seasonal_cycle.get_axis_num(sample_dim)
    if sample_dim_pos != 0:
        seasonal_cycle = seasonal_cycle.transpose(
            *([sample_dim] + feature_dims))

    n_samples = seasonal_cycle.sizes[sample_dim]

    if feature_dims:
        n_features = np.product(original_shape)
    else:
        n_features = 1

    flat_data = seasonal_cycle.values.reshape((n_samples, n_features))

    rdu.check_fixed_missing_values(flat_data, axis=0)

    valid_data, missing_features = rdu.remove_missing_features(flat_data)
    valid_features = [d for d in range(n_features)
                      if d not in missing_features]

    valid_data = valid_data.swapaxes(0, 1)

    if rdu.is_dask_array(valid_data):

        spectrum = da.fft.rfft(valid_data, axis=-1)

        def _filter(freqs):
            n_freqs = freqs.shape[0]
            return da.concatenate(
                [freqs[:n_harmonics], da.zeros(n_freqs - n_harmonics)])

        filtered_spectrum = da.apply_along_axis(_filter, axis=-1, arr=spectrum)

        filtered_valid_data = da.fft.irfft(
            filtered_spectrum, n=n_samples, axis=-1).swapaxes(0, 1)

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
        spectrum = rfft(valid_data, axis=-1, workers=workers)
        spectrum[..., n_harmonics:] = 0.0

        filtered_valid_data = irfft(
            spectrum, n=n_samples, axis=-1, workers=workers).swapaxes(0, 1)

        filtered_data = np.full((n_samples, n_features), np.NaN)
        filtered_data[:, valid_features] = filtered_valid_data

    if original_shape:
        filtered_data = filtered_data.reshape([n_samples] + original_shape)
        filtered_dims = [sample_dim] + feature_dims
    else:
        filtered_data = filtered_data.ravel()
        filtered_dims = [sample_dim]

    filtered_coords = deepcopy(seasonal_cycle.coords)

    smooth_seasonal_cycle = xr.DataArray(
        filtered_data, coords=filtered_coords, dims=filtered_dims)

    if sample_dim_pos != 0:
        smooth_seasonal_cycle = smooth_seasonal_cycle.transpose(
            *seasonal_cycle.dims)

    return smooth_seasonal_cycle


def _subtract_seasonal_cycle(data, base_period=None,
                             n_harmonics=4, workers=None, time_name=None):
    """Remove time mean and leading harmonics of annual cycle."""

    # Ensure input data is a data array.
    data = rdu.ensure_data_array(data)

    time_name = time_name if time_name is not None else rdu.get_time_name(data)

    base_period = rdu.check_base_period(data, base_period=base_period,
                                        time_name=time_name)

    # Restrict to base period for computing annual cycle.
    base_period_data = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    input_frequency = rdu.detect_frequency(data, time_name=time_name)

    if input_frequency == 'daily':
        base_period_groups = base_period_data[time_name].dt.dayofyear
        groups = data[time_name].dt.dayofyear
        sample_dim = 'dayofyear'
    elif input_frequency == 'monthly':
        base_period_groups = base_period_data[time_name].dt.month
        groups = data[time_name].dt.month
        sample_dim = 'month'
    else:
        raise ValueError("Unsupported sampling rate '%r'" % input_frequency)

    seasonal_cycle = base_period_data.groupby(
        base_period_groups).mean(time_name)

    smooth_seasonal_cycle = _compute_smooth_seasonal_cycle(
        seasonal_cycle, n_harmonics=n_harmonics, sample_dim=sample_dim)

    # Subtract seasonal cycle and leading higher harmonics
    return data.groupby(groups) - smooth_seasonal_cycle


def _floor_to_daily_resolution(t):
    """Ensure data is given at daily resolution."""

    if isinstance(t, np.ndarray):
        t = t.item()

    if isinstance(t, np.datetime64):

        return np.datetime64(pd.to_datetime(t).floor('D'))

    elif isinstance(t, cftime.datetime):

        return rdu.floor_cftime(t, freq='D')

    raise ValueError("Unsupported time value '%r'" % t)


def _subtract_monthly_linear_regression(data, predictor,
                                        base_period=None, time_name=None):
    """Subtract regressed values of predictand at each grid point."""

    data = rdu.ensure_data_array(data)
    predictor = rdu.ensure_data_array(predictor)

    time_name = time_name if time_name is not None else rdu.get_time_name(data)

    if time_name not in predictor.dims:
        raise ValueError(
            "Could not find time dimension '%s' in predictor array" %
            time_name)

    base_period = rdu.check_base_period(data, base_period=base_period,
                                        time_name=time_name)

    outcome_frequency = rdu.detect_frequency(data, time_name=time_name)

    if outcome_frequency != 'daily':
        raise ValueError('Outcome data must be daily resolution')

    predictor_frequency = rdu.detect_frequency(predictor, time_name=time_name)

    if predictor_frequency != 'daily':
        predictor = predictor.resample(
            {time_name: '1D'}).interpolate('linear')

    # Align daily mean predictor times with outcome times
    outcomes_start_date = _floor_to_daily_resolution(
        data[time_name].min().values)
    outcomes_end_date = _floor_to_daily_resolution(
        data[time_name].max().values)

    predictor = predictor.where(
        (predictor[time_name] >= outcomes_start_date) &
        (predictor[time_name] <= outcomes_end_date), drop=True)

    if predictor.sizes[time_name] != data.sizes[time_name]:
        raise RuntimeError('Incorrect number of predictor values given')

    predictor[time_name] = data[time_name]

    base_period_outcome = data.where(
        (data[time_name] >= base_period[0]) &
        (data[time_name] <= base_period[1]), drop=True)

    base_period_predictor = predictor.where(
        (predictor[time_name] >= base_period[0]) &
        (predictor[time_name] <= base_period[1]), drop=True)

    # Ensure time-points are aligned.
    base_period_outcome = base_period_outcome.resample(
        {time_name: '1D'}).mean()
    base_period_predictor = base_period_predictor.resample(
        {time_name: '1D'}).mean()

    # Fit seasonal regression relationship for each month
    slopes = xr.zeros_like(
        base_period_outcome.isel(
            {time_name: 0}, drop=True)).expand_dims(
                {time_name: pd.date_range('2000-01-01', '2001-01-01',
                                          freq='1MS')})
    intercepts = xr.zeros_like(
        base_period_outcome.isel(
            {time_name: 0}, drop=True)).expand_dims(
                {time_name: pd.date_range('2000-01-01', '2001-01-01',
                                          freq='1MS')})

    if dask.is_dask_collection(base_period_outcome):
        _apply_along_axis = da.apply_along_axis
        apply_kwargs = dict(dtype=base_period_outcome.dtype, shape=(2,))
    else:
        _apply_along_axis = np.apply_along_axis
        apply_kwargs = {}

    def _fit_along_axis(outcomes, predictor, axis=-1, **kwargs):
        def _fit(y):
            return np.array(linregress(predictor, y)[:2])
        return _apply_along_axis(_fit, axis=axis, arr=outcomes, **kwargs)

    for month in range(1, 13):

        predictor_vals = base_period_predictor.where(
            base_period_predictor[time_name].dt.month == month, drop=True)
        outcome_vals = base_period_outcome.where(
            base_period_outcome[time_name].dt.month == month, drop=True)

        fit_result = xr.apply_ufunc(
            _fit_along_axis, outcome_vals, predictor_vals,
            input_core_dims=[[time_name], [time_name]],
            output_core_dims=[['fit_coefficients']],
            dask='allowed', kwargs=apply_kwargs,
            output_dtypes=[outcome_vals.dtype])

        slopes = xr.where(slopes[time_name].dt.month == month,
                          fit_result.isel(fit_coefficients=0),
                          slopes)
        intercepts = xr.where(intercepts[time_name].dt.month == month,
                              fit_result.isel(fit_coefficients=1),
                              intercepts)

    slopes = slopes.resample(
        {time_name: '1D'}).interpolate('linear').isel(
            {time_name: slice(0, -1)})
    slopes = slopes.assign_coords(
        {time_name: slopes[time_name].dt.dayofyear}).rename(
            {time_name: 'dayofyear'})

    intercepts = intercepts.resample(
        {time_name: '1D'}).interpolate('linear').isel(
            {time_name: slice(0, -1)})
    intercepts = intercepts.assign_coords(
        {time_name: intercepts[time_name].dt.dayofyear}).rename(
            {time_name: 'dayofyear'})

    subtracted = data.groupby(data[time_name].dt.dayofyear) - intercepts

    def _subtract(x):
        day = x[time_name].dt.dayofyear[0]
        slope = slopes.sel(dayofyear=day)
        predictor_values = predictor.where(
            predictor[time_name].dt.dayofyear == day, drop=True)
        return x - predictor_values * slope

    subtracted = subtracted.groupby(
        subtracted[time_name].dt.dayofyear).map(_subtract)

    return subtracted


def _subtract_running_mean(data, n_steps, time_name=None):
    """Subtract running mean from data."""

    data = rdu.ensure_data_array(data)

    time_name = time_name if time_name is not None else rdu.get_time_name(data)

    if dask.is_dask_collection(data):
        chunk_sizes = data.data.chunksize

        if chunk_sizes[data.get_axis_num(time_name)] < n_steps:
            data = data.chunk({time_name: n_steps})

    running_mean = data.rolling({time_name: n_steps}).mean()

    return data - running_mean


def wh_rmm_anomalies(olr, u850, u200, base_period=None,
                     enso_index=None, subtract_running_mean=True,
                     n_running_mean_steps=120, lat_bounds=None,
                     lat_name=None, lon_name=None, time_name=None):
    """Calculate anomaly inputs for Wheeler-Hendon MJO index.

    Parameters
    ----------
    olr : xarray.DataArray
        Array containing top-of-atmosphere OLR values.

    u850 : xarray.DataArray
        Array containing values of zonal wind at 850 hPa.

    u200 : xarray.DataArray
        Array containing values of zonal wind at 200 hPa.

    base_period : list
        Earliest and latest times to use for standardization.

    enso_index : xarray.DataArray, optional
        If given, an array containing an ENSO index time-series
        used to remove ENSO related variability (e.g., monthly
        SST1 index values).

    subtract_running_mean : boolean, optional
        If True (default), subtract running mean of the previous
        days for each time-step.

    n_running_mean_steps : integer
        Number of days to include in running mean, by default 120.

    lat_bounds : list
        Latitude range over which to compute meridional mean,
        by default 15S to 15N.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    olr_anom : xarray.DataArray
        Array containing the values of the OLR anomalies.

    u850_anom : xarray.DataArray
        Array containing the values of the 850 hPa zonal wind
        anomalies.

    u200_anom : xarray.DataArray
        Array containing the values of the 200 hPa zonal wind anomalies.
    """

    # Ensure all inputs are data arrays.
    olr = rdu.ensure_data_array(olr)
    u850 = rdu.ensure_data_array(u850)
    u200 = rdu.ensure_data_array(u200)

    if enso_index is not None:
        enso_index = rdu.ensure_data_array(enso_index)

    # Index is defined in terms of daily resolution data.
    if not rdu.is_daily_data(olr):
        raise ValueError('Initial OLR data must be daily resolution')

    if not rdu.is_daily_data(u850):
        raise ValueError('Initial u850 data must be daily resolution')

    if not rdu.is_daily_data(u200):
        raise ValueError('Initial u200 data must be daily resolution')

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(olr)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(olr)
    time_name = time_name if time_name is not None else rdu.get_time_name(olr)

    # For convenience, ensure all inputs have same coordinate names.
    _check_consistent_coordinate_names(
        olr, u850, u200, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name)

    # Restrict to equatorial region.
    if lat_bounds is None:
        lat_bounds = [-15.0, 15.0]
    else:
        if len(lat_bounds) != 2:
            raise ValueError(
                'Latitude boundaries must be a 2 element list, but got %r' %
                lat_bounds)
        lat_bounds = sorted(lat_bounds)

    olr = olr.where(
        (olr[lat_name] >= lat_bounds[0]) &
        (olr[lat_name] <= lat_bounds[1]), drop=True).squeeze()
    u850 = u850.where(
        (u850[lat_name] >= lat_bounds[0]) &
        (u850[lat_name] <= lat_bounds[1]), drop=True).squeeze()
    u200 = u200.where(
        (u200[lat_name] >= lat_bounds[0]) &
        (u200[lat_name] <= lat_bounds[1]), drop=True).squeeze()

    # Ensure inputs cover same time period.
    start_time = max(
        [olr[time_name].min(), u850[time_name].min(),
         u200[time_name].min()])
    end_time = min(
        [olr[time_name].max(), u850[time_name].max(),
         u200[time_name].max()])

    def _restrict_time_period(data):
        return data.where(
            (data[time_name] >= start_time) &
            (data[time_name] <= end_time), drop=True)

    olr = _restrict_time_period(olr)
    u850 = _restrict_time_period(u850)
    u200 = _restrict_time_period(u200)

    base_period = rdu.check_base_period(olr, base_period=base_period,
                                        time_name=time_name)

    # Remove seasonal cycle
    olr_anom = _subtract_seasonal_cycle(
        olr, base_period=base_period, time_name=time_name)
    u850_anom = _subtract_seasonal_cycle(
        u850, base_period=base_period, time_name=time_name)
    u200_anom = _subtract_seasonal_cycle(
        u200, base_period=base_period, time_name=time_name)

    # If required, remove ENSO variability
    if enso_index is not None:

        olr_anom = _subtract_monthly_linear_regression(
            olr_anom, predictor=enso_index,
            base_period=base_period, time_name=time_name)
        u850_anom = _subtract_monthly_linear_regression(
            u850_anom, predictor=enso_index,
            base_period=base_period, time_name=time_name)
        u200_anom = _subtract_monthly_linear_regression(
            u200_anom, predictor=enso_index,
            base_period=base_period, time_name=time_name)

    # If required, apply running mean
    if subtract_running_mean:

        olr_anom = _subtract_running_mean(
            olr_anom, n_steps=n_running_mean_steps,
            time_name=time_name).dropna(
                time_name, how='all')
        u850_anom = _subtract_running_mean(
            u850_anom, n_steps=n_running_mean_steps,
            time_name=time_name).dropna(
                time_name, how='all')
        u200_anom = _subtract_running_mean(
            u200_anom, n_steps=n_running_mean_steps,
            time_name=time_name).dropna(
                time_name, how='all')

    olr_anom = rdu.meridional_mean(olr_anom, lat_bounds=lat_bounds,
                                   lat_name=lat_name)
    u850_anom = rdu.meridional_mean(u850_anom, lat_bounds=lat_bounds,
                                    lat_name=lat_name)
    u200_anom = rdu.meridional_mean(u200_anom, lat_bounds=lat_bounds,
                                    lat_name=lat_name)

    return olr_anom, u850_anom, u200_anom


def wh_rmm_eofs(olr_anom, u850_anom, u200_anom, n_modes=2,
                lon_name=None, time_name=None):
    """Calculate combined EOFs of OLR and zonal wind anomalies.

    Parameters
    ----------
    olr_anom : xarray.DataArray
        Array containing values of OLR anomalies.

    u850_anom : xarray.DataArray
        Array containing values of 850 hPa zonal wind anomalies.

    u200_anom : xarray.DataArray
        Array containing values of 200 hPa zonal wind anomalies.

    n_modes : integer
        Number of EOF modes to calculate. If None, by default only
        the leading two modes are computed.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    olr_eofs : xarray.Dataset
        Dataset containing the calculated OLR EOFs.

    u850_eofs : xarray.Dataset
        Dataset containing the calculated 850 hPa zonal wind EOFs.

    u200_eofs : xarray.Dataset
        Dataset containing the calculated 200 hPa zonal wind EOFs.
    """

    olr_anom = rdu.ensure_data_array(olr_anom)
    u850_anom = rdu.ensure_data_array(u850_anom)
    u200_anom = rdu.ensure_data_array(u200_anom)

    lon_name = (lon_name if lon_name is not None else
                rdu.get_lon_name(olr_anom))
    time_name = (time_name if time_name is not None else
                 rdu.get_time_name(olr_anom))

    if lon_name not in u850_anom.dims or lon_name not in u200_anom.dims:
        raise ValueError(
            "Could not find longitude coordinate '%s' in zonal wind data" %
            time_name)

    if time_name not in u850_anom.dims or time_name not in u200_anom.dims:
        raise ValueError(
            "Could not find time coordinate '%s' in zonal wind data" %
            time_name)

    if n_modes is None:
        n_modes = 2

    if not rdu.is_integer(n_modes) or n_modes < 1:
        raise ValueError('Number of modes must be a positive integer.')

    # Ensure input to EOFs has zero-mean
    olr_anom = olr_anom - olr_anom.mean(time_name)
    u850_anom = u850_anom - u850_anom.mean(time_name)
    u200_anom = u200_anom - u200_anom.mean(time_name)

    olr_eofs, u850_eofs, u200_eofs = rdu.eofs(
        olr_anom, u850_anom, u200_anom,
        n_modes=n_modes, sample_dim=time_name)

    # Define the leading mode such that it corresponds to positive zonal
    # wind anomalies over the Indian ocean sector and negative zonal wind
    # anomalies over the Pacific sector
    lon_bounds = [50.0, 120.0]

    leading_u850_eof = u850_eofs['EOFs'].sel(mode=0)
    u850_max_anom = leading_u850_eof.where(
        ((leading_u850_eof[lon_name] % 360) >= lon_bounds[0]) &
        ((leading_u850_eof[lon_name] % 360) <= lon_bounds[1])).max().item()
    u850_min_anom = leading_u850_eof.where(
        ((leading_u850_eof[lon_name] % 360) >= lon_bounds[0]) &
        ((leading_u850_eof[lon_name] % 360) <= lon_bounds[1])).min().item()

    if np.abs(u850_max_anom) > np.abs(u850_min_anom):
        must_flip = u850_max_anom < 0
    else:
        must_flip = u850_min_anom < 0

    if must_flip:
        olr_eofs['EOFs'] = xr.where(
            olr_eofs['mode'] == 0, -olr_eofs['EOFs'], olr_eofs['EOFs'])
        olr_eofs['PCs'] = xr.where(
            olr_eofs['mode'] == 0, -olr_eofs['PCs'], olr_eofs['PCs'])

        u850_eofs['EOFs'] = xr.where(
            u850_eofs['mode'] == 0, -u850_eofs['EOFs'], u850_eofs['EOFs'])
        u850_eofs['PCs'] = xr.where(
            u850_eofs['mode'] == 0, -u850_eofs['PCs'], u850_eofs['PCs'])

        u200_eofs['EOFs'] = xr.where(
            u200_eofs['mode'] == 0, -u200_eofs['EOFs'], u200_eofs['EOFs'])
        u200_eofs['PCs'] = xr.where(
            u200_eofs['mode'] == 0, -u200_eofs['PCs'], u200_eofs['PCs'])

    # Similarly, define the second leading mode to have
    # positive zonal wind anomalies over the maritime continent.
    if n_modes > 1:
        lon_bounds = [100.0, 210.0]

        second_u850_eof = u850_eofs['EOFs'].sel(mode=1)

        u850_max_anom = second_u850_eof.where(
            ((second_u850_eof[lon_name] % 360) >= lon_bounds[0]) &
            ((second_u850_eof[lon_name] % 360) <= lon_bounds[1])).max().item()
        u850_min_anom = second_u850_eof.where(
            ((second_u850_eof[lon_name] % 360) >= lon_bounds[0]) &
            ((second_u850_eof[lon_name] % 360) <= lon_bounds[1])).min().item()

        if np.abs(u850_max_anom) > np.abs(u850_min_anom):
            must_flip = u850_max_anom < 0
        else:
            must_flip = u850_min_anom < 0

        if must_flip:
            olr_eofs['EOFs'] = xr.where(
                olr_eofs['mode'] == 1, -olr_eofs['EOFs'], olr_eofs['EOFs'])
            olr_eofs['PCs'] = xr.where(
                olr_eofs['mode'] == 1, -olr_eofs['PCs'], olr_eofs['PCs'])

            u850_eofs['EOFs'] = xr.where(
                u850_eofs['mode'] == 1, -u850_eofs['EOFs'], u850_eofs['EOFs'])
            u850_eofs['PCs'] = xr.where(
                u850_eofs['mode'] == 1, -u850_eofs['PCs'], u850_eofs['PCs'])

            u200_eofs['EOFs'] = xr.where(
                u200_eofs['mode'] == 1, -u200_eofs['EOFs'], u200_eofs['EOFs'])
            u200_eofs['PCs'] = xr.where(
                u200_eofs['mode'] == 1, -u200_eofs['PCs'], u200_eofs['PCs'])

    return olr_eofs, u850_eofs, u200_eofs


def wh_rmm(olr, u850, u200, enso_index=None, base_period=None,
           subtract_running_mean=True, n_running_mean_steps=120,
           n_modes=2, lat_bounds=None, lat_name=None, lon_name=None,
           time_name=None):
    """Calculates the Wheeler-Hendon MJO index.

    See Wheeler, M. C. and Hendon, H., "An All-Season Real-Time Multivariate
    MJO Index: Development of an Index for Monitoring and Prediction",
    Monthly Weather Review 132, 1917 - 1932 (2004),
    doi:

    Parameters
    ----------
    olr : xarray.DataArray
        Array containing top-of-atmosphere OLR values.

    u850 : xarray.DataArray
        Array containing values of zonal wind at 850 hPa.

    u200 : xarray.DataArray
        Array containing values of zonal wind at 200 hPa.

    enso_index : xarray.DataArray, optional
        If given, an array containing an ENSO index time-series
        used to remove ENSO related variability (e.g., monthly
        SST1 index values).

    base_period : list
        Earliest and latest times to use for standardization.

    subtract_running_mean : boolean, optional
        If True (default), subtract running mean of the previous
        days for each time-step.

    n_running_mean_steps : integer
        Number of days to include in running mean, by default 120.

    lat_bounds : list
        Latitude range over which to compute meridional mean,
        by default 15S to 15N.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the RMM EOF patterns and the components of the
        index.
    """

    # Ensure all inputs are data arrays.
    olr = rdu.ensure_data_array(olr)
    u850 = rdu.ensure_data_array(u850)
    u200 = rdu.ensure_data_array(u200)

    if enso_index is not None:
        enso_index = rdu.ensure_data_array(enso_index)

    # Index is defined in terms of daily resolution data.
    if not rdu.is_daily_data(olr):
        raise ValueError('Initial OLR data must be daily resolution')

    if not rdu.is_daily_data(u850):
        raise ValueError('Initial u850 data must be daily resolution')

    if not rdu.is_daily_data(u200):
        raise ValueError('Initial u200 data must be daily resolution')

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(olr)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(olr)
    time_name = time_name if time_name is not None else rdu.get_time_name(olr)

    # For convenience, ensure all inputs have same coordinate names.
    _check_consistent_coordinate_names(
        olr, u850, u200, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name)

    if n_modes is None:
        n_modes = 2

    if not rdu.is_integer(n_modes) or n_modes < 1:
        raise ValueError('Number of modes must be a positive integer.')

    # Ensure inputs cover same time period.
    start_time = max(
        [olr[time_name].min(), u850[time_name].min(),
         u200[time_name].min()])
    end_time = min(
        [olr[time_name].max(), u850[time_name].max(),
         u200[time_name].max()])

    if enso_index is not None:
        start_time = max(start_time, enso_index[time_name].min())
        end_time = min(end_time, enso_index[time_name].max())

    def _restrict_time_period(data):
        return data.where(
            (data[time_name] >= start_time) &
            (data[time_name] <= end_time), drop=True)

    olr = _restrict_time_period(olr)
    u850 = _restrict_time_period(u850)
    u200 = _restrict_time_period(u200)

    base_period = rdu.check_base_period(olr, base_period=base_period,
                                        time_name=time_name)

    olr_anom, u850_anom, u200_anom = wh_rmm_anomalies(
        olr, u850, u200, base_period=base_period,
        enso_index=enso_index, subtract_running_mean=subtract_running_mean,
        n_running_mean_steps=n_running_mean_steps, lat_bounds=lat_bounds,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    base_period_olr_anom = olr_anom.where(
        (olr_anom[time_name] >= base_period[0]) &
        (olr_anom[time_name] <= base_period[1]), drop=True)
    base_period_u850_anom = u850_anom.where(
        (u850_anom[time_name] >= base_period[0]) &
        (u850_anom[time_name] <= base_period[1]), drop=True)
    base_period_u200_anom = u200_anom.where(
        (u200_anom[time_name] >= base_period[0]) &
        (u200_anom[time_name] <= base_period[1]), drop=True)

    olr_normalization = np.std(base_period_olr_anom)
    u850_normalization = np.std(base_period_u850_anom)
    u200_normalization = np.std(base_period_u200_anom)

    base_period_olr_anom = base_period_olr_anom / olr_normalization
    base_period_u850_anom = base_period_u850_anom / u850_normalization
    base_period_u200_anom = base_period_u200_anom / u200_normalization

    olr_eofs, u850_eofs, u200_eofs = wh_rmm_eofs(
        base_period_olr_anom, base_period_u850_anom, base_period_u200_anom,
        n_modes=n_modes, time_name=time_name)

    rmm1_normalization = olr_eofs['PCs'].sel(mode=0).std(time_name).item()
    rmm2_normalization = olr_eofs['PCs'].sel(mode=1).std(time_name).item()

    olr_pcs = (olr_anom / olr_normalization).dot(olr_eofs['EOFs'])
    u850_pcs = (u850_anom / u850_normalization).dot(u850_eofs['EOFs'])
    u200_pcs = (u200_anom / u200_normalization).dot(u200_eofs['EOFs'])

    rmm1 = (olr_pcs.sel(mode=0, drop=True) +
            u850_pcs.sel(mode=0, drop=True) +
            u200_pcs.sel(mode=0, drop=True)) / rmm1_normalization
    rmm2 = (olr_pcs.sel(mode=1, drop=True) +
            u850_pcs.sel(mode=1, drop=True) +
            u200_pcs.sel(mode=1, drop=True)) / rmm2_normalization

    rmm_olr_eofs = olr_eofs['EOFs'].reset_coords(
        [d for d in olr_eofs['EOFs'].coords if d not in ('mode', lon_name)],
        drop=True)
    rmm_u850_eofs = u850_eofs['EOFs'].reset_coords(
        [d for d in u850_eofs['EOFs'].coords if d not in ('mode', lon_name)],
        drop=True)
    rmm_u200_eofs = u200_eofs['EOFs'].reset_coords(
        [d for d in u200_eofs['EOFs'].coords if d not in ('mode', lon_name)],
        drop=True)

    data_vars = {'olr_eofs': rmm_olr_eofs,
                 'u850_eofs': rmm_u850_eofs,
                 'u200_eofs': rmm_u200_eofs,
                 'rmm1': rmm1,
                 'rmm2': rmm2}

    ds = xr.Dataset(data_vars)

    ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    ds.attrs['subtract_running_mean'] = '{}'.format(subtract_running_mean)
    ds.attrs['n_running_mean_steps'] = '{:d}'.format(n_running_mean_steps)
    ds.attrs['n_modes'] = '{:d}'.format(n_modes)

    return ds
