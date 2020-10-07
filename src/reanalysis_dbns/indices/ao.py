"""
Provides routines for computing AO indices.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
import xarray as xr

import reanalysis_dbns.utils as rdu


def _fix_ao_pc_phase(eofs_ds, ao_mode, lat_name=None):
    """Fixes the sign of the AO mode and PCs to chosen convention."""

    if lat_name is None:
        lat_name = rdu.get_lat_name(eofs_ds)

    ao_eof = eofs_ds.sel(mode=ao_mode, drop=True)['EOFs']
    lat_max = ao_eof.where(ao_eof == ao_eof.max(), drop=True)[lat_name].item()
    lat_min = ao_eof.where(ao_eof == ao_eof.min(), drop=True)[lat_name].item()

    if lat_max > lat_min:
        eofs_ds['EOFs'] = xr.where(eofs_ds['mode'] == ao_mode,
                                   -eofs_ds['EOFs'], eofs_ds['EOFs'])
        eofs_ds['PCs'] = xr.where(eofs_ds['mode'] == ao_mode,
                                  -eofs_ds['PCs'], eofs_ds['PCs'])

    return eofs_ds


def ao_loading_pattern(hgt_anom, ao_mode=0, n_modes=None, weights=None,
                       lat_name=None, lon_name=None, time_name=None):
    """Calculate loading pattern for the principal component AO index.

    The positive phase of the AO is defined such that the anomalous
    lows occur at high latitudes.

    Parameters
    ----------
    hgt_anom : xarray.DataArray
        Array containing geopotential height anomalies.

    ao_mode : integer
        Mode whose PC is taken to be the index.

    n_modes : integer
        Number of modes to compute in EOFs calculation.
        If None, only the minimum number of modes needed to include the
        requested AO mode are computed.

    weights : xarray.DataArray
        Weights to apply in calculating the EOFs.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    pattern: xarray.Dataset
        Dataset containing the computed EOFs and PCs.
    """

    # Ensure that the data provided is a data array
    hgt_anom = rdu.ensure_data_array(hgt_anom)

    if lat_name is None:
        lat_name = rdu.get_lat_name(hgt_anom)

    if lon_name is None:
        lon_name = rdu.get_lon_name(hgt_anom)

    if time_name is None:
        time_name = rdu.get_time_name(hgt_anom)

    if not rdu.is_integer(ao_mode) or ao_mode < 0:
        raise ValueError(
            'Invalid AO mode: got %r but must be a non-negative integer')

    if n_modes is None:
        n_modes = ao_mode + 1

    eofs_ds = rdu.eofs(hgt_anom, sample_dim=time_name,
                       weight=weights, n_modes=n_modes)

    # Fix signs such that positive AO corresponds to anomalous
    # lows at higher latitude
    eofs_ds = _fix_ao_pc_phase(eofs_ds, ao_mode, lat_name=lat_name)

    return eofs_ds


def pc_ao(hgt, frequency=None, base_period=None, ao_mode=0, n_modes=None,
          low_latitude_boundary=20, lat_name=None, lon_name=None,
          time_name=None):
    """Calculate principal component based AO index.

    The returned AO index is taken to be the principal component (PC)
    obtained by projecting geopotential height anomalies onto
    the chosen empirical orthogonal function (EOF) mode calculated from
    the anomalies. A square root of cos(latitude) weighting is applied
    when calculating the EOFs.

    Note that the AO index provided by the NOAA CPC adopts a
    normalization convention in which the index is normalized by the
    standard deviation of the monthly mean PCs over the base period
    used to compute the AO pattern.

    Parameters
    ----------
    input_data : xarray.DataArray
        Array containing geopotential height values.

    frequency : None | 'daily' | 'monthly'
        Frequency to calculate index at.

    base_period : list
        If given, a two element list containing the
        earliest and latest dates to include when calculating the
        EOFs. If None, the EOFs are computed using the full
        dataset.

    ao_mode : integer
        Mode whose PC is taken to be the index.

    n_modes : integer
        Number of modes to compute in EOFs calculation.
        If None, only the minimum number of modes needed to include the
        requested AO mode are computed.

    low_latitude_boundary : float
        Low-latitude bound for analysis region.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the values of the AO loading pattern and index.
    """

    # Ensure that the data provided is a data array
    hgt = rdu.ensure_data_array(hgt)

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(hgt)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(hgt)
    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    # Restrict to data polewards of boundary latitude
    hgt = hgt.where(hgt[lat_name] >= low_latitude_boundary, drop=True)

    # Get subset of data to use for computing anomalies and EOFs.
    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    # Calculate base period climatology and anomalies.
    base_period_hgt = hgt.where(
        (hgt[time_name] >= base_period[0]) &
        (hgt[time_name] <= base_period[1]), drop=True)

    if input_frequency == 'daily':

        base_period_monthly_hgt = rdu.downsample_data(
            base_period_hgt, frequency='monthly', time_name=time_name)

        monthly_clim = base_period_monthly_hgt.groupby(
            base_period_monthly_hgt[time_name].dt.month).mean(time_name)

        monthly_hgt_anom = (
            base_period_monthly_hgt.groupby(
                base_period_monthly_hgt[time_name].dt.month) -
            monthly_clim)

        if frequency == 'daily':

            daily_clim = base_period_hgt.groupby(
                base_period_hgt[time_name].dt.dayofyear).mean(time_name)

            hgt_anom = hgt.groupby(hgt[time_name].dt.dayofyear) - daily_clim

        elif frequency == 'monthly':

            hgt = rdu.downsample_data(hgt, frequency='monthly',
                                      time_name=time_name)

            hgt_anom = hgt.groupby(hgt[time_name].dt.month) - monthly_clim

    else:

        if frequency == 'daily':
            raise RuntimeError(
                'Attempting to calculate daily index from monthly data')

        monthly_clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - monthly_clim

        monthly_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights.
    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.)**0.5

    # Calculate loading pattern from monthly anomalies in base period.
    loadings_ds = ao_loading_pattern(
        monthly_hgt_anom, ao_mode=ao_mode, n_modes=n_modes,
        weights=scos_weights, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name)

    pcs_std = loadings_ds.sel(mode=ao_mode)['PCs'].std(ddof=1)

    # Project weighted anomalies onto AO mode
    ao_eof = loadings_ds.sel(mode=ao_mode)['EOFs']
    index = ((hgt_anom * scos_weights).fillna(0)).dot(
        ao_eof.fillna(0)).rename('ao_index') / pcs_std

    index_ds = index.to_dataset(name='ao_index')

    loadings_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    loadings_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    loadings_ds.attrs['low_latitude_boundary'] = '{:16.8e}'.format(
        low_latitude_boundary)
    loadings_ds.attrs['n_modes'] = '{:d}'.format(loadings_ds.sizes['mode'])
    loadings_ds.attrs['ao_mode'] = '{:d}'.format(ao_mode)

    index_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    index_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    index_ds.attrs['low_latitude_boundary'] = '{:16.8e}'.format(
        low_latitude_boundary)
    index_ds.attrs['n_modes'] = '{:d}'.format(loadings_ds.sizes['mode'])
    index_ds.attrs['ao_mode'] = '{:d}'.format(ao_mode)
    index_ds.attrs['index_normalization'] = '{:16.8e}'.format(pcs_std.values)

    return loadings_ds, index_ds
