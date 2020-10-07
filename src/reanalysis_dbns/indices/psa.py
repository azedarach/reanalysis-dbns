"""
Provides routines for calculating indices of the PSA.
"""

# License: MIT

from __future__ import absolute_import, division

import dask.array
import numpy as np
import scipy.linalg
import xarray as xr

import reanalysis_dbns.utils as rdu


def _project_onto_subspace(data, eofs, weight=None, sample_dim=None,
                           mode_dim='mode'):
    """Project given data onto (possibly non-orthogonal) basis."""

    # Ensure given data is a data array
    data = rdu.ensure_data_array(data)

    sample_dim = (sample_dim if sample_dim is not None else
                  rdu.get_time_name(data))

    if mode_dim is None:
        raise ValueError('No mode dimension given')

    if weight is not None:
        data = data * weight

    feature_dims = [d for d in data.dims if d != sample_dim]
    original_shape = [data.sizes[d] for d in feature_dims]

    if data.get_axis_num(sample_dim) != 0:
        data = data.transpose(*([sample_dim] + feature_dims))

    n_samples = data.sizes[sample_dim]
    n_features = np.product(original_shape)

    flat_data = data.values.reshape((n_samples, n_features))
    valid_data, missing_features = rdu.remove_missing_features(flat_data)
    nonmissing_features = [d for d in range(n_features)
                           if d not in missing_features]
    valid_data = valid_data.swapaxes(0, 1)

    if eofs.get_axis_num(mode_dim) != 0:
        eofs = eofs.transpose(*([mode_dim] + feature_dims))

    n_modes = eofs.sizes[mode_dim]

    flat_eofs = eofs.values.reshape((n_modes, n_features))
    valid_eofs = flat_eofs[:, nonmissing_features].swapaxes(0, 1)

    if rdu.is_dask_array(flat_data):

        projection = dask.array.linalg.lstsq(valid_eofs, valid_data)

    else:

        projection = scipy.linalg.lstsq(valid_eofs, valid_data)[0]

    projection = xr.DataArray(
        projection.swapaxes(0, 1),
        coords={sample_dim: data[sample_dim],
                mode_dim: eofs[mode_dim]},
        dims=[sample_dim, mode_dim])

    return projection


def _fix_psa1_phase(eofs_ds, psa1_mode, lat_name=None, lon_name=None):
    """Fix PSA1 phase such that negative anomalies occur in Pacific sector."""

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(eofs_ds)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(eofs_ds)

    psa1_pattern = eofs_ds['EOFs'].sel(mode=psa1_mode, drop=True).squeeze()

    box = rdu.select_latlon_box(
        psa1_pattern, lat_bounds=[-70.0, -50.0],
        lon_bounds=[210.0, 270.0],
        lat_name=lat_name, lon_name=lon_name)

    box_max = max(box.max().item(), 0)
    box_min = min(box.min().item(), 0)

    if np.abs(box_max) > np.abs(box_min):
        must_flip = box_max > 0
    else:
        must_flip = box_min > 0

    if must_flip:
        eofs_ds['EOFs'] = xr.where(eofs_ds['mode'] == psa1_mode,
                                   -eofs_ds['EOFs'], eofs_ds['EOFs'])
        eofs_ds['PCs'] = xr.where(eofs_ds['mode'] == psa1_mode,
                                  -eofs_ds['PCs'], eofs_ds['PCs'])

    return eofs_ds


def _fix_psa2_phase(eofs_ds, psa2_mode, lat_name=None, lon_name=None):
    """Fix PSA2 phase such that negative anomalies occur in Atlantic sector."""

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(eofs_ds)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(eofs_ds)

    psa2_pattern = eofs_ds['EOFs'].sel(mode=psa2_mode, drop=True).squeeze()

    box = rdu.select_latlon_box(
        psa2_pattern, lat_bounds=[-70.0, -50.0],
        lon_bounds=[240.0, 300.0],
        lat_name=lat_name, lon_name=lon_name)

    box_max = box.max().item()
    box_min = box.min().item()

    if np.abs(box_max) > np.abs(box_min):
        must_flip = box_max > 0
    else:
        must_flip = box_min > 0

    if must_flip:
        eofs_ds['EOFs'] = xr.where(eofs_ds['mode'] == psa2_mode,
                                   -eofs_ds['EOFs'], eofs_ds['EOFs'])
        eofs_ds['PCs'] = xr.where(eofs_ds['mode'] == psa2_mode,
                                  -eofs_ds['PCs'], eofs_ds['PCs'])

    return eofs_ds


def _calculate_hgt_anom(hgt, base_period=None, frequency=None, time_name=None):
    """Calculate height anomalies with given frequency."""

    # Ensure that the data provided is a data array.
    hgt = rdu.ensure_data_array(hgt)

    # Get coordinate names.
    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly', 'seasonal'):
        raise ValueError("Unsupported output frequency '%s'" % frequency)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported input frequency '%s'" % input_frequency)

    if input_frequency in ('monthly', 'seasonal') and frequency == 'daily':
        raise ValueError(
            'Cannot compute daily anomalies from monthly or seasonal data')

    must_downsample = ((input_frequency == 'daily' and
                        frequency in ('monthly', 'seasonal')) or
                       (input_frequency == 'monthly' and
                        frequency == 'seasonal'))

    if must_downsample:
        hgt = rdu.downsample_data(
            hgt, frequency=frequency, time_name=time_name)

    base_period_hgt = hgt.where(
        (hgt[time_name] >= base_period[0]) &
        (hgt[time_name] <= base_period[1]), drop=True)

    if frequency == 'daily':

        groups = '{}.dayofyear'.format(time_name)

    elif frequency == 'monthly':

        groups = '{}.month'.format(time_name)

    else:

        groups = '{}.season'.format(time_name)

    clim = base_period_hgt.groupby(groups).mean(time_name)
    anom = hgt.groupby(groups) - clim

    return anom


def _check_frequency(frequency):
    """Check given frequency is valid."""

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    return frequency


def _check_eofs_frequency(eofs_frequency):
    """Check given EOFs frequency is valid."""

    if eofs_frequency is None:
        eofs_frequency = 'monthly'

    if eofs_frequency not in ('daily', 'monthly', 'seasonal'):
        raise ValueError("Unsupported EOF frequency '%r'" % eofs_frequency)

    return eofs_frequency


def _check_input_frequency(hgt, time_name=None):
    """Check input frequency is valid."""

    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    return input_frequency


def _check_psa_modes(psa1_mode, psa2_mode):
    """Check given PSA modes are valid."""

    if psa1_mode is None:
        psa1_mode = 1

    if not rdu.is_integer(psa1_mode) or psa1_mode < 0:
        raise ValueError('PSA1 mode must be a positive integer')

    if psa2_mode is None:
        psa2_mode = 2

    if not rdu.is_integer(psa2_mode) or psa2_mode < 0:
        raise ValueError('PSA2 mode must be a positive integer')

    return psa1_mode, psa2_mode


def _check_number_of_modes(n_modes, psa1_mode, psa2_mode):
    """Check number of EOF modes is valid."""

    min_modes = max(psa1_mode, psa2_mode)

    if n_modes is None:
        n_modes = min_modes + 1

    if not rdu.is_integer(n_modes) or n_modes <= min_modes:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         min_modes)

    return n_modes


def _check_number_of_rotated_modes(n_rotated_modes, n_modes,
                                   psa1_mode, psa2_mode):
    """Check number of rotated modes is valid."""

    min_modes = max(psa1_mode, psa2_mode)

    if n_rotated_modes is None:
        n_rotated_modes = n_modes

    if not rdu.is_integer(n_rotated_modes) or n_rotated_modes <= min_modes:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         min_modes)

    return n_rotated_modes


def real_pc_psa(hgt, frequency='monthly', base_period=None, rotate=False,
                psa1_mode=1, psa2_mode=2, n_modes=None, n_rotated_modes=None,
                eofs_season='ALL', eofs_frequency='monthly',
                low_latitude_boundary=-20,
                lat_name=None, lon_name=None, time_name=None):
    """Calculate real principal component based PSA indices.

    Parameters
    ----------
    hgt : xarray.DataArray
        Array containing geopotential height values.

    frequency : 'daily' | 'monthly'
        Sampling rate at which to calculate the index.

    base_period : list
        If given, a two element list containing the earliest and
        latest dates to include in base period for standardization and
        calculation of EOFs. If None, then the full time period is used.

    rotate : bool
        If True, calculate rotated EOFs.

    psa1_mode : integer
        Mode to take as corresponding to the real PSA1 mode.

    psa2_mode : integer
        Mode to take as corresponding to the real PSA2 mode.

    n_modes : integer
        Number of EOF modes to calculate.

    n_rotated_modes : integer
        If computing rotated EOFs, number of modes to include in
        rotation.

    eofs_season : 'ALL' | 'DJF' | 'MAM' | 'JJA' | 'SON'
        Season to use for calculation of EOFs.

    eofs_frequency : 'daily' | 'monthly' | 'seasonal'
        Sampling rate of anomalies used to calculate the PSA patterns.

    low_latitude_boundary : float
        Low-latitude boundary for analysis region.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    index : xarray.Dataset
        Dataset containing the PSA1 index and loading pattern.
    """

    # Ensure that the data provided is a data array.
    hgt = rdu.ensure_data_array(hgt)

    # Get coordinate names.
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(hgt)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(hgt)
    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    # Restrict to data polewards of boundary latitude
    hgt = hgt.where(hgt[lat_name] <= low_latitude_boundary, drop=True)

    # Get subset of data to use for computing anomalies and EOFs.
    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    frequency = _check_frequency(frequency)
    eofs_frequency = _check_eofs_frequency(eofs_frequency)

    psa1_mode, psa2_mode = _check_psa_modes(psa1_mode, psa2_mode)
    n_modes = _check_number_of_modes(n_modes, psa1_mode, psa2_mode)
    n_rotated_modes = _check_number_of_rotated_modes(
        n_rotated_modes, n_modes, psa1_mode, psa2_mode)

    # Calculate anomalies to use for calculating EOFs.
    eof_hgt_anom = _calculate_hgt_anom(
        hgt, base_period=base_period, frequency=eofs_frequency,
        time_name=time_name)

    eof_hgt_anom = eof_hgt_anom.where(
        (eof_hgt_anom[time_name] >= base_period[0]) &
        (eof_hgt_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights.
    scos_weights = np.cos(
        np.deg2rad(eof_hgt_anom[lat_name])).clip(0., 1.)**0.5

    # Loading pattern is computed using austral winter anomalies.
    if eofs_season != 'ALL':
        eof_hgt_anom = eof_hgt_anom.where(
            eof_hgt_anom[time_name].dt.season == eofs_season, drop=True)

    # Calculate loading pattern from monthly anomalies in base period.
    if rotate:
        eofs_ds = rdu.reofs(eof_hgt_anom, sample_dim=time_name,
                            weight=scos_weights, n_modes=n_modes,
                            n_rotated_modes=n_rotated_modes)
    else:
        eofs_ds = rdu.eofs(eof_hgt_anom, sample_dim=time_name,
                           weight=scos_weights, n_modes=n_modes)

    eofs_ds = _fix_psa1_phase(eofs_ds, psa1_mode=psa1_mode,
                              lat_name=lat_name, lon_name=lon_name)
    eofs_ds = _fix_psa2_phase(eofs_ds, psa2_mode=psa2_mode,
                              lat_name=lat_name, lon_name=lon_name)

    # Calculate height anomalies to use in projection.
    hgt_anom = _calculate_hgt_anom(
        hgt, base_period=base_period, frequency=frequency,
        time_name=time_name)

    # Project weighted anomalies onto PSA modes.
    pcs = _project_onto_subspace(
        hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
        sample_dim=time_name)

    psa1_index = pcs.sel(mode=psa1_mode, drop=True).squeeze()
    psa2_index = pcs.sel(mode=psa2_mode, drop=True).squeeze()

    # Normalize indices by standard deviation of base period PCs.
    psa1_pcs_std = eofs_ds['PCs'].sel(mode=psa1_mode).std(ddof=1).values
    psa2_pcs_std = eofs_ds['PCs'].sel(mode=psa2_mode).std(ddof=1).values

    psa1_index = psa1_index / psa1_pcs_std
    psa2_index = psa2_index / psa2_pcs_std

    indices_ds = xr.Dataset({'psa1_index': psa1_index,
                             'psa2_index': psa2_index})

    eofs_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    eofs_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    eofs_ds.attrs['low_latitude_boundary'] = '{:16.8e}'.format(
        low_latitude_boundary)
    eofs_ds.attrs['n_modes'] = '{:d}'.format(eofs_ds.sizes['mode'])
    eofs_ds.attrs['psa1_mode'] = '{:d}'.format(psa1_mode)
    eofs_ds.attrs['psa2_mode'] = '{:d}'.format(psa2_mode)

    indices_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    indices_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    indices_ds.attrs['low_latitude_boundary'] = '{:16.8e}'.format(
        low_latitude_boundary)
    indices_ds.attrs['n_modes'] = '{:d}'.format(eofs_ds.sizes['mode'])
    indices_ds.attrs['psa1_mode'] = '{:d}'.format(psa1_mode)
    indices_ds.attrs['psa2_mode'] = '{:d}'.format(psa2_mode)
    indices_ds.attrs['psa1_index_normalization'] = '{:16.8e}'.format(
        psa1_pcs_std)
    indices_ds.attrs['psa2_index_normalization'] = '{:16.8e}'.format(
        psa2_pcs_std)

    return eofs_ds, indices_ds
