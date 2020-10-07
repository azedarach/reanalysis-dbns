"""
Provides routines for calculating indices of the PNA.
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


def _fix_pna_phase(eofs_ds, pna_mode, lat_name=None, lon_name=None):
    """Fix PNA phase such that negative anomalies occur in Pacific sector."""

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(eofs_ds)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(eofs_ds)

    pna_pattern = eofs_ds['EOFs'].sel(mode=pna_mode, drop=True).squeeze()

    box = rdu.select_latlon_box(
        pna_pattern, lat_bounds=[40.0, 60.0], lon_bounds=[160.0, 200.0],
        lat_name=lat_name, lon_name=lon_name)

    box_max = box.max().item()
    box_min = box.min().item()

    if np.abs(box_max) > np.abs(box_min):
        must_flip = box_max > 0
    else:
        must_flip = box_min > 0

    if must_flip:
        eofs_ds['EOFs'] = xr.where(eofs_ds['mode'] == pna_mode,
                                   -eofs_ds['EOFs'], eofs_ds['EOFs'])
        eofs_ds['PCs'] = xr.where(eofs_ds['mode'] == pna_mode,
                                  -eofs_ds['PCs'], eofs_ds['PCs'])

    return eofs_ds


def _check_frequency(frequency):
    """Check given frequency is valid."""

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    return frequency


def _check_input_frequency(hgt, time_name=None):
    """Check input frequency is valid."""

    time_name = (time_name if time_name is not None else
                 rdu.get_time_name(hgt))

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    return input_frequency


def _check_pna_mode(pna_mode):
    """Check given mode is valid."""
    if pna_mode is None:
        pna_mode = 1

    if not rdu.is_integer(pna_mode) or pna_mode < 0:
        raise ValueError('PNA mode must be a positive integer')

    return pna_mode


def _check_number_of_modes(n_modes, pna_mode):
    """Check number of EOF modes is valid."""

    if n_modes is None:
        n_modes = 10

    if not rdu.is_integer(n_modes) or n_modes <= pna_mode:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         pna_mode)

    return n_modes


def _check_number_of_rotated_modes(n_rotated_modes, pna_mode, n_modes):
    """Check number of rotated modes is valid."""

    if n_rotated_modes is None:
        n_rotated_modes = n_modes

    if not rdu.is_integer(n_rotated_modes) or n_rotated_modes <= pna_mode:
        raise ValueError('Number of modes must be an integer greater than %d' %
                         pna_mode)

    return n_rotated_modes


def pc_pna(hgt, frequency='monthly', base_period=None, rotate=True,
           pna_mode=1, n_modes=None, n_rotated_modes=None,
           low_latitude_boundary=20, lat_name=None, lon_name=None,
           time_name=None):
    """Calculate principal component based PNA index.

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

    pna_mode : integer
        Mode to take as corresponding to the PNA mode.

    n_modes : integer
        Number of EOF modes to calculate.

    n_rotated_modes : integer
        If computing rotated EOFs, number of modes to include in
        rotation.

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
        Dataset containing the PNA index and loading pattern.
    """

    # Ensure that the data provided is a data array.
    hgt = rdu.ensure_data_array(hgt)

    # Get coordinate names.
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(hgt)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(hgt)
    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    # Restrict to data polewards of boundary latitude
    hgt = hgt.where(hgt[lat_name] >= low_latitude_boundary, drop=True)

    # Get subset of data to use for computing anomalies and EOFs.
    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    frequency = _check_frequency(frequency)
    input_frequency = _check_input_frequency(hgt, time_name=time_name)

    pna_mode = _check_pna_mode(pna_mode)
    n_modes = _check_number_of_modes(n_modes, pna_mode)
    n_rotated_modes = _check_number_of_rotated_modes(
        n_rotated_modes, pna_mode, n_modes)

    # Note that EOFs are computed using monthly mean data.
    if input_frequency == 'daily' and frequency == 'daily':

        hgt_anom = rdu.standardized_anomalies(
            hgt, base_period=base_period, standardize_by='dayofyear',
            time_name=time_name)

        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        base_period_hgt = rdu.downsample_data(
            base_period_hgt, frequency='monthly', time_name=time_name)

        base_period_hgt_anom = rdu.standardized_anomalies(
            base_period_hgt, base_period=base_period,
            standardize_by='month', time_name=time_name)

    elif input_frequency == 'daily' and frequency == 'monthly':

        hgt = rdu.downsample_data(
            hgt, frequency='monthly', time_name=time_name)

        hgt_anom = rdu.standardized_anomalies(
            hgt, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise ValueError('Cannot calculate daily index from monthly data')

    else:

        hgt_anom = rdu.standardized_anomalies(
            hgt, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_hgt_anom = hgt_anom.where(
            (hgt_anom[time_name] >= base_period[0]) &
            (hgt_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights.
    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.)**0.5

    # Loading pattern is computed using boreal winter anomalies.
    winter_hgt_anom = base_period_hgt_anom.where(
        base_period_hgt_anom[time_name].dt.season == 'DJF', drop=True)

    # Calculate loading pattern from monthly anomalies in base period.
    if rotate:
        eofs_ds = rdu.reofs(winter_hgt_anom, sample_dim=time_name,
                            weight=scos_weights, n_modes=n_modes,
                            n_rotated_modes=n_rotated_modes)
    else:
        eofs_ds = rdu.eofs(winter_hgt_anom, sample_dim=time_name,
                           weight=scos_weights, n_modes=n_modes)

    eofs_ds = _fix_pna_phase(eofs_ds, pna_mode=pna_mode,
                             lat_name=lat_name, lon_name=lon_name)

    # Project weighted anomalies onto PNA mode.
    pcs = _project_onto_subspace(
        hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
        sample_dim=time_name)

    index = pcs.sel(mode=pna_mode, drop=True)

    # Normalize index by base period monthly means and
    # standard deviations.
    if frequency == 'monthly':
        base_period_index = index.where(
            (index[time_name] >= base_period[0]) &
            (index[time_name] <= base_period[1]), drop=True)

    else:

        base_period_index = _project_onto_subspace(
            base_period_hgt_anom, eofs_ds['EOFs'], weight=scos_weights,
            sample_dim=time_name)

        base_period_index = base_period_index.sel(mode=pna_mode, drop=True)

    index_mean = base_period_index.groupby(
        base_period_index[time_name].dt.month).mean(time_name)

    index_std = base_period_index.groupby(
        base_period_index[time_name].dt.month).std(time_name)

    index = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        index.groupby(index[time_name].dt.month),
        index_mean, index_std, dask='allowed')

    index_ds = index.to_dataset(name='pna_index')

    eofs_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    eofs_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    eofs_ds.attrs['low_latitude_boundary'] = '{:16.8e}'.format(
        low_latitude_boundary)
    eofs_ds.attrs['n_modes'] = '{:d}'.format(n_modes)
    eofs_ds.attrs['rotate'] = '{}'.format(rotate)
    if rotate:
        eofs_ds.attrs['n_rotated_modes'] = '{:d}'.format(n_rotated_modes)
    eofs_ds.attrs['pna_mode'] = '{:d}'.format(pna_mode)

    index_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    index_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    index_ds.attrs['low_latitude_boundary'] = '{:16.8e}'.format(
        low_latitude_boundary)
    index_ds.attrs['n_modes'] = '{:d}'.format(n_modes)
    index_ds.attrs['rotate'] = '{}'.format(rotate)
    if rotate:
        index_ds.attrs['n_rotated_modes'] = '{:d}'.format(n_rotated_modes)
    index_ds.attrs['pna_mode'] = '{:d}'.format(pna_mode)

    return eofs_ds, index_ds
