"""
Provides routines for computing indices associated with Indo-Pacific SST.
"""

# License: MIT

from __future__ import absolute_import, division

import os

import dask.array
import geopandas as gp
import numpy as np
import regionmask as rm
import scipy.linalg
import xarray as xr

import reanalysis_dbns.utils as rdu


INDIAN_PACIFIC_OCEAN_REGION_SHP = os.path.join(
    os.path.dirname(__file__), 'indian_pacific_ocean.shp')


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
    valid_features = [d for d in range(n_features)
                      if d not in missing_features]
    valid_data = valid_data.swapaxes(0, 1)

    if eofs.get_axis_num(mode_dim) != 0:
        eofs = eofs.transpose(*([mode_dim] + feature_dims))

    n_modes = eofs.sizes[mode_dim]

    flat_eofs = eofs.values.reshape((n_modes, n_features))
    valid_eofs = flat_eofs[:, valid_features].swapaxes(0, 1)

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


def _fix_loading_pattern_phases(eofs, lon_name=None, lat_name=None):
    """Ensure maximum positive SST anomaly occurs in Nino region."""

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(eofs)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(eofs)

    min_lon = eofs[lon_name].min()
    max_lon = eofs[lon_name].max()

    if min_lon < 0 and max_lon <= 180:
        leading_eof = eofs['EOFs'].where(
            (eofs[lon_name] <= -120) &
            (eofs[lon_name] >= -170) &
            (eofs[lat_name] >= -5) &
            (eofs[lat_name] <= 5), drop=True).isel(mode=0)
    else:
        leading_eof = eofs['EOFs'].where(
            (eofs[lon_name] >= 190) &
            (eofs[lon_name] <= 240) &
            (eofs[lat_name] >= -5) &
            (eofs[lat_name] <= 5), drop=True).isel(mode=0)

    flipped_leading_eof = -leading_eof
    nino34_max_anom = leading_eof.max()
    flipped_nino34_max_anom = flipped_leading_eof.max()

    if flipped_nino34_max_anom > nino34_max_anom:

        eofs['EOFs'] = xr.where(eofs['mode'] == 0, -eofs['EOFs'], eofs['EOFs'])
        eofs['PCs'] = xr.where(eofs['mode'] == 0, -eofs['PCs'], eofs['PCs'])

    return eofs


def dc_sst_loading_pattern(sst_anom, n_modes=None, n_rotated_modes=12,
                           weights=None, lat_name=None, lon_name=None,
                           time_name=None):
    """Calculate rotated EOFs for the SST1 and SST2 index.

    Parameters
    ----------
    sst_anom : xarray.DataArray
        Array containing SST anomalies.

    n_modes : integer
        Number of modes to compute in EOFs calculation.
        If None, the minimum number of modes needed to perform the
        rotation are computed.

    n_rotated_modes : integer
        Number of modes to include in VARIMAX rotation.
        If None, all computed modes are included.

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
        Dataset containing the computed rotated EOFs and PCs.
    """

    # Ensure that the data provided is a data array
    sst_anom = rdu.ensure_data_array(sst_anom)

    lat_name = (lat_name if lat_name is not None else
                rdu.get_lat_name(sst_anom))
    lon_name = (lon_name if lon_name is not None else
                rdu.get_lon_name(sst_anom))
    time_name = (time_name if time_name is not None else
                 rdu.get_time_name(sst_anom))

    if n_modes is None and n_rotated_modes is None:
        n_modes = 12
        n_rotated_modes = 12
    elif n_modes is None and n_rotated_modes is not None:
        n_modes = n_rotated_modes
    elif n_modes is not None and n_rotated_modes is None:
        n_rotated_modes = n_modes

    if not rdu.is_integer(n_modes) or n_modes < 1:
        raise ValueError(
            'Invalid number of modes: got %r but must be at least 1')

    if not rdu.is_integer(n_rotated_modes) or n_rotated_modes < 1:
        raise ValueError(
            'Invalid number of rotated modes: got %r but must be at least 1')

    if n_rotated_modes > n_modes:
        raise ValueError(
            'Number of rotated modes must not be greater than'
            ' number of unrotated modes')

    eofs_ds = rdu.reofs(sst_anom, sample_dim=time_name, weight=weights,
                        n_modes=n_modes, n_rotated_modes=n_rotated_modes)

    eofs_ds.attrs['n_modes'] = '{:d}'.format(n_modes)
    eofs_ds.attrs['n_rotated_modes'] = '{:d}'.format(n_rotated_modes)

    return _fix_loading_pattern_phases(
        eofs_ds, lat_name=lat_name, lon_name=lon_name)


def dc_sst(sst, frequency='monthly', n_modes=None, n_rotated_modes=12,
           base_period=None, lat_name=None, lon_name=None, time_name=None):
    """Calculate the SST1 and SST2 index of Indo-Pacific SST.

    The indices are defined as the PCs associated with the
    first and second rotated EOFs of standardized
    Indo-Pacific SST anomalies.

    See Drosdowsky, W. and Chambers, L. E., "Near-Global Sea Surface
    Temperature Anomalies as Predictors of Australian Seasonal
    Rainfall", Journal of Climate 14, 1677 - 1687 (2001).

    Parameters
    ----------
    sst : xarray.DataArray
        Array containing SST values.

    frequency : str
        If given, downsample data to requested frequency.

    n_modes : integer
        Number of EOFs to retain before rotation.

    n_rotated_modes : integer
        Number of EOFs to include in VARIMAX rotation.

    base_period : list
        Earliest and latest times to use for standardization.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray.Dataset
        Dataset containing the SST index loading patterns and indices.
    """

    # Ensure that the data provided is a data array
    sst = rdu.ensure_data_array(sst)

    # Get coordinate names
    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(sst)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(sst)
    time_name = time_name if time_name is not None else rdu.get_time_name(sst)

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    base_period = rdu.check_base_period(sst, base_period=base_period,
                                        time_name=time_name)

    shape_data = gp.read_file(INDIAN_PACIFIC_OCEAN_REGION_SHP)

    region_mask = rm.Regions(shape_data['geometry'], numbers=[0])

    # Generate mask for SST region
    mask = region_mask.mask(sst[lon_name] % 360, sst[lat_name],
                            lon_name=lon_name, lat_name=lat_name)
    mask[lon_name] = sst[lon_name]

    sst = sst.where(mask == 0)

    # EOFs are computed based on standardized monthly anomalies
    input_frequency = rdu.detect_frequency(sst, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    if input_frequency == 'daily' and frequency == 'daily':

        base_period_monthly_sst = sst.where(
            (sst[time_name] >= base_period[0]) &
            (sst[time_name] <= base_period[1]), drop=True).resample(
                {time_name: '1MS'}).mean()

        base_period_monthly_sst_anom = rdu.standardized_anomalies(
            base_period_monthly_sst, base_period=base_period,
            standardize_by='month', time_name=time_name)

        sst_anom = rdu.standardized_anomalies(
            sst, base_period=base_period, standardize_by='dayofyear',
            time_name=time_name)

    elif input_frequency == 'daily' and frequency == 'monthly':

        sst = sst.resample({time_name: '1MS'}).mean()

        sst_anom = rdu.standardized_anomalies(
            sst, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_monthly_sst_anom = sst_anom.where(
            (sst_anom[time_name] >= base_period[0]) &
            (sst_anom[time_name] <= base_period[1]), drop=True)

    elif input_frequency == 'monthly' and frequency == 'daily':

        raise RuntimeError(
            'Attempting to calculate daily index from monthly data')

    else:

        sst_anom = rdu.standardized_anomalies(
            sst, base_period=base_period, standardize_by='month',
            time_name=time_name)

        base_period_monthly_sst_anom = sst_anom.where(
            (sst_anom[time_name] >= base_period[0]) &
            (sst_anom[time_name] <= base_period[1]), drop=True)

    # Get square root of cos(latitude) weights
    scos_weights = np.cos(np.deg2rad(sst_anom[lat_name])).clip(0., 1.) ** 0.5

    # Calculate VARIMAX rotated EOFs of standardized monthly anomalies.
    loadings_ds = dc_sst_loading_pattern(
        base_period_monthly_sst_anom, n_modes=n_modes,
        n_rotated_modes=n_rotated_modes, weights=scos_weights,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name)

    # Project weighted anomalies onto first and second REOFs.
    rotated_pcs = _project_onto_subspace(
        sst_anom, loadings_ds['EOFs'], weight=scos_weights,
        sample_dim=time_name)

    sst1_index = rotated_pcs.sel(mode=0).drop_vars('mode').rename('sst1_index')
    sst2_index = rotated_pcs.sel(mode=1).drop_vars('mode').rename('sst2_index')

    index_ds = xr.Dataset({'sst1_index': sst1_index,
                           'sst2_index': sst2_index})

    loadings_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    loadings_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')

    index_ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    index_ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    index_ds.attrs['n_modes'] = loadings_ds.attrs['n_modes']
    index_ds.attrs['n_rotated_modes'] = loadings_ds.attrs['n_rotated_modes']

    return loadings_ds, index_ds
