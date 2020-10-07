""""
Provides routines for calculating NH teleconnection indices.
"""

# License: MIT

from __future__ import absolute_import, division

import itertools
import os

import dask.array
import numpy as np
import scipy.linalg
import xarray as xr

from sklearn.cluster import KMeans

import reanalysis_dbns.utils as rdu


N_REFERENCE_PATTERNS = 4
REFERENCE_PATTERNS = os.path.join(
    os.path.dirname(__file__), 'nhtele_patterns.nc')


def _project_onto_subspace(data, basis, weight=None, sample_dim=None,
                           mode_dim='mode', simultaneous=False):
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

    if basis.get_axis_num(mode_dim) != 0:
        basis = basis.transpose(*([mode_dim] + feature_dims))

    n_modes = basis.sizes[mode_dim]

    flat_eofs = basis.values.reshape((n_modes, n_features))
    valid_eofs = flat_eofs[:, nonmissing_features].swapaxes(0, 1)

    if simultaneous:
        if rdu.is_dask_array(flat_data):

            projection = dask.array.linalg.lstsq(valid_eofs, valid_data)

        else:

            projection = scipy.linalg.lstsq(valid_eofs, valid_data)[0]

    else:

        projection = valid_eofs.T.dot(valid_data)

    projection = xr.DataArray(
        projection.swapaxes(0, 1),
        coords={sample_dim: data[sample_dim],
                mode_dim: basis[mode_dim]},
        dims=[sample_dim, mode_dim])

    return projection


def _fix_cluster_ordering(labels, cluster_centers):
    """Impose fixed ordering on labelling of clusters."""

    # Order clusters by magnitude of PC1
    ordering = np.argsort(-np.abs(cluster_centers[:, 0]))

    n_clusters = cluster_centers.shape[0]
    n_samples = labels.shape[0]

    one_hot_labels = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        one_hot_labels[labels == i, i] = 1

    one_hot_labels = one_hot_labels[:, ordering]

    labels = np.argmax(one_hot_labels, axis=1)
    cluster_centers = cluster_centers[ordering, :]

    return labels, cluster_centers


def _order_by_reference_patterns(composites, lat_name=None, lon_name=None):
    """Order patterns according to similarity with reference patterns."""

    lat_name = (lat_name if lat_name is not None else
                rdu.get_lat_name(composites))
    lon_name = (lon_name if lon_name is not None else
                rdu.get_lon_name(composites))

    n_clusters = composites.sizes['cluster']

    with xr.open_dataset(REFERENCE_PATTERNS) as reference_ds:

        n_reference_clusters = reference_ds.sizes['cluster']

        if n_clusters > n_reference_clusters:
            raise ValueError(
                'Too few reference clusters to perform ordering')

        if lat_name != 'lat':
            reference_ds = reference_ds.rename({'lat': lat_name})

        if lon_name != 'lon':
            reference_ds = reference_ds.rename({'lon': lon_name})

        reference_composites = reference_ds['composites'].interp_like(
            composites)

        similarities = np.zeros((n_clusters, n_reference_clusters))
        for i in range(n_clusters):
            for j in range(n_reference_clusters):

                similarities[i, j] = rdu.pattern_correlation(
                    composites.sel(cluster=i),
                    reference_composites.sel(cluster=j),
                    core_dims=[lat_name, lon_name])

        orderings = itertools.permutations(
            range(n_reference_clusters), r=n_clusters)

        max_similarity = -np.inf
        ordering = None
        for o in orderings:

            ordering_similarity = 0.0
            for orig, dest in enumerate(o):
                ordering_similarity += np.abs(similarities[orig, dest])

            if ordering is None or ordering_similarity > max_similarity:
                ordering = o
                max_similarity = ordering_similarity

        ordered_composites = xr.zeros_like(composites)
        for orig, dest in enumerate(ordering):

            pattern_sign = np.sign(similarities[orig, dest])
            ordered_composites.loc[
                dict(cluster=dest)] = pattern_sign * composites.sel(
                    cluster=orig)

        return ordered_composites


def filter_hgt_field(hgt, window_length=None, time_name=None):
    """Apply rolling window filter to input data.

    Parameters
    ----------
    hgt : xarray DataArray
        Array containing geopotential height values.

    window_length : integer
        Length of moving average window.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    hgt_smoothed : xarray DataArray
        Array containing filtered geopotential height anomalies.
    """

    hgt = rdu.ensure_data_array(hgt)

    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency == 'daily':

        if window_length is None:
            window_length = 10

    elif input_frequency == 'monthly':

        if window_length is None:
            window_length = 1

    else:

        if window_length is None:
            window_length = 1

    if not rdu.is_integer(window_length) or window_length < 1:
        raise ValueError('Window length must be a positive integer')

    if window_length > 1:
        hgt = hgt.rolling(
            {time_name: window_length}).mean().dropna(time_name, how='all')

    return hgt


def _check_anomalies_frequency(frequency):
    """Check frequency for height anomalies is valid."""

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    return frequency


def _check_anomalies_input_frequency(hgt, frequency, time_name=None):
    """Check input frequency is valid."""

    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    if input_frequency == 'monthly' and frequency == 'daily':
        raise ValueError('Cannot calculate daily index from monthly data')

    return input_frequency


def _check_input_frequency(hgt, time_name=None):
    """Check input frequency is valid."""

    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    return input_frequency


def _calculate_composite_height_anomalies(hgt, season=None, base_period=None,
                                          time_name=None):
    """Calculate height anomalies within given groups."""

    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    input_frequency = _check_input_frequency(hgt, time_name=time_name)

    # Calculate seasonal height anomalies
    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    base_period_hgt = hgt.where(
        (hgt[time_name] >= base_period[0]) &
        (hgt[time_name] <= base_period[1]), drop=True)

    if season is None:
        season = 'ALL'

    valid_seasons = ('DJF', 'MAM', 'JJA', 'SON', 'ALL')
    if season not in valid_seasons:
        raise ValueError("Unrecognized season '%r'" % season)

    if season != 'ALL':
        base_period_hgt = base_period_hgt.where(
            base_period_hgt[time_name].dt.season == season, drop=True)

    if input_frequency == 'daily':

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.dayofyear).mean(time_name)

        hgt_anom = (base_period_hgt.groupby(
            base_period_hgt[time_name].dt.dayofyear) - clim)

        if 'dayofyear' not in hgt.coords:
            hgt_anom = hgt_anom.drop('dayofyear')

    elif input_frequency == 'monthly':

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = (base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month) - clim)

        if 'month' not in hgt.coords:
            hgt_anom = hgt_anom.drop('month')

    else:

        clim = base_period_hgt.mean(time_name)

        hgt_anom = base_period_hgt - clim

    return hgt_anom


def calculate_kmeans_pcs_anomalies(hgt, frequency=None, base_period=None,
                                   time_name=None):
    """Calculate anomalies from input data after applying smoothing.

    Parameters
    ----------
    hgt : xarray DataArray
        Array containing geopotential height values.

    window_length : integer
        Length of moving average window.

    base_period : list
        If given, a two element list containing the earliest and latest
        dates to include in the base period for calculating the climatology.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    hgt_anom : xarray DataArray
        Array containing geopotential height anomalies.

    base_period : list
        Base period used for climatology.
    """

    hgt = rdu.ensure_data_array(hgt)

    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    frequency = _check_anomalies_frequency(frequency)
    input_frequency = _check_anomalies_input_frequency(
        hgt, frequency, time_name=time_name)

    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    base_period_hgt = hgt.where(
        (hgt[time_name] >= base_period[0]) &
        (hgt[time_name] <= base_period[1]), drop=True)

    if input_frequency == 'daily' and frequency == 'daily':

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.dayofyear).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.dayofyear) - clim

        if 'dayofyear' not in hgt.coords:
            hgt_anom = hgt_anom.drop('dayofyear')

    elif input_frequency == 'monthly' and frequency == 'monthly':

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - clim

        if 'month' not in hgt.coords:
            hgt_anom = hgt_anom.drop('month')

    elif input_frequency == 'daily' and frequency == 'monthly':

        hgt = rdu.downsample_data(hgt, frequency='monthly',
                                  time_name=time_name)

        base_period_hgt = hgt.where(
            (hgt[time_name] >= base_period[0]) &
            (hgt[time_name] <= base_period[1]), drop=True)

        clim = base_period_hgt.groupby(
            base_period_hgt[time_name].dt.month).mean(time_name)

        hgt_anom = hgt.groupby(hgt[time_name].dt.month) - clim

        if 'month' not in hgt.coords:
            hgt_anom = hgt_anom.drop('month')

    else:

        clim = base_period_hgt.mean(time_name)

        hgt_anom = hgt - clim

    return hgt_anom


def kmeans_pc_clustering(hgt_anom, n_modes=20, n_clusters=4,
                         lat_name=None, lon_name=None,
                         time_name=None, **kwargs):
    """Perform k-means clustering of leading PCs.

    Parameters
    ----------
    hgt_anom : xarray DataArray
        Array containing geopotential height anomalies.

    n_modes : integer
        Number of EOF modes to compute.

    n_clusters : integer
        Number of clusters.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    result : xarray Dataset
        Dataset containing the result of the combined EOF and cluster
        analysis.
    """

    hgt_anom = rdu.ensure_data_array(hgt_anom)

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(hgt_anom)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(hgt_anom)
    time_name = (time_name if time_name is not None else
                 rdu.get_time_name(hgt_anom))

    if n_modes is None:
        n_modes = 20

    if not rdu.is_integer(n_modes) or n_modes < 1:
        raise ValueError('Number of modes must be a positive integer')

    if n_clusters is None:
        n_clusters = 4

    if not rdu.is_integer(n_clusters) or n_clusters < 1:
        raise ValueError('Number of clusters must be a positive integer')

    # EOF analysis is restricted to North Atlantic region.
    hgt_anom = rdu.select_latlon_box(
        hgt_anom, lat_bounds=[20.0, 80.0],
        lon_bounds=[-90.0, 30.0], lat_name=lat_name, lon_name=lon_name)

    scos_weights = np.cos(
        np.deg2rad(hgt_anom[lat_name])).clip(0., 1.)**0.5

    eofs_ds = rdu.eofs(hgt_anom, weight=scos_weights, n_modes=n_modes,
                       sample_dim=time_name)

    n_samples = eofs_ds[time_name].size
    if eofs_ds['PCs'].values.shape == (n_samples, n_modes):
        pcs_data = eofs_ds['PCs'].values
    else:
        pcs_data = eofs_ds['PCs'].values.T

    kmeans = KMeans(n_clusters=n_clusters, **kwargs).fit(pcs_data)

    inertia = kmeans.inertia_
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    labels, clusters_centers = _fix_cluster_ordering(
        labels, cluster_centers)

    eofs_ds['centroids'] = xr.DataArray(
        cluster_centers, coords={'cluster': np.arange(n_clusters),
                                 'mode': np.arange(n_modes)},
        dims=['cluster', 'mode'])
    eofs_ds['inertia'] = xr.DataArray(
        inertia, coords={time_name: eofs_ds[time_name]},
        dims=[time_name])
    eofs_ds['labels'] = xr.DataArray(
        labels, coords={time_name: eofs_ds[time_name]}, dims=[time_name])

    return eofs_ds


def kmeans_pcs_composites(hgt, season=None, n_modes=20, n_clusters=4,
                          base_period=None, use_reference_patterns=True,
                          lat_name=None, lon_name=None, time_name=None,
                          **kwargs):
    """Calculate composite fields based on cluster assignments.

    Parameters
    ----------
    hgt : xarray DataArray
        Array containing geopotential height values.

    season : None | 'DJF' | 'MAM' | 'JJA' | 'SON' | 'ALL'
        Season to restrict EOF analysis to. If None, all seasons are used.

    n_modes : integer
        Number of EOF modes to compute.

    n_clusters : integer
        Number of clusters.

    base_period : list
        If given, a two element list containing the earliest and latest
        dates to include in the EOF analysis.

    use_reference_patterns : bool, default: True
        If available, order calculated composites according to
        pattern correlation with reference patterns.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    composites : xarray DataArray
        Array containing the composited geopotential height anomalies.
    """

    hgt = rdu.ensure_data_array(hgt)

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(hgt)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(hgt)
    time_name = time_name if time_name is not None else rdu.get_time_name(hgt)

    hgt_anom = _calculate_composite_height_anomalies(
        hgt, season=season, base_period=base_period, time_name=time_name)

    # Calculate EOFs and PCs to perform clustering on
    eofs_ds = kmeans_pc_clustering(
        hgt_anom, n_modes=n_modes, n_clusters=n_clusters,
        lat_name=lat_name, lon_name=lon_name, time_name=time_name, **kwargs)

    cluster_assignments = eofs_ds['labels']

    # Select time-steps present in initial cluster analysis
    cluster_hgt_anom = hgt_anom.sel(
        {time_name: cluster_assignments[time_name]}, drop=True)

    scos_weights = np.cos(np.deg2rad(
        cluster_hgt_anom[lat_name])).clip(0., 1.)**0.5

    # Perform composite over each cluster state
    if n_clusters is None:
        n_clusters = np.size(np.unique(cluster_assignments.values))

    if not rdu.is_integer(n_clusters) or n_clusters < 1:
        raise ValueError('Number of clusters must be a positive integer')

    composite_shape = cluster_hgt_anom.isel({time_name: 0}).shape
    composite_coords = dict(c for c in cluster_hgt_anom.coords.items()
                            if c != time_name)
    composite_coords['cluster'] = np.arange(n_clusters)

    composite_dims = cluster_hgt_anom.isel({time_name: 0}).dims
    composite_dims = ('cluster',) + composite_dims

    composites = xr.DataArray(np.empty((n_clusters,) + composite_shape),
                              coords=composite_coords,
                              dims=composite_dims)

    for k in range(n_clusters):
        cluster_members = cluster_hgt_anom.where(
            cluster_assignments == k, drop=True)

        cluster_mean = cluster_members.mean(time_name)

        weighted_mean = scos_weights * cluster_mean

        if weighted_mean.shape != cluster_mean.shape:
            weighted_mean = weighted_mean.transpose(*(cluster_mean.dims))

        weighted_mean = weighted_mean / np.sqrt(np.sum(weighted_mean ** 2))

        composites.loc[dict(cluster=k)] = weighted_mean

    if use_reference_patterns and n_clusters <= N_REFERENCE_PATTERNS:
        composites = _order_by_reference_patterns(
            composites, lat_name=lat_name, lon_name=lon_name)

    return composites


def kmeans_pcs(hgt, frequency='monthly', base_period=None, n_modes=20,
               n_clusters=4, window_length=None, season='DJF',
               lat_name=None, lon_name=None, time_name=None, **kwargs):
    """Calculate k-means based teleconnection indices.

    Parameters
    ----------
    hgt : xarray DataArray
        Array containing geopotential height values.

    frequency : 'daily' | 'monthly'
        Sampling rate at which to calculate the index.

    base_period : list
        If given, a two element list containing the earliest and
        latest dates to include in base period for standardization and
        calculation of EOFs. If None, then the full time period is used.

    n_modes : integer
        Number of EOF modes to calculate.

    n_clusters : integer
        Number of clusters to use in k-means clustering.

    window_length : integer
        Length of window used for rolling mean. If None and daily data
        is given, a length of 10 days is used. Otherwise, no rolling
        average is performed.

    season : None | 'DJF' | 'MAM' | 'JJA' | 'SON' | 'ALL'
        Season to perform EOF analysis on.

    lat_name : str
        Name of the latitude coordinate.

    lon_name : str
        Name of the longitude coordinate.

    time_name : str
        Name of the time coordinate.

    Returns
    -------
    ds : xarray Dataset
        Dataset containing the indices and loading patterns.
    """

    # Ensure that given data is a data array.
    hgt = rdu.ensure_data_array(hgt)

    lat_name = lat_name if lat_name is not None else rdu.get_lat_name(hgt)
    lon_name = lon_name if lon_name is not None else rdu.get_lon_name(hgt)
    time_name = (time_name if time_name is not None else
                 rdu.get_time_name(hgt))

    if frequency is None:
        frequency = 'monthly'

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unsupported frequency '%r'" % frequency)

    input_frequency = rdu.detect_frequency(hgt, time_name=time_name)

    if input_frequency not in ('daily', 'monthly'):
        raise RuntimeError(
            'Can only calculate index for daily or monthly data')

    if input_frequency == 'monthly' and frequency == 'daily':
        raise ValueError('Cannot calculate daily index from monthly data')

    hgt = filter_hgt_field(hgt, window_length=window_length,
                           time_name=time_name)

    base_period = rdu.check_base_period(hgt, base_period=base_period,
                                        time_name=time_name)

    composites = kmeans_pcs_composites(
        hgt, season=season, n_modes=n_modes, n_clusters=n_clusters,
        base_period=base_period, lat_name=lat_name, lon_name=lon_name,
        time_name=time_name, **kwargs)

    hgt_anom = calculate_kmeans_pcs_anomalies(
        hgt, frequency=frequency, base_period=base_period, time_name=time_name)

    scos_weights = np.cos(np.deg2rad(hgt_anom[lat_name])).clip(0., 1.)**0.5

    indices = _project_onto_subspace(
        hgt_anom, composites, weight=scos_weights,
        sample_dim=time_name, mode_dim='cluster')

    # Standardize indices by monthly means and standard deviations
    # within base period.
    if frequency == 'monthly':

        base_period_indices = indices.where(
            (indices[time_name] >= base_period[0]) &
            (indices[time_name] <= base_period[1]), drop=True)

    else:

        monthly_hgt_anom = calculate_kmeans_pcs_anomalies(
            hgt, frequency='monthly', base_period=base_period,
            time_name=time_name)

        monthly_indices = _project_onto_subspace(
            monthly_hgt_anom, composites, weight=scos_weights,
            sample_dim=time_name, mode_dim='cluster')

        base_period_indices = monthly_indices.where(
            (monthly_indices[time_name] >= base_period[0]) &
            (monthly_indices[time_name] <= base_period[1]), drop=True)

    indices_mean = base_period_indices.groupby(
        base_period_indices[time_name].dt.month).mean(time_name)

    indices_std = base_period_indices.groupby(
        base_period_indices[time_name].dt.month).std(time_name)

    indices = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        indices.groupby(indices[time_name].dt.month),
        indices_mean, indices_std, dask='allowed')

    data_vars = {'composites': composites, 'indices': indices}

    ds = xr.Dataset(data_vars)

    ds.attrs['base_period_start'] = rdu.datetime_to_string(
        base_period[0], '%Y%m%d')
    ds.attrs['base_period_end'] = rdu.datetime_to_string(
        base_period[1], '%Y%m%d')
    ds.attrs['n_modes'] = '{:d}'.format(n_modes)
    ds.attrs['n_clusters'] = '{:d}'.format(n_clusters)
    if window_length is not None:
        ds.attrs['window_length'] = '{:d}'.format(window_length)
    ds.attrs['season'] = season

    return ds
