"""
Provides unit tests for EOF functions.
"""

# License: MIT

from __future__ import absolute_import

import os

import dask.array
import numpy as np
import pytest
import sklearn.decomposition as sd
import xarray as xr

import reanalysis_dbns.utils as rdu

from reanalysis_dbns.utils.eofs import (_apply_weights_to_arrays,
                                        _bsv_orthomax_rotation,
                                        _ensure_leading_sample_dim,
                                        _ensure_leading_sample_dims,
                                        _expand_and_weight_datasets)


TESTS_DIR = os.path.realpath(os.path.dirname(__file__))
TEST_SWISS_DATA_FILE = os.path.join(TESTS_DIR, 'swiss.csv')


def test_ensure_leading_sample_dim():
    """Test enforcing sample dimension being the first axis."""

    x = np.random.uniform(size=(10, 2, 4))
    x_checked = _ensure_leading_sample_dim(x, sample_dim=0)
    assert np.allclose(x, x_checked)

    x = np.random.uniform(size=(3, 20, 4))
    x_checked = _ensure_leading_sample_dim(x, sample_dim=1)
    assert x_checked.shape == (20, 3, 4)
    assert np.allclose(x_checked, x.transpose((1, 0, 2)))

    x = np.random.uniform(size=(3, 5, 4))
    x_checked = _ensure_leading_sample_dim(x, sample_dim=-1)
    assert x_checked.shape == (4, 3, 5)
    assert np.allclose(x_checked, x.transpose((2, 0, 1)))

    x = [np.random.uniform(size=(2, 20, 5)),
         np.random.uniform(size=(4, 20, 2))]
    x_checked = _ensure_leading_sample_dims(x, sample_dim=1)
    for i, xi in enumerate(x):
        assert np.allclose(x_checked[i], xi.transpose((1, 0, 2)))

    with dask.config.set(scheduler='single-threaded'):
        x = dask.array.from_array(np.random.uniform(size=(20, 2, 3)))
        x_checked = _ensure_leading_sample_dim(x, sample_dim=0)
        assert np.allclose(x, x_checked)

        x = dask.array.from_array(np.random.uniform(size=(20, 2, 30)))
        x_checked = _ensure_leading_sample_dim(x, sample_dim=2)
        assert x_checked.shape == (30, 20, 2)
        assert np.allclose(x_checked, x.transpose((2, 0, 1)))

        x = dask.array.from_array(np.random.uniform(size=(20, 2, 30)))
        x_checked = _ensure_leading_sample_dim(x, sample_dim=-2)
        assert x_checked.shape == (2, 20, 30)
        assert np.allclose(x_checked, x.transpose((1, 0, 2)))

    x = xr.DataArray(np.random.uniform(size=(15, 5, 3)),
                     coords={'x': np.arange(15),
                             'y': np.arange(5),
                             'z': np.arange(3)},
                     dims=['x', 'y', 'z'])
    x_checked = _ensure_leading_sample_dim(x, sample_dim='x')
    assert x_checked.equals(x)

    x = xr.DataArray(np.random.uniform(size=(10, 25, 3)),
                     coords={'x': np.arange(10),
                             'y': np.arange(25),
                             'z': np.arange(3)},
                     dims=['x', 'y', 'z'])
    x_checked = _ensure_leading_sample_dim(x, sample_dim='y')
    assert x_checked.equals(x.transpose('y', 'x', 'z'))


def test_expand_and_weight_datasets():
    """Test expanding and weighting datasets."""

    x = [xr.DataArray(np.random.uniform(size=(10, 3, 5)),
                      coords={'x': np.arange(10),
                              'y': np.arange(3),
                              'z': np.arange(5)},
                      dims=['x', 'y', 'z'])]

    x_weighted = _expand_and_weight_datasets(x, sample_dim='x')

    assert x[0].equals(x_weighted[0])

    weights = xr.DataArray(np.arange(3), coords={'y': np.arange(3)},
                           dims=['y'])

    x_weighted = _expand_and_weight_datasets(
        x, sample_dim='x', weight=weights)
    expected = (weights * x[0]).transpose('x', 'y', 'z')
    assert x_weighted[0].equals(expected)

    x1 = xr.DataArray(np.random.uniform(size=(30, 2, 4)),
                      coords={'x': np.arange(30),
                              'y': np.arange(2),
                              'z': np.arange(4)}, dims=['x', 'y', 'z'])
    x2 = xr.DataArray(np.random.uniform(size=(4, 30)),
                      coords={'x': np.arange(30),
                              'z': np.arange(4)}, dims=['z', 'x'])

    ds = xr.Dataset(data_vars={'x1': x1, 'x2': x2})

    x_weighted = _expand_and_weight_datasets([ds], sample_dim='x')

    assert len(x_weighted) == 2
    assert x_weighted[0].equals(x1)
    assert x_weighted[1].equals(x2.transpose('x', 'z'))

    x3 = xr.DataArray(np.random.uniform(size=(4, 2, 30)),
                      coords={'x': np.arange(30),
                              'y': np.arange(2),
                              'z': np.arange(4)}, dims=['z', 'y', 'x'])

    x_weighted = _expand_and_weight_datasets([ds, x3], sample_dim='x')

    assert len(x_weighted) == 3
    assert x_weighted[0].equals(x1)
    assert x_weighted[1].equals(x2.transpose('x', 'z'))
    assert x_weighted[2].equals(x3.transpose('x', 'z', 'y'))

    weights = xr.DataArray(
        np.arange(4), coords={'z': np.arange(4)}, dims=['z'])

    x_weighted = _expand_and_weight_datasets(
        [ds], sample_dim='x', weight=weights)

    assert len(x_weighted) == 2
    assert x_weighted[0].equals((weights * x1).transpose('x', 'y', 'z'))
    assert x_weighted[1].equals((weights * x2).transpose('x', 'z'))

    x_weighted = _expand_and_weight_datasets(
        [ds, x3], sample_dim='x', weight=weights)

    assert len(x_weighted) == 3
    assert x_weighted[0].equals((weights * x1).transpose('x', 'y', 'z'))
    assert x_weighted[1].equals((weights * x2).transpose('x', 'z'))
    assert x_weighted[2].equals((weights * x3).transpose('x', 'z', 'y'))

    w1 = xr.DataArray(
        np.arange(4), coords={'z': np.arange(4)}, dims=['z'])
    w2 = None

    x_weighted = _expand_and_weight_datasets(
        [ds, x3], sample_dim='x', weight=[w1, w2])

    assert len(x_weighted) == 3
    assert x_weighted[0].equals((w1 * x1).transpose('x', 'y', 'z'))
    assert x_weighted[1].equals((w1 * x2).transpose('x', 'z'))
    assert x_weighted[2].equals(x3.transpose('x', 'z', 'y'))


def test_apply_weights_to_arrays():
    """Test weighting plain arrays."""

    x = np.random.uniform(size=(30, 3, 4))
    w = np.full((3, 4), 2.0)

    weighted = _apply_weights_to_arrays([x])
    assert np.allclose(weighted[0], x)

    weighted = _apply_weights_to_arrays([x], weights=[w])
    expected = x * w[np.newaxis, ...]
    assert np.allclose(weighted[0], expected)

    x1 = np.random.uniform(size=(30, 3, 4))
    x2 = np.random.uniform(size=(30, 5, 6))

    weighted = _apply_weights_to_arrays([x1, x2])
    assert np.allclose(weighted[0], x1)
    assert np.allclose(weighted[1], x2)

    w1 = np.full((3, 4), 5.0)
    w2 = np.full((5, 6), 2.0)

    weighted = _apply_weights_to_arrays([x1, x2], weights=[w1, None])
    assert np.allclose(weighted[0], x1 * w1[np.newaxis, ...])
    assert np.allclose(weighted[1], x2)

    weighted = _apply_weights_to_arrays([x1, x2], weights=[w1, w2])
    assert np.allclose(weighted[0], x1 * w1[np.newaxis, ...])
    assert np.allclose(weighted[1], x2 * w2[np.newaxis, ...])

    with pytest.raises(ValueError):
        _apply_weights_to_arrays([x1, x2], weights=[w1])

    with dask.config.set(scheduler='single-threaded'):
        x = dask.array.from_array(np.random.uniform(size=(30, 3, 4)))
        w = dask.array.from_array(np.full((3, 4), 2.0))

        weighted = _apply_weights_to_arrays([x])
        assert np.allclose(weighted[0], x)

        weighted = _apply_weights_to_arrays([x], weights=[w])
        expected = x * w[np.newaxis, ...]
        assert np.allclose(weighted[0], expected)

        x1 = dask.array.from_array(np.random.uniform(size=(30, 3, 4)))
        x2 = dask.array.from_array(np.random.uniform(size=(30, 5, 6)))

        weighted = _apply_weights_to_arrays([x1, x2])
        assert np.allclose(weighted[0], x1)
        assert np.allclose(weighted[1], x2)

        w1 = dask.array.from_array(np.full((3, 4), 5.0))
        w2 = dask.array.from_array(np.full((5, 6), 2.0))

        weighted = _apply_weights_to_arrays([x1, x2], weights=[w1, None])
        assert np.allclose(weighted[0], x1 * w1[np.newaxis, ...])
        assert np.allclose(weighted[1], x2)

        weighted = _apply_weights_to_arrays([x1, x2], weights=[w1, w2])
        assert np.allclose(weighted[0], x1 * w1[np.newaxis, ...])
        assert np.allclose(weighted[1], x2 * w2[np.newaxis, ...])


def test_eofs_numpy_arrays():
    """Test implementation of EOFs for plain arrays."""

    x = np.random.uniform(size=(10, 5))
    x = x - x.mean(axis=0, keepdims=True)

    k = 3
    eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k)

    u, s, vh = rdu.calc_truncated_svd(x, k)
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(x, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    w = 2.0 * np.ones(5)

    k = 4
    eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k, weight=w)

    u, s, vh = rdu.calc_truncated_svd((x * w[np.newaxis, :]), k)
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = (
        mode_var / np.sum(np.var((x * w[np.newaxis, :]), axis=0, ddof=1)))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = x - x.mean(axis=1, keepdims=True)

    k = 2
    eofs_ds = rdu.eofs(x, n_modes=k, sample_dim=-1)

    u, s, vh = rdu.calc_truncated_svd(x.transpose(), k)
    mode_var = s**2 / (x.shape[1] - 1)
    explained_var = mode_var / np.sum(np.var(x, axis=1, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x1 = np.random.uniform(size=(30, 1))
    x1 = x1 - x1.mean(axis=0, keepdims=True)

    x2 = np.random.uniform(size=(30, 1))
    x2 = x2 - x2.mean(axis=0, keepdims=True)

    x3 = np.random.uniform(size=(30, 1))
    x3 = x3 - x3.mean(axis=0, keepdims=True)

    k = 3
    eofs_ds1, eofs_ds2, eofs_ds3 = rdu.eofs(
        x1, x2, x3, sample_dim=0, n_modes=k)

    x = np.hstack([x1, x2, x3])
    u, s, vh = rdu.calc_truncated_svd(x, k)
    pcs = np.dot(u, np.diag(s))
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(x, axis=0, ddof=1))

    assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
    assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
    assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)

    assert np.allclose(pcs, eofs_ds1['PCs'].data)
    assert np.allclose(pcs, eofs_ds2['PCs'].data)
    assert np.allclose(pcs, eofs_ds3['PCs'].data)

    assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds3['explained_var'].data)

    weights = np.array([2.0])
    eofs_ds1, eofs_ds2, eofs_ds3 = rdu.eofs(
        x1, x2, x3, sample_dim=0, n_modes=k, weight=weights)

    x = np.hstack([x1, x2, x3])
    w = np.full((x1.shape[0], 3), 2.0)
    u, s, vh = rdu.calc_truncated_svd(w * x, k)
    pcs = np.dot(u, np.diag(s))
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(w * x, axis=0, ddof=1))

    assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
    assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
    assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)

    assert np.allclose(pcs, eofs_ds1['PCs'].data)
    assert np.allclose(pcs, eofs_ds2['PCs'].data)
    assert np.allclose(pcs, eofs_ds3['PCs'].data)

    assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds3['explained_var'].data)

    weights = [None, np.array([2.0]), np.array([3.0])]
    eofs_ds1, eofs_ds2, eofs_ds3 = rdu.eofs(
        x1, x2, x3, sample_dim=0, n_modes=k, weight=weights)

    x = np.hstack([x1, x2, x3])
    w = np.array([1.0, 2.0, 3.0])
    u, s, vh = rdu.calc_truncated_svd(w[np.newaxis, :] * x, k)
    pcs = np.dot(u, np.diag(s))
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(w * x, axis=0, ddof=1))

    assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
    assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
    assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)

    assert np.allclose(pcs, eofs_ds1['PCs'].data)
    assert np.allclose(pcs, eofs_ds2['PCs'].data)
    assert np.allclose(pcs, eofs_ds3['PCs'].data)

    assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds3['explained_var'].data)

    x1 = np.random.uniform(size=(1, 30))
    x1 = x1 - x1.mean(axis=1, keepdims=True)

    x2 = np.random.uniform(size=(1, 30))
    x2 = x2 - x2.mean(axis=1, keepdims=True)

    x3 = np.random.uniform(size=(1, 30))
    x3 = x3 - x3.mean(axis=1, keepdims=True)

    x4 = np.random.uniform(size=(1, 30))
    x4 = x4 - x4.mean(axis=1, keepdims=True)

    k = 3
    eofs_ds1, eofs_ds2, eofs_ds3, eofs_ds4 = rdu.eofs(
        x1, x2, x3, x4, sample_dim=1, n_modes=k)

    x = np.vstack([x1, x2, x3, x4])
    u, s, vh = rdu.calc_truncated_svd(x.transpose(), k)
    pcs = np.dot(u, np.diag(s))
    mode_var = s**2 / (x.shape[1] - 1)
    explained_var = mode_var / np.sum(np.var(x, axis=1, ddof=1))

    assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
    assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
    assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)
    assert np.allclose(vh[:, 3].reshape((k, 1)), eofs_ds4['EOFs'].data)

    assert np.allclose(pcs, eofs_ds1['PCs'].data)
    assert np.allclose(pcs, eofs_ds2['PCs'].data)
    assert np.allclose(pcs, eofs_ds3['PCs'].data)
    assert np.allclose(pcs, eofs_ds4['PCs'].data)

    assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds3['explained_var'].data)
    assert np.allclose(explained_var, eofs_ds4['explained_var'].data)

    x = np.random.uniform(size=(40, 5, 3))
    x = x - x.mean(axis=0)

    k = 2
    eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k)

    u, s, vh = rdu.calc_truncated_svd(x.reshape((40, 15)), k)
    vh = vh.reshape((k, 5, 3))
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = (
        mode_var / np.sum(np.var(x.reshape((40, 15)), axis=0, ddof=1)))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    weights = np.tile(np.arange(3), (5, 1))

    k = 2
    eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k, weight=weights)

    weighted_x = np.broadcast_to(weights, (40, 5, 3)) * x
    u, s, vh = rdu.calc_truncated_svd(weighted_x.reshape((40, 15)), k)
    vh = vh.reshape((k, 5, 3))
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = (
        mode_var / np.sum(np.var(weighted_x.reshape((40, 15)),
                                 axis=0, ddof=1)))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = np.random.uniform(size=(4, 30, 6))
    x = x - x.mean(axis=1, keepdims=True)

    weights = np.random.uniform(size=(4, 6))

    k = 2
    eofs_ds = rdu.eofs(x, sample_dim=1, n_modes=k, weight=weights)

    weighted_x = np.broadcast_to(weights, (30, 4, 6)) * x.transpose((1, 0, 2))
    u, s, vh = rdu.calc_truncated_svd(weighted_x.reshape((30, 24)), k)
    vh = vh.reshape((k, 4, 6))
    mode_var = s**2 / (x.shape[1] - 1)
    explained_var = (
        mode_var / np.sum(np.var(weighted_x.reshape((30, 24)),
                                 axis=0, ddof=1)))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x1 = np.random.uniform(size=(4, 30, 6))
    x1 = x1 - x1.mean(axis=1, keepdims=True)

    w1 = np.random.uniform(size=(4, 6))

    x2 = np.random.uniform(size=(5, 30, 3))
    x2 = x2 - x2.mean(axis=1, keepdims=True)

    w2 = None

    k = 2
    eofs_ds1, eofs_ds2 = rdu.eofs(
        x1, x2, sample_dim=1, n_modes=k, weight=[w1, w2])

    weighted_x1 = np.broadcast_to(w1, (30, 4, 6)) * x1.transpose((1, 0, 2))
    weighted_x2 = x2.transpose((1, 0, 2))
    weighted_x = np.hstack([weighted_x1.reshape((30, 24)),
                            weighted_x2.reshape((30, 15))])
    u, s, vh = rdu.calc_truncated_svd(weighted_x, k)
    vh1 = vh[:, :24].reshape((k, 4, 6))
    vh2 = vh[:, 24:].reshape((k, 5, 3))
    mode_var = s**2 / (x.shape[1] - 1)
    explained_var = (
        mode_var / np.sum(np.var(weighted_x,
                                 axis=0, ddof=1)))

    assert np.allclose(vh1, eofs_ds1['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds1['PCs'].data)
    assert np.allclose(explained_var, eofs_ds1['explained_var'].data)

    assert np.allclose(vh2, eofs_ds2['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds2['PCs'].data)
    assert np.allclose(explained_var, eofs_ds2['explained_var'].data)

    x = np.random.uniform(size=(20, 5, 6))
    x = x - x.mean(axis=0, keepdims=True)
    x[:, 3, 2] = np.NaN

    k = 4
    eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k)

    flat_x = x.reshape((20, 30))
    nonmissing_mask = np.logical_not(np.all(np.isnan(flat_x), axis=0))
    nonmissing_x = flat_x[:, nonmissing_mask]

    u, s, nonmissing_vh = rdu.calc_truncated_svd(nonmissing_x, k)
    vh = np.full((k, 30), np.NaN)
    vh[:, nonmissing_mask] = nonmissing_vh
    vh = vh.reshape((k, 5, 6))
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = (
        mode_var / np.sum(np.var(nonmissing_x,
                                 axis=0, ddof=1)))

    assert np.allclose(vh, eofs_ds['EOFs'].data, equal_nan=True)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)


def test_eofs_dask_arrays():
    """Test implementation of EOFs for plain arrays."""

    with dask.config.set(scheduler='single-threaded'):

        x = dask.array.from_array(np.random.uniform(size=(10, 5)))
        x = x - x.mean(axis=0, keepdims=True)

        k = 3
        eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k)

        u, s, vh = rdu.calc_truncated_svd(x, k)
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = mode_var / np.sum(np.var(x, axis=0, ddof=1))

        assert np.allclose(vh, eofs_ds['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)

        w = dask.array.from_array(2.0 * np.ones(5))

        k = 4
        eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k, weight=w)

        u, s, vh = rdu.calc_truncated_svd((x * w[np.newaxis, :]), k)
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = (
            mode_var / np.sum(np.var((x * w[np.newaxis, :]), axis=0, ddof=1)))

        assert np.allclose(vh, eofs_ds['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)

        x = x - x.mean(axis=1, keepdims=True)

        k = 2
        eofs_ds = rdu.eofs(x, n_modes=k, sample_dim=-1)

        u, s, vh = rdu.calc_truncated_svd(x.transpose(), k)
        mode_var = s**2 / (x.shape[1] - 1)
        explained_var = mode_var / np.sum(np.var(x, axis=1, ddof=1))

        assert np.allclose(vh, eofs_ds['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)

        x1 = dask.array.from_array(np.random.uniform(size=(30, 1)))
        x1 = x1 - x1.mean(axis=0, keepdims=True)

        x2 = dask.array.from_array(np.random.uniform(size=(30, 1)))
        x2 = x2 - x2.mean(axis=0, keepdims=True)

        x3 = dask.array.from_array(np.random.uniform(size=(30, 1)))
        x3 = x3 - x3.mean(axis=0, keepdims=True)

        k = 3
        eofs_ds1, eofs_ds2, eofs_ds3 = rdu.eofs(
            x1, x2, x3, sample_dim=0, n_modes=k)

        x = dask.array.hstack([x1, x2, x3])
        u, s, vh = rdu.calc_truncated_svd(x, k)
        pcs = np.dot(u, np.diag(s))
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = mode_var / np.sum(np.var(x, axis=0, ddof=1))

        assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
        assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
        assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)

        assert np.allclose(pcs, eofs_ds1['PCs'].data)
        assert np.allclose(pcs, eofs_ds2['PCs'].data)
        assert np.allclose(pcs, eofs_ds3['PCs'].data)

        assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds3['explained_var'].data)

        weights = dask.array.from_array(np.array([2.0]))
        eofs_ds1, eofs_ds2, eofs_ds3 = rdu.eofs(
            x1, x2, x3, sample_dim=0, n_modes=k, weight=weights)

        x = dask.array.hstack([x1, x2, x3])
        w = dask.array.from_array(np.full((x1.shape[0], 3), 2.0))
        u, s, vh = rdu.calc_truncated_svd(w * x, k)
        pcs = np.dot(u, np.diag(s))
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = mode_var / np.sum(np.var(w * x, axis=0, ddof=1))

        assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
        assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
        assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)

        assert np.allclose(pcs, eofs_ds1['PCs'].data)
        assert np.allclose(pcs, eofs_ds2['PCs'].data)
        assert np.allclose(pcs, eofs_ds3['PCs'].data)

        assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds3['explained_var'].data)

        weights = [None, np.array([2.0]), np.array([3.0])]
        eofs_ds1, eofs_ds2, eofs_ds3 = rdu.eofs(
            x1, x2, x3, sample_dim=0, n_modes=k, weight=weights)

        x = dask.array.hstack([x1, x2, x3])
        w = dask.array.from_array(np.array([1.0, 2.0, 3.0]))
        u, s, vh = rdu.calc_truncated_svd(w[np.newaxis, :] * x, k)
        pcs = np.dot(u, np.diag(s))
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = mode_var / np.sum(np.var(w * x, axis=0, ddof=1))

        assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
        assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
        assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)

        assert np.allclose(pcs, eofs_ds1['PCs'].data)
        assert np.allclose(pcs, eofs_ds2['PCs'].data)
        assert np.allclose(pcs, eofs_ds3['PCs'].data)

        assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds3['explained_var'].data)

        x1 = dask.array.from_array(np.random.uniform(size=(1, 30)))
        x1 = x1 - x1.mean(axis=1, keepdims=True)

        x2 = dask.array.from_array(np.random.uniform(size=(1, 30)))
        x2 = x2 - x2.mean(axis=1, keepdims=True)

        x3 = dask.array.from_array(np.random.uniform(size=(1, 30)))
        x3 = x3 - x3.mean(axis=1, keepdims=True)

        x4 = dask.array.from_array(np.random.uniform(size=(1, 30)))
        x4 = x4 - x4.mean(axis=1, keepdims=True)

        k = 3
        eofs_ds1, eofs_ds2, eofs_ds3, eofs_ds4 = rdu.eofs(
            x1, x2, x3, x4, sample_dim=1, n_modes=k)

        x = dask.array.vstack([x1, x2, x3, x4])
        u, s, vh = rdu.calc_truncated_svd(x.transpose(), k)
        pcs = np.dot(u, np.diag(s))
        mode_var = s**2 / (x.shape[1] - 1)
        explained_var = mode_var / np.sum(np.var(x, axis=1, ddof=1))

        assert np.allclose(vh[:, 0].reshape((k, 1)), eofs_ds1['EOFs'].data)
        assert np.allclose(vh[:, 1].reshape((k, 1)), eofs_ds2['EOFs'].data)
        assert np.allclose(vh[:, 2].reshape((k, 1)), eofs_ds3['EOFs'].data)
        assert np.allclose(vh[:, 3].reshape((k, 1)), eofs_ds4['EOFs'].data)

        assert np.allclose(pcs, eofs_ds1['PCs'].data)
        assert np.allclose(pcs, eofs_ds2['PCs'].data)
        assert np.allclose(pcs, eofs_ds3['PCs'].data)
        assert np.allclose(pcs, eofs_ds4['PCs'].data)

        assert np.allclose(explained_var, eofs_ds1['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds2['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds3['explained_var'].data)
        assert np.allclose(explained_var, eofs_ds4['explained_var'].data)

        x = dask.array.from_array(np.random.uniform(size=(40, 5, 3)))
        x = x - x.mean(axis=0)

        k = 2
        eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k)

        u, s, vh = rdu.calc_truncated_svd(x.reshape((40, 15)), k)
        vh = vh.reshape((k, 5, 3))
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = (
            mode_var / np.sum(np.var(x.reshape((40, 15)), axis=0, ddof=1)))

        assert np.allclose(vh, eofs_ds['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)

        weights = dask.array.from_array(np.tile(np.arange(3), (5, 1)))

        k = 2
        eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k, weight=weights)

        weighted_x = np.broadcast_to(weights, (40, 5, 3)) * x
        u, s, vh = rdu.calc_truncated_svd(weighted_x.reshape((40, 15)), k)
        vh = vh.reshape((k, 5, 3))
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = (
            mode_var / np.sum(np.var(weighted_x.reshape((40, 15)),
                                     axis=0, ddof=1)))

        assert np.allclose(vh, eofs_ds['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)

        x = dask.array.from_array(np.random.uniform(size=(4, 30, 6)))
        x = x - x.mean(axis=1, keepdims=True)

        weights = dask.array.from_array(np.random.uniform(size=(4, 6)))

        k = 2
        eofs_ds = rdu.eofs(x, sample_dim=1, n_modes=k, weight=weights)

        weighted_x = (np.broadcast_to(weights, (30, 4, 6)) *
                      x.transpose((1, 0, 2)))
        u, s, vh = rdu.calc_truncated_svd(weighted_x.reshape((30, 24)), k)
        vh = vh.reshape((k, 4, 6))
        mode_var = s**2 / (x.shape[1] - 1)
        explained_var = (
            mode_var / np.sum(np.var(weighted_x.reshape((30, 24)),
                                     axis=0, ddof=1)))

        assert np.allclose(vh, eofs_ds['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)

        x1 = dask.array.from_array(np.random.uniform(size=(4, 30, 6)))
        x1 = x1 - x1.mean(axis=1, keepdims=True)

        w1 = dask.array.from_array(np.random.uniform(size=(4, 6)))

        x2 = dask.array.from_array(np.random.uniform(size=(5, 30, 3)))
        x2 = x2 - x2.mean(axis=1, keepdims=True)

        w2 = None

        k = 2
        eofs_ds1, eofs_ds2 = rdu.eofs(
            x1, x2, sample_dim=1, n_modes=k, weight=[w1, w2])

        weighted_x1 = (np.broadcast_to(w1, (30, 4, 6)) *
                       x1.transpose((1, 0, 2)))
        weighted_x2 = x2.transpose((1, 0, 2))
        weighted_x = dask.array.hstack([weighted_x1.reshape((30, 24)),
                                        weighted_x2.reshape((30, 15))])
        u, s, vh = rdu.calc_truncated_svd(weighted_x, k)
        vh1 = vh[:, :24].reshape((k, 4, 6))
        vh2 = vh[:, 24:].reshape((k, 5, 3))
        mode_var = s**2 / (x.shape[1] - 1)
        explained_var = (
            mode_var / np.sum(np.var(weighted_x,
                                     axis=0, ddof=1)))

        assert np.allclose(vh1, eofs_ds1['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds1['PCs'].data)
        assert np.allclose(explained_var, eofs_ds1['explained_var'].data)

        assert np.allclose(vh2, eofs_ds2['EOFs'].data)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds2['PCs'].data)
        assert np.allclose(explained_var, eofs_ds2['explained_var'].data)

        x = np.random.uniform(size=(20, 5, 6))
        x = x - x.mean(axis=0, keepdims=True)
        x[:, 3, 2] = np.NaN
        x = dask.array.from_array(x)

        k = 4
        eofs_ds = rdu.eofs(x, sample_dim=0, n_modes=k)

        flat_x = x.reshape((20, 30))
        nonmissing_mask = np.logical_not(np.all(np.isnan(flat_x), axis=0))
        nonmissing_x = flat_x[:, nonmissing_mask]

        u, s, nonmissing_vh = rdu.calc_truncated_svd(nonmissing_x, k)
        vh = np.full((k, 30), np.NaN)
        vh[:, nonmissing_mask] = nonmissing_vh
        vh = dask.array.from_array(vh.reshape((k, 5, 6)))
        mode_var = s**2 / (x.shape[0] - 1)
        explained_var = (
            mode_var / np.sum(np.var(nonmissing_x,
                                     axis=0, ddof=1)))

        assert np.allclose(vh, eofs_ds['EOFs'].data, equal_nan=True)
        assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
        assert np.allclose(explained_var, eofs_ds['explained_var'].data)


def test_eofs_xarray_arrays():
    """Test implementation of EOFs for xarray DataArrays."""

    x = xr.DataArray(np.random.uniform(size=(40, 12)),
                     coords={'time': np.arange(40), 'x': np.arange(12)},
                     dims=['time', 'x'])
    x = x - x.mean('time')

    k = 5
    eofs_ds = rdu.eofs(x, n_modes=k)

    u, s, vh = rdu.calc_truncated_svd(x.data, k)
    mode_var = s**2 / (x.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(x.data, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = x.transpose('x', 'time')
    eofs_ds = rdu.eofs(x, n_modes=k)

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = x.transpose('time', 'x')

    w = xr.DataArray(np.arange(12), coords={'x': np.arange(12)}, dims=['x'])

    eofs_ds = rdu.eofs(x, n_modes=k, weight=w)

    weighted_data = np.broadcast_to(w.data, (40, 12)) * x.data

    u, s, vh = rdu.calc_truncated_svd(weighted_data, k)
    mode_var = s**2 / (weighted_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(weighted_data, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = xr.DataArray(np.random.uniform(size=(40, 12, 10)),
                     coords={'time': np.arange(40),
                             'lat': np.arange(12),
                             'lon': np.arange(10)},
                     dims=['time', 'lat', 'lon'])
    x = x - x.mean('time')

    k = 5
    eofs_ds = rdu.eofs(x, n_modes=k)

    flat_data = x.data.reshape((40, 120))
    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh = vh.reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    w = np.cos(np.deg2rad(x['lat'])).clip(0., 1.)**0.5

    eofs_ds = rdu.eofs(x, n_modes=k, weight=w)

    flat_data = (w * x).transpose('time', 'lat', 'lon').data.reshape((40, 120))
    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh = vh.reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = x - x.mean('lat')

    k = 5
    eofs_ds = rdu.eofs(x, n_modes=k, sample_dim='lat')

    flat_data = x.transpose('lat', 'time', 'lon').data.reshape((12, 400))
    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh = vh.reshape((k, 40, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = np.random.uniform(size=(30, 6, 3))
    x = x - x.mean(axis=0, keepdims=True)
    x[:, 1, 2] = np.NaN
    x[:, 4, 1] = np.NaN

    x = xr.DataArray(x, coords={'time': np.arange(30),
                                'lat': np.arange(6),
                                'lon': np.arange(3)},
                     dims=['time', 'lat', 'lon'])

    eofs_ds = rdu.eofs(x, n_modes=k)

    flat_data = x.data.reshape(30, 18)
    nonmissing_mask = np.logical_not(np.all(np.isnan(flat_data), axis=0))
    valid_data = flat_data[:, nonmissing_mask]

    u, s, valid_vh = rdu.calc_truncated_svd(valid_data, k)

    vh = np.full((k, 18), np.NaN)
    vh[:, nonmissing_mask] = valid_vh
    vh = vh.reshape((k, 6, 3))
    mode_var = s**2 / (valid_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(valid_data, axis=0, ddof=1))

    assert np.allclose(vh, eofs_ds['EOFs'].data, equal_nan=True)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds['PCs'].data)
    assert np.allclose(explained_var, eofs_ds['explained_var'].data)

    x = xr.DataArray(np.random.uniform(size=(50, 3, 12, 10)),
                     coords={'time': np.arange(50),
                             'level': np.arange(3),
                             'lat': np.arange(12),
                             'lon': np.arange(10)},
                     dims=['time', 'level', 'lat', 'lon'])
    x = x - x.mean('time')

    y = xr.DataArray(np.random.uniform(size=(50, 12, 10)),
                     coords={'time': np.arange(50),
                             'lat': np.arange(12),
                             'lon': np.arange(10)},
                     dims=['time', 'lat', 'lon'])
    y = y - y.mean('time')

    k = 5
    eofs_ds_x, eofs_ds_y = rdu.eofs(x, y, n_modes=k)

    flat_x_data = x.data.reshape(50, 360)
    flat_y_data = y.data.reshape(50, 120)

    flat_data = np.hstack([flat_x_data, flat_y_data])

    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh_x = vh[:, :360].reshape((k, 3, 12, 10))
    vh_y = vh[:, 360:].reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh_x, eofs_ds_x['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_x['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_x['explained_var'].data)

    assert np.allclose(vh_y, eofs_ds_y['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_y['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_y['explained_var'].data)

    w = np.cos(np.deg2rad(x['lat'])).clip(0., 1.)**0.5

    eofs_ds_x, eofs_ds_y = rdu.eofs(x, y, n_modes=k, weight=w)

    flat_x_data = (w * x).transpose(
        'time', 'level', 'lat', 'lon').data.reshape(50, 360)
    flat_y_data = (w * y).transpose(
        'time', 'lat', 'lon').data.reshape(50, 120)

    flat_data = np.hstack([flat_x_data, flat_y_data])

    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh_x = vh[:, :360].reshape((k, 3, 12, 10))
    vh_y = vh[:, 360:].reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh_x, eofs_ds_x['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_x['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_x['explained_var'].data)

    assert np.allclose(vh_y, eofs_ds_y['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_y['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_y['explained_var'].data)

    w1 = np.cos(np.deg2rad(x['lat'])).clip(0., 1.)**0.5
    w2 = np.cos(np.deg2rad(y['lon'])).clip(0., 1.)**0.5

    eofs_ds_x, eofs_ds_y = rdu.eofs(x, y, n_modes=k, weight=[w1, w2])

    flat_x_data = (w1 * x).transpose(
        'time', 'level', 'lat', 'lon').data.reshape(50, 360)
    flat_y_data = (w2 * y).transpose(
        'time', 'lat', 'lon').data.reshape(50, 120)

    flat_data = np.hstack([flat_x_data, flat_y_data])

    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh_x = vh[:, :360].reshape((k, 3, 12, 10))
    vh_y = vh[:, 360:].reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh_x, eofs_ds_x['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_x['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_x['explained_var'].data)

    assert np.allclose(vh_y, eofs_ds_y['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_y['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_y['explained_var'].data)

    ds = xr.Dataset({'x': x, 'y': y})

    eofs_ds_x, eofs_ds_y = rdu.eofs(ds, n_modes=k)

    flat_x_data = x.data.reshape(50, 360)
    flat_y_data = y.data.reshape(50, 120)

    flat_data = np.hstack([flat_x_data, flat_y_data])

    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh_x = vh[:, :360].reshape((k, 3, 12, 10))
    vh_y = vh[:, 360:].reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh_x, eofs_ds_x['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_x['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_x['explained_var'].data)

    assert np.allclose(vh_y, eofs_ds_y['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_y['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_y['explained_var'].data)

    w = np.cos(np.deg2rad(x['lat'])).clip(0., 1.)**0.5

    eofs_ds_x, eofs_ds_y = rdu.eofs(ds, n_modes=k, weight=w)

    flat_x_data = (w * x).transpose(
        'time', 'level', 'lat', 'lon').data.reshape(50, 360)
    flat_y_data = (w * y).transpose(
        'time', 'lat', 'lon').data.reshape(50, 120)

    flat_data = np.hstack([flat_x_data, flat_y_data])

    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh_x = vh[:, :360].reshape((k, 3, 12, 10))
    vh_y = vh[:, 360:].reshape((k, 12, 10))
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh_x, eofs_ds_x['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_x['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_x['explained_var'].data)

    assert np.allclose(vh_y, eofs_ds_y['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_y['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_y['explained_var'].data)

    z = xr.DataArray(np.random.uniform(size=(3, 50)),
                     coords={'station': np.arange(3), 'time': np.arange(50)},
                     dims=['station', 'time'])
    z = z - z.mean('time')

    eofs_ds_x, eofs_ds_y, eofs_ds_z = rdu.eofs(
        ds, z, n_modes=k, weight=[w, None])

    flat_x_data = (w * x).transpose(
        'time', 'level', 'lat', 'lon').data.reshape(50, 360)
    flat_y_data = (w * y).transpose(
        'time', 'lat', 'lon').data.reshape(50, 120)
    flat_z_data = z.transpose('time', 'station').data

    flat_data = np.hstack([flat_x_data, flat_y_data, flat_z_data])

    u, s, vh = rdu.calc_truncated_svd(flat_data, k)
    vh_x = vh[:, :360].reshape((k, 3, 12, 10))
    vh_y = vh[:, 360:-3].reshape((k, 12, 10))
    vh_z = vh[:, -3:]
    mode_var = s**2 / (flat_data.shape[0] - 1)
    explained_var = mode_var / np.sum(np.var(flat_data, axis=0, ddof=1))

    assert np.allclose(vh_x, eofs_ds_x['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_x['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_x['explained_var'].data)

    assert np.allclose(vh_y, eofs_ds_y['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_y['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_y['explained_var'].data)

    assert np.allclose(vh_z, eofs_ds_z['EOFs'].data)
    assert np.allclose(np.dot(u, np.diag(s)), eofs_ds_z['PCs'].data)
    assert np.allclose(explained_var, eofs_ds_z['explained_var'].data)


def test_eofs_matches_sklearn():
    """Test result from eofs routine matches reference implementation."""

    x = np.random.uniform(size=(100, 20))

    k = 10
    eofs_result = rdu.eofs(x, n_modes=k)

    model = sd.PCA(n_components=k)
    pcs = model.fit_transform(x)

    assert np.allclose(model.components_, eofs_result['EOFs'].data)
    assert np.allclose(pcs, eofs_result['PCs'].data)
    assert np.allclose(
        model.explained_variance_ratio_, eofs_result['explained_var'].data)


def test_bsv_orthomax_rotation():
    """Test BSV orthomax rotation algorithm."""

    # Test recovers perfect singular structure using QUARTIMAX
    A0 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]])
    Z = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(Z)

    A = np.dot(A0, Q)

    max_iter = 500
    T, n_iter = _bsv_orthomax_rotation(A, gamma=0.0, max_iter=max_iter)

    AT = np.dot(A, T)

    assert n_iter <= max_iter
    assert np.allclose(np.max(np.abs(A0), axis=1),
                       np.max(np.abs(AT), axis=1))
    assert np.allclose(np.sum(np.abs(AT) > 0.1, axis=1), 1.0)

    with dask.config.set(scheduler='single-threaded'):
        A = dask.array.from_array(A)

        max_iter = 500
        T, n_iter = _bsv_orthomax_rotation(A, gamma=0.0, max_iter=max_iter)

        AT = np.dot(A, T)

        assert n_iter <= max_iter
        assert np.allclose(np.max(np.abs(A0), axis=1),
                           np.max(np.abs(AT), axis=1))
        assert np.allclose(np.sum(np.abs(AT) > 0.1, axis=1), 1.0)

    # Test recovers perfect singular structure using VARIMAX
    A0 = np.array([[1.0, 0.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, -1.0]])
    Z = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(Z)

    A = np.dot(A0, Q)

    max_iter = 500
    T, n_iter = _bsv_orthomax_rotation(A, gamma=1.0, max_iter=max_iter)

    AT = np.dot(A, T)

    assert n_iter <= max_iter
    assert np.allclose(np.max(np.abs(A0), axis=1),
                       np.max(np.abs(AT), axis=1))
    assert np.allclose(np.sum(np.abs(AT) > 0.1, axis=1), 1.0)


def test_reofs_numpy_arrays():
    """Test implementation of rotated EOFs for plain arrays."""

    data = np.genfromtxt(TEST_SWISS_DATA_FILE, delimiter=',',
                         skip_header=1, usecols=[1, 2, 3, 4, 5, 6])

    data = ((data - data.mean(axis=0, keepdims=True)) /
            data.std(axis=0, ddof=1, keepdims=True))

    k = 3
    reofs_result = rdu.reofs(data, n_modes=k, scale_eofs=False)

    expected_reofs = np.array([[-0.2375930, 0.53422316, -0.02983113],
                               [-0.5724203, -0.15164041, 0.01054517],
                               [0.5018115, -0.11544691, -0.13652139],
                               [0.4935835, -0.20726360, 0.48509548],
                               [-0.2134739, 0.07819156, 0.86203593],
                               [0.2736312, 0.79322802, 0.04401419]])
    expected_expl_var = np.array([0.16667, 0.16667, 0.16667])

    # Ensure fixed ordering of REOFs for comparisons
    expected_order = np.argsort(np.abs(expected_reofs[0]))
    result_order = np.argsort(np.abs(reofs_result['EOFs'].data[:, 0]))

    for i in range(k):
        expected_loadings = expected_reofs[:, expected_order[i]]
        loadings = reofs_result['EOFs'].data[result_order[i]]
        assert np.allclose(loadings, expected_loadings, atol=0.01)

    assert np.allclose(reofs_result['explained_var'].data, expected_expl_var,
                       atol=1e-4)

    k = data.shape[1]
    reofs_result = rdu.reofs(data, n_modes=k, scale_eofs=False)

    expected_reofs = np.array([[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    expected_expl_var = np.array([0.16667, 0.16667, 0.16667,
                                  0.16667, 0.16667, 0.16667])

    result_order = np.zeros(k, dtype=int)
    result_order[0] = np.argmax(np.abs(reofs_result['EOFs'].data[:, 0]))
    result_order[1] = np.argmax(np.abs(reofs_result['EOFs'].data[:, 5]))
    result_order[2] = np.argmax(np.abs(reofs_result['EOFs'].data[:, 4]))
    result_order[3] = np.argmax(np.abs(reofs_result['EOFs'].data[:, 1]))
    result_order[4] = np.argmax(np.abs(reofs_result['EOFs'].data[:, 2]))
    result_order[5] = np.argmax(np.abs(reofs_result['EOFs'].data[:, 3]))
    for i in range(k):
        expected_loadings = expected_reofs[:, i]
        loadings = reofs_result['EOFs'].data[result_order[i]]
        assert np.allclose(np.abs(loadings), np.abs(expected_loadings),
                           atol=0.001)

    assert np.allclose(reofs_result['explained_var'].data, expected_expl_var,
                       atol=1e-4)

    assert np.allclose(data, np.dot(reofs_result['PCs'].data,
                                    reofs_result['EOFs'].data))

    reofs_result = rdu.reofs(data, n_modes=k, scale_eofs=True)

    assert np.allclose(data, np.dot(reofs_result['PCs'].data,
                                    reofs_result['EOFs'].data))
