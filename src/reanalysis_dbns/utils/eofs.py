"""
Provides routines for performing EOF analysis.
"""

# License: MIT

from __future__ import absolute_import, division

import warnings

import dask.array
import numpy as np
import xarray as xr

from .computation import calc_truncated_svd
from .defaults import get_time_name
from .validation import (check_fixed_missing_values,
                         is_dask_array, is_data_array, is_dataset,
                         is_integer, is_xarray_object,
                         remove_missing_features,
                         restore_missing_features)


def _get_other_dims(ds, dim):
    """Return all dimensions other than the given dimension."""

    all_dims = list(ds.dims)

    return [d for d in all_dims if d != dim]


def _check_either_all_xarray_or_plain_arrays(objects):
    """Check if all elements of list are xarray objects or plain arrays."""

    has_xarray_input = False
    has_plain_array_input = False

    for obj in objects:
        if is_xarray_object(obj):
            has_xarray_input = True
        else:
            has_plain_array_input = True

    if has_xarray_input and has_plain_array_input:
        raise NotImplementedError(
            'mixed xarray and plain array input not supported')

    return has_xarray_input


def _ensure_leading_sample_dim(arr, sample_dim=0):
    """Transpose single array to ensure leading sample dimension."""

    if is_xarray_object(arr) and is_integer(sample_dim):
        # Find name of dimension corresponding to integer
        # axis dimension.
        dims = list(arr.dims)
        axis_nums = [arr.get_axis_num(d) for d in dims]
        sample_dim = dims[axis_nums.index(sample_dim)]

    if is_xarray_object(arr):

        if arr.get_axis_num(sample_dim) != 0:
            other_dims = _get_other_dims(arr, sample_dim)
            arr = arr.transpose(*([sample_dim] + other_dims))

    else:

        if sample_dim != 0:

            n_dims = arr.ndim
            sample_dim = sample_dim % n_dims

            other_dims = [d for d in range(n_dims) if d != sample_dim]
            transposed_dims = [sample_dim] + other_dims

            arr = np.transpose(arr, transposed_dims)

    return arr


def _ensure_leading_sample_dims(arrays, sample_dim=0):
    """Transpose objects to have leading sample dimension."""
    return [_ensure_leading_sample_dim(a, sample_dim=sample_dim)
            for a in arrays]


def _weight_with_fixed_dimension_order(arr, weight=None):
    """Apply weights to an array, preserving dimension order."""

    if weight is None:
        return arr

    return (weight.fillna(0) * arr).transpose(*arr.dims)


def _expand_and_weight_datasets(objects, sample_dim=0, weight=None):
    """Replace xarray Datasets in a list by separate DataArrays.

    Each dataset in the list is replaced by separate arrays,
    one for each data variable.

    Parameters
    ----------
    objects : list of xarray objects
        A list of DataArrays and Datasets to combine.

    sample_dim : int or str
        Dimension to treat as the sampling dimension.

    weight : xarray.DataArray or list of xarray.DataArrays, optional
        If given, either a single DataArray or one DataArray
        for each entry in the list.

    Returns
    -------
    arrays : list
        List of xarray.DataArrays corresponding to the input
        data, multiplied by the given weights.
    """

    arrays = []
    weights = []

    if weight is None or is_data_array(weight):
        has_common_weight = True
    elif is_dataset(weight):
        raise ValueError(
            'Weights must either be an xarray DataArray'
            ' or list of DataArrays')
    else:
        if len(weight) != len(objects):
            raise ValueError(
                'Number of weight arrays does not match number'
                ' of input arrays '
                '(got len(weight)=%d but len(objects)=%d)' %
                (len(weight), len(objects)))
        has_common_weight = False

    for idx, obj in enumerate(objects):
        if is_data_array(obj):
            arrays.append(obj)
            if has_common_weight:
                weights.append(weight)
            else:
                weights.append(weight[idx])
        else:
            for v in obj.data_vars:
                arrays.append(obj[v])
                if has_common_weight:
                    weights.append(weight)
                else:
                    weights.append(weight[idx])

    arrays = [_weight_with_fixed_dimension_order(arrays[i], w)
              for i, w in enumerate(weights)]

    arrays = _ensure_leading_sample_dims(arrays, sample_dim=sample_dim)

    return arrays


def _get_data_values(X):
    """Get numerical values from object."""

    if is_data_array(X):
        return X.values

    return X


def _apply_weights_to_arrays(arrays, weights=None, sample_dim=0):
    """Apply weights to the corresponding input array.

    Given a list of arrays and an equal length list of
    weights, apply the weights to the corresponding array
    element-wise.

    Each given array is assumed to have shape (N, ...) where N
    is the number of samples. The corresponding element in the
    list of weights must either be None, in which case no weighting
    is applied, or must have shape compatible with the remaining
    dimensions of the input array.

    Parameters
    ----------
    arrays : list of arrays
        List of arrays to apply weights to.

    weights : None or list of arrays
        List of weights to apply. If None, no weighting is applied
        and the original arrays are returned. If not None, each
        element of the list should be either None or broadcastable
        to the shape of the corresponding list of input arrays.

    sample_dim : int
        Dimension to treat as sampling dimension.

    Returns
    -------
    weighted_arrays : list of arrays
        List of arrays with element-wise weights applied.
    """

    arrays = _ensure_leading_sample_dims(
        arrays, sample_dim=sample_dim)

    if weights is None:
        return arrays

    n_arrays = len(arrays)

    if not isinstance(weights, tuple) and not isinstance(weights, list):
        # Assume a single common set of weights is given.
        weights_list = []
        for i in range(n_arrays):
            weights_list.append(weights)
    else:
        if len(arrays) != len(weights):
            raise ValueError(
                'Number of weight arrays does not match number of '
                ' input arrays '
                '(got len(weights)=%d but len(arrays)=%d)' %
                (len(weights), len(arrays)))

        weights_list = weights

    has_dask_arrays = np.any([is_dask_array(a) for a in arrays])

    if has_dask_arrays:
        broadcast_arrays = dask.array.broadcast_arrays
        ones_like = dask.array.ones_like
    else:
        broadcast_arrays = np.broadcast_arrays
        ones_like = np.ones_like

    weights_to_apply = [broadcast_arrays(
        arrays[i][0:1], w)[1][0]
                        if w is not None
                        else ones_like(arrays[i][0:1])
                        for i, w in enumerate(weights_list)
                        ]

    return [a * weights_to_apply[i] for i, a in enumerate(arrays)]


def _check_eofs_input_arrays(arrays):
    """Check input arrays have consistent shapes."""

    # All inputs must have the same number of samples, i.e.,
    # same size for the first dimension.
    n_samples = None
    for a in arrays:
        if n_samples is None:
            n_samples = a.shape[0]
        else:
            if a.shape[0] != n_samples:
                raise ValueError(
                    'Numbers of samples do not agree for all arrays')

        check_fixed_missing_values(a, axis=0)

    return n_samples


def _check_zero_feature_means(data):
    """Check that input features have zero mean."""
    nonzero_column_means = np.any(np.abs(data.mean(axis=0)) > 1e-3)
    if nonzero_column_means:
        warnings.warn(
            'Input data has features with non-zero means '
            '(got max(abs(data.mean(axis=0)))=%.2f)' %
            np.max(np.abs(data.mean(axis=0))),
            UserWarning)


def _check_degrees_of_freedom(ddof, bias):
    """Check denominator degrees of freedom."""

    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1

    return ddof


def _eofs_impl(*arrays, n_modes=None, center=True,
               bias=False, ddof=None, skipna=True):
    """Perform standard empirical orthogonal function analysis.

    Given one or more arrays, combines the contents of the arrays
    to form a single dataset and performs a standard empirical orthogonal
    function (EOF) analysis on the combined dataset.

    Each array is assumed to have shape (N, ...), where N is the number
    of observations or samples, and hence all input arrays must have the
    same number of elements in the first dimension. The remaining dimensions
    for each array may differ.

    Parameters
    ----------
    *arrays : arrays of shape (N, ...)
        Arrays for perform EOF analysis on. All of the input arrays must
        have the same length first dimension.

    n_modes : None or integer, optional
        If an integer, the number of EOF modes to compute. If None, the
        maximum possible number of modes is computed.

    center : boolean, default: True
        If True, center the inputs by subtracting the column means.

    bias : boolean, default: False
        If False, normalize the covariance matrix by N - 1.
        If True, normalize by N. These values may be overridden
        by using the ddof argument.

    ddof : integer or None
        If an integer, normalize the covariance matrix by N - ddof.
        Note that this overrides the value implied by the bias argument.

    skipna : boolean, optional
       If True, ignore NaNs for the purpose of computing the total variance.

    Returns
    -------
    eofs : list of arrays of shape (n_modes, ...)
        List of arrays containing the components of the EOFs associated
        with each of the input arrays.

    pcs : array, shape (N, n_modes)
        Array containing the corresponding principal components.

    lambdas : array, shape (n_modes,)
        Array containing the eigenvalues of the covariance matrix.

    explained_var : array, shape (n_modes,)
        Array containing the fraction of the total variance associated with
        each mode.
    """

    n_arrays = len(arrays)

    n_samples = _check_eofs_input_arrays(arrays)
    has_dask_array = any([is_dask_array(a) for a in arrays])

    # Retain original shapes for reshaping after computing EOFs.
    original_shapes = [a.shape[1:] for a in arrays]
    original_sizes = [np.product(shp) for shp in original_shapes]

    # Reshape to 2D to perform SVD on.
    flat_arrays = [a.reshape((n_samples, original_sizes[i]))
                   for i, a in enumerate(arrays)]

    if has_dask_array:
        concatenate_arrays = dask.array.concatenate
    else:
        concatenate_arrays = np.concatenate

    combined_dataset = concatenate_arrays(flat_arrays, axis=1)

    # Remove missing features from inputs to SVD.
    combined_dataset, missing_features = remove_missing_features(
        combined_dataset)

    if has_dask_array:
        # Initial workaround to avoid error if the concatenated
        # array is not tall-and-skinny as required by dask.linalg.svd.
        combined_dataset = combined_dataset.rechunk('auto')

    if n_modes is None:
        n_modes = min(combined_dataset.shape)

    if center:
        combined_dataset = (combined_dataset -
                            combined_dataset.mean(axis=0, keepdims=True))
    else:
        _check_zero_feature_means(combined_dataset)

    u, s, vh = calc_truncated_svd(combined_dataset, k=n_modes)

    ddof = _check_degrees_of_freedom(ddof, bias=bias)

    lambdas = s ** 2 / (n_samples * 1. - ddof)

    if skipna:
        calc_variance = np.nanvar if not has_dask_array else dask.array.nanvar
    else:
        calc_variance = np.var if not has_dask_array else dask.array.var

    variances = calc_variance(combined_dataset, ddof=ddof, axis=0)
    explained_var = lambdas / variances.sum()

    pcs = (np.dot(u, np.diag(s)) if not has_dask_array
           else dask.array.dot(u, dask.array.diag(s)))

    # Restore missing features in EOFs and reshape back to original shape.
    eofs_2d = restore_missing_features(vh, missing_features)

    eofs_results = []
    idx = 0
    for i in range(n_arrays):
        start_col = idx
        end_col = start_col + original_sizes[i]
        eofs_results.append(
            eofs_2d[:, start_col:end_col].reshape(
                (n_modes,) + original_shapes[i]))
        idx = end_col

    return eofs_results, pcs, lambdas, explained_var


def _bsv_orthomax_rotation(A, T0=None, gamma=1.0, tol=1e-6, max_iter=500):
    """Returns optimal rotation found by BSV algorithm.

    Given an initial column-wise orthonormal matrix A of dimension
    p x k, p >= k, the orthomax family of criteria seek an
    orthogonal matrix T that maximizes the objective function

        Q(Lambda) = (1 / 4) Tr[(L ^ 2)^T (L^2 - gamma * \bar{L^2})]

    where L = A * T, L^2 denotes the element-wise square
    of L, and \bar{L^2} denotes the result of replacing each
    element of L^2 by the mean of its corresponding column. The
    parameter gamma defines the particular objective function to be
    maximized; for gamma = 1, the procedure corresponds to
    VARIMAX rotation.

    Parameters
    ----------
    A : array-like, shape (n_features, n_components)
        The matrix to be rotated.

    T0 : None or array-like, shape (n_components, n_components)
        If given, an initial guess for the rotation matrix T.

    gamma : float, default: 1.0
        Objective function parameter.

    tol : float, default: 1e-6
        Tolerance of the stopping condition.

    max_iter : integer, default: 500
        Maximum number of iterations before stopping.

    Returns
    -------
    T : array-like, shape (n_components, n_components)
        Approximation to the optimal rotation.

    n_iter : integer
        The actual number of iterations.

    References
    ----------
    R. I. Jennrich, "A simple general procedure for orthogonal rotation",
    Psychometrika 66, 2 (2001), 289-306
    """

    n_features, n_components = A.shape

    if n_components > n_features:
        raise ValueError(
            'Number of rows in input array must be greater than '
            'or equal to number of columns, got A.shape = %r' %
            A.shape)

    if T0 is None:
        if is_dask_array(A):
            T = dask.array.eye(n_components, dtype=A.dtype)
        else:
            T = np.eye(n_components, dtype=A.dtype)
    else:
        if T0.shape != (n_components, n_components):
            raise ValueError(
                'Array with wrong shape passed to %s. '
                'Expected %s, but got %s' %
                ('_bsv_orthomax_rotation',
                 (n_components, n_components), A.shape))
        T = T0.copy()

    if is_dask_array(A):
        to_diag = dask.array.diag
        calc_svd = dask.array.linalg.svd
    else:
        to_diag = np.diag
        calc_svd = np.linalg.svd

    delta = 0
    for n_iter in range(max_iter):
        delta_old = delta

        Li = A.dot(T)

        grad = Li ** 3 - gamma * Li.dot(to_diag((Li ** 2).mean(axis=0)))
        G = (A.T).dot(grad)

        u, s, vt = calc_svd(G)

        T = u.dot(vt)

        delta = s.sum()
        if delta < delta_old * (1 + tol):
            break

    if n_iter == max_iter and tol > 0:
        warnings.warn('Maximum number of iterations %d reached.' % max_iter,
                      UserWarning)

    return T, n_iter


def _reorder_eofs(input_eofs, pcs, lambdas, explained_variance):
    """Sorts modes in descending order of explained variance."""

    n_components = np.size(explained_variance)
    sort_order = np.flip(np.argsort(explained_variance))

    perm_inv = np.zeros((n_components, n_components))
    for i in range(n_components):
        perm_inv[i, sort_order[i]] = 1

    return (input_eofs[sort_order], pcs[:, sort_order],
            lambdas[sort_order], explained_variance[sort_order])


def _check_number_of_rotated_modes(n_rotated_modes, n_modes):
    """Check number of rotated modes is valid."""

    if n_rotated_modes is None:
        return n_modes

    is_valid = (is_integer(n_rotated_modes) and
                n_rotated_modes > 0 and
                n_rotated_modes <= n_modes)
    if not is_valid:
        raise ValueError(
            'Number of rotated modes must be an integer '
            'between 1 and %d '
            '(got n_rotated_modes=%r)' %
            (n_modes, n_rotated_modes))

    return n_rotated_modes


def _varimax_reofs_impl(*arrays, n_modes=None, n_rotated_modes=None,
                        scale_eofs=True, kaiser_normalize=True,
                        T0=None, tol=1e-6, max_iter=500, center=True,
                        bias=False, ddof=None, skipna=True):
    """Perform VARIMAX rotated empirical orthogonal function analysis.

    Given one or more arrays, combines the contents of the arrays
    to form a single dataset and first performs a standard empirical orthogonal
    function (EOF) analysis on the combined dataset. The obtained EOFs
    are then rotated using the VARIMAX rotation criterion.

    Each array is assumed to have shape (N, ...), where N is the number
    of observations or samples, and hence all input arrays must have the
    same number of elements in the first dimension. The remaining dimensions
    for each array may differ.

    Parameters
    ----------
    *arrays : arrays of shape (N, ...)
        Arrays for perform EOF analysis on. All of the input arrays must
        have the same length first dimension.

    n_modes : None or integer, optional
        If an integer, the number of EOF modes to compute. If None, the
        maximum possible number of modes is computed.

    n_rotated_modes : None or integer, optional
        If an integer, the number of modes to retain when performing
        rotation. If None, all modes are included in the rotation.

    scale_eofs : boolean, optional
        If True, scale the EOFs by multiplying by the square root
        of the corresponding eigenvalue of the covariance matrix.
        In this case, the rotated PCs are uncorrelated.

    kaiser_normalize : boolean, optional
        Use Kaiser normalization for EOFs.

    T0 : None or array-like, shape (n_rotated_modes, n_rotated_modes)
        If given, an initial guess for the rotation matrix to be
        used.

    tol : float, optional
        Stopping tolerance for calculating rotation matrix.

    max_iter : integer, optional
        The maximum number of iterations to allow in calculating
        the rotation matrix.

    center : boolean, default: True
        If True, center the inputs by subtracting the column means.

    bias : boolean, default: False
        If False, normalize the covariance matrix by N - 1.
        If True, normalize by N. These values may be overridden
        by using the ddof argument.

    ddof : integer or None
        If an integer, normalize the covariance matrix by N - ddof.
        Note that this overrides the value implied by the bias argument.

    skipna : boolean, optional
       If True, ignore NaNs for the purpose of computing the total variance.

    Returns
    -------
    reofs : list of arrays of shape (n_rotated_modes, ...)
        List of arrays containing the components of the EOFs associated
        with each of the input arrays.

    pcs : array, shape (N, n_rotated_modes)
        Array containing the corresponding principal components.

    lambdas : array, shape (n_rotated_modes,)
        Array containing the eigenvalues of the covariance matrix.

    explained_var : array, shape (n_rotated_modes,)
        Array containing the fraction of the total variance associated with
        each mode.

    rotation : array, shape (n_rotated_modes, n_rotated_modes)
        Array containing the orthogonal rotation matrix used to
        obtain the rotated EOFs.
    """

    varimax_gamma = 1.0

    n_arrays = len(arrays)

    eofs_results, pcs, lambdas, explained_var = _eofs_impl(
        *arrays, n_modes=n_modes, center=center, bias=bias,
        ddof=ddof, skipna=skipna)

    has_dask_array = any([is_dask_array(a) for a in eofs_results])

    n_modes = pcs.shape[1]
    n_rotated_modes = _check_number_of_rotated_modes(
        n_rotated_modes, n_modes=n_modes)

    # Explained variance is the ratio of the normalized
    # eigenvalues to the total variance.
    total_variance = lambdas[0] / explained_var[0]

    # Select subset of modes for rotation.
    eofs_to_rotate = [a[:n_rotated_modes] for a in eofs_results]
    pcs = pcs[:, :n_rotated_modes]
    lambdas = lambdas[:n_rotated_modes]
    explained_var = explained_var[:n_rotated_modes]

    # Retain original shapes for reshaping after computing rotated EOFs.
    original_shapes = [a.shape[1:] for a in eofs_to_rotate]
    original_sizes = [np.product(shp) for shp in original_shapes]

    # Reshape to 2D to perform rotation on.
    flat_arrays = [a.reshape((n_rotated_modes, original_sizes[i]))
                   for i, a in enumerate(eofs_to_rotate)]

    if has_dask_array:
        concatenate_arrays = dask.array.concatenate
    else:
        concatenate_arrays = np.concatenate

    # Remove fixed missing values and store indices of corresponding
    # features for later replacement.
    combined_eofs = concatenate_arrays(flat_arrays, axis=1)

    combined_eofs, missing_features = remove_missing_features(
        combined_eofs)

    if has_dask_array:
        # Initial workaround to avoid error if the concatenated
        # array is not tall-and-skinny as required by dask.linalg.svd.
        combined_eofs = combined_eofs.rechunk('auto')

    if scale_eofs:
        combined_eofs *= np.sqrt(lambdas[:, np.newaxis])
        pcs /= np.sqrt(lambdas[np.newaxis, :])

    if kaiser_normalize:
        normalization = np.sqrt((combined_eofs ** 2).sum(axis=0))
        normalization[normalization == 0] = 1
        combined_eofs /= normalization

    # Perform rotation on combined EOFs.
    rotation, _ = _bsv_orthomax_rotation(
        combined_eofs.T, T0=T0, gamma=varimax_gamma,
        tol=tol, max_iter=max_iter)

    pcs = pcs.dot(rotation)
    rotated_combined_eofs = (rotation.T).dot(combined_eofs)

    if kaiser_normalize:
        rotated_combined_eofs *= normalization

    # Recompute amount of variance explained by each of the individual
    # rotated modes. Note that this only strictly meaningful if
    # the EOFs are scaled so that the rotated modes remain uncorrelated.
    lambdas = (rotated_combined_eofs ** 2).sum(axis=1)
    explained_var = lambdas / total_variance

    # Ensure returned EOFs and PCs are still ordered according
    # to the amount of variance explained.
    rotated_combined_eofs, pcs, lambdas, explained_var = _reorder_eofs(
        rotated_combined_eofs, pcs, lambdas, explained_var)

    # Restore missing features and reshape to original shapes.
    reofs_2d = restore_missing_features(
        rotated_combined_eofs, missing_features)

    reofs_results = []
    idx = 0
    for i in range(n_arrays):
        start_col = idx
        end_col = start_col + original_sizes[i]
        reofs_results.append(
            reofs_2d[:, start_col:end_col].reshape(
                (n_modes,) + original_shapes[i]))
        idx = end_col

    return reofs_results, pcs, lambdas, explained_var, rotation


def _get_default_sample_dim(objects):
    """Get default sampling dimension for objects."""

    is_xarray_input = _check_either_all_xarray_or_plain_arrays(objects)

    if not is_xarray_input:
        # By default, assume the leading dimension is the sampling
        # dimension
        return 0

    for obj in objects:
        if is_xarray_object(obj):
            try:
                sample_dim = get_time_name(obj)
            except ValueError:
                continue

            if sample_dim is not None:
                break

    if sample_dim is None:
        raise RuntimeError(
            'Unable to automatically determine sampling dimension')

    return sample_dim


def _check_sample_dim(sample_dim, objects):
    """Check given sampling dimension is valid."""

    if sample_dim is None:
        return _get_default_sample_dim(objects)

    is_xarray_input = _check_either_all_xarray_or_plain_arrays(objects)

    if is_xarray_input:
        for obj in objects:
            if sample_dim not in obj.dims:
                raise ValueError(
                    "Sampling dimension '%s' not found "
                    "(got obj.dims=%r)" % (obj.dims))

    else:
        if not is_integer(sample_dim):
            raise ValueError(
                'Sampling dimension must be an integer for plain arrays '
                '(got sample_dim=%r)' % sample_dim)

    return sample_dim


def _construct_eofs_dataset(input_array, eofs_array, pcs, lambdas, expl_var,
                            sample_dim, copy_attrs=True):
    """Construct Dataset from EOFs results."""

    n_modes = pcs.shape[1]

    if is_data_array(input_array):
        input_dims = _get_other_dims(input_array, sample_dim)

        data_vars = {'EOFs': (['mode'] + input_dims, eofs_array),
                     'PCs': ([sample_dim, 'mode'], pcs),
                     'lambdas': (['mode'], lambdas),
                     'explained_var': (['mode'], expl_var)
                     }

        coords = dict(input_array.coords.items())
        coords['mode'] = np.arange(n_modes)

        ds = xr.Dataset(data_vars, coords=coords)

        if copy_attrs:
            for attr in input_array.attrs:
                ds.attrs[attr] = input_array.attrs[attr]

        return ds

    input_shape = input_array.shape[1:]
    input_dims = ['axis_{:d}'.format(i + 1)
                  for i in range(len(input_shape))]

    data_vars = {'EOFs': (['mode'] + input_dims, eofs_array),
                 'PCs': ([sample_dim, 'mode'], pcs),
                 'lambdas': (['mode'], lambdas),
                 'explained_var': (['mode'], expl_var)
                 }

    coords = {d: np.arange(input_shape[i])
              for i, d in enumerate(input_dims)}
    coords['mode'] = np.arange(n_modes)

    return xr.Dataset(data_vars, coords=coords)


def eofs(*objects, sample_dim=None, weight=None, n_modes=None, center=True):
    """Perform standard empirical orthogonal function analysis.

    Given one or more objects, perform a standard empirical orthogonal
    function (EOF) analysis on the dataset formed from the combined objects.

    Parameters
    ----------
    *objects : arrays
        Objects containing data for perform EOF analysis on.

    sample_dim : str or integer, optional
        Axis corresponding to the sampling dimension.

    weight : arrays, optional
        If given, weights to apply to the data. If multiple objects are
        given, the number of elements of weight must be the same as the
        given number of objects. For each object used in the analysis,
        the given weight, if not None, must be broadcastable onto the
        corresponding data object.

    n_modes : None or integer, optional
        The number of EOFs to retain. If None, then the maximum
        possible number of EOFs will be retained.

    center : boolean, default: True
        If True, center the inputs by subtracting the column means.

    Returns
    -------
    eofs : xarray Dataset
        Dataset containing the following variables:

        - 'EOFs' : array containing the empirical orthogonal functions

        - 'PCs' : array containing the associated principal components

        - 'lambdas' : array containing the eigenvalues of the covariance
          matrix of the input data

        - 'explained_var' : array containing the fraction of the total
          variance explained by each mode
    """

    # Currently, mixed xarray and plain array inputs are not supported.
    is_xarray_input = _check_either_all_xarray_or_plain_arrays(objects)

    sample_dim = _check_sample_dim(sample_dim, objects)

    if is_xarray_input:
        arrays = _expand_and_weight_datasets(
            objects, weight=weight, sample_dim=sample_dim)
    else:
        arrays = _apply_weights_to_arrays(
            objects, weights=weight, sample_dim=sample_dim)

    # Convert to plain arrays for computation.
    plain_arrays = [_get_data_values(a) for a in arrays]

    eofs_arr, pcs_arr, lambdas_arr, expl_var_arr = _eofs_impl(
        *plain_arrays, n_modes=n_modes, center=center)

    # Assemble results into datasets.
    result = []
    for idx, arr in enumerate(arrays):
        result.append(
            _construct_eofs_dataset(
                arrays[idx], eofs_arr[idx], pcs_arr,
                lambdas_arr, expl_var_arr, sample_dim))

    if len(result) == 1:
        return result[0]

    return result


def reofs(*objects, sample_dim=None, weight=None, n_modes=None,
          n_rotated_modes=None, scale_eofs=True, kaiser_normalize=True,
          T0=None, tol=1e-6, max_iter=500, center=True):
    """Perform rotated empirical orthogonal function analysis.

    Given one or more objects, perform a VARIMAX rotated empirical orthogonal
    function (EOF) analysis on the dataset formed from the combined objects.

    Parameters
    ----------
    *objects : arrays
        Objects containing data for perform EOF analysis on.

    sample_dim : str or integer, optional
        Axis corresponding to the sampling dimension.

    weight : arrays, optional
        If given, weights to apply to the data. If multiple objects are
        given, the number of elements of weight  must be the same as the
        given number of objects. For each object used in the analysis,
        the given weight, if not None, must be broadcastable onto the
        corresponding data object.

    n_modes : None or integer, optional
        The number of EOFs to retain. If None, then the maximum
        possible number of EOFs will be retained.

    n_rotated_modes : None or integer, optional
        If an integer, the number of modes to retain when performing
        rotation. If None, all modes are included in the rotation.

    scale_eofs : boolean, optional
        If True, scale the EOFs by multiplying by the square root
        of the corresponding eigenvalue of the covariance matrix.
        In this case, the rotated PCs are uncorrelated.

    kaiser_normalize : boolean, optional
        Use Kaiser normalization for EOFs.

    T0 : None or array-like, shape (n_rotated_modes, n_rotated_modes)
        If given, an initial guess for the rotation matrix to be
        used.

    tol : float, optional
        Stopping tolerance for calculating rotation matrix.

    max_iter : integer, optional
        The maximum number of iterations to allow in calculating
        the rotation matrix.

    center : boolean, default: True
        If True, center the inputs by subtracting the column means.

    Returns
    -------
    eofs : xarray Dataset
        Dataset containing the following variables:

        - 'EOFs' : array containing the rotated empirical orthogonal functions

        - 'PCs' : array containing the associated principal components

        - 'lambdas' : array containing the eigenvalues of the covariance
          matrix of the input data

        - 'explained_var' : array containing the fraction of the total
          variance explained by each mode
    """

    # Currently, mixed xarray and plain array inputs are not supported.
    is_xarray_input = _check_either_all_xarray_or_plain_arrays(objects)

    sample_dim = _check_sample_dim(sample_dim, objects)

    if is_xarray_input:
        arrays = _expand_and_weight_datasets(
            objects, sample_dim=sample_dim, weight=weight)
    else:
        arrays = _apply_weights_to_arrays(objects, weights=weight)

    # Convert to plain arrays for computation.
    plain_arrays = [_get_data_values(a) for a in arrays]

    eofs_arr, pcs_arr, lambdas_arr, expl_var_arr, _ = _varimax_reofs_impl(
        *plain_arrays, n_modes=n_modes, n_rotated_modes=n_rotated_modes,
        kaiser_normalize=kaiser_normalize, scale_eofs=scale_eofs,
        T0=T0, tol=tol, max_iter=max_iter, center=center)

    # Assemble results into datasets.
    result = []
    for idx, arr in enumerate(arrays):
        result.append(
            _construct_eofs_dataset(
                arrays[idx], eofs_arr[idx], pcs_arr,
                lambdas_arr, expl_var_arr, sample_dim))

    if len(result) == 1:
        return result[0]

    return result
