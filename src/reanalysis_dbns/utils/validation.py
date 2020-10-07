"""
Helper routines for validating inputs.
"""

# License: MIT

from __future__ import absolute_import

import collections
import numbers
import warnings

import dask.array
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.utils import check_array

from .defaults import get_coordinate_standard_name


INTEGER_TYPES = (numbers.Integral, np.integer)


def is_integer(x):
    """Check if x is an integer."""
    return isinstance(x, INTEGER_TYPES)


def is_scalar(x):
    """Check is x is a scalar value."""
    return isinstance(x, numbers.Number)


def is_data_array(data):
    """Check if object is an xarray DataArray."""

    return isinstance(data, xr.DataArray)


def is_dataset(data):
    """Check if object is an xarray Dataset."""

    return isinstance(data, xr.Dataset)


def is_xarray_object(data):
    """Check if object is an xarray DataArray or Dataset."""

    return is_data_array(data) or is_dataset(data)


def is_dask_array(data):
    """Check if object is a dask array."""

    return isinstance(data, dask.array.Array)


def is_pandas_series(data):
    """Check if object is a pandas Series."""

    return isinstance(data, pd.Series)


def is_pandas_dataframe(data):
    """Check if object is a pandas DataFrame."""

    return isinstance(data, pd.DataFrame)


def is_pandas_object(data):
    """Check if object is a pandas Series or DataFrame."""

    return is_pandas_series(data) or is_pandas_dataframe(data)


def _check_non_negative_integer(x, full_name, short_name):
    """Check if object is a non-negative integer."""
    if not is_integer(x) or x < 0:
        raise ValueError(
            "%s must be a non-negative integer "
            "(got %s=%r)" % (full_name, short_name, x))

    return x


def _check_positive_integer(x, full_name, short_name):
    """Check if object is a positive integer."""
    if not is_integer(x) or x < 1:
        raise ValueError(
            "%s must be a positive integer "
            "(got %s=%r)" % (full_name, short_name, x))

    return x


def _check_lower_bound(x, lower, full_name, short_name,
                       inclusive_bound=True):
    """Check object satisfies lower bound."""

    if inclusive_bound:
        if x < lower:
            raise ValueError(
                "%s must be at least %r "
                "(got %s=%r)" %
                (full_name, lower, short_name, x))
    else:
        if x <= lower:
            raise ValueError(
                "%s must be greater than %r "
                "(got %s=%r)" %
                (full_name, lower, short_name, x))

    return x


def _check_upper_bound(x, upper, full_name, short_name,
                       inclusive_bound=True):
    """Check object satisfies upper bound."""

    if inclusive_bound:
        if x > upper:
            raise ValueError(
                "%s must be at most %r "
                "(got %s=%r)" %
                (full_name, upper, short_name, x))
    else:
        if x >= upper:
            raise ValueError(
                "%s must be lessthan %r "
                "(got %s=%r)" %
                (full_name, upper, short_name, x))

    return x


def _check_bounded_integer(x, full_name, short_name,
                           lower=None, upper=None,
                           inclusive_bounds=True):
    """Check if object is a bounded integer."""

    if lower is None and upper is None:
        raise ValueError('Lower or upper bound must be given')

    if not is_integer(x):
        raise ValueError(
            "%s must be an integer "
            "(got %s=%r)" % (full_name, short_name, x))

    if lower is not None:
        x = _check_lower_bound(
            x, lower, full_name, short_name,
            inclusive_bound=inclusive_bounds)

    if upper is not None:
        x = _check_upper_bound(
            x, upper, full_name, short_name,
            inclusive_bound=inclusive_bounds)

    return x


def _check_non_negative_scalar(x, full_name, short_name):
    """Check object is a non-negative scalar."""

    if not is_scalar(x) or x <= 0.0:
        raise ValueError(
            "%s must be a non-negative scalar "
            "(got %s=%r)" % (full_name, short_name, x))

    return x


def check_number_of_chains(n_chains):
    """Check number of chains is valid."""
    return _check_positive_integer(
        n_chains, 'Number of chains', 'n_chains')


def check_number_of_initializations(n_init):
    """Check number of initializations is valid."""
    return _check_positive_integer(
        n_init, 'Number of initializations', 'n_init')


def check_number_of_iterations(n_iter):
    """Check number of iterations is valid."""
    return _check_positive_integer(
        n_iter, 'Number of iterations', 'n_iter')


def check_tolerance(tol):
    """Check tolerance parameter is valid."""
    return _check_non_negative_scalar(
        tol, 'Tolerance parameter', 'tol')


def check_warmup(warmup, n_iter):
    """Check number of warmup iterations is valid."""
    n_iter = check_number_of_iterations(n_iter)
    return _check_bounded_integer(
        warmup, 'Number of warmup iterations', 'warmup',
        lower=0, upper=(n_iter - 1))


def ensure_variables_in_data(data, variables):
    """Check data contains expected variables.

    Parameters
    ----------
    data : object
        Object supporting membership testing with the in operator,
        representing a dataset.

    variables : list
        List of variables to check for in data.

    Returns
    -------
    data : dict-like
        The input dataset.
    """

    if data is None:
        raise ValueError('No data given')

    for v in variables:
        if v not in data:
            raise ValueError(
                "Variable '%s' not present in given data." % v)

    return data


def check_max_memory(variable_names, max_memory):
    """Check given maximum memory is valid.

    The maximum memory for the given variables may be either
    specified as an integer, a collection of integers, or
    an array of the same length as the number of variables.
    In the first case, the same maximum memory is used for
    all of the variables. In the second case, each variable
    must be present as a key, with the corresponding value
    representing the maximum memory for that variable.
    Finally, if max_memory is an array, then each element of
    the array is taken to be the memory for the corresponding
    variable.

    Parameters
    ----------
    variable_names : list
        List containing the variable names.

    max_memory : int, collection, or array-like
        Maximum memory for the listed variables.

    Returns
    -------
    max_memory : dict
        Dict with keys corresponding to the given variables
        and values equal to the maximum memory for that variable.
    """

    n_variables = len(variable_names)

    if max_memory is None:
        raise ValueError('Maximum memory must not be None')

    if is_integer(max_memory):
        max_memory = _check_non_negative_integer(
            max_memory, 'Maximum memory', 'max_memory')

        return {v: max_memory for v in variable_names}

    if isinstance(max_memory, collections.abc.Mapping):

        valid_max_memory = {}
        for v in variable_names:
            if v not in max_memory:
                raise ValueError(
                    "Maximum memory must be given for each variable "
                    "(maximum memory missing for variable '%s')" % v)

            valid_max_memory[v] = _check_non_negative_integer(
                max_memory[v], 'Maximum memory', 'max_memory')

        return valid_max_memory

    max_memory = check_array(
        max_memory, dtype=[np.int32, np.int64],
        ensure_2d=False, allow_nd=False)

    if len(max_memory) != n_variables:
        raise ValueError(
            'Maximum memory must be given for each variable '
            '(got len(max_memory)=%d but n_variables=%d)' %
            (len(max_memory), n_variables))

    valid_max_memory = {}
    for i, v in enumerate(variable_names):

        valid_max_memory[v] = _check_non_negative_integer(
            max_memory[i], 'Maximum memory', 'max_memory')

    return valid_max_memory


def check_max_parents(variable_names, max_parents):
    """Check given maximum number of parents is valid.

    The maximum number of parents for the given variables
    may be either specified as an integer, a collection of
    integers, or an array of the same length as the number
    of variables. In the first case, the same maximum memory
    is used for all of the variables. In the second case,
    each variable must be present as a key, with the
    corresponding value representing the maximum memory for
    that variable. Finally, if max_memory is an array, then
    each element of the array is taken to be the memory for
    the corresponding variable.

    Parameters
    ----------
    variable_names : list
        List containing the variable names.

    max_parents : None, int, collection, or array-like
        Maximum number of parents for the listed variables.

    Returns
    -------
    max_parents : dict
        Dict with keys corresponding to the given variables
        and values equal to the maximum number of parents for
        that variable.
    """

    if max_parents is None:
        return {v: max_parents for v in variable_names}

    n_variables = len(variable_names)

    if is_integer(max_parents):
        max_parents = _check_non_negative_integer(
            max_parents, 'Maximum number of parent nodes',
            'max_parents')

        return {v: max_parents for v in variable_names}

    if isinstance(max_parents, collections.abc.Mapping):

        valid_max_parents = {}
        for v in variable_names:
            if v not in max_parents:
                raise ValueError(
                    "Maximum number of parent nodes must be given "
                    "for each variable "
                    "(max_parents missing for variable '%s')" % v)

            valid_max_parents[v] = _check_non_negative_integer(
                max_parents[v], 'Maximum number of parent nodes',
                'max_parents')

        return valid_max_parents

    max_parents = check_array(
        max_parents, dtype=[np.int32, np.int64],
        ensure_2d=False, allow_nd=False)

    if len(max_parents) != n_variables:
        raise ValueError(
            'Maximum number of parent nodes must be given '
            'for each variable '
            '(got len(max_parents)=%d but n_variables=%d)' %
            (len(max_parents), n_variables))

    valid_max_parents = {}
    for i, v in enumerate(variable_names):

        valid_max_parents[v] = _check_non_negative_integer(
            max_parents[i], 'Maximum number of parent nodes',
            'max_parents')

    return valid_max_parents


def _guess_sampling_frequency(data, time_name=None):
    """Attempt to calculate sampling frequency for data."""

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    # Take the difference between the second and first time values
    dt = data[time_name][1] - data[time_name][0]

    # Covert and evaluate
    dt = dt.data.astype('timedelta64[D]').astype(int)

    if not np.isscalar(dt):
        dt = dt.item()

    if dt == 1:
        return 'daily'

    if 28 <= dt < 365:
        return 'monthly'

    if dt >= 365:
        return 'yearly'


def _convert_freq_string_to_description(freq):
    """Get long description for frequency string."""

    if freq in ('H', '1H'):
        return 'hourly'

    if freq in ('D', '1D'):
        return 'daily'

    if freq in ('W', '1W', 'W-SUN'):
        return 'weekly'

    if freq in ('1M', '1MS', 'MS'):
        return 'monthly'

    if freq in ('1A', '1AS', '1BYS', 'A-DEC'):
        return 'yearly'

    raise ValueError("Unsupported frequency string '%r'" % freq)


def detect_frequency(data, time_name=None):
    """Detect if the data is sampled at daily or monthly resolution."""

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    try:
        inferred_frequency = pd.infer_freq(data[time_name].values[:3])
    except TypeError:
        inferred_frequency = None

    if inferred_frequency is None:
        warnings.warn(
            'Unable to infer frequency for data. '
            'Falling back to calculating interval between samples.',
            UserWarning)

        return _guess_sampling_frequency(data, time_name=time_name)

    return _convert_freq_string_to_description(inferred_frequency)


def is_daily_data(data, time_name=None):
    """Check if data is sampled at daily resolution."""
    return detect_frequency(data, time_name=time_name) == 'daily'


def is_monthly_data(data, time_name=None):
    """Check if data is sampled at monthly resolution."""
    return detect_frequency(data, time_name=time_name) == 'monthly'


def check_array_shape(x, shape, whom):
    """Check array has the desired shape."""

    if x.shape != shape:
        raise ValueError(
            'Array with wrong shape passed to %s. '
            'Expected %s, but got %s' % (whom, shape, x.shape))


def ensure_data_array(obj, variable=None):
    """Ensure that object is an xarray DataArray."""

    if is_data_array(obj):
        return obj

    if is_dataset(obj) and variable is not None:
        return obj[variable]

    input_type = type(obj)
    raise TypeError("Given object is of type '%r'" % input_type)


def check_base_period(data, base_period=None, time_name=None):
    """Get list containing start and end points of base period.

    Parameters
    ----------
    data : xarray object
        DataArray or Dataset to compute base period for.

    base_period : 2-element list, optional
        If given, the first and last time included in the
        base period.

    time_name : str, optional
        The name of the time coordinate.

    Returns
    -------
    base_period : 2-element list
        List containing the first and last time included
        in the base period.
    """

    if time_name is None:
        time_name = get_coordinate_standard_name(data, 'time')

    if base_period is None:
        if issubclass(data[time_name].dtype.type, np.datetime64):
            base_period = [data[time_name].min().values,
                           data[time_name].max().values]
        else:
            base_period = [data[time_name].min().item(),
                           data[time_name].max().item()]
    else:
        if len(base_period) != 2:
            raise ValueError(
                'Incorrect length for base period: '
                'expected length 2 list '
                'but got %r' % base_period)

        base_period = sorted(base_period)

    return base_period


def has_fixed_missing_values(X, axis=0):
    """Check if NaN values occur in fixed features throughout array."""

    if is_dask_array(X):
        nan_mask = dask.array.isnan(X)
    else:
        nan_mask = np.isnan(X)

    return (nan_mask.any(axis=axis) == nan_mask.all(axis=axis)).all()


def check_fixed_missing_values(X, axis=0):
    """Check if array has fixed missing values."""

    if not has_fixed_missing_values(X, axis=axis):
        raise ValueError(
            'Variable has partial missing values')

    return X


def remove_missing_features(X):
    """Remove features that have only missing values.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Array containing the data to check.

    Returns
    -------
    nonmissing_data : array-like, shape (n_samples, n_nonmissing)
        Array with features containing only missing values removed.

    missing_features : array-like, shape (n_missing)
        Array containing the column indices corresponding to
        the removed features.
    """

    # Only features that are complete or are all missing are handled
    check_fixed_missing_values(X, axis=0)

    n_features = X.shape[1]

    if is_dask_array(X):
        missing_features = dask.array.nonzero(
            dask.array.isnan(X[0]))[0].compute()
    else:
        missing_features = np.where(np.isnan(X[0]))[0]

    nonmissing_features = [i for i in range(n_features)
                           if i not in missing_features]
    nonmissing_data = X[:, nonmissing_features]

    return nonmissing_data, missing_features


def restore_missing_features(nonmissing_X, missing_features):
    """Insert columns corresponding to missing features.

    Parameters
    ----------
    nonmissing_X : array-like, shape (n_samples, n_nonmissing)
        Array containing data with missing features removed.

    missing_features : array-like, shape (n_missing,)
        Array containing the column indices in the original
        data that correspond to missing features.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Array with missing features inserted.
    """

    if missing_features is None:
        missing_features = []

    n_samples, n_nonmissing = nonmissing_X.shape
    n_missing = len(missing_features)

    n_features = n_missing + n_nonmissing

    # Ensure missing indices are consistent with the
    # inferred number of features.
    if len(missing_features) > 0:
        if min(missing_features) < 0 or max(missing_features) >= n_features:
            raise ValueError(
                'Missing features are inconsistent with '
                'number of features in complete data')

    if is_dask_array(nonmissing_X):
        cols = []
        idx = 0
        for i in range(n_features):
            if i in missing_features:
                cols.append(dask.array.full((n_samples, 1), np.NaN))
            else:
                cols.append(nonmissing_X[:, idx].reshape((n_samples, 1)))
                idx += 1

        X = dask.array.hstack(cols)

    else:
        nonmissing_features = [i for i in range(n_features)
                               if i not in missing_features]

        X = np.full((n_samples, n_features), np.NaN)
        X[:, nonmissing_features] = nonmissing_X

    return X
