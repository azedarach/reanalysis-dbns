"""
Provides unit tests for validation functions.
"""

# License: MIT

from __future__ import absolute_import

import dask.array
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import reanalysis_dbns.utils as rdu


def test_is_integer():
    """Test checking argument is an integer."""

    assert rdu.is_integer(1)
    assert not rdu.is_integer(1.0)
    assert not rdu.is_integer(np.array([1]))
    assert not rdu.is_integer(np.array(1))


def test_is_scalar():
    """Test checking argument is a scalar."""

    assert rdu.is_scalar(0)
    assert rdu.is_scalar(3.2)
    assert not rdu.is_scalar([1])
    assert not rdu.is_scalar({'a': 2})
    assert not rdu.is_scalar(np.array([2.0]))
    assert not rdu.is_scalar(np.array(-2))


def test_is_data_array():
    """Test checking argument is a DataArray."""

    da = xr.DataArray(np.zeros((4, 4)))

    assert rdu.is_data_array(da)

    ds = xr.Dataset({'a': da})

    assert not rdu.is_data_array(ds)

    assert not rdu.is_data_array(pd.Series(np.zeros(4)))
    assert not rdu.is_data_array(pd.DataFrame({'a': np.zeros(3)}))
    assert not rdu.is_data_array(np.zeros((3, 3)))


def test_is_dataset():
    """Test checking argument is a Dataset."""

    da = xr.DataArray(np.zeros((23, 21)))

    assert not rdu.is_dataset(da)

    ds = xr.Dataset({'a': da})

    assert rdu.is_dataset(ds)

    assert not rdu.is_dataset(pd.Series(np.zeros(4)))
    assert not rdu.is_dataset(pd.DataFrame({'a': np.zeros(3)}))
    assert not rdu.is_dataset(np.zeros((34, 2)))


def test_is_xarray_object():
    """Test checking argument is an xarray object."""

    da = xr.DataArray(np.zeros((4, 4)))

    assert rdu.is_xarray_object(da)

    ds = xr.Dataset({'a': da})

    assert rdu.is_xarray_object(ds)

    assert not rdu.is_xarray_object(pd.Series(np.zeros(4)))
    assert not rdu.is_xarray_object(pd.DataFrame({'a': np.zeros(3)}))
    assert not rdu.is_xarray_object(np.zeros((34, 2)))


def test_is_dask_array():
    """Test checking argument is a Dask array."""

    a = dask.array.from_array(np.zeros((3, 4)))

    assert rdu.is_dask_array(a)

    assert not rdu.is_dask_array(np.zeros((3, 4)))

    da = xr.DataArray(np.zeros((4, 4)))

    assert not rdu.is_dask_array(da)

    da = xr.DataArray(a)

    assert not rdu.is_dask_array(da)


def test_is_pandas_dataframe():
    """Test checking argument is a pandas DataFrame."""

    da = xr.DataArray(np.zeros((4, 4)))

    assert not rdu.is_pandas_dataframe(da)

    ds = xr.Dataset({'a': da})

    assert not rdu.is_pandas_dataframe(ds)

    assert not rdu.is_pandas_dataframe(pd.Series(np.zeros(4)))
    assert rdu.is_pandas_dataframe(pd.DataFrame({'a': np.zeros(3)}))


def test_is_pandas_object():
    """Test checking argument is a pandas object."""

    da = xr.DataArray(np.zeros((4, 4)))

    assert not rdu.is_pandas_object(da)

    ds = xr.Dataset({'a': da})

    assert not rdu.is_pandas_object(ds)

    assert rdu.is_pandas_object(pd.Series(np.zeros(4)))
    assert rdu.is_pandas_object(pd.DataFrame({'a': np.zeros(3)}))


def test_is_pandas_series():
    """Test checking argument is a pandas Series."""

    da = xr.DataArray(np.zeros((4, 4)))

    assert not rdu.is_pandas_series(da)

    ds = xr.Dataset({'a': da})

    assert not rdu.is_pandas_series(ds)

    assert rdu.is_pandas_series(pd.Series(np.zeros(4)))
    assert not rdu.is_pandas_series(pd.DataFrame({'a': np.zeros(3)}))


def test_check_number_of_chains():
    """Test checking number of MCMC chains."""

    assert rdu.check_number_of_chains(1) == 1

    with pytest.raises(ValueError):
        rdu.check_number_of_chains(0)

    with pytest.raises(ValueError):
        rdu.check_number_of_chains(10.0)

    with pytest.raises(ValueError):
        rdu.check_number_of_chains(-1)


def test_check_number_of_initializations():
    """Test checking number of initializations."""

    assert rdu.check_number_of_initializations(1) == 1

    with pytest.raises(ValueError):
        rdu.check_number_of_initializations(0)

    with pytest.raises(ValueError):
        rdu.check_number_of_initializations(23.0)

    with pytest.raises(ValueError):
        rdu.check_number_of_initializations(-10)


def test_check_number_of_iterations():
    """Test checking number of iterations."""

    assert rdu.check_number_of_iterations(1) == 1

    with pytest.raises(ValueError):
        rdu.check_number_of_iterations(0)

    with pytest.raises(ValueError):
        rdu.check_number_of_iterations(3.0)

    with pytest.raises(ValueError):
        rdu.check_number_of_iterations(-3)


def test_check_tolerance():
    """Test checking tolerance parameter."""

    assert rdu.check_tolerance(1) == 1

    with pytest.raises(ValueError):
        rdu.check_tolerance(0.0)

    with pytest.raises(ValueError):
        rdu.check_tolerance(-1)


def test_check_warmup():
    """Test checking number of warmup samples."""

    assert rdu.check_warmup(0, 10) == 0
    assert rdu.check_warmup(2, 3) == 2

    with pytest.raises(ValueError):
        rdu.check_warmup(-1, 10)

    with pytest.raises(ValueError):
        rdu.check_warmup(10, 10)

    with pytest.raises(ValueError):
        rdu.check_warmup(10, -10)

    with pytest.raises(ValueError):
        rdu.check_warmup(10.0, 20)


def test_ensure_variables_in_data():
    """Test ensuring dataset contains given variables."""

    with pytest.raises(ValueError):
        rdu.ensure_variables_in_data(None, ['a', 'b'])

    data = {'a': [1, 2, 3]}

    assert rdu.ensure_variables_in_data(data, ['a']) == data
    assert rdu.ensure_variables_in_data(data, []) == data

    with pytest.raises(ValueError):
        rdu.ensure_variables_in_data(data, ['a', 'x'])


def test_check_max_memory():
    """Test checking maximum memory for variables."""

    variable_names = ['a', 'b', 'c']

    result = rdu.check_max_memory(variable_names, 0)
    expected = {'a': 0, 'b': 0, 'c': 0}

    for v in variable_names:
        assert result[v] == expected[v]

    result = rdu.check_max_memory(variable_names, 1)
    expected = {'a': 1, 'b': 1, 'c': 1}

    for v in variable_names:
        assert result[v] == expected[v]

    with pytest.raises(ValueError):
        rdu.check_max_memory(variable_names, -1)

    with pytest.raises(ValueError):
        rdu.check_max_memory(variable_names, {'a': 0, 'b': 1})

    max_memory = {'a': 3, 'b': 0, 'c': 1}
    result = rdu.check_max_memory(variable_names, max_memory)
    for v in result:
        assert result[v] == max_memory[v]

    max_memory = [3, 1, 0]
    result = rdu.check_max_memory(variable_names, max_memory)
    for i, v in enumerate(variable_names):
        assert result[v] == max_memory[i]

    with pytest.raises(ValueError):
        rdu.check_max_memory(variable_names, [3, 2])

    with pytest.raises(ValueError):
        rdu.check_max_memory(variable_names, [3, -1, 2])


def test_check_max_parents():
    """Test checking maximum number of parent nodes for variables."""

    variable_names = ['a', 'b', 'c']

    result = rdu.check_max_parents(variable_names, None)
    expected = {'a': None, 'b': None, 'c': None}

    for v in variable_names:
        assert result[v] == expected[v]

    result = rdu.check_max_parents(variable_names, 0)
    expected = {'a': 0, 'b': 0, 'c': 0}

    for v in variable_names:
        assert result[v] == expected[v]

    result = rdu.check_max_parents(variable_names, 1)
    expected = {'a': 1, 'b': 1, 'c': 1}

    for v in variable_names:
        assert result[v] == expected[v]

    with pytest.raises(ValueError):
        rdu.check_max_parents(variable_names, -1)

    with pytest.raises(ValueError):
        rdu.check_max_parents(variable_names, {'a': 0, 'b': 1})

    max_parents = {'a': 3, 'b': 0, 'c': 1}
    result = rdu.check_max_parents(variable_names, max_parents)
    for v in result:
        assert result[v] == max_parents[v]

    max_parents = [3, 1, 0]
    result = rdu.check_max_parents(variable_names, max_parents)
    for i, v in enumerate(variable_names):
        assert result[v] == max_parents[i]

    with pytest.raises(ValueError):
        rdu.check_max_parents(variable_names, [3, 2])

    with pytest.raises(ValueError):
        rdu.check_max_parents(variable_names, [3, -1, 2])


def test_detect_frequency():
    """Test detecting sampling frequency."""

    t = pd.date_range('2010-01-01', freq='1D', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.detect_frequency(da) == 'daily'

    t = pd.date_range('2010-01-01', freq='1H', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.detect_frequency(da) == 'hourly'

    t = pd.date_range('2010-01-01', freq='1MS', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.detect_frequency(da) == 'monthly'

    t = pd.date_range('2010-01-01', freq='1A', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.detect_frequency(da) == 'yearly'

    t = np.array([np.datetime64('2010-01-01T09:00:00'),
                  np.datetime64('2010-01-02T10:00:00'),
                  np.datetime64('2010-01-03T09:00:00')])

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.detect_frequency(da) == 'daily'


def test_is_daily_data():
    """Test checking if data is sampled at daily resolution."""

    t = pd.date_range('2010-01-01', freq='1D', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.is_daily_data(da)

    t = pd.date_range('2010-01-01', freq='1W', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert not rdu.is_daily_data(da)


def test_is_monthly_data():
    """Test checking if data is sampled at monthly resolution."""

    t = pd.date_range('2010-01-01', freq='1MS', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert rdu.is_monthly_data(da)

    t = pd.date_range('2010-01-01', freq='1D', periods=3)

    da = xr.DataArray(
        np.zeros(3), coords={'time': t}, dims=['time'])

    assert not rdu.is_monthly_data(da)


def test_check_array_shape():
    """Test checking array shape."""

    x = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        rdu.check_array_shape(x, (3, 4), 'test')


def test_ensure_data_array():
    """Test ensuring input is an xarray DataArray."""

    da = xr.DataArray(np.zeros((3, 4)))

    assert rdu.ensure_data_array(da).equals(da)

    ds = xr.Dataset({'a': da})

    assert rdu.ensure_data_array(ds, 'a').equals(da)

    with pytest.raises(TypeError):
        rdu.ensure_data_array(np.zeros((3, 3)))


def test_check_base_period():
    """Test checking base period."""

    t = pd.date_range('1982-03-02', freq='1D', periods=5)
    da = xr.DataArray(np.zeros(5), coords={'time': t}, dims=['time'])

    base_period = rdu.check_base_period(da)

    assert len(base_period) == 2
    assert base_period[0] == t[0]
    assert base_period[1] == t[-1]

    t = np.array([np.datetime64('2010-01-01'),
                  np.datetime64('2010-06-01'),
                  np.datetime64('2011-01-01')])

    da = xr.DataArray(np.zeros(3), coords={'time': t}, dims=['time'])

    base_period = rdu.check_base_period(da)

    assert len(base_period) == 2
    assert base_period[0] == t[0]
    assert base_period[1] == t[-1]

    target_base_period = [np.datetime64('2010-03-01'),
                          np.datetime64('2011-06-01')]

    base_period = rdu.check_base_period(da, base_period=target_base_period)

    assert base_period[0] == target_base_period[0]
    assert base_period[1] == target_base_period[1]


def test_has_fixed_missing_values():
    """Test checking an array has fixed missing values."""

    x = np.array([[1, np.NaN, 3], [4, np.NaN, 6]])

    assert rdu.has_fixed_missing_values(x, axis=0)

    x[0, 1] = 2.0

    assert not rdu.has_fixed_missing_values(x, axis=0)


def test_check_fixed_missing_values():
    """Test checking an array has fixed missing values."""

    x = np.array([[1, np.NaN, 3], [4, np.NaN, 6]])

    result = rdu.check_fixed_missing_values(x, axis=0)

    assert np.allclose(result[:, 0], x[:, 0])
    assert np.allclose(result[:, 2], x[:, 2])

    x[0, 1] = 2.0

    with pytest.raises(ValueError):
        rdu.check_fixed_missing_values(x, axis=0)


def test_remove_and_restore_missing_features():
    """Test removing and restoring missing features."""

    x = np.array([[1, np.NaN, 3], [4, np.NaN, 6]])

    result, missing_vars = rdu.remove_missing_features(x)

    assert result.shape == (2, 2)
    assert missing_vars.shape == (1,)

    assert missing_vars[0] == 1

    assert np.allclose(result[:, 0], x[:, 0])
    assert np.allclose(result[:, 1], x[:, 2])

    x[0, 1] = 2.0

    with pytest.raises(ValueError):
        rdu.remove_missing_features(x)

    nonmissing_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    missing_vars = np.array([1, 3])
    result = rdu.restore_missing_features(nonmissing_x, missing_vars)

    assert result.shape == (2, 4)

    assert np.allclose(result[:, 0], nonmissing_x[:, 0])
    assert np.all(np.isnan(result[:, 1]))
    assert np.allclose(result[:, 2], nonmissing_x[:, 1])
    assert np.all(np.isnan(result[:, 3]))

    with pytest.raises(ValueError):
        missing_vars = np.array([1, 3, 5])
        rdu.restore_missing_features(nonmissing_x, missing_vars)

    x = np.array([[np.NaN, 2, 3], [np.NaN, 4, 5], [np.NaN, 6, 7]])

    nonmissing_x, missing_vars = rdu.remove_missing_features(x)
    missing_x = rdu.restore_missing_features(nonmissing_x, missing_vars)

    assert np.all(np.isnan(missing_x[:, 0]))
    assert np.allclose(missing_x[:, 1:], x[:, 1:])

    nonmissing_x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    missing_vars = []
    x = rdu.restore_missing_features(nonmissing_x, missing_vars)
    assert np.allclose(x, nonmissing_x)


def test_remove_and_restore_missing_features_dask():
    """Test removing and restoring missing features."""

    with dask.config.set(scheduler='single-threaded'):
        x = dask.array.from_array(np.array([[1, np.NaN, 3], [4, np.NaN, 6]]))

        result, missing_vars = rdu.remove_missing_features(x)

        assert result.shape == (2, 2)
        assert missing_vars.shape == (1,)

        assert missing_vars[0] == 1

        assert np.allclose(result[:, 0], x[:, 0])
        assert np.allclose(result[:, 1], x[:, 2])

        x = dask.array.from_array(np.array([[1, np.NaN, 3], [4, 2.0, 6]]))

        with pytest.raises(ValueError):
            rdu.remove_missing_features(x)

        nonmissing_x = dask.array.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]]))
        missing_vars = np.array([1, 3])
        result = rdu.restore_missing_features(nonmissing_x, missing_vars)

        assert result.shape == (2, 4)

        assert np.allclose(result[:, 0], nonmissing_x[:, 0])
        assert np.all(np.isnan(result[:, 1]))
        assert np.allclose(result[:, 2], nonmissing_x[:, 1])
        assert np.all(np.isnan(result[:, 3]))

        with pytest.raises(ValueError):
            missing_vars = np.array([1, 3, 5])
            rdu.restore_missing_features(nonmissing_x, missing_vars)

        x = dask.array.from_array(
            np.array([[np.NaN, 2, 3], [np.NaN, 4, 5], [np.NaN, 6, 7]]))

        nonmissing_x, missing_vars = rdu.remove_missing_features(x)
        missing_x = rdu.restore_missing_features(nonmissing_x, missing_vars)

        assert np.all(np.isnan(missing_x[:, 0]))
        assert np.allclose(missing_x[:, 1:], x[:, 1:])
