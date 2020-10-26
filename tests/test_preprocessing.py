"""
Provides unit tests for preprocessing helper routines.
"""

# License: MIT

from __future__ import absolute_import, division

import numpy as np
import pandas as pd

from sklearn.utils import check_random_state

import reanalysis_dbns.utils as rdu


def test_construction_of_lagged_dataframe():
    """Test construction of lagged data when input is a pandas DataFrame."""

    index = np.arange(10)
    df = pd.DataFrame(
        {'x': np.arange(10), 'y': np.arange(10, 20)}, index=index)

    offsets = [('x', 0), ('y', -1)]

    lagged_df = rdu.construct_lagged_data(offsets, df)

    expected_index = np.arange(1, 10)
    expected_df = pd.DataFrame(
        {'x': np.arange(1, 10), 'y_lag_1': np.arange(10, 19, dtype='f8')},
        index=expected_index)

    assert lagged_df.equals(expected_df)

    index = pd.date_range('2000-01-01', freq='1D', periods=10)
    df = pd.DataFrame({'x': np.arange(10, dtype='f8'),
                       'y': np.arange(10, 20, dtype='f8')}, index=index)

    offsets = [('x', -1), ('y', -2)]

    lagged_df = rdu.construct_lagged_data(offsets, df)

    expected_index = index[2:]
    expected_df = pd.DataFrame(
        {'x_lag_1': np.arange(1, 9, dtype='f8'),
         'y_lag_2': np.arange(10, 18, dtype='f8')},
        index=expected_index)

    assert lagged_df.equals(expected_df)

    index = pd.date_range('2000-01-01', freq='1D', periods=5)
    df = pd.DataFrame(
        {'x': pd.Categorical(['a', 'b', 'a', 'b', 'c']),
         'y': np.arange(2, 7, dtype='f8')}, index=index)

    offsets = [('x', -1), ('y', 1)]

    lagged_df = rdu.construct_lagged_data(offsets, df)

    expected_index = index[1:]
    expected_df = pd.DataFrame(
        {'x_lag_1': pd.Categorical(['a', 'b', 'a', 'b'],
                                   categories=['a', 'b', 'c']),
         'y_lead_1': np.array([4.0, 5.0, 6.0, np.NaN])},
        index=expected_index)

    assert lagged_df.equals(expected_df)


def test_remove_polynomial_trend():
    """Test removal of polynomial trends from data."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 100
    t = pd.date_range('2010-01-01', periods=n_samples, freq='1D')
    x = -2.3 + 0.02 * np.arange(n_samples)
    data = pd.DataFrame({'x': x}, index=t)
    detrended_data = rdu.remove_polynomial_trend(data, trend_order=1)
    expected_df = pd.DataFrame({'x': np.zeros(n_samples)}, index=t)

    assert detrended_data.equals(expected_df)

    x += random_state.normal(size=(n_samples,))
    data = pd.DataFrame({'x': x}, index=t)
    detrended_data = rdu.remove_polynomial_trend(data, trend_order=1)

    assert np.abs(np.mean(detrended_data['x'])) < 1e-12

    x = -2.3 + 0.02 * np.arange(n_samples) - 0.001 * np.arange(n_samples)**2
    data = pd.DataFrame({'x': x}, index=t)
    detrended_data = rdu.remove_polynomial_trend(data, trend_order=2)
    expected_df = pd.DataFrame({'x': np.zeros(n_samples)}, index=t)

    assert np.allclose(detrended_data.to_numpy(), expected_df.to_numpy())

    x += random_state.normal(size=(n_samples,))
    data = pd.DataFrame({'x': x}, index=t)
    detrended_data = rdu.remove_polynomial_trend(data, trend_order=1)

    assert np.abs(np.mean(detrended_data['x'])) < 1e-12


def test_standardize_time_series():
    """Test standardization of time series data."""

    random_seed = 0
    random_state = check_random_state(random_seed)

    n_samples = 100
    t = pd.date_range('2010-01-01', periods=n_samples, freq='1D')
    x = random_state.normal(loc=2.0, scale=3.2, size=n_samples)
    data = pd.DataFrame({'x': x}, index=t)
    standardized_data = rdu.standardize_time_series(data)

    assert np.abs(np.mean(standardized_data['x'])) < 1e-12
    assert np.abs(np.std(standardized_data['x'], ddof=1) - 1) < 1e-12

    t = pd.DatetimeIndex(
        [pd.Timestamp('2009-12-01'), pd.Timestamp('2010-01-01'),
         pd.Timestamp('2010-02-01'), pd.Timestamp('2010-12-01'),
         pd.Timestamp('2011-01-01'), pd.Timestamp('2011-02-01'),
         pd.Timestamp('2011-12-01'), pd.Timestamp('2012-01-01'),
         pd.Timestamp('2012-02-01'), pd.Timestamp('2012-12-01'),
         pd.Timestamp('2013-01-01'), pd.Timestamp('2013-02-01')],
         )
    x = np.array(
        [random_state.normal(loc=1.0, scale=0.2),
         random_state.normal(loc=2.0, scale=3.0),
         random_state.normal(loc=-3.2, scale=0.1),
         random_state.normal(loc=1.0, scale=0.2),
         random_state.normal(loc=2.0, scale=3.0),
         random_state.normal(loc=-3.2, scale=0.1),
         random_state.normal(loc=1.0, scale=0.2),
         random_state.normal(loc=2.0, scale=3.0),
         random_state.normal(loc=-3.2, scale=0.1),
         random_state.normal(loc=1.0, scale=0.2),
         random_state.normal(loc=2.0, scale=3.0),
         random_state.normal(loc=-3.2, scale=0.1)])
    data = pd.DataFrame({'x': x}, index=t)
    standardized_data = rdu.standardize_time_series(
        data, standardize_by='month')

    assert np.all(
        np.abs(standardized_data['x'].groupby(
            standardized_data.index.month).mean()) < 1e-12)
    assert np.all(
        np.abs(standardized_data['x'].groupby(
            standardized_data.index.month).std(ddof=1) - 1) < 1e-12)

    n_samples = 1000
    t = pd.date_range('2010-01-01', periods=n_samples, freq='1D')
    x = random_state.normal(loc=3.2, scale=4.0, size=n_samples)
    data = pd.DataFrame({'x': x}, index=t)
    standardized_data = rdu.standardize_time_series(
        data, standardize_by='dayofyear')

    assert np.all(
        np.abs(standardized_data['x'].groupby(
            standardized_data.index.dayofyear).mean()) < 1e-12)
    assert np.all(
        np.abs(standardized_data['x'].groupby(
            standardized_data.index.dayofyear).std(ddof=1) - 1) < 1e-12)
