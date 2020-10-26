"""
Provides helper routines for preprocessing.
"""

# License: MIT

from __future__ import absolute_import, division


import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as ss

from .validation import (is_integer, is_pandas_dataframe,
                         is_pandas_series)


def get_offset_variable_name(variable, offset):
    """Get placeholder name for lagged variable."""

    if offset < 0:
        return '{}_lag_{:d}'.format(variable, abs(offset))

    if offset == 0:
        return variable

    return '{}_lead_{:d}'.format(variable, offset)


def _check_presample_length(presample_length, offsets):
    """Check presample length is consistent with the given offsets."""

    max_lag = min(offsets)

    if presample_length is None:
        presample_length = abs(max_lag)
    else:
        if not is_integer(presample_length) or presample_length < max_lag:
            raise ValueError(
                'Presample length must be an integer greater '
                'than or equal to the maximum lag '
                '(got presample_length=%r but min(offsets)=%d)' %
                (presample_length, max_lag))

    return presample_length


def _construct_lagged_dataframe(variables_and_offsets, data,
                                presample_length=None,
                                **kwargs):
    """Construct pandas DataFrame containing lagged variables.

    Parameters
    ----------
    variables_and_offsets : list
        List of tuples of the form (variable name, offset).

    data : pandas DataFrame
        Dataframe containing the values of each variable.

    presample_length : int
        Minimum number of rows to treat as pre-sample values.

    Returns
    -------
    lagged : pandas DataFrame
        Dataframe with columns containing offset values
        for each variable and offset pair.
    """

    if not is_pandas_dataframe(data):
        raise ValueError(
            'Input data must be a pandas DataFrame '
            '(got type(data)=%r)' % type(data))

    offsets = [v[1] for v in variables_and_offsets]
    presample_length = _check_presample_length(presample_length, offsets)

    lagged_series = {
        get_offset_variable_name(*v): data[v[0]].shift(
            periods=-v[1], **kwargs)
        for v in variables_and_offsets}

    return pd.DataFrame(lagged_series).iloc[presample_length:]


def construct_lagged_data(variables_and_lags, data, presample_length=None,
                          **kwargs):
    """Construct dataset containing lagged variables.

    Parameters
    ----------
    variables_and_lags : list
        List of tuples of the form (variable name, lag).

    data : dict-like
        Object containing the values of each variable.

    presample_length : int
        Minimum number of rows to treat as pre-sample values.

    Returns
    -------
    lagged : dict-like
        Object with keys corresponding to lagged values
        for each variable and lag pair.
    """

    if is_pandas_dataframe(data):
        return _construct_lagged_dataframe(
            variables_and_lags, data, presample_length=presample_length,
            **kwargs)

    raise NotImplementedError(
        'Construction of lagged data not supported for given data type '
        '(got type(data)=%r)' % type(data))


def _check_standardization_interval(standardize_by):
    """Check standardization interval is valid."""

    valid_intervals = ['dayofyear', 'month']
    is_valid = standardize_by is None or standardize_by in valid_intervals

    if not is_valid:
        raise ValueError(
            "Unrecognized standardization interval '%r'" % standardize_by)

    return standardize_by


def _standardize_time_series_dataframe(data, base_period=None,
                                       standardize_by=None,
                                       resample=False):
    """Standardize time series of index values."""

    if base_period is None:
        base_period = [data.index.min(), data.index.max()]

    standardize_by = _check_standardization_interval(standardize_by)

    if standardize_by is None:

        base_period_data = data[(data.index >= base_period[0]) &
                                (data.index <= base_period[1])]

        return (data - base_period_data.mean()) / base_period_data.std(ddof=1)

    if standardize_by == 'dayofyear':

        must_resample = pd.infer_freq(data.index) not in ('D', '1D')
        if resample and must_resample:
            data = data.resample('1D').mean()

        groups = data.index.dayofyear

    else:

        must_resample = (pd.infer_freq(data.index) not in
                         ('M', 'MS', '1M', '1MS'))
        if resample and must_resample:
            data = data.resample('1M').mean()

        groups = data.index.month

    def standardize_within_base_period(x):
        mask = (x.index >= base_period[0]) & (x.index <= base_period[1])
        return (x - x[mask].mean()) / (x[mask].std(ddof=1))

    return data.groupby(groups).apply(standardize_within_base_period)


def _standardize_time_series_series(data, base_period=None,
                                    standardize_by=None,
                                    resample=False):
    """Standardize time series of index values."""

    if data.name:
        input_name = data.name
    else:
        input_name = 'x'

    df = pd.DataFrame({input_name: data})

    standardized_df = _standardize_time_series_dataframe(
        df, base_period=base_period, standardize_by=standardize_by,
        resample=resample)

    standardized = standardized_df[input_name]
    if not data.name:
        standardized.name = None

    return standardized


def standardize_time_series(data, base_period=None, standardize_by=None,
                            resample=False):
    """Standardize input time series."""

    if is_pandas_dataframe(data):
        return _standardize_time_series_dataframe(
            data, base_period=base_period, standardize_by=standardize_by,
            resample=resample)

    if is_pandas_series(data):
        return _standardize_time_series_series(
            data, base_period=base_period, standardize_by=standardize_by,
            resample=resample)

    raise NotImplementedError(
        'Construction of standardized data not supported for given data type '
        '(got type(data)=%r)' % type(data))


def _remove_polynomial_trend_series(data, trend_order=1):
    """Remove polynomial trend from data."""

    n_samples = data.shape[0]

    nonmissing_mask = pd.notna(data)
    t = np.arange(n_samples)
    t_valid = t[nonmissing_mask]
    valid_data = data.where(nonmissing_mask).dropna().array

    if trend_order == 1:

        slope, intcpt, _, _, _ = ss.linregress(t_valid, valid_data)

        trend = pd.Series(
            slope * t + intcpt, index=data.index, name=data.name)

        detrended = data - trend

    else:

        def polynomial_trend(x, *a):
            val = np.full(x.shape, a[0])
            for i in range(1, len(a)):
                val += a[i] * x ** i
            return val

        initial_guess = np.ones((trend_order + 1,))

        fitted_coeffs, fitted_cov = so.curve_fit(
            polynomial_trend, t_valid, valid_data,
            p0=initial_guess)

        trend = pd.Series(
            polynomial_trend(t, *fitted_coeffs),
            index=data.index, name=data.name)

        detrended = data - trend

    return detrended


def _remove_polynomial_trend_dataframe(data, trend_order=1):
    """Remove polynomial trends from elements of DataFrame."""

    vars = list(data.keys())

    detrended = {}
    for v in vars:
        detrended[v] = _remove_polynomial_trend_series(
            data[v], trend_order=trend_order)

    return pd.DataFrame(detrended, index=data.index)


def remove_polynomial_trend(data, trend_order=1):
    """Remove polynomial trends from data."""

    if is_pandas_dataframe(data):
        return _remove_polynomial_trend_dataframe(
            data, trend_order=trend_order)
    elif is_pandas_series(data):
        return _remove_polynomial_trend_series(
            data, trend_order=trend_order)

    raise NotImplementedError(
        'Construction of detrended data not supported for given data type '
        '(got type(data)=%r)' % type(data))
