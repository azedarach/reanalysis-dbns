"""
Provides helper routines for reanalysis DBNs study.
"""

# License: MIT


from __future__ import absolute_import

from .computation import (calc_truncated_svd, downsample_data,
                          meridional_mean,
                          pattern_correlation, select_lat_band,
                          select_latlon_box, select_lon_band,
                          standardized_anomalies, zonal_mean)
from .defaults import (get_coordinate_standard_name,
                       get_default_indicator_name, get_lat_name,
                       get_level_name, get_lon_name, get_time_name)
from .eofs import (eofs, reofs)
from .preprocessing import (construct_lagged_data,
                            get_offset_variable_name,
                            remove_polynomial_trend,
                            standardize_time_series)
from .time_helpers import datetime_to_string
from .validation import (check_array_shape, check_base_period,
                         check_fixed_missing_values,
                         check_max_memory, check_max_parents,
                         check_number_of_chains,
                         check_number_of_initializations,
                         check_number_of_iterations,
                         check_tolerance, check_warmup,
                         detect_frequency, ensure_data_array,
                         ensure_variables_in_data,
                         has_fixed_missing_values,
                         is_daily_data,
                         is_dask_array, is_data_array, is_dataset,
                         is_integer, is_monthly_data, is_pandas_dataframe,
                         is_pandas_object, is_pandas_series, is_scalar,
                         is_xarray_object, remove_missing_features,
                         restore_missing_features)

__all__ = [
    'calc_truncated_svd',
    'check_array_shape',
    'check_fixed_missing_values',
    'check_base_period',
    'check_max_memory',
    'check_max_parents',
    'check_number_of_chains',
    'check_number_of_initializations',
    'check_number_of_iterations',
    'check_tolerance',
    'check_warmup',
    'construct_lagged_data',
    'datetime_to_string',
    'detect_frequency',
    'downsample_data',
    'ensure_data_array',
    'ensure_variables_in_data',
    'eofs',
    'get_coordinate_standard_name',
    'get_default_indicator_name',
    'get_lat_name',
    'get_level_name',
    'get_lon_name',
    'get_offset_variable_name',
    'get_time_name',
    'get_valid_variables',
    'has_fixed_missing_values',
    'is_daily_data',
    'is_dask_array',
    'is_data_array',
    'is_dataset',
    'is_integer',
    'is_monthly_data',
    'is_pandas_dataframe',
    'is_pandas_object',
    'is_pandas_series',
    'is_scalar',
    'is_xarray_object',
    'meridional_mean',
    'pattern_correlation',
    'remove_missing_features',
    'remove_polynomial_trend',
    'restore_missing_features',
    'reofs',
    'select_lat_band',
    'select_latlon_box',
    'select_lon_band',
    'standardized_anomalies',
    'standardize_time_series',
    'zonal_mean'
]
