"""
Provides default coordinate names and conventions.
"""

# License: MIT


def get_default_indicator_name(predictor):
    """Get default name for indicator variable."""
    return 'i_{}'.format(predictor)


def get_coordinate_standard_name(obj, coord):
    """Return standard name for coordinate.

    Parameters
    ----------
    obj : xarray object
        Object to search for coordinate name in.

    coord : 'lat' | 'lon' | 'level' | 'time'
        Coordinate to search for.

    Returns
    -------
    name : str
        Standard name of coordinate if found
    """

    valid_keys = ['lat', 'lon', 'level', 'time']

    if coord not in valid_keys:
        raise ValueError("Unrecognized coordinate key '%r'" % coord)

    if coord == 'lat':

        candidates = ['lat', 'latitude', 'g0_lat_1', 'g0_lat_2',
                      'lat_2', 'yt_ocean']

    elif coord == 'lon':

        candidates = ['lon', 'longitude', 'g0_lon_2', 'g0_lon_3',
                      'lon_2', 'xt_ocean']

    elif coord == 'level':

        candidates = ['level']

    elif coord == 'time':

        candidates = ['time', 'initial_time0_hours']

    for c in candidates:
        if c in obj.dims:
            return c

    raise ValueError("Unable to find coordinate '%s'" % coord)


def get_lat_name(obj):
    """Return name of latitude coordinate.

    Parameters
    ----------
    obj : xarray object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of latitude coordinate if found
    """
    return get_coordinate_standard_name(obj, 'lat')


def get_level_name(obj):
    """Return name of pressure level coordinate.

    Parameters
    ----------
    obj : xarray object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of pressure level coordinate if found
    """
    return get_coordinate_standard_name(obj, 'level')


def get_lon_name(obj):
    """Return name of longitude coordinate.

    Parameters
    ----------
    obj : xarray object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of longitude coordinate if found
    """
    return get_coordinate_standard_name(obj, 'lon')


def get_time_name(obj):
    """Return name of time coordinate.

    Parameters
    ----------
    obj : xarray object
        Object to search for coordinate name in.

    Returns
    -------
    name : str
        Standard name of time coordinate if found
    """
    return get_coordinate_standard_name(obj, 'time')
