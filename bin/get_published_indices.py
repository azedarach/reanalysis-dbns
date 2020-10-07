#!/usr/bin/env python

"""Fetch and format published indices."""

# License: MIT

from __future__ import absolute_import


import argparse
import logging
import os
import urllib.request

import numpy as np
import pandas as pd


BOM_FTP_ROOT = 'ftp://ftp.bom.gov.au'
BOM_HTTP_ROOT = 'http://www.bom.gov.au'
CDG_UCAR_HTTPS_ROOT = 'https://climatedataguide.ucar.edu'
NOAA_CPC_FTP_ROOT = 'ftp://ftp.cpc.ncep.noaa.gov'
NOAA_CPC_HTTPS_ROOT = 'https://www.cpc.ncep.noaa.gov'
NOAA_ESRL_HTTPS_ROOT = 'https://www.esrl.noaa.gov'


def format_cpc_index(src_file, target_file, frequency='daily', **kwargs):
    """Format CPC index data."""

    if frequency not in ('daily', 'monthly'):
        raise ValueError("Unrecognized frequency argument '%s'" % frequency)

    if frequency == 'monthly':
        columns = ['year', 'month', 'value']
        widths = [5, 5, 14]
        dtype = {'year': 'i8', 'month': 'i8', 'value': 'f8'}
    else:
        columns = ['year', 'month', 'day', 'value']
        widths = [4, 3, 3, 7]
        dtype = {'year': 'i8', 'month': 'i8', 'day': 'i8', 'value': 'f8'}

    logging.info('Reading source file: %s', src_file)
    input_data = pd.read_fwf(src_file, na_values='*******', header=None,
                             widths=widths, names=columns, dtype=dtype)

    n_samples = input_data.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = input_data['year'].values
    data[:, 1] = input_data['month'].values

    if frequency == 'daily':
        data[:, 2] = input_data['day'].values

    data[:, 3] = input_data['value'].values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_standardized_monthly_cpc_index(src_file, target_file,
                                          **unused_kwargs):
    """Format CPC index data."""

    logging.info('Reading source file: %s', src_file)
    input_data = pd.read_csv(src_file, skiprows=8, na_values='-99.90',
                             header=0, delim_whitespace=True)
    input_data = input_data.rename(
        columns={'YEAR': 'year', 'MONTH': 'month', 'INDEX': 'value'})

    n_samples = input_data.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = input_data['year'].values
    data[:, 1] = input_data['month'].values
    data[:, 3] = input_data['value'].values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_hurrell_nao_index(src_file, target_file):
    """Format Hurrell station-based NAO index."""

    logging.info('Reading source file: %s', src_file)
    input_data = pd.read_csv(src_file, skiprows=1, delim_whitespace=True,
                             na_values='-999.')

    stacked_data = input_data.stack()
    times = np.array([pd.to_datetime('{} {}'.format(i[0], i[1]))
                      for i in stacked_data.index])
    values = stacked_data.values

    n_samples = times.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = np.array([dt.year for dt in times])
    data[:, 1] = np.array([dt.month for dt in times])
    data[:, 3] = values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_marshall_sam_index(src_file, target_file):
    """Format Marshall SAM index."""

    logging.info('Reading source file: %s', src_file)
    input_data = pd.read_csv(src_file, delim_whitespace=True)

    stacked_data = input_data.stack()
    times = np.array([pd.to_datetime('{} {}'.format(i[0], i[1]))
                      for i in stacked_data.index])
    values = stacked_data.values

    n_samples = times.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = np.array([dt.year for dt in times])
    data[:, 1] = np.array([dt.month for dt in times])
    data[:, 3] = values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_cpc_soi(src_file, target_file):
    """Format CPC SOI data."""

    index_table_start = None
    with open(src_file, 'r') as ifs:
        for i, line in enumerate(ifs):
            if 'STANDARDIZED' in line:
                index_table_start = i + 2

    if index_table_start is None:
        raise RuntimeError('Start of standardized data not found')

    logging.info('Reading source file: %s', src_file)
    widths = [4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
    input_data = pd.read_fwf(
        src_file, header=0, widths=widths, na_values='-999.9', index_col=0,
        skiprows=index_table_start)

    stacked_data = input_data.stack()
    times = np.array([pd.to_datetime('{} {}'.format(i[0], i[1]))
                      for i in stacked_data.index])
    values = stacked_data.values

    n_samples = times.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = np.array([dt.year for dt in times])
    data[:, 1] = np.array([dt.month for dt in times])
    data[:, 3] = values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_cpc_nino_indices(src_file, target_files):
    """Format combined Nino indices."""

    col_titles = None
    with open(src_file, 'r') as ifs:
        header = ifs.readline()
        col_titles = [field.strip() for field in header.split()]

    logging.info('Reading source file: %s', src_file)

    for index in target_files:

        target_file = target_files[index]
        target_col = col_titles.index(index) + 1

        input_data = pd.read_csv(src_file, header=0,
                                 usecols=[0, 1, target_col],
                                 names=['year', 'month', 'value'],
                                 delim_whitespace=True)

        n_samples = input_data.shape[0]
        logging.info('Number of samples: %d', n_samples)

        data = np.ones((n_samples, 4))
        data[:, 0] = input_data['year'].values
        data[:, 1] = input_data['month'].values
        data[:, 3] = input_data['value'].values

        header = 'year,month,day,value'
        fmt = '%d,%d,%d,%16.8e'

        logging.info('Writing formatted data to: %s', target_file)
        np.savetxt(target_file, data, header=header, fmt=fmt)


def read_esrl_timeseries(src_file):
    """Read datafile formatted according to ESRL time-series format."""

    logging.info('Reading source file: %s', src_file)
    with open(src_file, 'r') as ifs:

        years_range = [int(f.strip()) for f in ifs.readline().split()]
        n_years = years_range[1] - years_range[0] + 1

        data = np.empty((n_years * 12, 3))
        for i, line in enumerate(ifs):
            fields = [f.strip() for f in line.split()]
            n_fields = len(fields)

            if n_fields == 13:
                year = int(fields[0])
                for month in range(1, 13):
                    data[12 * i + month - 1, 0] = year
                    data[12 * i + month - 1, 1] = month
                    data[12 * i + month - 1, 2] = fields[month]
            elif n_fields == 1:
                missing_value = float(fields[0])
                data[data == missing_value] = np.NaN
                break
            else:
                raise RuntimeError(
                    'Incorrect number of fields: expected 1 or 13 but got %d' %
                    n_fields)

        return pd.DataFrame(data, columns=['year', 'month', 'value'])


def format_esrl_mei(src_file, target_file):
    """Format ESRL MEIv2 index data."""

    input_data = read_esrl_timeseries(src_file)
    n_samples = input_data.shape[0]

    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = input_data['year'].values
    data[:, 1] = input_data['month'].values
    data[:, 3] = input_data['value'].values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_bom_soi(src_file, target_file):
    """Format monthly BOM SOI data."""

    logging.info('Reading source file: %s', src_file)

    # Skip HTML tags
    skiprows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    skipfooter = 3

    input_data = pd.read_csv(src_file, delim_whitespace=True,
                             skiprows=skiprows, skipfooter=skipfooter,
                             index_col=0, engine='python')

    stacked_data = input_data.stack()
    times = np.array([pd.to_datetime('{} {}'.format(i[0], i[1]))
                      for i in stacked_data.index])
    values = stacked_data.values

    n_samples = times.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = np.array([dt.year for dt in times])
    data[:, 1] = np.array([dt.month for dt in times])
    data[:, 3] = values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_bom_mjo(src_file, target_file):
    """Format daily WH MJO index from BOM."""

    cols = [0, 1, 2, 3, 4, 5, 6]
    names = ['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude']
    na_values = ['1E+36', '1.E+36', '999', 999,
                 '9.99999962E+35', 9.99999962E+35]

    logging.info('Reading source file: %s', src_file)

    input_data = pd.read_csv(src_file, skiprows=[0, 1], na_values=na_values,
                             usecols=cols, names=names, delim_whitespace=True)

    n_samples = input_data.shape[0]

    logging.info('Number of samples: %d', n_samples)

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    for field in target_file:

        data = np.ones((n_samples, 4))
        data[:, 0] = input_data['year']
        data[:, 1] = input_data['month']
        data[:, 2] = input_data['day']
        data[:, 3] = input_data[field]

        logging.info('Writing formatted data to: %s', target_file[field])
        np.savetxt(target_file[field], data, header=header, fmt=fmt)


def format_bom_dmi(src_file, target_file):
    """Format weekly BOM DMI index."""

    cols = [0, 2]
    names = ['date', 'value']

    logging.info('Reading source file: %s', src_file)

    input_data = pd.read_csv(src_file, usecols=cols, names=names,
                             parse_dates=[0])

    n_samples = input_data.shape[0]

    logging.info('Number of samples: %d', n_samples)

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    data = np.ones((n_samples, 4))
    data[:, 0] = pd.DatetimeIndex(input_data['date']).year
    data[:, 1] = pd.DatetimeIndex(input_data['date']).month
    data[:, 2] = pd.DatetimeIndex(input_data['date']).day
    data[:, 3] = input_data['value']

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def format_cpc_pna4(src_file, target_file):
    """Format CPC modified pointwise PNA index."""

    logging.info('Reading source file: %s', src_file)

    names = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
             'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    input_data = pd.read_csv(src_file, delim_whitespace=True,
                             header=None, names=names, index_col=0)

    stacked_data = input_data.stack()
    times = np.array([pd.to_datetime('{} {}'.format(i[0], i[1]))
                      for i in stacked_data.index])
    values = stacked_data.values

    n_samples = times.shape[0]
    logging.info('Number of samples: %d', n_samples)

    data = np.ones((n_samples, 4))
    data[:, 0] = np.array([dt.year for dt in times])
    data[:, 1] = np.array([dt.month for dt in times])
    data[:, 3] = values

    header = 'year,month,day,value'
    fmt = '%d,%d,%d,%16.8e'

    logging.info('Writing formatted data to: %s', target_file)
    np.savetxt(target_file, data, header=header, fmt=fmt)


def download(source_url, destination, use_request_api=False):
    """Download data from source to destination."""

    logging.info('Source URL: %s', source_url)
    logging.info('Downloading to: %s', destination)

    if use_request_api:
        req = urllib.request.Request(
            source_url, headers={'User-Agent': 'Mozilla/5.0'})

        data = urllib.request.urlopen(req).read()

        with open(destination, 'wb') as ofs:
            ofs.write(data)

    else:

        urllib.request.urlretrieve(source_url, destination, )


def download_and_format_index(source_url, download_kwargs, raw_file,
                              formatted_file,
                              format_func, format_func_kwargs):
    """Download and format index."""

    download(source_url, raw_file, **download_kwargs)
    format_func(raw_file, formatted_file, **format_func_kwargs)


def prepare_directories(data_dir, results_dir):
    """Check and create output directories."""

    if not os.path.exists(data_dir):
        logging.info('Creating directory: %s', data_dir)
        os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(results_dir):
        logging.info('Creating directory: %s', results_dir)
        os.makedirs(results_dir, exist_ok=True)


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Fetch and format published indices')

    parser.add_argument('data_dir', help='location to save raw data to')
    parser.add_argument('results_dir',
                        help='location to save formatted data to')

    args = parser.parse_args()

    return args.data_dir, args.results_dir


def main():
    """Fetch and format published indices."""

    data_dir, results_dir = parse_cmd_line_args()

    indices_to_download = [
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'cwlinks',
                 'norm.daily.ao.index.b500101.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.daily.ao.index.b500101.current.ascii'),
            'formatted_file': os.path.join(results_dir, 'cpc.ao.daily.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'daily'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'products', 'precip', 'CWlink',
                 'daily_ao_index',
                 'monthly.ao.index.b50.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'monthly.ao.index.b50.current.ascii'),
            'formatted_file': os.path.join(results_dir, 'cpc.ao.monthly.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'monthly'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'cwlinks',
                 'norm.daily.aao.index.b790101.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.daily.aao.index.b790101.current.ascii'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.sam.daily.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'daily'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'products', 'precip', 'CWlink',
                 'daily_ao_index', 'aao',
                 'monthly.aao.index.b79.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'monthly.aao.index.b79.current.ascii'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.sam.monthly.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'monthly'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'cwlinks',
                 'norm.daily.nao.index.b500101.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.daily.nao.index.b500101.current.ascii'),
            'formatted_file': os.path.join(results_dir, 'cpc.nao.daily.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'daily'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'products', 'precip', 'CWlink',
                 'pna', 'norm.nao.monthly.b5001.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.nao.monthly.b5001.current.ascii'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.nao.monthly.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'monthly'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'nao_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'nao_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.nao.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'cwlinks',
                 'norm.daily.pna.index.b500101.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.daily.pna.index.b500101.current.ascii'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.pna.daily.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'daily'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'products', 'precip', 'CWlink',
                 'pna', 'norm.pna.monthly.b5001.current.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.pna.monthly.b5001.current.ascii'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.pna.monthly.csv'),
            'format_func': format_cpc_index,
            'format_func_kwargs': {'frequency': 'monthly'}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'pna_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'pna_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.pna.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'scand_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'scand_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.scand.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'ea_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'ea_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.ea.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'eawr_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'eawr_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.eawr.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'poleur_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'poleur_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.poleur.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'wp_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'wp_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.wp.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'epnp_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'epnp_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.epnp.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'tnh_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'tnh_index.tim'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.tnh.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_FTP_ROOT, 'wd52dg', 'data', 'indices',
                 'pt_index.tim']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'pt_index.tim'),
            'formatted_file':
                os.path.join(
                    results_dir, 'cpc.pt.monthly.standardized.csv'),
            'format_func': format_standardized_monthly_cpc_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [CDG_UCAR_HTTPS_ROOT, 'sites', 'default', 'files',
                 'nao_station_monthly.txt']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'nao_station_monthly.txt'),
            'formatted_file': os.path.join(
                results_dir, 'hurrell.nao.monthly.csv'),
            'format_func': format_hurrell_nao_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                ['http://www.nerc-bas.ac.uk', 'public', 'icd', 'gjma',
                 'newsam.1957.2007.txt']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'newsam.1957.2007.txt'),
            'formatted_file': os.path.join(
                results_dir, 'marshall.sam.monthly.csv'),
            'format_func': format_marshall_sam_index,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'data', 'indices', 'soi']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'soi'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.soi.monthly.standardized.csv'),
            'format_func': format_cpc_soi,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'data', 'indices',
                 'ersst5.nino.mth.81-10.ascii']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'ersst5.nino.mth.81-10.ascii'),
            'formatted_file': {
                'NINO1+2': os.path.join(
                    results_dir, 'cpc.nino1+2.monthly.csv'),
                'NINO3': os.path.join(
                    results_dir, 'cpc.nino3.monthly.csv'),
                'NINO4': os.path.join(
                    results_dir, 'cpc.nino4.monthly.csv'),
                'NINO3.4': os.path.join(
                    results_dir, 'cpc.nino3.4.monthly.csv')},
            'format_func': format_cpc_nino_indices,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_ESRL_HTTPS_ROOT, 'psd', 'enso', 'mei', 'data',
                 'meiv2.data']),
            'download_kwargs': {},
            'raw_file': os.path.join(data_dir, 'meiv2.data'),
            'formatted_file': os.path.join(
                results_dir, 'esrl.mei.bimonthly.csv'),
            'format_func': format_esrl_mei,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [BOM_FTP_ROOT, 'anon', 'home', 'ncc', 'www', 'sco',
                 'soi', 'soiplaintext.html']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'soiplaintext.html'),
            'formatted_file': os.path.join(
                results_dir, 'bom.soi.monthly.csv'),
            'format_func': format_bom_soi,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [BOM_HTTP_ROOT, 'climate', 'mjo', 'graphics',
                 'rmm.74toRealtime.txt']),
            'download_kwargs': {'use_request_api': True},
            'raw_file': os.path.join(data_dir, 'rmm.74toRealtime.txt'),
            'formatted_file': {
                'RMM1': os.path.join(
                    results_dir, 'bom.rmm1.daily.csv'),
                'RMM2': os.path.join(
                    results_dir, 'bom.rmm2.daily.csv'),
                'phase': os.path.join(
                    results_dir, 'bom.mjo_phase.daily.csv'),
                'amplitude': os.path.join(
                    results_dir, 'bom.mjo_amplitude.daily.csv')
                               },
            'format_func': format_bom_mjo,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [BOM_HTTP_ROOT, 'climate', 'enso', 'iod_1.txt']),
            'download_kwargs': {'use_request_api': True},
            'raw_file': os.path.join(data_dir, 'iod_1.txt'),
            'formatted_file': os.path.join(
                results_dir, 'bom.dmi.weekly.csv'),
            'format_func': format_bom_dmi,
            'format_func_kwargs': {}
        },
        {
            'source_url': '/'.join(
                [NOAA_CPC_HTTPS_ROOT, 'products', 'precip', 'CWlink',
                 'pna', 'norm.mon.pna.wg.jan1950-current.ascii.table']),
            'download_kwargs': {},
            'raw_file': os.path.join(
                data_dir, 'norm.mon.pna.wg.jan1950-current.ascii.table'),
            'formatted_file': os.path.join(
                results_dir, 'cpc.pna4.monthly.csv'),
            'format_func': format_cpc_pna4,
            'format_func_kwargs': {}
        }
    ]

    prepare_directories(data_dir=data_dir, results_dir=results_dir)

    for index in indices_to_download:

        download_and_format_index(**index)


if __name__ == '__main__':
    main()
