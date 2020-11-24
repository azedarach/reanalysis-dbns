"""
Fit Bayes regression model to monthly JRA-55 indices.
"""

# License: MIT

from __future__ import absolute_import, division


import argparse
import os

import numpy as np
import pandas as pd
import scipy.stats as ss

import reanalysis_dbns.models as rdm
import reanalysis_dbns.utils as rdu


PROJECT_DIR = os.path.abspath(
    os.path.realpath(
        os.path.join(os.path.dirname(__file__), '..')))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')

JRA55_RESULTS_DIR = os.path.join(RESULTS_DIR, 'jra55')
JRA55_INDICES_DIR = os.path.join(JRA55_RESULTS_DIR, 'indices')

REANALYSIS = 'jra55'
BASE_PERIOD_STR = '19790101_20011230'
INDICES_FREQUENCY = 'monthly'

INDICES_INPUT_FILES = {
    'AO': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.ao.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'DMI': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.dmi.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'MEI': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.mei.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'NHTELE1': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.nhtele1.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'NHTELE2': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.nhtele2.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'NHTELE3': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.nhtele3.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'NHTELE4': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.nhtele4.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'PNA': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.pna.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'PSA1': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.psa1_index.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'PSA2': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.psa2_index.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'SAM': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.sam.{}.csv'.format(
            REANALYSIS, BASE_PERIOD_STR, INDICES_FREQUENCY)),
    'RMM1': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.rmm1.daily.csv'.format(REANALYSIS, BASE_PERIOD_STR)),
    'RMM2': os.path.join(
        JRA55_INDICES_DIR, 'csv',
        '{}.{}.rmm2.daily.csv'.format(REANALYSIS, BASE_PERIOD_STR))
}

FIT_PERIOD = [pd.Timestamp('1960-01-01'), pd.Timestamp('2005-11-30')]
INDICES_TO_INCLUDE = ['AO', 'DMI', 'MEI', 'NHTELE1', 'NHTELE2', 'NHTELE3',
                      'NHTELE4', 'PNA', 'PSA1', 'PSA2', 'SAM', 'RMM1', 'RMM2']


def get_nu_sq_value(a_tau, b_tau, R=4, q=0.99):
    """Calculate value of SNR hyperparameter."""
    df = 2 * a_tau
    t_crit = ss.t.ppf(0.5 * (1 - q), df)
    return a_tau * b_tau * R**2 / t_crit**2


def get_rmm_magnitude_and_phase(rmm1, rmm2):
    """Calculate continuous measures of MJO amplitude and phase."""

    mjo_amplitude = np.sqrt(rmm1**2 + rmm2**2)
    mjo_phase = np.arctan2(rmm2, rmm1)

    return mjo_amplitude, mjo_phase


def read_indices_data(input_files, standardize=True, detrend=True,
                      trend_order=1):
    """Read index values from file."""

    indices = {}
    for index in input_files:

        index_ds = pd.read_csv(
            input_files[index], comment='#', index_col='date',
            parse_dates=True)

        mask = ((index_ds.index >= FIT_PERIOD[0]) &
                (index_ds.index <= FIT_PERIOD[1]))

        index_ds = index_ds[mask]

        index_ds = index_ds.resample('1MS').mean()

        indices[index] = index_ds['value'].rename(index)

    # Calculate MJO phase and amplitude.
    rmm1_input_file = input_files['RMM1']
    rmm2_input_file = input_files['RMM2']

    rmm1 = pd.read_csv(rmm1_input_file, comment='#',
                       index_col='date', parse_dates=True)
    rmm2 = pd.read_csv(rmm2_input_file, comment='#',
                       index_col='date', parse_dates=True)

    mask = (rmm1.index >= FIT_PERIOD[0]) & (rmm1.index <= FIT_PERIOD[1])
    rmm1 = rmm1[mask]

    mask = (rmm2.index >= FIT_PERIOD[0]) & (rmm2.index <= FIT_PERIOD[1])
    rmm2 = rmm2[mask]

    input_amp, input_phase = get_rmm_magnitude_and_phase(
        rmm1['value'], rmm2['value'])

    amp = input_amp.resample('1MS').mean()
    phase = input_phase.resample('1MS').mean()

    indices['RMM_amplitude'] = amp.rename('RMM_amplitude')
    indices['RMM_phase'] = phase.rename('RMM_phase')

    if detrend:

        for index in indices:
            indices[index] = rdu.remove_polynomial_trend(
                indices[index], trend_order=trend_order)

        # MJO phase and amplitude calculated from detrended RMM1 and RMM2
        rmm1 = pd.read_csv(rmm1_input_file, comment='#',
                           index_col='date', parse_dates=True)
        rmm2 = pd.read_csv(rmm2_input_file, comment='#',
                           index_col='date', parse_dates=True)

        mask = (rmm1.index >= FIT_PERIOD[0]) & (rmm1.index <= FIT_PERIOD[1])
        rmm1 = rmm1[mask]

        mask = (rmm2.index >= FIT_PERIOD[0]) & (rmm2.index <= FIT_PERIOD[1])
        rmm2 = rmm2[mask]

        rmm1 = rdu.remove_polynomial_trend(rmm1, trend_order=trend_order)
        rmm2 = rdu.remove_polynomial_trend(rmm2, trend_order=trend_order)

        input_amp, input_phase = get_rmm_magnitude_and_phase(
            rmm1['value'], rmm2['value'])

        amp = input_amp.resample('1MS').mean()
        phase = input_phase.resample('1MS').mean()

        indices['RMM_amplitude'] = amp.rename('RMM_amplitude')
        indices['RMM_phase'] = phase.rename('RMM_phase')

    if standardize:

        for index in indices:
            indices[index] = rdu.standardize_time_series(
                indices[index])

    return indices


def get_season_data(data, season='ALL'):
    """Get data within given season."""

    if season == 'ALL':
        return data

    if season == 'DJF':
        return data[(data.index.month == 12) |
                    (data.index.month == 1) |
                    (data.index.month == 2)]

    if season == 'MAM':
        return data[(data.index.month == 3) |
                    (data.index.month == 4) |
                    (data.index.month == 5)]

    if season == 'JJA':
        return data[(data.index.month == 6) |
                    (data.index.month == 7) |
                    (data.index.month == 8)]

    if season == 'SON':
        return data[(data.index.month == 9) |
                    (data.index.month == 10) |
                    (data.index.month == 11)]

    raise ValueError("Unrecognized season '%r'" % season)


def estimate_thinning_parameter(model_spec, a_tau, b_tau, nu_sq, data=None,
                                n_chains=4, n_iter=2500, warmup=None,
                                thin=1, max_terms=4, n_jobs=-1,
                                reduction_frac=0.01,
                                random_seed=None):
    """Estimate required thinning parameter."""

    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    samples = model.sample_structures_posterior(
        model_spec, data=data, max_terms=max_terms,
        n_chains=n_chains, n_iter=n_iter,
        warmup=warmup, thin=thin, n_jobs=n_jobs,
        generate_posterior_predictive=False,
        random_state=random_seed)

    convergence_rate = rdm.structure_sample_convergence_rate(
        samples.posterior, max_nonzero=max_terms,
        combine_chains=True, sparse=True)

    return int(np.ceil(np.log(reduction_frac) /
                       np.log(np.real(convergence_rate))))


def fit_stepwise_regression(model_spec, a_tau, b_tau, nu_sq, data=None,
                            n_chains=4, n_iter=2000, max_terms=4,
                            warmup=None, thin=1, restart_file=None,
                            n_jobs=-1, random_seed=None):
    """Fit stepwise Bayes regression."""

    model = rdm.StepwiseBayesRegression(a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    samples = model.sample_structures_posterior(
        model_spec, data=data, max_terms=max_terms,
        n_chains=n_chains, n_iter=n_iter, warmup=warmup,
        thin=thin, n_jobs=n_jobs,
        restart_file=restart_file,
        generate_posterior_predictive=True,
        random_state=random_seed)

    return samples


def get_fit_output_file(output_dir, outcome, max_lag, season,
                        max_terms, a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                        thin=1, restart_number=None):
    """Get filename for output."""

    prefix = '.'.join([REANALYSIS, BASE_PERIOD_STR, season,
                       INDICES_FREQUENCY])
    suffix = '.'.join(['stepwise_bayes_regression',
                       'max_lag-{:d}'.format(max_lag),
                       'max_terms-{:d}'.format(max_terms),
                       'a_tau-{:.3f}'.format(a_tau),
                       'b_tau-{:.3f}'.format(b_tau),
                       'nu_sq-{:.3f}'.format(nu_sq),
                       'thin-{:d}'.format(thin),
                       outcome, 'posterior_samples'])

    if restart_number is not None and restart_number > 0:
        suffix = '.'.join([suffix, 'restart-{:d}'.format(restart_number)])

    filename = '.'.join([prefix, suffix, 'nc'])

    return os.path.join(output_dir, filename)


def get_restart_file(restart_number, *args, **kwargs):
    """Get name of file to use for restarting sampling."""

    if restart_number is None or restart_number == 0:
        raise ValueError(
            'Run is not a restart of a previous run')

    if restart_number == 1:
        return get_fit_output_file(*args, restart_number=None, **kwargs)

    return get_fit_output_file(*args, restart_number=(restart_number - 1),
                               **kwargs)


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Fit Bayes regression model to reanalysis indices.')

    parser.add_argument('outcome', help='outcome variable')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('--a-tau', dest='a_tau', type=float,
                        default=1.0, help='precision shape hyperparameter')
    parser.add_argument('--b-tau', dest='b_tau', type=float,
                        default=1.0, help='precision scale hyperparameter')
    parser.add_argument('-R', dest='R', type=float,
                        default=4.0, help='coefficient range')
    parser.add_argument('-q', dest='q', type=float,
                        default=0.99, help='coefficient interval probability')
    parser.add_argument('--season', dest='season',
                        choices=['ALL', 'DJF', 'MAM', 'JJA', 'SON'],
                        default='ALL', help='season to fit')
    parser.add_argument('--max-lag', dest='max_lag', type=int,
                        default=6, help='maximum lag')
    parser.add_argument('--max-terms', dest='max_terms', type=int,
                        default=4, help='maximum number of terms')
    parser.add_argument('--n-chains', dest='n_chains', type=int,
                        default=4, help='number of chains')
    parser.add_argument('--n-iter', dest='n_iter', type=int,
                        default=2000, help='number of iterations')
    parser.add_argument('--warmup', dest='warmup', type=int,
                        default=None, help='number of warmup iterations')
    parser.add_argument('--thin', dest='thin', type=int,
                        default=None, help='thinning parameter')
    parser.add_argument('--restart-number', dest='restart_number',
                        type=int, default=None, help='restart number')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        default=-1, help='number of jobs')
    parser.add_argument('--random-seed', dest='random_seed', type=int,
                        default=None, help='random seed')

    return parser.parse_args()


def main():
    """Fit Bayes regression model to reanalysis indices."""

    args = parse_cmd_line_args()

    indices = read_indices_data(
        INDICES_INPUT_FILES, standardize=True, detrend=True)

    indices_data = pd.DataFrame(
        {index: indices[index] for index in INDICES_TO_INCLUDE})

    max_lag = args.max_lag
    data_vars = [(v, -lag) for v in INDICES_TO_INCLUDE
                 for lag in range(max_lag + 1)]

    data = rdu.construct_lagged_data(
        data_vars, data=indices_data, presample_length=args.max_lag)

    data = get_season_data(data, season=args.season)

    predictors = ['{}_lag_{}'.format(index, lag)
                  for index in INDICES_TO_INCLUDE
                  for lag in range(1, max_lag + 1)]

    # The complete set of models that we consider include all main
    # effects from lagged variables.
    model_spec = '{} ~ {}'.format(
        args.outcome, ' + '.join(predictors))

    # Get hyperparameter values.
    a_tau = args.a_tau
    b_tau = args.b_tau
    nu_sq = get_nu_sq_value(a_tau, b_tau, R=args.R, q=args.q)

    # First run several short chains to estimate tuning parameters.
    if args.thin is not None:
        thin = args.thin
    else:
        thin = estimate_thinning_parameter(
            model_spec, a_tau, b_tau, nu_sq, data=data,
            n_chains=args.n_chains, max_terms=args.max_terms,
            n_jobs=args.n_jobs, random_seed=0)

    # Then, run full fits.
    if args.restart_number is None or args.restart_number == 0:
        warmup = args.warmup
        restart_file = None
    else:
        warmup = 0
        restart_file = get_restart_file(
            args.restart_number, args.output_dir, args.outcome,
            args.max_lag, args.season,
            args.max_terms, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
            thin=thin)

    samples = fit_stepwise_regression(
        model_spec, a_tau, b_tau, nu_sq, data=data,
        n_chains=args.n_chains, n_iter=args.n_iter,
        max_terms=args.max_terms, thin=thin,
        warmup=warmup,
        restart_file=restart_file,
        n_jobs=args.n_jobs, random_seed=args.random_seed)

    output_file = get_fit_output_file(
        args.output_dir, args.outcome, args.max_lag, args.season,
        args.max_terms, a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
        thin=thin, restart_number=args.restart_number)

    samples.to_netcdf(output_file)


if __name__ == '__main__':
    main()
