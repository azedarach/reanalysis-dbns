"""
Sample from conditional posterior for parameters.
"""

# License: MIT

from __future__ import absolute_import, division

import argparse
import glob
import re
import os

import arviz as az
import pandas as pd
import xarray as xr

import reanalysis_dbns.models as rdm


BASE_PERIOD = [pd.Timestamp('1979-01-01'), pd.Timestamp('2001-12-30')]


def get_model_fit_pattern(outcome=None, a_tau=1.0, b_tau=1.0, nu_sq=20.0,
                          max_terms=10, max_lag=6, season='ALL',
                          frequency='monthly',
                          base_period=BASE_PERIOD):
    """Get fit output filename pattern."""

    base_period_str = '{}_{}'.format(
        base_period[0].strftime('%Y%m%d'),
        base_period[1].strftime('%Y%m%d'))

    outcome_str = outcome if outcome is not None else '[A-Za-z0-9]+'
    prefix = r'.+\.{}\.{}\.{}'.format(base_period_str, season, frequency)
    suffix = r'\.'.join(['stepwise_bayes_regression',
                         'max_lag-{:d}'.format(max_lag),
                         'max_terms-{:d}'.format(max_terms),
                         'a_tau-{:.3f}'.format(a_tau),
                         'b_tau-{:.3f}'.format(b_tau),
                         'nu_sq-{:.3f}'.format(nu_sq),
                         'thin-[0-9]+', '(' + outcome_str + ')',
                         'posterior_samples'])

    return r'\.'.join([prefix, suffix]) + r'(\.restart-[0-9]+)?' + r'\.nc'


def get_fit_output_files(models_dir, model_pattern):
    """Get fit output files matching pattern."""

    all_files = sorted(glob.glob(os.path.join(models_dir, '*')))

    pattern = re.compile(model_pattern)

    matching_files = {}
    for f in all_files:
        match = pattern.search(f)

        if not match:
            continue

        outcome = match[1]

        if outcome in matching_files:
            matching_files[outcome].append(f)
        else:
            matching_files[outcome] = [f]

    return matching_files


def get_posterior_mode_fit(output_dir, outcome, model_files,
                           a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                           n_chains=4, n_iter=2000,
                           thin=1, warmup=None, n_jobs=-1,
                           random_seed=0):
    """Sample from the conditional posterior distribution."""

    n_model_files = len(model_files)

    if n_model_files == 1:
        fit = az.from_netcdf(model_files[0])
    else:

        fit = az.from_netcdf(model_files[0])

        for i in range(1, n_model_files):
            restart_fit = az.from_netcdf(model_files[i])

            fit.posterior = xr.concat([fit.posterior, restart_fit.posterior],
                                      dim='draw')
            fit.sample_stats = xr.concat(
                [fit.sample_stats, restart_fit.sample_stats],
                dim='draw')

        # Extend warmup to cover first half of all combined
        # samples.
        n_warmup = fit.warmup_posterior.sizes['draw']
        n_draws = fit.posterior.sizes['draw']
        n_total = n_warmup + n_draws
        n_keep = n_total // 2

        fit.posterior = fit.posterior.isel(
            draw=slice(n_keep - n_warmup, None))
        fit.sample_stats = fit.sample_stats.isel(
            draw=slice(n_keep - n_warmup, None))

    output_basename = os.path.basename(model_files[0]).replace('.nc', '')
    output_basename = os.path.join(output_dir, output_basename)

    data = {f: fit.constant_data[f].data
            for f in fit.constant_data.data_vars}
    for f in fit.observed_data.data_vars:
        data[f] = fit.observed_data[f].data

    posterior_mode_draw = fit.sample_stats['lp'].argmax(
        dim=['chain', 'draw'])

    posterior_mode = fit.posterior['k'].isel(posterior_mode_draw)

    posterior_mode_terms = []
    for i, ki in enumerate(posterior_mode.data):
        if ki > 0:
            posterior_mode_terms.append(posterior_mode['term'].values[i])

    posterior_model_spec = '{} ~ {}'.format(
        outcome, ' + '.join(posterior_mode_terms))

    print('* Outcome: {}'.format(outcome))
    for f in model_files:
        print('\t- Source: ', f)
    print('\t- Posterior mode structure: ', posterior_model_spec)

    model = rdm.StepwiseBayesRegression(
        a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq)

    posterior_mode_prior_samples = model.sample_parameters_prior(
        posterior_model_spec, data=data,
        n_chains=n_chains, n_iter=n_iter, n_jobs=n_jobs,
        generate_prior_predictive=True,
        random_state=random_seed)

    output_file = '.'.join([output_basename, 'posterior_mode',
                            'parameter_prior_samples', 'nc'])

    posterior_mode_prior_samples.to_netcdf(output_file)

    posterior_mode_posterior_samples = model.sample_parameters_posterior(
        posterior_model_spec, data=data,
        n_chains=n_chains, n_iter=n_iter, thin=thin,
        generate_posterior_predictive=True, n_jobs=n_jobs,
        warmup=warmup, random_state=random_seed)

    output_file = '.'.join([output_basename, 'posterior_mode',
                            'parameter_posterior_samples', 'nc'])

    posterior_mode_posterior_samples.to_netcdf(output_file)

    return posterior_mode_posterior_samples


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Generate summaries of structure fits.')

    parser.add_argument('models_dir', help='directory containing fits')
    parser.add_argument('output_dir', help='directory to write fit output to')
    parser.add_argument('summary_dir', help='directory to write summary to')
    parser.add_argument('--outcome', dest='outcome', default=None,
                        help='name of outcome variable')
    parser.add_argument('--a-tau', dest='a_tau', type=float,
                        default=1.0, help='precision shape hyperparameter')
    parser.add_argument('--b-tau', dest='b_tau', type=float,
                        default=1.0, help='precision scale hyperparameter')
    parser.add_argument('--nu-sq', dest='nu_sq', type=float,
                        default=1.0, help='SNR hyperparameter')
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
                        default=10000, help='number of iterations')
    parser.add_argument('--thin', dest='thin', type=int, default=1,
                        help='thinning parameter')

    return parser.parse_args()


def main():
    """Sample from conditional posterior for parameters."""

    args = parse_cmd_line_args()

    # Get hyperparameter values.
    a_tau = args.a_tau
    b_tau = args.b_tau
    nu_sq = args.nu_sq

    model_pattern = get_model_fit_pattern(
        outcome=args.outcome,
        a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
        max_terms=args.max_terms, max_lag=args.max_lag,
        season=args.season)

    model_files = get_fit_output_files(
        args.models_dir, model_pattern)

    for outcome in model_files:

        outcome_model_files = model_files[outcome]

        if any(['restart' in f for f in outcome_model_files]):
            # Ensure correct sort order.
            def restart_number(f):
                pattern = re.compile('restart-([0-9]+)')
                match = pattern.search(f)
                if match:
                    return int(match[1])
                return 0
            outcome_model_files = sorted(
                outcome_model_files, key=restart_number)

        print('* Outcome: ', outcome)
        for f in outcome_model_files:
            print('\t- ', f)

        fit = get_posterior_mode_fit(
            args.output_dir, outcome, outcome_model_files,
            a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
            n_chains=args.n_chains, n_iter=args.n_iter,
            thin=args.thin)

        output_basename = os.path.basename(
            outcome_model_files[0]).replace('.nc', '')
        output_basename = os.path.join(args.summary_dir, output_basename)

        summary_file = '.'.join([output_basename, 'posterior_mode',
                                 'parameter_posterior_samples',
                                 'summary', 'csv'])

        summary = az.summary(fit.posterior)

        summary.to_csv(summary_file)


if __name__ == '__main__':
    main()
