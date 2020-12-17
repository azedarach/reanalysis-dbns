"""
Generate trace plots of indicator variables.
"""

# License: MIT

from __future__ import absolute_import, division


import argparse
import glob
import itertools
import re
import os

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


BASE_PERIOD = [pd.Timestamp('1979-01-01'), pd.Timestamp('2001-12-30')]

INDEX_COLORS = {
    'AO': '#f2d21d',
    'NHTELE2': '#fb9a99',
    'DMI': '#1f78b4',
    'MEI': '#a6cee3',
    'NHTELE4': '#fdbf6f',
    'NHTELE1': '#b15928',
    'PNA': '#ff7f00',
    'PSA1': '#cab2d6',
    'PSA2': '#6a3d9a',
    'RMM1': '#33a02c',
    'RMM2': '#b2df8a',
    'SAM': '#83858a',
    'NHTELE3': '#e31a1c'
}

INDEX_NAMES = {
    'AO': 'AO',
    'NHTELE2': 'AR',
    'DMI': 'DMI',
    'MEI': 'MEI',
    'NHTELE4': 'NAO$^-$',
    'NHTELE1': 'NAO$^+$',
    'PNA': 'PNA',
    'PSA1': 'PSA1',
    'PSA2': 'PSA2',
    'RMM1': 'RMM1',
    'RMM2': 'RMM2',
    'SAM': 'SAM',
    'NHTELE3': 'SCAND'
}


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


def get_variables_and_lags(fit):
    """Get variables and lags in fit."""

    indicator_pattern = 'i_([a-zA-Z0-9]+)_lag_([0-9]+)'

    summary = az.summary(fit.posterior, var_names=indicator_pattern,
                         filter_vars='regex')

    indicator_vars = np.array([re.match(indicator_pattern, i)[1]
                               for i in summary.index])
    indicator_lags = np.array([int(re.match(indicator_pattern, i)[2])
                               for i in summary.index])

    unique_vars = np.array(sorted(np.unique(indicator_vars),
                                  key=lambda v: INDEX_NAMES[v]))
    unique_lags = np.unique(indicator_lags)

    return unique_vars, unique_lags


def calculate_chain_estimates(fit, batch_size=None):
    """Calculate estimates from individual chains."""

    unique_vars, unique_lags = get_variables_and_lags(fit)

    n_lags = unique_lags.shape[0]

    samples = xr.concat(
        [fit.warmup_posterior, fit.posterior], dim='draw')

    n_chains = samples.sizes['chain']
    n_draws = samples.sizes['draw']

    if batch_size is None:
        batch_size = int((n_draws // 2) // 30)

    n_batches = int(np.ceil((n_draws // 2) / batch_size))

    batches = np.empty((n_batches,))
    post_probs = {v: np.empty((n_lags, n_chains, n_batches))
                  for v in unique_vars}
    for q in range(1, n_batches + 1):
        for k in range(n_chains):

            batch_stop = min(n_draws, 2 * q * batch_size)

            batch_samples = samples.isel(chain=slice(k, k + 1),
                                         draw=slice(0, batch_stop))

            warmup = int(batch_stop // 2)
            batch_kept = batch_samples.isel(draw=slice(warmup, None))

            batch_summary = az.summary(batch_kept, kind='stats')

            batches[q - 1] = batch_stop
            for v in unique_vars:
                for i, lag in enumerate(unique_lags):

                    ind = 'i_{}_lag_{}'.format(v, lag)
                    mask = batch_summary.index == ind

                    post_probs[v][i, k, q - 1] = batch_summary['mean'][mask][0]

    return batches, post_probs


def calculate_combined_estimates(fit, batch_size=None):
    """Calculate estimates by combining all chains."""

    unique_vars, unique_lags = get_variables_and_lags(fit)

    n_lags = unique_lags.shape[0]

    samples = xr.concat(
        [fit.warmup_posterior, fit.posterior], dim='draw')

    n_draws = samples.sizes['draw']

    if batch_size is None:
        batch_size = int((n_draws // 2) // 30)

    n_batches = int(np.ceil((n_draws // 2) / batch_size))

    batches = np.empty((n_batches,))
    post_probs = {v: np.empty((n_lags, n_batches)) for v in unique_vars}
    for q in range(1, n_batches + 1):

        batch_stop = min(n_draws, 2 * q * batch_size)

        batch_samples = samples.isel(draw=slice(0, batch_stop))

        warmup = int(batch_stop // 2)
        batch_kept = batch_samples.isel(draw=slice(warmup, None))

        batch_summary = az.summary(batch_kept)

        batches[q - 1] = batch_stop
        for v in unique_vars:
            for i, lag in enumerate(unique_lags):

                ind = 'i_{}_lag_{}'.format(v, lag)
                mask = batch_summary.index == ind

                post_probs[v][i, q - 1] = batch_summary['mean'][mask][0]

    return batches, post_probs


def plot_indicator_probability_trace(fit, batch_size=None):
    """Plot posterior probability trace."""

    unique_vars, unique_lags = get_variables_and_lags(fit)

    chain_batches, chain_post_probs = calculate_chain_estimates(
        fit, batch_size=batch_size)
    combined_batches, combined_post_probs = calculate_combined_estimates(
        fit, batch_size=batch_size)

    n_vars = len(unique_vars)
    n_chains = fit.posterior.sizes['chain']
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols,
                           figsize=(14 * n_cols, 6 * n_rows),
                           squeeze=False)
    fig.subplots_adjust(hspace=0.6, wspace=0.17)

    row_index = 0
    col_index = 0

    for i, v in enumerate(unique_vars):

        c = INDEX_COLORS[v]

        markers = itertools.cycle(('o', 'x', 's', 'd', '+', 'v'))

        if i == n_vars - 1:
            col_index = 1
            ax[n_rows - 1, 0].set_visible(False)
            ax[n_rows - 1, -1].set_visible(False)

        for j, lag in enumerate(unique_lags):

            m = next(markers)

            for k in range(n_chains):
                ax[row_index, col_index].semilogy(
                    chain_batches, chain_post_probs[v][j, k],
                    color=c, marker=m, ls='-', alpha=0.2)

            ax[row_index, col_index].semilogy(
                combined_batches, combined_post_probs[v][j],
                color=c, marker=m, ls='-', lw=2,
                label='{}, lag {:d}'.format(INDEX_NAMES[v], lag))

        ax[row_index, col_index].grid(ls='--', color='gray', alpha=0.5)
        ax[row_index, col_index].legend(ncol=3, loc='upper center',
                                        fontsize=18,
                                        bbox_to_anchor=(0.5, 1.4))

        ax[row_index, col_index].tick_params(axis='both', labelsize=18)

        ax[row_index, col_index].set_ylabel(
            'Estimated probability', fontsize=20)
        ax[row_index, col_index].set_xlabel(
            'Total number of draws', fontsize=20)

        col_index += 1
        if col_index == n_cols:
            col_index = 0
            row_index += 1

    return fig


def plot_trace_plots(reanalysis, output_dir,
                     outcome, model_files,
                     a_tau=1.0, b_tau=1.0, nu_sq=1.0,
                     batch_size=None):
    """Generate trace plots of indicator variables."""

    n_model_files = len(model_files)

    if n_model_files == 1:
        fit = az.from_netcdf(model_files[0])
    else:

        fit = az.from_netcdf(model_files[0])

        for i in range(1, n_model_files):
            restart_fit = az.from_netcdf(model_files[i])

            fit.posterior = xr.concat([fit.posterior, restart_fit.posterior],
                                      dim='draw')
            fit.sample_stats = xr.concat([fit.sample_stats,
                                          restart_fit.sample_stats],
                                         dim='draw')

    plot_basename = os.path.basename(model_files[0]).replace('.nc', '')
    plot_basename = os.path.join(output_dir, plot_basename)

    title_str = \
        r'{} {} $a_\tau = {:.3f}$, $b_\tau = {:.3f}$, $\nu^2 = {:.3f}$'.format(
            reanalysis.upper(), INDEX_NAMES[outcome], a_tau, b_tau, nu_sq)

    fig = plot_indicator_probability_trace(  # noqa: F841
        fit, batch_size=batch_size)

    plot_filename = '.'.join(
        [plot_basename, 'indicator_posterior_probability_trace_combined',
         'pdf'])

    plt.suptitle(title_str, fontsize=24, y=0.95)

    plt.savefig(plot_filename, bbox_inches='tight', facecolor='white')
    plt.savefig(plot_filename.replace('.pdf', '.png'),
                bbox_inches='tight', facecolor='white')

    plt.close()


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Generate trace plots of indicator variables.')

    parser.add_argument('reanalysis', help='reanalysis name')
    parser.add_argument('models_dir', help='directory containing fits')
    parser.add_argument('output_dir', help='directory to write fit output to')
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

    return parser.parse_args()


def main():
    """Generate trace plots of indicator variables."""

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

        plot_trace_plots(
            args.reanalysis, args.output_dir,
            outcome, outcome_model_files,
            a_tau=a_tau, b_tau=b_tau, nu_sq=nu_sq,
            max_terms=args.max_terms)


if __name__ == '__main__':
    main()
