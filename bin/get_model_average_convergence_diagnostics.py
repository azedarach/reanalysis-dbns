"""
Generate plots of model averaged convergence diagnostics.
"""

# License: MIT


from __future__ import absolute_import, division

import argparse
import glob
import os
import re

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import reanalysis_dbns.models as rdm


BASE_PERIOD = [pd.Timestamp('1979-01-01'), pd.Timestamp('2001-12-30')]


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

    for outcome in matching_files:
        matching_files[outcome] = sorted(matching_files[outcome])

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


def calculate_edge_probability(fit, batch_size=None):
    """Calculate edge posterior probabilities."""

    unique_vars, unique_lags = get_variables_and_lags(fit)

    n_lags = unique_lags.shape[0]

    samples = xr.concat(
        [fit.warmup_posterior, fit.posterior], dim='draw')

    n_draws = samples.sizes['draw']

    if batch_size is None:
        batch_size = int((n_draws // 2) // 20)

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


def plot_pval_diagnostics(batch_sizes, estimates,
                          chi2_results, ks_results, level=0.05):
    """Plot traces of p-values."""

    term_pattern = re.compile(r'([a-zA-Z0-9]+)_lag_([0-9]+)')

    terms = [t for t in chi2_results]

    predictors = np.array(
        sorted(np.unique([term_pattern.search(t)[1] for t in terms]),
               key=lambda v: INDEX_NAMES[v]))
    lags = np.unique([term_pattern.search(t)[2] for t in terms])

    n_predictors = len(predictors)
    n_lags = len(lags)

    fig, ax = plt.subplots(n_predictors, n_lags,
                           figsize=(7 * n_lags, 5 * n_predictors),
                           squeeze=False)
    fig.subplots_adjust(hspace=0.3)

    for i, p in enumerate(predictors):
        for j, lag in enumerate(lags):

            term = '{}_lag_{}'.format(p, lag)

            post_prob = estimates[p][j]

            ax[i, j].plot(batch_sizes, post_prob, 'k-', label=r'$\hat{\pi}$')

            chi2_samples = chi2_results[term][2]
            chi2_pvals = chi2_results[term][1]

            ax[i, j].plot(chi2_samples, chi2_pvals, '-',
                          label=r'$\chi^2$ test')

            ks_samples = ks_results[term][2]
            ks_pvals = ks_results[term][1]

            n_comparisons = ks_pvals.shape[1]

            for k in range(n_comparisons):
                if k == 0:
                    ax[i, j].plot(ks_samples, ks_pvals[:, k], ls='--',
                                  alpha=0.7, label='KS test')
                else:
                    ax[i, j].plot(ks_samples, ks_pvals[:, k], ls='--',
                                  alpha=0.7)

            ax[i, j].axhline(level, ls='-.', color='k')

            ax[i, j].grid(ls='--', color='gray', alpha=0.5)
            ax[i, j].legend()

            ax[i, j].set_ylim(0, 1.05)
            ax[i, j].tick_params(axis='both', labelsize=13)

            ax[i, j].set_xlabel('Number of draws', fontsize=14)
            ax[i, j].set_ylabel('$p$-value', fontsize=14)
            ax[i, j].set_title(
                r'{} lag {}'.format(INDEX_NAMES[p], lag), fontsize=15)

    return fig


def plot_convergence_diagnostics(output_dir, outcome, model_files):
    """Generate plots of convergence diagnostics."""

    n_model_files = len(model_files)

    if n_model_files == 1:
        fit = az.from_netcdf(model_files[0])
    else:

        fit = az.from_netcdf(model_files[0])

        for i in range(1, n_model_files):
            restart_fit = az.from_netcdf(model_files[i])

            fit.posterior = xr.concat([fit.posterior, restart_fit.posterior],
                                      dim='draw')

    plot_basename = os.path.basename(model_files[0]).replace('.nc', '')
    plot_basename = os.path.join(output_dir, plot_basename)

    # Calculate posterior probabilities for each edge.
    batch_sizes, post_probs = calculate_edge_probability(fit)

    try:
        # First plot diagnostics without splitting chains.
        indicator_chi2_results = \
            rdm.structure_sample_marginal_chi2(
                fit, batch=True, split=False)

        indicator_ks_results = \
            rdm.structure_sample_marginal_ks(
                fit, batch=True, split=False)
    except ValueError:
        return

    fig = plot_pval_diagnostics(  # noqa: F841
        batch_sizes, post_probs,
        indicator_chi2_results, indicator_ks_results)

    plt.suptitle(outcome, fontsize=16, y=0.9)

    output_file = '.'.join(
        [plot_basename, 'model_average_convergence_diagnostics.pdf'])
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.pdf', '.png'),
                bbox_inches='tight', facecolor='white')

    plt.close()

    try:
        # Then generate plots with split diagnostics.
        indicator_chi2_results = \
            rdm.structure_sample_marginal_chi2(
                fit, batch=True, split=True)

        indicator_ks_results = \
            rdm.structure_sample_marginal_ks(
                fit, batch=True, split=True)
    except ValueError:
        return

    fig = plot_pval_diagnostics(  # noqa: F841
        batch_sizes, post_probs,
        indicator_chi2_results, indicator_ks_results)

    plt.suptitle(outcome + ' (split)', fontsize=16)

    output_file = '.'.join(
        [plot_basename,
         'split_model_average_convergence_diagnostics.pdf'])
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.pdf', '.png'),
                bbox_inches='tight', facecolor='white')

    plt.close()


def parse_cmd_line_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description='Generate plots of convergence diagnostics')

    parser.add_argument('models_dir', help='directory containing fits')
    parser.add_argument('output_dir', help='directory to write output to')
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
    """Generate plots of convergence diagnostics."""

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

        plot_convergence_diagnostics(
            args.output_dir, outcome, outcome_model_files)


if __name__ == '__main__':
    main()
