"""
Generate plots of convergence diagnostics.
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


def get_model_fit_pattern(outcome=None, a_tau=1.0, b_tau=1.0, nu_sq=20.0,
                          max_terms=10, max_lag=6, season='ALL',
                          frequency='monthly',
                          base_period=BASE_PERIOD):
    """Get fit output filename pattern."""

    base_period_str = '{}_{}'.format(
        base_period[0].strftime('%Y%m%d'),
        base_period[1].strftime('%Y%m%d'))

    outcome_str = outcome if outcome is not None else '[A-Za-z0-9]+'
    prefix = '.+\.{}\.{}\.{}'.format(base_period_str, season, frequency)
    suffix = '\.'.join(['stepwise_bayes_regression',
                        'max_lag-{:d}'.format(max_lag),
                        'max_terms-{:d}'.format(max_terms),
                        'a_tau-{:.3f}'.format(a_tau),
                        'b_tau-{:.3f}'.format(b_tau),
                        'nu_sq-{:.3f}'.format(nu_sq),
                        'thin-[0-9]+', '(' + outcome_str + ')',
                        'posterior_samples'])

    return '\.'.join([prefix, suffix]) + '(\.restart-[0-9]+)?' + '\.nc'


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


def plot_pval_diagnostics(chi2_samples, chi2_pvals, ks_samples, ks_pvals,
                          level=0.05):
    """Plot traces of p-values."""

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)

    ax[0, 0].plot(chi2_samples, chi2_pvals, '-')
    ax[0, 0].axhline(level, ls='-.', color='k')

    ax[0, 0].grid(ls='--', color='gray', alpha=0.5)

    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].tick_params(axis='both', labelsize=13)

    ax[0, 0].set_xlabel('Number of draws', fontsize=14)
    ax[0, 0].set_ylabel('$p$-value', fontsize=14)
    ax[0, 0].set_title(r'$\chi^2$ test', fontsize=15)

    n_comparisons = ks_pvals.shape[1]

    for i in range(n_comparisons):
        ax[0, 1].plot(ks_samples, ks_pvals[:, i])

    ax[0, 1].axhline(level, ls='-.', color='k')

    ax[0, 1].grid(ls='--', color='gray', alpha=0.5)

    ax[0, 1].set_ylim(0, 1.05)
    ax[0, 1].tick_params(axis='both', labelsize=13)

    ax[0, 1].set_xlabel('Number of draws', fontsize=14)
    ax[0, 1].set_ylabel('$p$-value', fontsize=14)
    ax[0, 1].set_title(r'KS test', fontsize=15)

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

    try:
        # First plot diagnostics without splitting chains.
        _, chi2_pvals, chi2_samples = \
            rdm.structure_sample_chi2_convergence_diagnostics(
                fit, batch=True, split=False)

        _, ks_pvals, ks_samples = \
            rdm.structure_sample_ks_convergence_diagnostics(
                fit, batch=True, split=False)
    except:
        return

    fig = plot_pval_diagnostics(chi2_samples, chi2_pvals,
                                ks_samples, ks_pvals)

    plt.suptitle(outcome, fontsize=16)

    output_file = '.'.join([plot_basename, 'convergence_diagnostics.pdf'])
    plt.savefig(output_file, bbox_inches='tight', facecolor='white')
    plt.savefig(output_file.replace('.pdf', '.png'),
                bbox_inches='tight', facecolor='white')

    plt.close()

    try:
        # Then generate plots with split diagnostics.
        _, chi2_pvals, chi2_samples = \
            rdm.structure_sample_chi2_convergence_diagnostics(
                fit, batch=True, split=True)

        _, ks_pvals, ks_samples = \
            rdm.structure_sample_ks_convergence_diagnostics(
                fit, batch=True, split=True)
    except:
        return

    fig = plot_pval_diagnostics(chi2_samples, chi2_pvals,
                                ks_samples, ks_pvals)

    plt.suptitle(outcome + ' (split)', fontsize=16)

    output_file = '.'.join([plot_basename, 'split_convergence_diagnostics.pdf'])
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
